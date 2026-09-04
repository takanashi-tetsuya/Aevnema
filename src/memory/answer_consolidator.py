from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
import unicodedata
from typing import Any

from config.prompt_config import (
    INLINE_MEMORY_PROMPT,
    MEMORY_CONSOLIDATION_SYSTEM,
    knowledge_consolidation_query,
    memory_consolidation_prompt,
)
from src.llm.engine import Message

from .identity import PlatformIdentity
from .service import MemoryRoute


INLINE_MEMORY_OPEN = "<assistant_memory>"
INLINE_MEMORY_CLOSE = "</assistant_memory>"
_INLINE_MEMORY_PATTERN = re.compile(
    rf"{re.escape(INLINE_MEMORY_OPEN)}\s*(.*?)\s*{re.escape(INLINE_MEMORY_CLOSE)}",
    re.DOTALL | re.IGNORECASE,
)
_CJK_RUN_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_LATIN_WORD_PATTERN = re.compile(r"[a-z0-9_]{2,}")

KNOWLEDGE_INFERENCE_TYPES = frozenset(
    {
        "causal",
        "motivation",
        "contrast",
        "trait",
        "identity",
        "temporal",
        "relationship",
        "recall_trigger",
        "thematic",
        "evidence_bridge",
    }
)
MIN_KNOWLEDGE_CONFIDENCE = 0.72


def _parse_json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("memory consolidator did not return a JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("memory consolidator response is not an object")
    return payload


@dataclass(slots=True, frozen=True)
class KnowledgeCandidate:
    claim: str
    premise_episode_ids: tuple[int, ...]
    confidence: float
    reason: str
    inference_type: str


@dataclass(slots=True, frozen=True)
class ContextualRecallCandidate:
    """A local retrieval hint, never a factual or semantic claim."""

    anchor_type: str
    anchor_id: int
    target_episode_id: int
    context_query_id: str
    need_query_id: str
    slot_id: str = ""
    source_request_hash: str = ""
    reason: str = "recovered_missing_evidence"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class PrivateCandidate:
    claim: str
    reason: str


@dataclass(slots=True)
class ConsolidationDecision:
    knowledge_candidates: list[KnowledgeCandidate] = field(default_factory=list)
    private_candidates: list[PrivateCandidate] = field(default_factory=list)
    rejected_claims: list[dict[str, str]] = field(default_factory=list)

    @property
    def knowledge_query(self) -> str:
        if not self.knowledge_candidates:
            return ""
        rows = [
            {
                "candidate_inference": item.claim,
                "foreground_premise_episode_ids": list(
                    item.premise_episode_ids
                ),
                "inference_type": item.inference_type,
                "confidence": item.confidence,
            }
            for item in self.knowledge_candidates
        ]
        return knowledge_consolidation_query(rows)

    def as_dict(self) -> dict[str, Any]:
        return {
            "knowledge_candidates": [
                asdict(item) for item in self.knowledge_candidates
            ],
            "private_candidates": [
                asdict(item) for item in self.private_candidates
            ],
            "rejected_claims": list(self.rejected_claims),
            "knowledge_query": self.knowledge_query,
        }


def inline_memory_prompt() -> str:
    """Ask the existing chat call for locally verifiable memory candidates.

    This does not trigger a second model request.  The tagged object is removed
    before the answer reaches the user or conversation journal.
    """

    return INLINE_MEMORY_PROMPT


def derive_evidence_bridge_candidate(
    question: str,
    raw_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a retrieval-only bridge from the reranker's coverage proof.

    The reranker has already paid to map separate answer slots to visible
    Episodes. Reusing that map is more reliable than asking the final roleplay
    model to emit a hidden machine footer. This function never invents a
    factual relation: it only proposes two direct Episodes as a reusable lookup
    pair, and the normal primary plus adversarial growth audits remain required.
    """

    raw = raw_result if isinstance(raw_result, dict) else {}
    domains = raw.get("domains") or {}
    knowledge = domains.get("knowledge") if isinstance(domains, dict) else None
    if not isinstance(knowledge, dict):
        return None
    rerank = knowledge.get("rerank_trace") or {}
    episodes: dict[int, dict[str, Any]] = {}
    for row in list(knowledge.get("evidence_episodes") or []):
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        try:
            episode_id = int(row["id"])
            generation = int(row.get("generation", 0) or 0)
        except (TypeError, ValueError):
            continue
        if generation != 0 or str(
            row.get("evidence_origin", "")
        ).casefold() not in {"source", "direct", "imported"}:
            continue
        episodes[episode_id] = row

    selected: list[tuple[str, dict[str, Any]]] = []
    selection_reason = ""
    coverage_payload = rerank.get("merged_coverage") or rerank.get("initial")
    if isinstance(coverage_payload, dict):
        # Expanded deep queries may report missing optional details even when
        # two original answer slots already have explicit, distinct evidence.
        # A retrieval-only bridge may reuse those proven slots; it does not
        # claim that the entire synthesis was complete.
        used_ids: set[int] = set()
        for slot in list(coverage_payload.get("coverage") or []):
            if not isinstance(slot, dict):
                continue
            slot_text = " ".join(str(slot.get("query", "")).split())[:140]
            if not slot_text:
                continue
            for raw_id in list(slot.get("episode_ids") or []):
                try:
                    episode_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                episode = episodes.get(episode_id)
                if episode is None or episode_id in used_ids:
                    continue
                selected.append((slot_text, episode))
                used_ids.add(episode_id)
                break
            if len(selected) == 2:
                selection_reason = "two explicit reranker coverage slots"
                break

    if len(selected) != 2:
        planning = raw.get("intent_planning") or {}
        raw_plan = planning.get("raw_plan") if isinstance(planning, dict) else {}
        slots = [
            " ".join(str(value).split())[:140]
            for value in list(
                raw_plan.get("answer_slots") or []
                if isinstance(raw_plan, dict)
                else []
            )
            if str(value).strip()
        ][:4]

        def features(value: str) -> set[str]:
            normalized = unicodedata.normalize("NFKC", value).casefold()
            result = {
                f"w:{word}" for word in _LATIN_WORD_PATTERN.findall(normalized)
            }
            for run in _CJK_RUN_PATTERN.findall(normalized):
                result.update(f"c:{char}" for char in run)
                result.update(
                    f"n2:{run[index:index + 2]}"
                    for index in range(max(0, len(run) - 1))
                )
            return result

        slot_features = [features(value) for value in slots]
        episode_rows = list(episodes.values())
        episode_features = [
            features(str(row.get("text", ""))) for row in episode_rows
        ]

        def coverage_score(query_features: set[str], row_features: set[str]) -> float:
            if not query_features:
                return 0.0
            return len(query_features.intersection(row_features)) / len(query_features)

        best: tuple[float, int, int, int, int] | None = None
        for left_slot_index in range(len(slots)):
            for right_slot_index in range(left_slot_index + 1, len(slots)):
                for left_episode_index, left_row_features in enumerate(
                    episode_features
                ):
                    left_score = coverage_score(
                        slot_features[left_slot_index], left_row_features
                    )
                    if left_score < 0.35:
                        continue
                    for right_episode_index, right_row_features in enumerate(
                        episode_features
                    ):
                        if right_episode_index == left_episode_index:
                            continue
                        right_score = coverage_score(
                            slot_features[right_slot_index], right_row_features
                        )
                        if right_score < 0.35:
                            continue
                        cross_score = coverage_score(
                            slot_features[right_slot_index], left_row_features
                        ) + coverage_score(
                            slot_features[left_slot_index], right_row_features
                        )
                        score = left_score + right_score - 0.15 * cross_score
                        candidate = (
                            score,
                            -left_episode_index,
                            -right_episode_index,
                            left_slot_index,
                            right_slot_index,
                        )
                        if best is None or candidate > best:
                            best = candidate
                            selected = [
                                (slots[left_slot_index], episode_rows[left_episode_index]),
                                (slots[right_slot_index], episode_rows[right_episode_index]),
                            ]
                            selection_reason = (
                                "two distinct planned answer slots with local lexical proof"
                            )
    if len(selected) != 2:
        return None

    def observation(row: dict[str, Any]) -> str:
        text = " ".join(str(row.get("text", "")).split())
        return text[:220].rstrip("，,；;：: ")

    left_slot, left = selected[0]
    right_slot, right = selected[1]
    claim = (
        f"检索证据桥：问题“{' '.join(question.split())[:180]}”中的槽“{left_slot}”"
        f"由 Episode {int(left['id'])} 直接支持，其观察为：{observation(left)}；"
        f"槽“{right_slot}”由 Episode {int(right['id'])} 直接支持，其观察为："
        f"{observation(right)}；此边只用于共同检索两个答案槽，不断言两项观察之间"
        "存在因果、身份或机制关系。"
    )[:700]
    return {
        "claim": claim,
        "premise_episode_ids": [int(left["id"]), int(right["id"])],
        "confidence": 0.82,
        "reason": selection_reason,
        "inference_type": "evidence_bridge",
    }


def _knowledge_trace(raw_result: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_result if isinstance(raw_result, dict) else {}
    domains = raw.get("domains") or {}
    knowledge = domains.get("knowledge") if isinstance(domains, dict) else None
    return knowledge if isinstance(knowledge, dict) else {}


def _trace_slot_rows(knowledge: dict[str, Any]) -> list[dict[str, Any]]:
    trace = knowledge.get("evidence_slot_trace") or {}
    rows: list[dict[str, Any]] = []
    if not isinstance(trace, dict):
        return rows
    deterministic = trace.get("deterministic") or {}
    if isinstance(deterministic, dict):
        for key in ("constraint_slots", "atomic_slots"):
            values = deterministic.get(key) or []
            if isinstance(values, list):
                rows.extend(item for item in values if isinstance(item, dict))
    values = trace.get("coverage_slots") or []
    if isinstance(values, list):
        rows.extend(item for item in values if isinstance(item, dict))
    return rows


def _slot_episode_ids(slot: dict[str, Any]) -> set[int]:
    values: set[int] = set()
    for raw in slot.get("episode_ids", []) if isinstance(slot, dict) else []:
        try:
            values.add(int(raw))
        except (TypeError, ValueError):
            continue
    return values


def _int_set(values: Any) -> set[int]:
    """Best-effort conversion for retrieval receipts from older/partial runs.

    Retrieval metadata is deliberately treated as untrusted input.  A malformed
    row should reduce the set of usable candidates, not abort answer handling or
    the background consolidation job.
    """

    if not isinstance(values, (list, tuple, set)):
        return set()
    result: set[int] = set()
    for value in values:
        try:
            result.add(int(value))
        except (TypeError, ValueError):
            continue
    return result


def _int_list(values: Any) -> list[int]:
    """Best-effort ordered conversion for paired receipt arrays."""

    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[int] = []
    for value in values:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return result


def derive_contextual_recall_candidates(
    question: str,
    raw_result: dict[str, Any] | None,
    *,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    """Derive local double-key candidates from a finished evidence trace.

    The function deliberately consumes only serializable retrieval metadata:
    an independently retrieved base Episode, a newly selected direct Episode,
    and a query/slot mapping.  It never reads the answer prose and never asks
    a model to invent a relation.
    """

    knowledge = _knowledge_trace(raw_result)
    contextual = knowledge.get("contextual_association") or {}
    if not isinstance(contextual, dict) or not contextual.get("enabled"):
        return []
    base_ids = _int_set(contextual.get("base_episode_ids", []))
    contextual_ids = _int_set(contextual.get("contextual_episode_ids", []))
    if not base_ids:
        return []
    rows = [row for row in knowledge.get("evidence_episodes", []) if isinstance(row, dict)]
    direct_ids: set[int] = set()
    for row in rows:
        try:
            row_id = int(row["id"])
            generation = int(row.get("generation", 0) or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if (
            generation == 0
            and str(row.get("evidence_origin", "")).casefold()
            in {"source", "direct", "imported"}
        ):
            direct_ids.add(row_id)
    target_ids = [value for value in direct_ids if value not in base_ids | contextual_ids]
    if not target_ids:
        return []
    metadata = [item for item in contextual.get("query_vectors", []) if isinstance(item, dict)]
    query_by_text = {
        " ".join(str(item.get("text", "")).casefold().split()): item
        for item in metadata
        if str(item.get("text", "")).strip()
    }
    whole_id = str(contextual.get("context_query_id", ""))
    if not whole_id:
        return []
    fallback_need = next(
        (item for item in metadata if str(item.get("role", "")) != "whole"),
        None,
    )
    candidates: list[ContextualRecallCandidate] = []
    for target_id in target_ids:
        matching_slot: dict[str, Any] | None = None
        for slot in _trace_slot_rows(knowledge):
            episode_ids = _slot_episode_ids(slot)
            if target_id in episode_ids:
                matching_slot = slot
                break
        slot_text = " ".join(str((matching_slot or {}).get("query", "")).split())
        need = query_by_text.get(slot_text.casefold()) if slot_text else None
        need = need or fallback_need
        if not need or not str(need.get("query_id", "")).strip():
            continue
        anchor_id = next((value for value in sorted(base_ids) if value != target_id), None)
        if anchor_id is None:
            continue
        candidates.append(
            ContextualRecallCandidate(
                anchor_type="episode",
                anchor_id=anchor_id,
                target_episode_id=target_id,
                context_query_id=whole_id,
                need_query_id=str(need["query_id"]),
                slot_id=slot_text[:180],
                source_request_hash=hashlib.sha256(
                    " ".join(str(question).split()).encode("utf-8")
                ).hexdigest(),
            )
        )
        if len(candidates) >= max(0, int(max_candidates)):
            break
    return [item.as_dict() for item in candidates]


def derive_contextual_utility_observations(
    raw_result: dict[str, Any] | None,
    *,
    query_hash: str,
) -> list[dict[str, Any]]:
    """Produce a conservative local Treatment/Masked observation receipt."""

    knowledge = _knowledge_trace(raw_result)
    contextual = knowledge.get("contextual_association") or {}
    if not isinstance(contextual, dict) or not contextual.get("enabled"):
        return []
    edges = _int_list(contextual.get("attached_edges", []))
    targets = _int_list(contextual.get("attached_episode_ids", []))
    selected = _int_set(knowledge.get("episode_ids", []))
    base = _int_set(contextual.get("base_episode_ids", []))
    slots = _trace_slot_rows(knowledge)
    observations: list[dict[str, Any]] = []
    for edge_id, target_id in zip(edges, targets):
        treatment_slots = [
            str(slot.get("query", ""))[:180]
            for slot in slots
            if target_id in _slot_episode_ids(slot)
        ]
        # A target not selected cannot have provided an answer-slot gain.  A
        # selected target that introduces a slot is sufficient; otherwise the
        # conservative outcome is redundant/no-op, never a success.
        base_slot_count = sum(
            1
            for slot in slots
            if base.intersection(_slot_episode_ids(slot))
        )
        outcome = "no_op"
        if target_id in selected and treatment_slots:
            outcome = "sufficient" if len(treatment_slots) > base_slot_count else "redundant"
        observations.append(
            {
                "association_id": edge_id,
                "query_hash": str(query_hash),
                "outcome": outcome,
                "delta_slots": len(treatment_slots),
                "treatment_episode_ids": [target_id],
                "masked_episode_ids": sorted(base),
            }
        )
    return observations


def _knowledge_candidate_rejection_reason(
    *,
    claim: str,
    premise_ids: tuple[int, ...],
    confidence: float,
    inference_type: str,
) -> str:
    """Apply a cheap, deterministic gate before any background model call."""

    if not claim or not premise_ids:
        return "missing premise"
    if len(premise_ids) != 2:
        return "knowledge edge requires exactly two premise episodes"
    if confidence < MIN_KNOWLEDGE_CONFIDENCE:
        return (
            "candidate confidence below local threshold "
            f"({confidence:.3f} < {MIN_KNOWLEDGE_CONFIDENCE:.3f})"
        )
    if inference_type not in KNOWLEDGE_INFERENCE_TYPES:
        return f"unsupported or non-inferential relation type: {inference_type or 'missing'}"
    if inference_type == "trait" and any(
        marker in claim for marker in ("且", "以及", "同时还", "并且")
    ):
        return "non-atomic trait candidate combines multiple conclusions"
    return ""


def _available_episode_ids(raw_result: dict[str, Any] | None) -> set[int]:
    raw = raw_result if isinstance(raw_result, dict) else {}
    domains = raw.get("domains", {})
    knowledge = domains.get("knowledge", {}) if isinstance(domains, dict) else {}
    rows = knowledge.get("evidence_episodes", []) if isinstance(knowledge, dict) else []
    values: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            values.add(int(row["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def split_inline_memory_response(
    response: str,
    raw_result: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    """Remove and validate a machine footer emitted by the chat response.

    Knowledge candidates fail closed unless every cited premise is an Episode
    retrieved from the shared knowledge domain in this exact turn.  The model
    cannot directly supply ``knowledge_query``; it is built here only from
    accepted candidates.
    """

    text = str(response or "")
    matches = list(_INLINE_MEMORY_PATTERN.finditer(text))
    if not matches:
        return text.strip(), ConsolidationDecision().as_dict()
    match = matches[-1]
    visible = _INLINE_MEMORY_PATTERN.sub("", text).strip()
    rejected: list[dict[str, str]] = []
    try:
        payload = _parse_json_object(match.group(1))
    except (ValueError, json.JSONDecodeError) as exc:
        decision = ConsolidationDecision(
            rejected_claims=[
                {"claim": "inline_memory", "reason": f"invalid JSON: {exc}"}
            ]
        )
        return visible, decision.as_dict()

    available_ids = _available_episode_ids(raw_result)
    knowledge: list[KnowledgeCandidate] = []
    raw_knowledge = payload.get("knowledge_candidates")
    if raw_knowledge is None:
        raw_knowledge = payload.get("k")
    for raw in list(raw_knowledge or [])[:3]:
        if not isinstance(raw, dict):
            continue
        claim = " ".join(str(raw.get("claim", raw.get("c", ""))).split())[:700]
        try:
            premise_ids = tuple(
                dict.fromkeys(
                    int(value)
                    for value in raw.get("premise_episode_ids", raw.get("e", []))
                )
            )[:12]
            confidence = max(
                0.0,
                min(1.0, float(raw.get("confidence", raw.get("q", 0.0)))),
            )
            inference_type = " ".join(
                str(
                    raw.get(
                        "inference_type",
                        raw.get("relation_type", raw.get("t", "")),
                    )
                ).casefold().split()
            )[:40]
        except (TypeError, ValueError):
            premise_ids = ()
            confidence = 0.0
            inference_type = ""
        rejection_reason = _knowledge_candidate_rejection_reason(
            claim=claim,
            premise_ids=premise_ids,
            confidence=confidence,
            inference_type=inference_type,
        )
        if rejection_reason:
            rejected.append(
                {"claim": claim or "(empty)", "reason": rejection_reason}
            )
            continue
        unknown = [value for value in premise_ids if value not in available_ids]
        if unknown:
            rejected.append(
                {
                    "claim": claim,
                    "reason": f"unavailable premise episode ids: {unknown}",
                }
            )
            continue
        knowledge.append(
            KnowledgeCandidate(
                claim=claim,
                premise_episode_ids=premise_ids,
                confidence=confidence,
                reason=" ".join(
                    str(raw.get("reason", raw.get("r", ""))).split()
                )[:400],
                inference_type=inference_type,
            )
        )

    private: list[PrivateCandidate] = []
    raw_private = payload.get("private_candidates")
    if raw_private is None:
        raw_private = payload.get("p")
    for raw in list(raw_private or [])[:3]:
        if isinstance(raw, str):
            claim = " ".join(raw.split())[:700]
            reason = "inline private event"
        elif isinstance(raw, dict):
            claim = " ".join(str(raw.get("claim", raw.get("c", ""))).split())[:700]
            reason = " ".join(
                str(raw.get("reason", raw.get("r", ""))).split()
            )[:400]
        else:
            continue
        if claim:
            private.append(
                PrivateCandidate(
                    claim=claim,
                    reason=reason,
                )
            )
    return visible, ConsolidationDecision(
        knowledge_candidates=knowledge,
        private_candidates=private,
        rejected_claims=rejected[:3],
    ).as_dict()


class AnswerMemoryConsolidator:
    """Split one generated answer into evidence-bound lore and private events."""

    def __init__(self, engine: Any):
        self.engine = engine

    @staticmethod
    def _available_episode_ids(evidence: dict[str, Any]) -> set[int]:
        values: set[int] = set()
        for row in evidence.get("knowledge_episodes", []):
            try:
                values.add(int(row["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        return values

    async def consolidate(
        self,
        identity: PlatformIdentity,
        question: str,
        assistant_response: str,
        route: MemoryRoute,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        schema = {
            "knowledge_candidates": [
                {
                    "claim": "可跨用户复用的稳定剧情推论",
                    "premise_episode_ids": [1, 2],
                    "inference_type": "causal",
                    "confidence": 0.8,
                    "reason": "两条剧情证据共同支持，但原文没有直接陈述",
                }
            ],
            "private_candidates": [
                {
                    "claim": "本轮新发生的任务、消息或互动",
                    "reason": "由角色扮演临时创造，只属于当前用户",
                }
            ],
            "rejected_claims": [
                {
                    "claim": "不应进入长期图的修辞或无证据断言",
                    "reason": "没有剧情前提或没有复用价值",
                }
            ],
        }
        prompt = memory_consolidation_prompt(
            schema=schema,
            platform=identity.platform,
            route={
                "reason": route.reason,
                "creative": route.creative,
                "policy": route.knowledge_write_policy,
            },
            question=question,
            assistant_response=assistant_response,
            evidence=evidence,
        )
        response = await self.engine.generate_response(
            [Message(role="user", content=prompt)],
            system_prompt=MEMORY_CONSOLIDATION_SYSTEM,
            max_retries=2,
            task_context=f"{identity.key}:memory-consolidation",
        )
        payload = _parse_json_object(response)
        available_ids = self._available_episode_ids(evidence)
        knowledge: list[KnowledgeCandidate] = []
        for raw in payload.get("knowledge_candidates", []):
            if not isinstance(raw, dict):
                continue
            claim = " ".join(str(raw.get("claim", "")).split())
            try:
                premise_ids = tuple(
                    dict.fromkeys(
                        int(value)
                        for value in raw.get("premise_episode_ids", [])
                    )
                )
                confidence = max(
                    0.0, min(1.0, float(raw.get("confidence", 0.0)))
                )
                inference_type = " ".join(
                    str(raw.get("inference_type", "")).casefold().split()
                )[:40]
            except (TypeError, ValueError):
                continue
            if any(value not in available_ids for value in premise_ids):
                continue
            if _knowledge_candidate_rejection_reason(
                claim=claim,
                premise_ids=premise_ids,
                confidence=confidence,
                inference_type=inference_type,
            ):
                continue
            knowledge.append(
                KnowledgeCandidate(
                    claim=claim[:700],
                    premise_episode_ids=premise_ids[:12],
                    confidence=confidence,
                    reason=" ".join(str(raw.get("reason", "")).split())[:400],
                    inference_type=inference_type,
                )
            )
        private: list[PrivateCandidate] = []
        for raw in payload.get("private_candidates", []):
            if not isinstance(raw, dict):
                continue
            claim = " ".join(str(raw.get("claim", "")).split())
            if claim:
                private.append(
                    PrivateCandidate(
                        claim=claim[:700],
                        reason=" ".join(
                            str(raw.get("reason", "")).split()
                        )[:400],
                    )
                )
        rejected = [
            {
                "claim": " ".join(str(raw.get("claim", "")).split())[:700],
                "reason": " ".join(str(raw.get("reason", "")).split())[:400],
            }
            for raw in payload.get("rejected_claims", [])
            if isinstance(raw, dict) and str(raw.get("claim", "")).strip()
        ]
        return ConsolidationDecision(
            knowledge_candidates=knowledge[:3],
            private_candidates=private[:3],
            rejected_claims=rejected[:3],
        ).as_dict()
