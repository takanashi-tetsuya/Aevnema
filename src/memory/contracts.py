from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from time import perf_counter
from typing import Any

from config.prompt_config import render_memory_context


@dataclass(frozen=True, slots=True)
class DeadlineBudget:
    """One monotonic deadline shared by planning, retrieval, and generation.

    A provider retry may only spend time left in this request; it never gets a
    fresh per-model timeout.  This object is request-local and deliberately
    has no persistence or prompt representation.
    """

    total_deadline: float = 40.0
    planning_deadline: float = 4.0
    retrieval_deadline: float = 22.0
    answer_deadline: float = 12.0
    fallback_reserve: float = 2.0
    started_at: float = field(default_factory=perf_counter, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "total_deadline",
            "planning_deadline",
            "retrieval_deadline",
            "answer_deadline",
            "fallback_reserve",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.total_deadline <= 0:
            raise ValueError("total_deadline must be positive")

    def remaining(self, now: float | None = None) -> float:
        current = perf_counter() if now is None else float(now)
        return max(0.0, self.total_deadline - (current - self.started_at))

    def planning_timeout(self, now: float | None = None) -> float:
        return min(self.planning_deadline, self.remaining(now))

    def retrieval_timeout(self, now: float | None = None) -> float:
        return max(
            0.0,
            min(
                self.retrieval_deadline,
                self.remaining(now) - self.answer_deadline - self.fallback_reserve,
            ),
        )

    def answer_timeout(self, now: float | None = None) -> float:
        return max(
            0.0,
            min(self.answer_deadline, self.remaining(now) - self.fallback_reserve),
        )

    def fallback_timeout(self, now: float | None = None) -> float:
        return min(self.fallback_reserve, self.remaining(now))

    def as_dict(self, now: float | None = None) -> dict[str, float]:
        return {
            "total_deadline": self.total_deadline,
            "planning_deadline": self.planning_deadline,
            "retrieval_deadline": self.retrieval_deadline,
            "answer_deadline": self.answer_deadline,
            "fallback_reserve": self.fallback_reserve,
            "remaining": round(self.remaining(now), 6),
        }


@dataclass(slots=True)
class RetrievedMemory:
    context: str
    raw_result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    domains: tuple[str, ...] = ()
    # Request-local normalized embedding used only by the semantic plan cache.
    # Keeping it outside ``raw_result`` avoids logs and prompt serialization.
    semantic_vector: Any | None = None
    # Request-local whole/atomic vectors shared by contextual retrieval.  This
    # field is intentionally excluded from raw_result/prompt serialization.
    query_vector_bundle: Any | None = None

    @property
    def available(self) -> bool:
        return bool(self.raw_result) and not self.error


@dataclass(slots=True, frozen=True)
class DomainRecallRequest:
    """Model-planned, request-local inputs for one physical memory domain."""

    query: str
    intent_override: dict[str, Any] = field(default_factory=dict)
    followup_queries: tuple[str, ...] | None = None
    evidence_goal: str = "lookup"
    absence_answerable: bool = False

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("domain recall query must not be empty")

    @property
    def cache_key(self) -> str:
        payload = {
            "query": " ".join(self.query.casefold().split()),
            "intent": self.intent_override,
            "followup": (
                list(self.followup_queries)
                if self.followup_queries is not None
                else None
            ),
            "evidence_goal": self.evidence_goal,
            "absence_answerable": self.absence_answerable,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(slots=True, frozen=True)
class RetrievalPlan:
    """Immutable request-level retrieval policy expanded from a preset."""

    preset: str
    query_planner: str
    graph_hops: int
    candidate_limit: int
    reranker: str
    evidence_slots: bool
    followup_policy: str
    verification: str
    deadline_seconds: float
    answer_episode_limit: int
    answer_concept_limit: int
    answer_path_limit: int

    def __post_init__(self) -> None:
        if self.preset not in {"light", "standard", "deep"}:
            raise ValueError("preset must be light, standard, or deep")
        if self.query_planner not in {"heuristic", "llm"}:
            raise ValueError("query_planner must be heuristic or llm")
        if self.reranker not in {"configured", "llm", "disabled"}:
            raise ValueError("reranker must be configured, llm, or disabled")
        if self.followup_policy not in {
            "never",
            "on_insufficient_evidence",
            "always",
        }:
            raise ValueError("invalid followup_policy")
        if self.verification not in {"local", "llm"}:
            raise ValueError("verification must be local or llm")
        if self.graph_hops < 0 or self.candidate_limit <= 0:
            raise ValueError("retrieval budgets must be positive")

    @property
    def cache_key(self) -> str:
        return "|".join(
            str(value)
            for value in (
                self.preset,
                self.query_planner,
                self.graph_hops,
                self.candidate_limit,
                self.reranker,
                self.evidence_slots,
                self.followup_policy,
                self.verification,
                self.answer_episode_limit,
                self.answer_concept_limit,
                self.answer_path_limit,
            )
        )


@dataclass(slots=True, frozen=True)
class RetrievalQuality:
    """Auditable evidence sufficiency signal; relevance alone is not enough."""

    entity_coverage: float
    claim_slot_coverage: float
    top_score: float
    score_margin: float
    source_diversity: int
    timeline_conflict: bool
    evidence_count: int
    entity_count: int
    claim_slot_count: int
    semantic_constraint_coverage: float
    semantic_constraint_count: int
    sufficient: bool
    reasons: tuple[str, ...] = ()
    contextual_candidate_count: int = 0
    contextual_selected_count: int = 0
    contextual_new_slot_count: int = 0
    contextual_harm_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value


def assess_retrieval_quality(
    question: str,
    result: dict[str, Any],
) -> RetrievalQuality:
    """Measure whether selected evidence covers the known answer contract.

    The first version is deliberately deterministic.  It consumes the
    engine's evidence-slot trace, selected entities, reranker scores and
    chronology diagnostics without spending another model call.
    """

    episodes = [
        item
        for item in (result.get("evidence_episodes") or [])
        if isinstance(item, dict)
    ]
    concepts = [
        item
        for item in (result.get("evidence_concepts") or [])
        if isinstance(item, dict)
    ]
    evidence_text = "\n".join(
        str(value)
        for value in [
            *(
                " ".join(
                    str(item.get(key, ""))
                    for key in ("text", "participants", "source_key")
                )
                for item in episodes
            ),
            *(
                " ".join(
                    str(item.get(key, ""))
                    for key in ("canonical_name", "description")
                )
                for item in concepts
            ),
        ]
    ).casefold()
    intent = result.get("intent") or {}
    raw_entities = intent.get("target_entities", []) if isinstance(intent, dict) else []
    entities = list(
        dict.fromkeys(
            str(value).strip()
            for value in raw_entities
            if str(value).strip()
        )
    )
    matched_entities = [
        value for value in entities if value.casefold() in evidence_text
    ]
    entity_coverage = (
        len(matched_entities) / len(entities) if entities else 1.0
    )

    slot_trace = result.get("evidence_slot_trace") or {}
    slots: list[dict[str, Any]] = []
    deterministic = slot_trace.get("deterministic", {})
    deterministic_constraint_keys: set[str] = set()
    if isinstance(deterministic, dict):
        for key in ("constraint_slots", "atomic_slots"):
            selected_slots = [
                item
                for item in deterministic.get(key, [])
                if isinstance(item, dict)
            ]
            slots.extend(selected_slots)
            if key == "constraint_slots":
                deterministic_constraint_keys.update(
                    str(item.get("query", "")).strip()
                    for item in selected_slots
                    if str(item.get("query", "")).strip()
                )
    slots.extend(
        item
        for item in slot_trace.get("coverage_slots", [])
        if isinstance(item, dict)
    )
    unique_slots: dict[str, bool] = {}
    for index, slot in enumerate(slots):
        key = str(slot.get("query", "")).strip() or f"slot:{index}"
        unique_slots[key] = bool(slot.get("satisfied"))
    claim_slot_coverage = (
        sum(unique_slots.values()) / len(unique_slots)
        if unique_slots
        else (1.0 if episodes else 0.0)
    )

    rerank_trace = result.get("rerank_trace") or {}
    scores = [
        float(value)
        for value in rerank_trace.get("cross_encoder_scores", [])
        if isinstance(value, (int, float))
    ]
    if not scores:
        scores = sorted(
            (
                float(item.get("score", 0.0))
                for item in episodes
                if isinstance(item.get("score", 0.0), (int, float))
            ),
            reverse=True,
        )
    top_score = scores[0] if scores else 0.0
    score_margin = top_score - scores[1] if len(scores) >= 2 else top_score
    semantic_constraints = list(
        dict.fromkeys(
            str(intent.get(key, "")).strip()
            for key in (
                "requested_relation",
                "temporal_constraint",
                "causal_constraint",
            )
            if isinstance(intent, dict) and str(intent.get(key, "")).strip()
        )
    )
    if (
        rerank_trace.get("backend") == "cross_encoder"
        and semantic_constraints
    ):
        # Deterministic floors protect recall budget; whether a floor happened
        # to survive final selection is not an independent relevance verdict.
        # The cross-encoder score below owns typed semantic coverage.
        for key in deterministic_constraint_keys:
            unique_slots.pop(key, None)
        claim_slot_coverage = (
            sum(unique_slots.values()) / len(unique_slots)
            if unique_slots
            else (1.0 if episodes else 0.0)
        )
    # A cross-encoder ranks relevance but does not perform the LLM coverage
    # audit. A typed relation/event constraint therefore needs stronger direct
    # relevance than a broad entity/profile query.
    cross_encoder_constraint_min_score = 0.80
    if not semantic_constraints:
        semantic_constraint_coverage = 1.0
    elif rerank_trace.get("backend") == "cross_encoder":
        semantic_constraint_coverage = float(
            top_score >= cross_encoder_constraint_min_score
        )
    else:
        semantic_constraint_coverage = claim_slot_coverage
    source_diversity = len(
        {
            str(item.get("source_key", "")).strip()
            for item in episodes
            if str(item.get("source_key", "")).strip()
        }
    )
    chronology = "\n".join(
        str(value) for value in (result.get("chronology_notes") or [])
    ).casefold()
    timeline_conflict = any(
        marker in chronology
        for marker in (
            "timeline conflict",
            "时间线冲突",
            "无法确定顺序",
            "形成环",
            "cycle",
        )
    )

    reasons: list[str] = []
    if not episodes:
        reasons.append("no_evidence")
    if rerank_trace.get("error"):
        reasons.append("reranker_error")
    if entities and entity_coverage < 1.0:
        reasons.append("entity_coverage_incomplete")
    if unique_slots and claim_slot_coverage < 1.0:
        reasons.append("claim_slot_coverage_incomplete")
    if semantic_constraints and semantic_constraint_coverage < 1.0:
        reasons.append("semantic_constraint_relevance_low")
    if timeline_conflict:
        reasons.append("timeline_conflict")
    return RetrievalQuality(
        entity_coverage=round(entity_coverage, 6),
        claim_slot_coverage=round(claim_slot_coverage, 6),
        top_score=round(top_score, 8),
        score_margin=round(score_margin, 8),
        source_diversity=source_diversity,
        timeline_conflict=timeline_conflict,
        evidence_count=len(episodes),
        entity_count=len(entities),
        claim_slot_count=len(unique_slots),
        semantic_constraint_coverage=round(
            semantic_constraint_coverage, 6
        ),
        semantic_constraint_count=len(semantic_constraints),
        sufficient=not reasons,
        reasons=tuple(reasons),
        contextual_candidate_count=int(
            (result.get("contextual_association") or {}).get("candidate_count", 0)
        ),
        contextual_selected_count=int(
            (result.get("contextual_association") or {}).get("selected_count", 0)
        ),
        contextual_new_slot_count=int(
            (result.get("contextual_association") or {}).get("new_slot_count", 0)
        ),
        contextual_harm_count=int(
            (result.get("contextual_association") or {}).get("harm_count", 0)
        ),
    )


def build_lore_fact_contract(
    result: dict[str, Any],
    quality: RetrievalQuality,
) -> dict[str, Any]:
    """Freeze what the answer model may treat as retrieved lore evidence."""

    episodes = [
        item
        for item in (result.get("evidence_episodes") or [])
        if isinstance(item, dict)
    ]
    paths = [
        item
        for item in (result.get("association_paths") or [])
        if isinstance(item, dict)
        and str(item.get("association_mode", "")) != "contextual_recall"
        and str(item.get("claim_level", "")) != "retrieval_only"
        and str(item.get("relation_key", "")) != "contextual_recall"
    ]
    return {
        "version": "lore-fact-contract-v1",
        "episode_ids": [
            int(item["id"]) for item in episodes if "id" in item
        ],
        "source_keys": list(
            dict.fromkeys(
                str(item.get("source_key", "")).strip()
                for item in episodes
                if str(item.get("source_key", "")).strip()
            )
        ),
        "association_ids": list(
            dict.fromkeys(
                int(item.get("association_id", item.get("id")))
                for item in paths
                if item.get("association_id", item.get("id")) is not None
            )
        ),
        "epistemic_statuses": list(
            dict.fromkeys(
                str(item.get("epistemic_status", "unknown"))
                for item in episodes
            )
        ),
        "max_generation": max(
            (int(item.get("generation", 0)) for item in episodes),
            default=0,
        ),
        "requires_uncertainty": not quality.sufficient,
        "unresolved_reasons": list(quality.reasons),
        "timeline_conflict": quality.timeline_conflict,
    }


def _retrieval_answer_shape(domain: str, *, lightweight: bool) -> str:
    """Describe the evidence task according to the isolated memory domain."""

    if lightweight:
        return "short_roleplay_context"
    if domain.startswith("user:"):
        return "private_memory_evidence"
    if domain == "knowledge":
        return "lore_evidence"
    return "shared_memory_evidence"


def format_memory_context(result: dict[str, Any], max_chars: int = 18_000) -> str:
    """Turn a retrieval trace into a bounded, evidence-labelled prompt section."""
    return render_memory_context(result, max_chars)


