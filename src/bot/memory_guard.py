from __future__ import annotations

from dataclasses import dataclass, field
import json
from time import perf_counter
from typing import Any, Callable

from config.prompt_config import (
    PRIVATE_MEMORY_AUDIT_SYSTEM,
    private_memory_audit_prompt,
    private_memory_contract_policy,
    private_memory_prompt_addendum,
    private_memory_rewrite_instruction,
)
from src.llm.engine import Message
from src.memory.service import MemoryRoute, RetrievedMemory


PRIVATE_RECALL_MARKERS = (
    "你记得",
    "还记得",
    "记不记得",
    "想得起来",
    "之前说",
    "以前说",
    "上次我们",
    "你之前",
    "曾经告诉你",
    "跟你说过",
    "暗号是什么",
    "暗号吗",
    "约定是什么",
    "最喜欢什么",
    "偏好是什么",
    "我喜欢什么",
    "我讨厌什么",
    "老师喜欢什么",
    "老师喜欢还是讨厌",
    "亲口说",
    "remember",
)

PRIVATE_TEACHING_MARKERS = (
    "请记住",
    "记住：",
    "记住:",
    "告诉你：",
    "告诉你:",
)


def _parse_json_object(value: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"passed": False, "violations": ["审计器没有返回合法 JSON"]}
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"passed": False, "violations": ["审计器返回的 JSON 无法解析"]}
    return payload if isinstance(payload, dict) else {
        "passed": False,
        "violations": ["审计器返回值不是对象"],
    }


@dataclass(slots=True, frozen=True)
class PrivateMemoryContract:
    state: str
    evidence: tuple[dict[str, Any], ...]
    required_semantics: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    non_private_evidence: tuple[dict[str, Any], ...] = ()
    private_retrieval_sufficient: bool = False
    knowledge_retrieval_sufficient: bool = False
    unresolved_reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "evidence": list(self.evidence),
            "required_semantics": list(self.required_semantics),
            "forbidden_claims": list(self.forbidden_claims),
            "non_private_evidence": list(self.non_private_evidence),
            "private_retrieval_sufficient": self.private_retrieval_sufficient,
            "knowledge_retrieval_sufficient": self.knowledge_retrieval_sufficient,
            "unresolved_reasons": list(self.unresolved_reasons),
        }


@dataclass(slots=True)
class GuardResult:
    text: str
    applied: bool = False
    rewritten: bool = False
    emergency_fallback: bool = False
    audits: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    contract_state: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "rewritten": self.rewritten,
            "emergency_fallback": self.emergency_fallback,
            "audits": self.audits,
            "timings": self.timings,
            "contract_state": self.contract_state,
        }


class PrivateMemoryResponseGuard:
    """Keep memory claims grounded while leaving roleplay wording to models."""

    def __init__(
        self,
        *,
        audit_engine: Any,
        rewrite_engine: Any,
        emergency_reply_factory: Callable[[], str],
    ):
        self.audit_engine = audit_engine
        self.rewrite_engine = rewrite_engine
        self.emergency_reply_factory = emergency_reply_factory

    @staticmethod
    def applies(question: str, route: MemoryRoute) -> bool:
        normalized = " ".join(question.casefold().split())
        if any(marker in normalized for marker in PRIVATE_TEACHING_MARKERS):
            return False
        return route.user and any(
            marker in normalized for marker in PRIVATE_RECALL_MARKERS
        )

    @staticmethod
    def build_contract(memory: RetrievedMemory) -> PrivateMemoryContract:
        domains = memory.raw_result.get("domains", {})
        private = domains.get("user", {}) if isinstance(domains, dict) else {}
        private_quality = (
            private.get("retrieval_quality") or {}
            if isinstance(private, dict)
            else {}
        )
        knowledge_quality: dict[str, Any] = {}
        non_private_evidence: list[dict[str, Any]] = []
        if isinstance(domains, dict):
            for domain_name in ("public", "knowledge"):
                domain = domains.get(domain_name, {})
                if not isinstance(domain, dict):
                    continue
                if domain_name == "knowledge":
                    knowledge_quality = domain.get("retrieval_quality") or {}
                for row in list(domain.get("evidence_episodes") or [])[:8]:
                    non_private_evidence.append(
                        {
                            "domain": domain_name,
                            "id": int(row.get("id", 0) or 0),
                            "text": " ".join(str(row.get("text", "")).split())[:420],
                            "origin": str(row.get("evidence_origin", "unknown")),
                            "status": str(row.get("epistemic_status", "unknown")),
                            "generation": int(row.get("generation", 0) or 0),
                        }
                    )
        rows = (
            list(private.get("evidence_episodes") or [])
            if isinstance(private, dict)
            else []
        )
        evidence: list[dict[str, Any]] = []
        direct = 0
        qualified = 0
        for row in rows[:8]:
            origin = str(row.get("evidence_origin", "unknown"))
            status = str(row.get("epistemic_status", "unknown"))
            generation = int(row.get("generation", 0) or 0)
            if origin == "source" and status in {"observed", "asserted"}:
                direct += 1
            else:
                qualified += 1
            evidence.append(
                {
                    "id": int(row.get("id", 0) or 0),
                    "text": " ".join(str(row.get("text", "")).split())[:420],
                    "origin": origin,
                    "status": status,
                    "generation": generation,
                    "note": " ".join(
                        str(row.get("epistemic_note", "")).split()
                    )[:120],
                }
            )
        if not evidence:
            return PrivateMemoryContract(
                state="missing",
                evidence=(),
                required_semantics=(
                    "自然承认目前没有可靠的这段私人记忆",
                    "可以邀请用户重新说明",
                ),
                forbidden_claims=(
                    "猜测任何具体暗号、饮料、偏好或答案候选",
                    "虚构以前看到、听到或共同经历过某件事",
                    "以模糊记得为名陈述具体的过去事实",
                ),
                non_private_evidence=tuple(non_private_evidence),
                private_retrieval_sufficient=bool(
                    private_quality.get("sufficient", False)
                ),
                knowledge_retrieval_sufficient=bool(
                    knowledge_quality.get("sufficient", False)
                ),
                unresolved_reasons=tuple(
                    dict.fromkeys(
                        [
                            *list(private_quality.get("reasons") or []),
                            *list(knowledge_quality.get("reasons") or []),
                        ]
                    )
                )[:6],
            )
        state = "conflicted_or_qualified" if qualified else "grounded"
        required_semantics, forbidden_claims = private_memory_contract_policy(
            has_importer=any(item["origin"] == "importer" for item in evidence)
        )
        return PrivateMemoryContract(
            state=state,
            evidence=tuple(evidence),
            required_semantics=required_semantics,
            forbidden_claims=forbidden_claims,
            non_private_evidence=tuple(non_private_evidence),
            private_retrieval_sufficient=bool(
                private_quality.get("sufficient", bool(evidence))
            ),
            knowledge_retrieval_sufficient=bool(
                knowledge_quality.get("sufficient", bool(non_private_evidence))
            ),
            unresolved_reasons=tuple(
                dict.fromkeys(
                    [
                        *list(private_quality.get("reasons") or []),
                        *list(knowledge_quality.get("reasons") or []),
                    ]
                )
            )[:6],
        )

    @classmethod
    def prompt_addendum(
        cls,
        question: str,
        memory: RetrievedMemory,
        route: MemoryRoute,
    ) -> str:
        """Give the writer a compact turn-specific boundary before drafting."""

        if not cls.applies(question, route):
            return ""
        contract = cls.build_contract(memory)
        return private_memory_prompt_addendum(
            question=question,
            state=contract.state,
            evidence=list(contract.evidence),
            has_non_private_evidence=bool(contract.non_private_evidence),
            private_retrieval_sufficient=contract.private_retrieval_sufficient,
            knowledge_retrieval_sufficient=contract.knowledge_retrieval_sufficient,
            unresolved_reasons=list(contract.unresolved_reasons),
        )

    async def _audit(
        self,
        *,
        question: str,
        draft: str,
        contract: PrivateMemoryContract,
        task_context: str,
    ) -> dict[str, Any]:
        prompt = private_memory_audit_prompt(
            question=question,
            draft=draft,
            contract=contract.as_dict(),
        )
        try:
            response = await self.audit_engine.generate_response(
                [Message(role="user", content=prompt)],
                system_prompt=PRIVATE_MEMORY_AUDIT_SYSTEM,
                max_retries=1,
                task_context=f"{task_context}:private-memory-audit",
            )
        except TypeError:
            response = await self.audit_engine.generate_response(
                [Message(role="user", content=prompt)],
                system_prompt=PRIVATE_MEMORY_AUDIT_SYSTEM,
            )
        except Exception as exc:
            return {
                "passed": False,
                "memory_claims": [],
                "violations": [f"审计调用失败：{type(exc).__name__}"],
                "reason": str(exc),
            }
        parsed = _parse_json_object(response)
        parsed["passed"] = parsed.get("passed") is True
        violations = parsed.get("violations")
        if not isinstance(violations, list):
            parsed["violations"] = [str(violations or "审计结果格式错误")]
            parsed["passed"] = False
        else:
            parsed["violations"] = [str(value) for value in violations if str(value)]

        allowed_lore_ids = {
            int(item.get("id", 0) or 0)
            for item in contract.non_private_evidence
            if int(item.get("id", 0) or 0) > 0
        }
        allowed_private_ids = {
            int(item.get("id", 0) or 0)
            for item in contract.evidence
            if int(item.get("id", 0) or 0) > 0
        }
        private_claims = parsed.get("private_claims")
        if not isinstance(private_claims, list):
            parsed["private_claims"] = []
            parsed["violations"].append(
                "审计器没有返回合法 private_claims"
            )
            parsed["passed"] = False
        else:
            unsupported_private: list[str] = []
            for row in private_claims:
                if not isinstance(row, dict):
                    unsupported_private.append("格式错误的私人主张")
                    continue
                claim = " ".join(str(row.get("claim", "")).split())
                raw_ids = row.get("supporting_private_evidence_ids", [])
                valid_ids: list[int] = []
                if isinstance(raw_ids, list):
                    for value in raw_ids:
                        try:
                            evidence_id = int(value)
                        except (TypeError, ValueError):
                            continue
                        if evidence_id in allowed_private_ids:
                            valid_ids.append(evidence_id)
                row["supporting_private_evidence_ids"] = list(
                    dict.fromkeys(valid_ids)
                )
                if claim and not valid_ids:
                    unsupported_private.append(claim[:120])
            if unsupported_private:
                parsed["violations"].extend(
                    f"私人主张缺少直接证据：{claim}"
                    for claim in unsupported_private[:3]
                )
                parsed["passed"] = False
        lore_claims = parsed.get("lore_claims")
        if lore_claims is None and not contract.non_private_evidence:
            parsed["lore_claims"] = []
        elif not isinstance(lore_claims, list):
            parsed["lore_claims"] = []
            parsed["violations"].append("审计器没有返回合法 lore_claims")
            parsed["passed"] = False
        else:
            unsupported: list[str] = []
            for row in lore_claims:
                if not isinstance(row, dict):
                    unsupported.append("格式错误的剧情主张")
                    continue
                claim = " ".join(str(row.get("claim", "")).split())
                raw_ids = row.get("supporting_non_private_evidence_ids", [])
                valid_ids: list[int] = []
                if isinstance(raw_ids, list):
                    for value in raw_ids:
                        try:
                            evidence_id = int(value)
                        except (TypeError, ValueError):
                            continue
                        if evidence_id in allowed_lore_ids:
                            valid_ids.append(evidence_id)
                row["supporting_non_private_evidence_ids"] = list(
                    dict.fromkeys(valid_ids)
                )
                if claim and not valid_ids:
                    unsupported.append(claim[:120])
            if unsupported:
                parsed["violations"].extend(
                    f"剧情主张缺少直接证据：{claim}"
                    for claim in unsupported[:3]
                )
                parsed["passed"] = False
        if parsed["violations"]:
            parsed["passed"] = False
        return parsed

    async def _rewrite(
        self,
        *,
        messages: list[Any],
        system_prompt: str,
        draft: str,
        audit: dict[str, Any],
        contract: PrivateMemoryContract,
        task_context: str,
    ) -> str:
        instruction = private_memory_rewrite_instruction(
            contract=contract.as_dict(),
            draft=draft,
            violations=list(audit.get("violations", [])),
        )
        try:
            return await self.rewrite_engine.generate_response(
                messages=messages,
                system_prompt=system_prompt + instruction,
                max_retries=1,
                task_context=f"{task_context}:private-memory-rewrite",
            )
        except TypeError:
            return await self.rewrite_engine.generate_response(
                messages=messages,
                system_prompt=system_prompt + instruction,
            )

    async def enforce(
        self,
        *,
        question: str,
        draft: str,
        memory: RetrievedMemory,
        route: MemoryRoute,
        messages: list[Any],
        system_prompt: str,
        task_context: str,
    ) -> GuardResult:
        if not self.applies(question, route):
            return GuardResult(text=draft)
        started = perf_counter()
        contract = self.build_contract(memory)
        contract_finished = perf_counter()
        first = await self._audit(
            question=question,
            draft=draft,
            contract=contract,
            task_context=task_context,
        )
        first_audit_finished = perf_counter()
        timings = {
            "contract_seconds": round(contract_finished - started, 6),
            "audit_1_seconds": round(
                first_audit_finished - contract_finished, 6
            ),
        }
        if first.get("passed") is True:
            timings["total_seconds"] = round(
                first_audit_finished - started, 6
            )
            return GuardResult(
                text=draft,
                applied=True,
                audits=[first],
                timings=timings,
                contract_state=contract.state,
            )
        try:
            rewrite_started = perf_counter()
            rewritten = await self._rewrite(
                messages=messages,
                system_prompt=system_prompt,
                draft=draft,
                audit=first,
                contract=contract,
                task_context=task_context,
            )
            rewrite_finished = perf_counter()
            timings["rewrite_seconds"] = round(
                rewrite_finished - rewrite_started, 6
            )
            second = await self._audit(
                question=question,
                draft=rewritten,
                contract=contract,
                task_context=task_context,
            )
            second_audit_finished = perf_counter()
            timings["audit_2_seconds"] = round(
                second_audit_finished - rewrite_finished, 6
            )
        except Exception as exc:
            second = {
                "passed": False,
                "violations": [f"重写调用失败：{type(exc).__name__}"],
                "reason": str(exc),
            }
            rewritten = ""
            second_audit_finished = perf_counter()
        timings["total_seconds"] = round(
            second_audit_finished - started, 6
        )
        if second.get("passed") is True:
            return GuardResult(
                text=rewritten,
                applied=True,
                rewritten=True,
                audits=[first, second],
                timings=timings,
                contract_state=contract.state,
            )
        if contract.state == "grounded":
            emergency_text = (
                "唔，老师。阿洛娜确实记得这件事，但刚才没能把内容说准确。"
                "让我再整理一下，然后认真回答您，好吗？"
            )
        elif contract.state == "conflicted_or_qualified":
            emergency_text = (
                "老师，这段记忆里直接说过的话和后来的推测混在一起了。"
                "阿洛娜不想把它们说反，能让我先确认清楚吗？"
            )
        else:
            emergency_text = self.emergency_reply_factory().strip()
        return GuardResult(
            text=emergency_text,
            applied=True,
            rewritten=True,
            emergency_fallback=True,
            audits=[first, second],
            timings=timings,
            contract_state=contract.state,
        )
