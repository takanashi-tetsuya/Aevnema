from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import json
import re
from time import perf_counter
from typing import Any

from config.prompt_config.memory_intent_prompts import (
    MEMORY_INTENT_SYSTEM,
    memory_intent_prompt,
)
from src.llm.engine import Message

from .contracts import DomainRecallRequest
from .routing import MemoryRoute


_INTENSITIES = {"light", "standard", "deep"}
_WRITE_POLICIES = {"none", "evidence_gated", "query_and_answer"}


def _json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1)
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(value[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("memory intent response must be a JSON object")
    return payload


@dataclass(slots=True, frozen=True)
class MemoryIntentPlan:
    route: MemoryRoute
    queries: dict[str, str]
    target_entities: tuple[str, ...]
    requested_relation: str
    temporal_constraint: str
    causal_constraint: str
    answer_slots: tuple[str, ...]
    uncertainty_required: bool
    evidence_goal: str = "lookup"
    absence_answerable: bool = False
    planning_seconds: float = 0.0
    raw: dict[str, Any] | None = None

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        fallback_question: str,
        planning_seconds: float = 0.0,
    ) -> "MemoryIntentPlan":
        raw_domains = payload.get("domains")
        domains = raw_domains if isinstance(raw_domains, dict) else {}
        needs_memory = bool(payload.get("needs_memory", True))
        user = needs_memory and bool(domains.get("private"))
        public = needs_memory and bool(domains.get("public"))
        knowledge = needs_memory and bool(domains.get("knowledge"))
        intensity = str(payload.get("intensity", "standard")).casefold()
        if intensity not in _INTENSITIES:
            intensity = "standard"
        creative = bool(payload.get("creative"))
        write_policy = str(
            payload.get("knowledge_write_policy", "evidence_gated")
        ).casefold()
        if write_policy not in _WRITE_POLICIES:
            write_policy = "evidence_gated"
        if not knowledge:
            write_policy = "none"
        raw_queries = payload.get("queries")
        query_payload = raw_queries if isinstance(raw_queries, dict) else {}
        queries = {
            domain: str(query_payload.get(domain, "")).strip()
            or fallback_question.strip()
            for domain, enabled in (
                ("private", user),
                ("public", public),
                ("knowledge", knowledge),
            )
            if enabled
        }
        entities = payload.get("target_entities")
        slots = payload.get("answer_slots")
        return cls(
            route=MemoryRoute(
                user,
                public,
                knowledge,
                "model_intent_planner",
                intensity,
                write_policy,
                creative,
            ),
            queries=queries,
            target_entities=tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in (entities if isinstance(entities, list) else [])
                    if str(value).strip()
                )
            )[:8],
            requested_relation=str(payload.get("requested_relation", "")).strip(),
            temporal_constraint=str(payload.get("temporal_constraint", "")).strip(),
            causal_constraint=str(payload.get("causal_constraint", "")).strip(),
            answer_slots=tuple(
                dict.fromkeys(
                    str(value).strip()
                    for value in (slots if isinstance(slots, list) else [])
                    if str(value).strip()
                )
            )[:12],
            uncertainty_required=bool(payload.get("uncertainty_required")),
            evidence_goal=str(payload.get("evidence_goal", "lookup")).strip()
            or "lookup",
            absence_answerable=bool(payload.get("absence_answerable")),
            planning_seconds=round(float(planning_seconds), 6),
            raw=dict(payload),
        )

    def intent_override(self, domain: str) -> dict[str, Any]:
        query = self.queries.get(domain, "")
        answer_shape = {
            "private": "private_memory_evidence",
            "public": "shared_memory_evidence",
            "knowledge": "lore_evidence",
        }[domain]
        return {
            "language": "auto",
            "target_entities": list(self.target_entities),
            "search_queries": [query, *self.answer_slots],
            "requested_relation": self.requested_relation,
            "temporal_constraint": self.temporal_constraint,
            "causal_constraint": self.causal_constraint,
            "answer_shape": answer_shape,
            "uncertainty_required": self.uncertainty_required,
        }

    def domain_request(self, domain: str) -> DomainRecallRequest:
        """Compile this semantic plan into one request-local domain query."""

        return DomainRecallRequest(
            query=self.queries[domain],
            intent_override=self.intent_override(domain),
            evidence_goal=self.evidence_goal,
            absence_answerable=self.absence_answerable,
        )

    def domain_requests(self) -> dict[str, DomainRecallRequest]:
        return {
            domain: self.domain_request(domain)
            for domain in self.queries
        }


class MemoryIntentPlanner:
    def __init__(self, engine: Any, *, cache_size: int = 128):
        self.engine = engine
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[str, MemoryIntentPlan] = OrderedDict()

    async def plan(
        self,
        current_message: str,
        recent_messages: list[dict[str, Any]] | None = None,
        domain_catalog: dict[str, Any] | None = None,
    ) -> MemoryIntentPlan:
        cache_key = json.dumps(
            {
                "message": " ".join(current_message.casefold().split()),
                "recent": recent_messages or [],
                "catalog": domain_catalog or {},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        started = perf_counter()
        response = await self.engine.generate_response(
            messages=[
                Message(
                    role="user",
                    content=memory_intent_prompt(
                        current_message, recent_messages, domain_catalog
                    ),
                )
            ],
            system_prompt=MEMORY_INTENT_SYSTEM,
            # Intent planning is an optimization, not a correctness boundary.
            # Fail quickly to the deterministic all-domain fallback instead of
            # letting one unhealthy provider stall the entire conversation.
            max_retries=1,
            task_context="memory-intent",
            request_options={
                "temperature": 0.0,
                "max_tokens": 700,
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            },
        )
        elapsed = perf_counter() - started
        planned = MemoryIntentPlan.from_payload(
            _json_object(response),
            fallback_question=current_message,
            planning_seconds=elapsed,
        )
        if self.cache_size > 0:
            self._cache[cache_key] = planned
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return planned
