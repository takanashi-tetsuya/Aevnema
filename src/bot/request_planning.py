"""Conversation-aware planning for one memory retrieval request.

The model planner owns semantic interpretation.  The deterministic path in
this module is intentionally small and exists only when planning is disabled
or unavailable.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from time import monotonic, perf_counter
from typing import Any, Awaitable, Callable, Sequence

from config.prompt_config import retrieval_question
from src.memory.contracts import DomainRecallRequest
from src.memory.conversation import ChatTurn, ConversationSessionBuffer
from src.memory.identity import PlatformIdentity
from src.memory.intent_planner import MemoryIntentPlan, MemoryIntentPlanner
from src.memory.routing import MemoryRoute, route_memory_query
from src.utils.logger import setup_logger


logger = setup_logger(__name__)


def needs_retrieval_history(text: str) -> bool:
    """Fallback-only test for a short, context-dependent utterance."""

    normalized = " ".join(text.split())
    if len(normalized) <= 8:
        return True
    if len(normalized) > 36:
        return False
    return normalized.casefold().startswith(
        (
            "他",
            "她",
            "它",
            "他们",
            "她们",
            "这个",
            "那个",
            "这件事",
            "那件事",
            "前者",
            "后者",
            "然后",
            "后来呢",
            "还有呢",
            "what about",
            "and then",
            "why did they",
            "why did he",
            "why did she",
        )
    )


def fallback_retrieval_question(
    turns: Sequence[ChatTurn],
    text: str,
    route: MemoryRoute | None = None,
) -> str:
    """Build the conservative query used only without model planning."""

    previous = list(turns)[-4:]
    if route is not None and route.creative:
        previous = [item for item in turns if item.role == "user"][-2:]
    if not previous or not (
        needs_retrieval_history(text)
        or (route is not None and route.creative)
    ):
        return text
    history = "\n".join(f"{item.role}: {item.content}" for item in previous)
    return retrieval_question(history, text)


def bounded_recent_messages(
    turns: Sequence[ChatTurn],
    *,
    max_messages: int,
    max_chars: int,
) -> list[dict[str, str]]:
    """Return the newest complete context that fits the explicit egress cap."""

    if max_messages <= 0 or max_chars <= 0:
        return []
    selected: list[dict[str, str]] = []
    remaining = max_chars
    for turn in reversed(list(turns)[-max_messages:]):
        content = str(turn.content).strip()
        if not content:
            continue
        if len(content) > remaining:
            if remaining <= 0:
                break
            content = content[-remaining:]
        selected.append({"role": str(turn.role), "content": content})
        remaining -= len(content)
        if remaining <= 0:
            break
    selected.reverse()
    return selected


@dataclass(slots=True, frozen=True)
class PlannedChatRequest:
    route: MemoryRoute
    retrieval_question: str
    domain_requests: dict[str, DomainRecallRequest] = field(default_factory=dict)
    planning_seconds: float = 0.0
    planner: str = "fallback"
    error: str = ""
    raw_plan: dict[str, Any] = field(default_factory=dict)
    # Request-local vector created by a semantic-cache probe.  It is not part
    # of traces or prompts and can be reused by retrieval immediately.
    semantic_vector: Any | None = None

    def trace(self) -> dict[str, Any]:
        return {
            "planner": self.planner,
            "planning_seconds": self.planning_seconds,
            "error": self.error,
            "retrieval_question": self.retrieval_question,
            "domain_queries": {
                domain: request.query
                for domain, request in self.domain_requests.items()
            },
            "raw_plan": self.raw_plan,
        }


@dataclass(slots=True)
class _SemanticPlanEntry:
    stored_at: float
    session_key: str
    source_message: str
    vector: Any
    planned: PlannedChatRequest


class ChatRequestPlanner:
    """Convert a turn plus bounded conversation context into retrieval inputs."""

    def __init__(
        self,
        *,
        sessions: ConversationSessionBuffer,
        intent_planner: MemoryIntentPlanner | None,
        enabled: bool = True,
        context_messages: int = 6,
        context_chars: int = 4_000,
        semantic_embedder: Callable[[str], Awaitable[Any]] | None = None,
        association_route_matcher: (
            Callable[[str], Awaitable[dict[str, Any] | None]] | None
        ) = None,
        semantic_cache_size: int = 64,
        semantic_cache_ttl_seconds: float = 900.0,
        semantic_cache_similarity: float = 0.60,
    ):
        if context_messages < 0 or context_chars < 0:
            raise ValueError("memory intent context limits cannot be negative")
        self.sessions = sessions
        self.intent_planner = intent_planner
        self.enabled = enabled
        self.context_messages = context_messages
        self.context_chars = context_chars
        self.semantic_embedder = semantic_embedder
        self.association_route_matcher = association_route_matcher
        self.semantic_cache_size = max(0, int(semantic_cache_size))
        self.semantic_cache_ttl_seconds = max(
            0.0, float(semantic_cache_ttl_seconds)
        )
        self.semantic_cache_similarity = max(
            -1.0, min(1.0, float(semantic_cache_similarity))
        )
        self._semantic_cache: OrderedDict[str, _SemanticPlanEntry] = OrderedDict()

    async def _semantic_cache_lookup(
        self,
        session_key: str,
        current_message: str,
    ) -> PlannedChatRequest | None:
        if self.semantic_embedder is None or not self._semantic_cache:
            return None
        now = monotonic()
        current_route = route_memory_query(current_message)
        candidates: list[tuple[str, _SemanticPlanEntry]] = []
        expired: list[str] = []
        for key, entry in self._semantic_cache.items():
            if (
                self.semantic_cache_ttl_seconds > 0
                and now - entry.stored_at > self.semantic_cache_ttl_seconds
            ):
                expired.append(key)
                continue
            if entry.session_key != session_key:
                continue
            cached_route = entry.planned.route
            # The cached model plan is more trustworthy than the fallback
            # router for domain selection.  The local router is used only as
            # a fail-closed creative/non-creative boundary because creative
            # turns have different write isolation semantics.
            if cached_route.creative != current_route.creative:
                continue
            candidates.append((key, entry))
        for key in expired:
            self._semantic_cache.pop(key, None)
        if not candidates:
            return None
        started = perf_counter()
        try:
            vector = await self.semantic_embedder(current_message)
        except Exception as exc:
            logger.warning("semantic plan cache probe failed: %s", exc)
            return None
        best_key = ""
        best_entry: _SemanticPlanEntry | None = None
        best_score = -1.0
        for key, entry in candidates:
            try:
                score = float(vector @ entry.vector)
            except (TypeError, ValueError):
                continue
            if score > best_score:
                best_key, best_entry, best_score = key, entry, score
        if best_entry is None or best_score < self.semantic_cache_similarity:
            return None
        self._semantic_cache.move_to_end(best_key)
        raw_plan = dict(best_entry.planned.raw_plan)
        raw_plan["semantic_plan_cache"] = {
            "hit": True,
            "cosine": round(best_score, 6),
            "source_message": best_entry.source_message,
        }
        current_requests: dict[str, DomainRecallRequest] = {}
        for domain, previous in best_entry.planned.domain_requests.items():
            previous_intent = dict(previous.intent_override)
            intent_override = {
                "language": previous_intent.get("language", "zh"),
                "target_entities": [],
                "search_queries": [current_message],
                "requested_relation": "",
                "temporal_constraint": "",
                "causal_constraint": "",
                "answer_shape": previous_intent.get(
                    "answer_shape", "evidence_bound_answer"
                ),
                "uncertainty_required": bool(
                    previous_intent.get("uncertainty_required", True)
                ),
            }
            current_requests[domain] = DomainRecallRequest(
                query=current_message,
                intent_override=intent_override,
                followup_queries=(),
                evidence_goal=previous.evidence_goal,
                absence_answerable=previous.absence_answerable,
            )
        cached_route = best_entry.planned.route
        effective_route = replace(
            cached_route,
            intensity=(
                "deep"
                if current_route.intensity == "deep"
                else "standard"
                if cached_route.intensity == "deep"
                else cached_route.intensity
            ),
            reason="semantic_plan_cache",
        )
        return replace(
            best_entry.planned,
            route=effective_route,
            retrieval_question=current_message,
            planning_seconds=round(perf_counter() - started, 6),
            planner="semantic_cache",
            error="",
            raw_plan=raw_plan,
            domain_requests=current_requests,
            semantic_vector=vector,
        )

    def remember_semantic_plan(
        self,
        *,
        session_key: str,
        current_message: str,
        planned: PlannedChatRequest,
        vector: Any,
        observed_domains: dict[str, Any] | None = None,
    ) -> None:
        """Remember an executed plan only after retrieval produced its vector."""

        planned_for_cache = planned
        if planned.planner == "fallback":
            observed = observed_domains or {}

            def has_sufficient_evidence(name: str) -> bool:
                value = observed.get(name)
                if not isinstance(value, dict):
                    return False
                quality = value.get("retrieval_quality")
                return bool(
                    isinstance(quality, dict)
                    and quality.get("sufficient") is True
                    and value.get("evidence_episodes")
                )

            selected = {
                "private": has_sufficient_evidence("user"),
                "public": has_sufficient_evidence("public"),
                "knowledge": has_sufficient_evidence("knowledge"),
            }
            if not any(selected.values()):
                return
            refined_route = MemoryRoute(
                user=selected["private"],
                public=selected["public"],
                knowledge=selected["knowledge"],
                reason="retrieval_evidence_refined_fallback",
                intensity="standard",
                knowledge_write_policy=planned.route.knowledge_write_policy,
                creative=False,
            )
            refined_requests = {
                domain: DomainRecallRequest(
                    query=current_message,
                    intent_override={"search_queries": [current_message]},
                    followup_queries=(),
                )
                for domain, enabled in selected.items()
                if enabled
            }
            planned_for_cache = replace(
                planned,
                route=refined_route,
                domain_requests=refined_requests,
            )
        if (
            self.semantic_cache_size <= 0
            or vector is None
            or planned_for_cache.planner
            not in {"model", "semantic_cache", "fallback"}
            or planned_for_cache.route.creative
            or not planned_for_cache.route.knowledge
        ):
            return
        key = f"{session_key}\0{' '.join(current_message.casefold().split())}"
        stored_vector = vector.copy() if hasattr(vector, "copy") else vector
        self._semantic_cache[key] = _SemanticPlanEntry(
            stored_at=monotonic(),
            session_key=session_key,
            source_message=current_message,
            vector=stored_vector,
            planned=planned_for_cache,
        )
        self._semantic_cache.move_to_end(key)
        while len(self._semantic_cache) > self.semantic_cache_size:
            self._semantic_cache.popitem(last=False)

    async def plan(
        self,
        *,
        identity: PlatformIdentity,
        current_message: str,
        conversation_key: str | None = None,
    ) -> PlannedChatRequest:
        session_key = conversation_key or identity.key
        turns = self.sessions.get(session_key)
        if self.enabled and self.intent_planner is not None:
            current_route = route_memory_query(current_message)
            if (
                self.association_route_matcher is not None
                and not current_route.creative
            ):
                try:
                    cue = await self.association_route_matcher(current_message)
                except Exception as exc:
                    logger.warning("association route cache probe failed: %s", exc)
                    cue = None
                if cue:
                    route = MemoryRoute(
                        user=False,
                        public=False,
                        knowledge=True,
                        reason="audited_association_cache",
                        intensity="standard",
                        knowledge_write_policy="evidence_gated",
                        creative=False,
                    )
                    return PlannedChatRequest(
                        route=route,
                        retrieval_question=current_message,
                        domain_requests={
                            "knowledge": DomainRecallRequest(
                                query=current_message,
                                intent_override={
                                    "search_queries": [current_message],
                                    "uncertainty_required": True,
                                },
                                followup_queries=(),
                            )
                        },
                        planner="association_cache",
                        raw_plan={"association_route_cache": dict(cue)},
                    )
            cached = await self._semantic_cache_lookup(
                session_key, current_message
            )
            if cached is not None:
                return cached
            # Let the semantic planner decide whether recent conversation is
            # relevant.  Phrase-level rules cannot reliably enumerate every
            # way a user may refer back to an earlier person or event.
            context = bounded_recent_messages(
                turns,
                max_messages=self.context_messages,
                max_chars=self.context_chars,
            )
            try:
                intent = await self.intent_planner.plan(
                    current_message,
                    recent_messages=context,
                    domain_catalog={
                        "private": "current platform user only",
                        "public": "shared across users",
                        "knowledge": "imported documents and derived lore",
                    },
                )
                return self._from_model(intent, current_message)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("memory intent planning failed; using fallback: %s", error)
                return self._fallback(turns, current_message, error=error)
        return self._fallback(turns, current_message)

    @staticmethod
    def _from_model(
        intent: MemoryIntentPlan,
        current_message: str,
    ) -> PlannedChatRequest:
        requests = intent.domain_requests()
        primary_query = (
            requests.get("knowledge")
            or requests.get("private")
            or requests.get("public")
        )
        return PlannedChatRequest(
            route=intent.route,
            retrieval_question=(
                primary_query.query if primary_query is not None else current_message
            ),
            domain_requests=requests,
            planning_seconds=intent.planning_seconds,
            planner="model",
            raw_plan=dict(intent.raw or {}),
        )

    @staticmethod
    def _fallback(
        turns: Sequence[ChatTurn],
        current_message: str,
        *,
        error: str = "",
    ) -> PlannedChatRequest:
        route = route_memory_query(current_message)
        query = fallback_retrieval_question(turns, current_message, route)
        return PlannedChatRequest(
            route=route,
            retrieval_question=query,
            planning_seconds=0.0,
            planner="fallback",
            error=error,
        )
