from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import inspect
import re
from time import perf_counter
from typing import Any, Awaitable, Callable

from config.prompt_config import (
    CACHED_RECALL_ADDENDUM,
    CREATIVE_DETAIL_ADDENDUM,
    UNRESOLVED_RECALL_ADDENDUM,
    creative_memory_context,
    knowledge_consolidation_query,
    route_prompt_addendum,
)
from src.memory.conversation import (
    ChatTurn,
    ConversationIngestionWorker,
    ConversationJournal,
    ConversationSessionBuffer,
)
from src.memory.identity import PlatformIdentity
from src.memory.growth import BackgroundGrowthWorker
from src.memory.answer_consolidator import (
    derive_evidence_bridge_candidate,
    derive_contextual_recall_candidates,
    derive_contextual_utility_observations,
    inline_memory_prompt,
    split_inline_memory_response,
)
from src.memory.service import (
    DeadlineBudget,
    MemoryRoute,
    RetrievedMemory,
    route_memory_query,
)
from src.bot.memory_guard import PrivateMemoryResponseGuard
from src.bot.request_planning import (
    ChatRequestPlanner,
    PlannedChatRequest,
    fallback_retrieval_question,
    needs_retrieval_history,
)
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

_BRIDGE_OBSERVATION_RE = re.compile(
    r"槽[“\"](?P<slot>.*?)[”\"].*?其观察为：(?P<observation>.*?)(?=；槽|；此边|$)",
    re.DOTALL,
)
_CAUSAL_ASSERTION_RE = re.compile(
    r"导致|造成|促成|引发|使得|源于|因此|所以"
)
_NEGATION_MARKERS = ("不", "未", "并非", "不能", "无法", "不足以", "没有")


@dataclass(slots=True)
class ChatReply:
    text: str
    route: MemoryRoute
    memory: RetrievedMemory
    memory_guard: dict[str, Any] = field(default_factory=dict)
    memory_consolidation: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    _finalizer: Callable[[], Awaitable[None]] | None = field(
        default=None, repr=False
    )
    _finalized: bool = field(default=False, repr=False)

    async def finalize(self) -> None:
        """Commit local conversation side effects at most once."""

        if self._finalized:
            return
        if self._finalizer is not None:
            await self._finalizer()
        self._finalized = True


class ConversationCoordinator:
    """One message lifecycle independent of Telegram and storage internals."""

    def __init__(
        self,
        *,
        chat_engine: Any,
        fast_chat_engine: Any | None = None,
        memory_system: Any,
        system_prompt_factory: Callable[[str], str],
        sessions: ConversationSessionBuffer,
        private_memory_guard: PrivateMemoryResponseGuard | None = None,
        journal: ConversationJournal | None = None,
        ingestion_worker: ConversationIngestionWorker | None = None,
        growth_worker: BackgroundGrowthWorker | None = None,
        request_planner: ChatRequestPlanner | None = None,
        generation_options_factory: Callable[[PlatformIdentity], dict[str, Any]]
        | None = None,
    ):
        self.chat_engine = chat_engine
        self.fast_chat_engine = fast_chat_engine
        self.memory_system = memory_system
        self.system_prompt_factory = system_prompt_factory
        self.sessions = sessions
        self.private_memory_guard = private_memory_guard
        self.journal = journal
        self.ingestion_worker = ingestion_worker
        self.growth_worker = growth_worker
        self.request_planner = request_planner or ChatRequestPlanner(
            sessions=sessions,
            intent_planner=None,
            enabled=False,
        )
        self.generation_options_factory = generation_options_factory

    @staticmethod
    def _accepts_keyword(call: Any, name: str) -> bool:
        """Keep older engines/test doubles compatible without masking call errors."""

        try:
            parameters = inspect.signature(call).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            item.kind is inspect.Parameter.VAR_KEYWORD or item.name == name
            for item in parameters
        )

    @staticmethod
    def route(text: str) -> MemoryRoute:
        return route_memory_query(text)

    @staticmethod
    def _needs_retrieval_history(text: str) -> bool:
        return needs_retrieval_history(text)

    def _retrieval_question(
        self,
        identity: PlatformIdentity,
        text: str,
        route: MemoryRoute | None = None,
        *,
        conversation_key: str | None = None,
    ) -> str:
        session = self.sessions.get(conversation_key or identity.key)
        return fallback_retrieval_question(session, text, route)

    @staticmethod
    def _route_prompt_addendum(route: MemoryRoute) -> str:
        return route_prompt_addendum(
            creative=route.creative,
            intensity=route.intensity,
        )

    @staticmethod
    def _creative_memory_context(
        memory: RetrievedMemory,
        focus_text: str = "",
    ) -> str:
        """Expose private events and lore vocabulary, not old plot events.

        A historical Episode is valid evidence for a factual question, but it
        is a dangerous template for “who messaged me just now?”.  Creative
        turns therefore receive current/private evidence verbatim while the
        shared story domain contributes only Concept names and descriptions.
        """

        return creative_memory_context(memory.raw_result, focus_text)

    @staticmethod
    def _all_recalled_domains_cached(
        memory: RetrievedMemory, route: MemoryRoute
    ) -> bool:
        domains = memory.raw_result.get("domains", {})
        if not isinstance(domains, dict):
            return False
        required = [
            name
            for name, enabled in (
                ("user", route.user),
                ("public", route.public),
                ("knowledge", route.knowledge),
            )
            if enabled
        ]
        return bool(required) and all(
            isinstance(domains.get(name), dict)
            and (
                bool(domains[name].get("service_cache_hit"))
                or (
                    (domains[name].get("rerank_trace") or {}).get(
                        "cache_kind"
                    )
                    == "audited_association_cue"
                )
            )
            for name in required
        )

    @staticmethod
    def _all_recalled_domains_unresolved(
        memory: RetrievedMemory, route: MemoryRoute
    ) -> bool:
        domains = memory.raw_result.get("domains", {})
        if not isinstance(domains, dict):
            return False
        required = [
            name
            for name, enabled in (
                ("user", route.user),
                ("public", route.public),
                ("knowledge", route.knowledge),
            )
            if enabled
        ]
        if not required:
            return False
        for name in required:
            value = domains.get(name)
            if not isinstance(value, dict):
                return False
            if value.get("empty_domain"):
                continue
            quality = value.get("retrieval_quality")
            if not isinstance(quality, dict):
                return False
            if quality.get("sufficient") is True:
                return False
        return True

    @staticmethod
    def _association_capsule_reply(memory: RetrievedMemory) -> str:
        """Render a dynamic, audited cached relation when the gateway is down."""

        domains = memory.raw_result.get("domains", {})
        if not isinstance(domains, dict):
            return ""
        for value in domains.values():
            if not isinstance(value, dict):
                continue
            rerank = value.get("rerank_trace") or {}
            if rerank.get("cache_kind") != "audited_association_cue":
                continue
            for capsule in rerank.get("association_capsules") or []:
                relation = str(capsule.get("relation_text") or "").strip()
                if relation:
                    if str(capsule.get("relation_key") or "").casefold() == (
                        "evidence_bridge"
                    ):
                        observations: list[str] = []
                        for match in _BRIDGE_OBSERVATION_RE.finditer(relation):
                            slot_chars = {
                                char
                                for char in match.group("slot")
                                if char.isalnum() or "\u3400" <= char <= "\u9fff"
                            }
                            sentences = [
                                value.strip("；。 ")
                                for value in re.split(
                                    r"[。！？!?]", match.group("observation")
                                )
                                if value.strip("；。 ")
                            ]
                            if not sentences:
                                continue
                            best = max(
                                sentences,
                                key=lambda value: (
                                    len(slot_chars.intersection(set(value))),
                                    -sentences.index(value),
                                ),
                            )
                            observations.append(best[:180])
                        if len(observations) >= 2:
                            return (
                                f"老师，我能确认两件事：{observations[0]}；另外，"
                                f"{observations[1]}。两者可以一起参考，但现有证据"
                                "不足以直接判为因果。"
                            )
                        facts = relation.removeprefix("查询综合推论：").strip()
                        for marker in (
                            "；此关联只用于",
                            "；该关联只用于",
                            "；只用于",
                        ):
                            facts = facts.split(marker, 1)[0].strip()
                        facts = facts.replace("一段证据记录", "", 1)
                        facts = facts.replace("，另一段记录", "；", 1)
                        facts = facts.strip("；。 ")
                        if facts:
                            return (
                                f"老师，我记得相关的两件事：{facts}。"
                                "它们可以一起作为线索，但现有证据不足以把二者直接判为因果。"
                            )
                    return f"老师，我记得这两件事是这样连起来的：{relation[:500]}"
        return ""

    @staticmethod
    def _direct_evidence_reply(memory: RetrievedMemory) -> str:
        """Return a bounded source-grounded answer when generation is down."""

        domains = memory.raw_result.get("domains", {})
        if not isinstance(domains, dict):
            return ""
        excerpts: list[str] = []
        seen: set[str] = set()
        for value in domains.values():
            if not isinstance(value, dict):
                continue
            for row in value.get("evidence_episodes") or []:
                if not isinstance(row, dict):
                    continue
                try:
                    generation = int(row.get("generation", 0) or 0)
                except (TypeError, ValueError):
                    continue
                origin = str(row.get("evidence_origin") or "source")
                if generation != 0 or origin not in {"", "source"}:
                    continue
                excerpt = re.sub(r"\s+", " ", str(row.get("text") or "")).strip()
                if not excerpt or excerpt in seen:
                    continue
                seen.add(excerpt)
                excerpts.append(excerpt[:360])
                if len(excerpts) >= 2:
                    break
            if len(excerpts) >= 2:
                break
        if not excerpts:
            return ""
        joined = "；另外，".join(excerpts)
        return (
            f"老师，回答模型暂时没有及时完成。我先依据检索到的直接资料说明："
            f"{joined}。"
        )

    @staticmethod
    def _deadline_limit_reply(memory: RetrievedMemory) -> str:
        """A deterministic last resort when no model time remains."""

        direct = ConversationCoordinator._direct_evidence_reply(memory)
        if direct:
            return direct
        return "老师，本轮检索或回答未能在时限内完成；我不会把未验证的内容当作结论。"

    @staticmethod
    def _evidence_bridge_boundary_violation(
        reply: str,
        memory: RetrievedMemory,
    ) -> bool:
        """Reject positive causal upgrades of a retrieval-only bridge."""

        domains = memory.raw_result.get("domains", {})
        if not isinstance(domains, dict):
            return False
        has_bridge = False
        for value in domains.values():
            if not isinstance(value, dict):
                continue
            rerank = value.get("rerank_trace") or {}
            if rerank.get("cache_kind") != "audited_association_cue":
                continue
            has_bridge = any(
                str(item.get("relation_key") or "").casefold()
                == "evidence_bridge"
                for item in list(rerank.get("association_capsules") or [])
                if isinstance(item, dict)
            )
            if has_bridge:
                break
        if not has_bridge:
            return False
        for match in _CAUSAL_ASSERTION_RE.finditer(reply):
            prefix = reply[max(0, match.start() - 8) : match.start()]
            if any(marker in prefix for marker in _NEGATION_MARKERS):
                continue
            return True
        return False

    async def handle(
        self,
        *,
        identity: PlatformIdentity,
        text: str,
        conversation_key: str | None = None,
        defer_commit: bool = False,
    ) -> ChatReply:
        started = perf_counter()
        deadline_budget = DeadlineBudget()

        async def await_with_timeout(
            operation: Awaitable[Any],
            timeout: float,
        ) -> Any:
            """Stop waiting at the budget even if a provider ignores cancel."""

            task = asyncio.ensure_future(operation)
            done, _pending = await asyncio.wait(
                {task}, timeout=max(0.0, timeout)
            )
            if done:
                return task.result()
            task.cancel()

            def consume_late_result(late_task: asyncio.Future[Any]) -> None:
                try:
                    late_task.result()
                except (asyncio.CancelledError, Exception):
                    pass

            task.add_done_callback(consume_late_result)
            raise TimeoutError("request phase deadline exhausted")

        clean_text = text.strip()
        session_key = conversation_key or identity.key
        planning_started = perf_counter()
        try:
            planned_request = await await_with_timeout(
                self.request_planner.plan(
                    identity=identity,
                    current_message=clean_text,
                    conversation_key=session_key,
                ),
                max(0.1, deadline_budget.planning_timeout()),
            )
        except TimeoutError:
            fallback_route = self.route(clean_text)
            planned_request = PlannedChatRequest(
                route=fallback_route,
                retrieval_question=fallback_retrieval_question(
                    self.sessions.get(session_key), clean_text, fallback_route
                ),
                planner="deadline_fallback",
                error="planning deadline exhausted",
            )
        planning_finished = perf_counter()
        route = planned_request.route
        query = planned_request.retrieval_question
        messages = self.sessions.get(session_key)
        messages.append(ChatTurn("user", clean_text))
        retrieval_started = perf_counter()
        recall = self.memory_system.recall
        recall_kwargs: dict[str, Any] = {"route": route}
        if self._accepts_keyword(recall, "deadline_budget"):
            recall_kwargs["deadline_budget"] = deadline_budget
        if self._accepts_keyword(recall, "domain_requests"):
            recall_kwargs["domain_requests"] = planned_request.domain_requests
        if (
            planned_request.semantic_vector is not None
            and self._accepts_keyword(recall, "query_embeddings_override")
        ):
            recall_kwargs["query_embeddings_override"] = {
                query: planned_request.semantic_vector
            }
        memory = await recall(identity, query, **recall_kwargs)
        remember_plan = getattr(
            self.request_planner, "remember_semantic_plan", None
        )
        if callable(remember_plan):
            remember_plan(
                session_key=session_key,
                current_message=clean_text,
                planned=planned_request,
                vector=memory.semantic_vector,
                observed_domains=(
                    memory.raw_result.get("domains")
                    if isinstance(memory.raw_result, dict)
                    else None
                ),
            )
        memory.raw_result.setdefault(
            "intent_planning", planned_request.trace()
        )
        memory.raw_result["request_deadline_budget"] = deadline_budget.as_dict()
        retrieval_finished = perf_counter()
        prompt_context = (
            self._creative_memory_context(memory, query)
            if route.creative
            else memory.context
        )
        prompt = self.system_prompt_factory(prompt_context)
        prompt += self._route_prompt_addendum(route)
        if route.creative:
            prompt += CREATIVE_DETAIL_ADDENDUM
        cached_recall = self._all_recalled_domains_cached(memory, route)
        unresolved_recall = self._all_recalled_domains_unresolved(
            memory, route
        )
        if route.intensity == "standard" and cached_recall:
            prompt += CACHED_RECALL_ADDENDUM
        elif route.intensity == "standard" and unresolved_recall:
            prompt += UNRESOLVED_RECALL_ADDENDUM
        if (
            route.knowledge
            and route.knowledge_write_policy != "none"
            and not cached_recall
        ):
            prompt += inline_memory_prompt()
        if self.private_memory_guard is not None:
            prompt += self.private_memory_guard.prompt_addendum(
                clean_text, memory, route
            )
        generation_started = perf_counter()
        local_cached_reply = (
            self._association_capsule_reply(memory)
            if cached_recall and route.knowledge and not route.creative
            else ""
        )
        if local_cached_reply:
            # A dual-audited capsule is already the compact answer contract.
            # Rephrasing it remotely repeatedly introduced unsupported causal
            # claims and was then discarded by the boundary guard.
            active_chat_engine = None

            async def generate(**_kwargs):
                return local_cached_reply

        else:
            active_chat_engine = (
                self.fast_chat_engine
                if self.fast_chat_engine is not None
                and (
                    route.intensity == "light"
                    or unresolved_recall
                    or (
                        route.intensity == "standard"
                        and cached_recall
                    )
                )
                else self.chat_engine
            )
            generate = active_chat_engine.generate_response
        generation_kwargs: dict[str, Any] = {
            "messages": messages,
            "system_prompt": prompt,
        }
        if self._accepts_keyword(generate, "task_context"):
            generation_kwargs["task_context"] = (
                f"{identity.key}:chat-fast"
                if active_chat_engine is self.fast_chat_engine
                else f"{identity.key}:chat"
            )
        if (
            self.generation_options_factory is not None
            and self._accepts_keyword(generate, "request_options")
        ):
            configured_options = self.generation_options_factory(identity)
            if hasattr(configured_options, "model_dump"):
                configured_options = configured_options.model_dump(
                    exclude_none=True
                )
            generation_kwargs["request_options"] = configured_options
        if (unresolved_recall or cached_recall) and self._accepts_keyword(
            generate, "request_options"
        ):
            compact_options = dict(
                generation_kwargs.get("request_options") or {}
            )
            compact_options["max_tokens"] = min(
                140, int(compact_options.get("max_tokens", 140))
            )
            temperature_limit = 0.2 if cached_recall else 0.4
            compact_options["temperature"] = min(
                temperature_limit,
                float(compact_options.get("temperature", temperature_limit)),
            )
            generation_kwargs["request_options"] = compact_options

        async def generate_with_budget(
            call: Any,
            kwargs: dict[str, Any],
            timeout: float,
        ) -> str:
            if timeout <= 0:
                raise TimeoutError("answer deadline exhausted")
            bounded_kwargs = dict(kwargs)
            if self._accepts_keyword(call, "deadline_seconds"):
                bounded_kwargs["deadline_seconds"] = max(0.1, timeout)
            return await await_with_timeout(call(**bounded_kwargs), timeout)

        try:
            raw_reply = await generate_with_budget(
                generate,
                generation_kwargs,
                deadline_budget.answer_timeout(),
            )
        except Exception as exc:
            if active_chat_engine is not self.fast_chat_engine:
                logger.warning(
                    "main chat generation failed; returning local bounded reply: %s",
                    exc,
                )
                raw_reply = self._deadline_limit_reply(memory)
            capsule_reply = self._association_capsule_reply(memory)
            if active_chat_engine is self.fast_chat_engine and capsule_reply:
                logger.warning(
                    "fast chat generation failed; returning audited association capsule: %s",
                    exc,
                )
                raw_reply = capsule_reply
            elif active_chat_engine is self.fast_chat_engine:
                direct_reply = self._direct_evidence_reply(memory)
                if direct_reply:
                    logger.warning(
                        "fast chat generation failed; returning direct evidence: %s",
                        exc,
                    )
                    raw_reply = direct_reply
                else:
                    # Without direct evidence, preserve the general roleplay
                    # fallback so an adapter event is not silently dropped.
                    logger.warning(
                        "fast chat generation failed; using main model: %s", exc
                    )
                    fallback_generate = self.chat_engine.generate_response
                    fallback_kwargs = dict(generation_kwargs)
                    if self._accepts_keyword(fallback_generate, "task_context"):
                        fallback_kwargs["task_context"] = (
                            f"{identity.key}:chat-fallback"
                        )
                    try:
                        raw_reply = await generate_with_budget(
                            fallback_generate,
                            fallback_kwargs,
                            deadline_budget.fallback_timeout(),
                        )
                    except Exception as fallback_exc:
                        logger.warning(
                            "fallback chat generation failed; returning local bounded reply: %s",
                            fallback_exc,
                        )
                        raw_reply = self._deadline_limit_reply(memory)
        reply, consolidation = split_inline_memory_response(
            raw_reply, memory.raw_result
        )
        if not local_cached_reply and self._evidence_bridge_boundary_violation(
            reply, memory
        ):
            capsule_reply = self._association_capsule_reply(memory)
            if capsule_reply:
                logger.warning(
                    "cached evidence bridge answer upgraded correlation to causation; "
                    "using local grounded rendering"
                )
                reply = capsule_reply
                consolidation = {
                    "knowledge_candidates": [],
                    "private_candidates": [],
                    "rejected_claims": [
                        {
                            "claim": "assistant draft",
                            "reason": "evidence bridge causal boundary violation",
                        }
                    ],
                    "knowledge_query": "",
                }
        if (
            route.knowledge
            and route.knowledge_write_policy != "none"
            and not cached_recall
            and not consolidation.get("knowledge_candidates")
        ):
            bridge = derive_evidence_bridge_candidate(query, memory.raw_result)
            if bridge is not None:
                consolidation["knowledge_candidates"] = [bridge]
                consolidation["knowledge_query"] = knowledge_consolidation_query(
                    [
                        {
                            "candidate_inference": bridge["claim"],
                            "foreground_premise_episode_ids": bridge[
                                "premise_episode_ids"
                            ],
                            "inference_type": bridge["inference_type"],
                            "confidence": bridge["confidence"],
                        }
                    ]
                )
        if not route.user:
            # A story answer is still journaled as dialogue, but the model may
            # not label canonical restatements as newly created private events
            # in the claim-level decision.
            consolidation["private_candidates"] = []
        if route.creative and not consolidation.get("private_candidates"):
            # The entire newly invented scene is private by route.  This local
            # fallback needs no second model call and can never authorize a
            # shared-lore write because it creates no knowledge candidate.
            consolidation["private_candidates"] = [
                {
                    "claim": reply[:700],
                    "reason": "creative turn is private by routing policy",
                }
            ]
        generation_finished = perf_counter()
        guard_data: dict[str, Any] = {}
        guard_started = perf_counter()
        if self.private_memory_guard is not None:
            guarded = await self.private_memory_guard.enforce(
                question=clean_text,
                draft=reply,
                memory=memory,
                route=route,
                messages=messages,
                system_prompt=prompt,
                task_context=f"{identity.key}:chat",
            )
            reply = guarded.text
            guard_data = guarded.as_dict()
            if guarded.rewritten or guarded.emergency_fallback:
                # Candidates described the discarded draft, not the visible
                # answer, so none may become durable memory.
                consolidation = {
                    "knowledge_candidates": [],
                    "private_candidates": [],
                    "rejected_claims": [
                        {
                            "claim": "assistant draft",
                            "reason": "private-memory guard replaced the draft",
                        }
                    ],
                    "knowledge_query": "",
                }
        # Contextual candidates are derived only after the visible answer has
        # survived the guard.  They contain IDs and query references, never
        # model-written claims; creative turns and guard rewrites therefore
        # cannot contaminate shared lore.
        if (
            route.knowledge
            and route.knowledge_write_policy != "none"
            and not route.creative
            and not guard_data.get("rewritten")
            and not guard_data.get("emergency_fallback")
        ):
            contextual_candidates = derive_contextual_recall_candidates(
                query,
                memory.raw_result,
            )
            contextual_observations = derive_contextual_utility_observations(
                memory.raw_result,
                query_hash=hashlib.sha256(
                    " ".join(query.casefold().split()).encode("utf-8")
                ).hexdigest(),
            )
            if contextual_candidates:
                consolidation["contextual_candidates"] = contextual_candidates
            if contextual_observations:
                consolidation["contextual_observations"] = contextual_observations
        guard_finished = perf_counter()
        generated_at = perf_counter()
        chat_reply: ChatReply

        async def finalize() -> None:
            postprocess_started = perf_counter()
            if self.growth_worker is not None:
                enqueue = self.growth_worker.enqueue
                growth_kwargs: dict[str, Any] = {}
                if self._accepts_keyword(enqueue, "consolidation_decision"):
                    growth_kwargs["consolidation_decision"] = consolidation
                if self._accepts_keyword(enqueue, "memory"):
                    growth_kwargs["memory"] = memory
                await enqueue(identity, query, route, **growth_kwargs)
            if self.journal is not None:
                ready = await self.journal.record_exchange(
                    identity, clean_text, reply
                )
                if self.ingestion_worker is not None:
                    try:
                        await self.ingestion_worker.enqueue(ready)
                    except Exception as exc:
                        # A rotated .ready file is crash-recoverable and will
                        # be discovered on the next worker start.
                        logger.error(
                            "对话批次已落盘，但当前入队失败：%s", exc
                        )
            self.sessions.add(session_key, "user", clean_text)
            self.sessions.add(session_key, "assistant", reply)
            committed_at = perf_counter()
            chat_reply.timings["postprocess_seconds"] = round(
                committed_at - postprocess_started, 6
            )
            chat_reply.timings["total_seconds"] = round(
                committed_at - started, 6
            )

        chat_reply = ChatReply(
            text=reply,
            route=route,
            memory=memory,
            memory_guard=guard_data,
            memory_consolidation=consolidation,
            timings={
                "memory_intent_seconds": round(
                    planning_finished - planning_started, 6
                ),
                "memory_recall_seconds": round(
                    retrieval_finished - retrieval_started, 6
                ),
                "chat_generation_seconds": round(
                    generation_finished - generation_started, 6
                ),
                "memory_guard_seconds": round(
                    guard_finished - guard_started, 6
                ),
                "deadline_remaining_seconds": round(
                    deadline_budget.remaining(), 6
                ),
                "postprocess_seconds": 0.0,
                "total_seconds": round(generated_at - started, 6),
            },
            _finalizer=finalize,
        )
        if not defer_commit:
            await chat_reply.finalize()
        return chat_reply

    def clear_recent_context(
        self,
        identity: PlatformIdentity,
        *,
        conversation_key: str | None = None,
    ) -> None:
        self.sessions.clear(conversation_key or identity.key)
