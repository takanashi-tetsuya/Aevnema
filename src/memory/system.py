from __future__ import annotations

import asyncio
from copy import deepcopy
import inspect
from pathlib import Path
from typing import Any

from config.prompt_config import render_domain_memory_section

from .config import MemorySystemConfig
from .contracts import DeadlineBudget, DomainRecallRequest, RetrievedMemory
from .domain import AssociativeMemoryService
from .identity import PlatformIdentity
from .routing import MemoryRoute, route_memory_query


class MemorySystem:
    """Three domains with fast foreground reads and isolated deep growth."""

    def __init__(self, config: MemorySystemConfig):
        self.config = config
        self.knowledge = AssociativeMemoryService(
            config.domain_config("knowledge", config.knowledge_database_path)
        )
        self.public = AssociativeMemoryService(
            config.domain_config("public", config.public_database_path)
        )
        self._users: dict[str, AssociativeMemoryService] = {}
        self._user_lock = asyncio.Lock()
        self._background: dict[str, AssociativeMemoryService] = {}
        self._background_lock = asyncio.Lock()

    async def initialize(self) -> None:
        self.config.user_database_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.gather(self.knowledge.initialize(), self.public.initialize())

    async def embed_query_text(self, text: str):
        """Use the shared knowledge embedding model for semantic cache probes."""

        return await self.knowledge.embed_query_text(text)

    async def match_association_cue(self, text: str) -> dict[str, Any] | None:
        """Return a fail-closed match from the shared lore graph, if any."""

        return await self.knowledge.match_association_cue(text)

    async def _user_service(
        self, identity: PlatformIdentity
    ) -> AssociativeMemoryService:
        key = identity.key
        existing = self._users.get(key)
        if existing is not None:
            return existing
        async with self._user_lock:
            existing = self._users.get(key)
            if existing is None:
                user_dir = (
                    self.config.user_database_dir
                    / identity.platform
                    / identity.storage_key
                )
                identity.write_metadata(user_dir)
                database = user_dir / "memory.db"
                existing = AssociativeMemoryService(
                    self.config.domain_config(f"user:{key}", database)
                )
                self._users[key] = existing
            return existing

    async def _background_service(
        self,
        domain: str,
        database_path: Path,
    ) -> AssociativeMemoryService:
        existing = self._background.get(domain)
        if existing is not None:
            return existing
        async with self._background_lock:
            existing = self._background.get(domain)
            if existing is None:
                existing = AssociativeMemoryService(
                    self.config.domain_config(
                        domain,
                        database_path,
                        background=True,
                    )
                )
                self._background[domain] = existing
            return existing

    async def _background_user_service(
        self,
        identity: PlatformIdentity,
    ) -> AssociativeMemoryService:
        user_dir = (
            self.config.user_database_dir
            / identity.platform
            / identity.storage_key
        )
        identity.write_metadata(user_dir)
        return await self._background_service(
            f"user:{identity.key}", user_dir / "memory.db"
        )

    async def recall(
        self,
        identity: PlatformIdentity,
        question: str,
        route: MemoryRoute | None = None,
        *,
        domain_requests: dict[str, DomainRecallRequest] | None = None,
        query_embeddings_override: dict[str, Any] | None = None,
        query_vector_bundle: Any | None = None,
        deadline_budget: DeadlineBudget | None = None,
    ) -> RetrievedMemory:
        selected = route or route_memory_query(question)
        requests = domain_requests or {}
        private_request = requests.get("private")
        public_request = requests.get("public")
        knowledge_request = requests.get("knowledge")
        private_question = private_request.query if private_request else question
        public_question = public_request.query if public_request else question
        knowledge_question = knowledge_request.query if knowledge_request else question

        def accepts_keyword(call: Any, name: str) -> bool:
            try:
                parameters = inspect.signature(call).parameters.values()
            except (TypeError, ValueError):
                return False
            return any(
                item.kind is inspect.Parameter.VAR_KEYWORD or item.name == name
                for item in parameters
            )

        async def recall_domain(service: Any, prompt: str, **kwargs: Any):
            if deadline_budget is not None:
                remaining = deadline_budget.retrieval_timeout()
                if remaining < 0.1:
                    return RetrievedMemory(
                        context="",
                        raw_result={"deadline_budget_exhausted": True},
                        error="request retrieval budget exhausted",
                    )
                if accepts_keyword(service.recall, "deadline_seconds"):
                    kwargs["deadline_seconds"] = remaining
            return await service.recall(prompt, **kwargs)

        def missing_slot_request(
            value: RetrievedMemory,
            original_request: DomainRecallRequest | None,
            fallback_question: str,
        ) -> DomainRecallRequest | None:
            """Make a deep repair request from auditable unsatisfied slots.

            A generic ``insufficient`` flag is deliberately not enough to
            restart the whole query.  The repair retains the first pass's
            intent and vector bundle, and names only slots that the first
            pass's coverage proof could not satisfy.
            """
            raw = value.raw_result if isinstance(value.raw_result, dict) else {}
            slot_trace = raw.get("evidence_slot_trace") or {}
            selector = (
                slot_trace.get("slot_selector_v2")
                if isinstance(slot_trace, dict)
                else None
            )
            queries: list[str] = []
            if isinstance(selector, dict):
                missing = {
                    str(item).strip()
                    for item in selector.get("masked_missing_slots", [])
                    if str(item).strip()
                }
                for slot in selector.get("slots", []):
                    if not isinstance(slot, dict):
                        continue
                    if str(slot.get("slot_id", "")).strip() not in missing:
                        continue
                    question = " ".join(str(slot.get("question", "")).split())
                    if question:
                        queries.append(question)
            if not queries and isinstance(slot_trace, dict):
                for slot in slot_trace.get("coverage_slots", []):
                    if not isinstance(slot, dict) or slot.get("satisfied", True):
                        continue
                    question = " ".join(str(slot.get("query", "")).split())
                    if question:
                        queries.append(question)
            queries = list(dict.fromkeys(queries))[:4]
            if not queries:
                return None
            prior_intent = raw.get("intent")
            intent = (
                deepcopy(prior_intent)
                if isinstance(prior_intent, dict)
                else deepcopy(original_request.intent_override)
                if original_request is not None
                else {}
            )
            return DomainRecallRequest(
                query="；".join(queries) or fallback_question,
                intent_override=intent,
                followup_queries=tuple(queries),
                evidence_goal=(
                    original_request.evidence_goal
                    if original_request is not None
                    else "lookup"
                ),
                absence_answerable=(
                    original_request.absence_answerable
                    if original_request is not None
                    else False
                ),
            )

        if (
            query_vector_bundle is None
            and getattr(getattr(self, "config", None), "contextual_association_enabled", False)
        ):
            try:
                bundle_timeout = (
                    deadline_budget.retrieval_timeout()
                    if deadline_budget is not None
                    else None
                )
                if bundle_timeout is not None and bundle_timeout < 0.1:
                    raise TimeoutError("request retrieval budget exhausted")
                bundle_call = self.knowledge.embed_query_bundle(
                    [
                        ("whole", question),
                        ("atomic", private_question),
                        ("atomic", public_question),
                        ("atomic", knowledge_question),
                    ]
                )
                query_vector_bundle = (
                    await asyncio.wait_for(bundle_call, timeout=bundle_timeout)
                    if bundle_timeout is not None
                    else await bundle_call
                )
            except Exception:
                # Contextual association is an optional local lane.  A failed
                # batch must degrade to ordinary retrieval without retries per
                # edge or turning a good answer into a service error.
                query_vector_bundle = None
        calls: list[tuple[str, Any]] = []
        if selected.user:
            calls.append(
                (
                    "user",
                    recall_domain(
                        await self._user_service(identity),
                        private_question,
                        request=private_request,
                        retrieval_intensity=selected.intensity,
                        auto_escalate=False,
                        query_embeddings_override=query_embeddings_override,
                        query_vector_bundle=query_vector_bundle,
                    ),
                )
            )
        if selected.public:
            calls.append(
                (
                    "public",
                    recall_domain(
                        self.public,
                        public_question,
                        request=public_request,
                        retrieval_intensity=selected.intensity,
                        auto_escalate=False,
                        query_embeddings_override=query_embeddings_override,
                        query_vector_bundle=query_vector_bundle,
                    ),
                )
            )
        if selected.knowledge:
            calls.append(
                (
                    "knowledge",
                    recall_domain(
                        self.knowledge,
                        knowledge_question,
                        request=knowledge_request,
                        retrieval_intensity=selected.intensity,
                        auto_escalate=False,
                        query_embeddings_override=query_embeddings_override,
                        query_vector_bundle=query_vector_bundle,
                    ),
                )
            )
        if not calls:
            return RetrievedMemory(
                context="",
                raw_result={"route": selected.__dict__ if hasattr(selected, "__dict__") else {
                    "user": selected.user,
                    "public": selected.public,
                    "knowledge": selected.knowledge,
                    "reason": selected.reason,
                    "intensity": selected.intensity,
                    "knowledge_write_policy": selected.knowledge_write_policy,
                    "creative": selected.creative,
                }},
            )
        values = await asyncio.gather(*(call for _domain, call in calls))
        if selected.intensity == "standard":
            primary_domains = {
                domain
                for domain, enabled in (
                    ("user", selected.user),
                    ("knowledge", selected.knowledge),
                )
                if enabled
            }
            if not primary_domains and selected.public:
                primary_domains.add("public")
            escalation_jobs: list[tuple[int, str, RetrievedMemory, Any, str]] = []
            escalation_skips: list[tuple[int, RetrievedMemory, str]] = []
            for index, ((domain, _call), value) in enumerate(
                zip(calls, values, strict=True)
            ):
                quality = value.raw_result.get("retrieval_quality") or {}
                if (
                    domain not in primary_domains
                    or value.error
                    or quality.get("sufficient", True)
                ):
                    continue
                target_service = (
                    await self._user_service(identity)
                    if domain == "user"
                    else self.knowledge
                    if domain == "knowledge"
                    else self.public
                )
                original_request = (
                    knowledge_request
                    if domain == "knowledge"
                    else private_request
                    if domain == "user"
                    else public_request
                )
                original_question = (
                    knowledge_question
                    if domain == "knowledge"
                    else private_question
                    if domain == "user"
                    else public_question
                )
                repair_request = missing_slot_request(
                    value,
                    original_request,
                    original_question,
                )
                if repair_request is None:
                    escalation_skips.append(
                        (index, value, "no_explicit_missing_evidence_slot")
                    )
                    continue
                if (
                    deadline_budget is not None
                    and deadline_budget.retrieval_timeout() < 0.1
                ):
                    escalation_skips.append(
                        (index, value, "deadline_budget_exhausted")
                    )
                    continue
                escalation_jobs.append(
                    (
                        index,
                        domain,
                        value,
                        recall_domain(
                            target_service,
                            repair_request.query,
                            request=repair_request,
                            retrieval_plan=target_service.retrieval_plan("deep"),
                            auto_escalate=False,
                            query_embeddings_override=query_embeddings_override,
                            query_vector_bundle=(
                                value.query_vector_bundle or query_vector_bundle
                            ),
                        ),
                        repair_request.query,
                    )
                )
            for index, initial_value, reason in escalation_skips:
                skipped_raw = deepcopy(initial_value.raw_result)
                skipped_raw["retrieval_escalation"] = {
                    "triggered": False,
                    "from": "standard",
                    "to": "deep",
                    "reason": reason,
                }
                values[index] = RetrievedMemory(
                    context=initial_value.context,
                    raw_result=skipped_raw,
                    error=initial_value.error,
                    domains=initial_value.domains,
                    semantic_vector=initial_value.semantic_vector,
                    query_vector_bundle=initial_value.query_vector_bundle,
                )
            if escalation_jobs:
                deep_values = await asyncio.gather(
                    *(
                        job
                        for _index, _domain, _value, job, _repair_query
                        in escalation_jobs
                    )
                )
                for (
                    index,
                    _domain,
                    initial_value,
                    _job,
                    repair_query,
                ), deep_value in zip(escalation_jobs, deep_values, strict=True):
                    initial_raw = initial_value.raw_result
                    initial_quality = initial_raw.get("retrieval_quality") or {}
                    escalation = {
                        "triggered": True,
                        "from": "standard",
                        "to": "deep",
                        "reasons": list(initial_quality.get("reasons") or []),
                        "initial_quality": deepcopy(initial_quality),
                        "initial_episode_ids": list(
                            initial_raw.get("episode_ids") or []
                        ),
                        "initial_evidence_slot_trace": deepcopy(
                            initial_raw.get("evidence_slot_trace") or {}
                        ),
                        "initial_contextual_association": deepcopy(
                            initial_raw.get("contextual_association") or {}
                        ),
                        "initial_timings": deepcopy(
                            initial_raw.get("timings") or {}
                        ),
                        "repair_query": repair_query,
                        "deep_error": deep_value.error,
                    }
                    if deep_value.raw_result and not deep_value.error:
                        deep_raw = deepcopy(deep_value.raw_result)
                        deep_raw["retrieval_escalation"] = escalation
                        values[index] = RetrievedMemory(
                            context=deep_value.context,
                            raw_result=deep_raw,
                            domains=deep_value.domains,
                            query_vector_bundle=deep_value.query_vector_bundle,
                        )
                    else:
                        fallback_raw = deepcopy(initial_raw)
                        fallback_raw["retrieval_escalation"] = escalation
                        values[index] = RetrievedMemory(
                            context=initial_value.context,
                            raw_result=fallback_raw,
                            error=initial_value.error,
                            domains=initial_value.domains,
                            query_vector_bundle=initial_value.query_vector_bundle,
                        )
        contexts: list[str] = []
        errors: list[str] = []
        raw_domains: dict[str, Any] = {}
        used_domains: list[str] = []
        semantic_vector = None
        for (domain, _call), value in zip(calls, values, strict=True):
            raw_domains[domain] = value.raw_result
            section = render_domain_memory_section(domain, value.context)
            if section:
                contexts.append(section)
                used_domains.append(domain)
            if value.error:
                errors.append(f"{domain}: {value.error}")
            if value.semantic_vector is not None and (
                semantic_vector is None or domain == "knowledge"
            ):
                semantic_vector = value.semantic_vector
        if contexts:
            total_budget = max(
                2_000,
                int(getattr(getattr(self, "config", None), "context_max_chars", 10_000)),
            )
            per_domain = max(1_200, total_budget // len(contexts))
            suffix = "\n[该记忆域其余候选已由证据预算截断]"
            contexts = [
                value
                if len(value) <= per_domain
                else value[: per_domain - len(suffix)].rstrip() + suffix
                for value in contexts
            ]
        return RetrievedMemory(
            context="\n\n".join(contexts),
            raw_result={
                "route": {
                    "user": selected.user,
                    "public": selected.public,
                    "knowledge": selected.knowledge,
                    "reason": selected.reason,
                    "intensity": selected.intensity,
                    "knowledge_write_policy": selected.knowledge_write_policy,
                    "creative": selected.creative,
                },
                "knowledge_query": knowledge_question if selected.knowledge else "",
                "domain_queries": {
                    "private": private_question if selected.user else "",
                    "public": public_question if selected.public else "",
                    "knowledge": knowledge_question if selected.knowledge else "",
                },
                "domains": raw_domains,
                "deadline_budget": (
                    deadline_budget.as_dict() if deadline_budget is not None else None
                ),
            },
            error="; ".join(errors),
            domains=tuple(used_domains),
            semantic_vector=semantic_vector,
            query_vector_bundle=query_vector_bundle,
        )

    async def grow(
        self,
        identity: PlatformIdentity,
        question: str,
        route: MemoryRoute,
        *,
        user_question: str | None = None,
        knowledge_question: str | None = None,
        knowledge_candidates: list[dict[str, Any]] | None = None,
        contextual_candidates: list[dict[str, Any]] | None = None,
        contextual_observations: list[dict[str, Any]] | None = None,
        query_vector_bundle: Any | None = None,
    ) -> dict[str, Any]:
        """Run expensive query-time Association growth outside the reply path.

        Public memory is intentionally read-only for ordinary conversations.
        Only explicit administrator imports may write to that shared domain.
        """

        calls: list[tuple[str, Any]] = []
        if route.user and self.config.user_growth_enabled:
            calls.append(
                (
                    "user",
                    (await self._background_user_service(identity)).recall(
                        user_question or question,
                        retrieval_intensity="deep",
                    ),
                )
            )
        knowledge_writable = (
            route.knowledge
            and route.knowledge_write_policy != "none"
            and bool((knowledge_question or question).strip())
        )
        if knowledge_writable and self.config.knowledge_growth_enabled:
            knowledge_service = await self._background_service(
                "knowledge", self.config.knowledge_database_path
            )
            calls.append(
                (
                    "knowledge",
                    knowledge_service.audit_candidates(knowledge_candidates)
                    if knowledge_candidates
                    else knowledge_service.recall(
                        knowledge_question or question,
                        retrieval_intensity="deep",
                        growth_persist_only_used_override=(
                            True if knowledge_question is not None else None
                        ),
                        growth_max_rounds_override=(
                            self.config.knowledge_growth_max_rounds
                            if knowledge_question is not None
                            else None
                        ),
                        query_vector_bundle=query_vector_bundle,
                    ),
                )
            )
        # Contextual plasticity is a local, retrieval-only lane.  It may run
        # even when no semantic knowledge candidate was authorized; it never
        # calls the consolidator and never receives the assistant prose.
        contextual_result: dict[str, Any] | None = None
        contextual_utility: dict[str, Any] | None = None
        if (
            knowledge_writable
            and self.config.contextual_association_enabled
            and (
                contextual_candidates
                or contextual_observations
            )
        ):
            if contextual_candidates:
                contextual_result = await self.knowledge.apply_contextual_plasticity(
                    contextual_candidates,
                    query_vector_bundle,
                )
            if contextual_observations:
                contextual_utility = await self.knowledge.record_contextual_utility(
                    contextual_observations
                )
        if not calls and contextual_result is None and contextual_utility is None:
            return {
                "question": question,
                "domains": {},
                "skipped": "no writable growth domain selected",
            }

        values = await asyncio.gather(*(call for _domain, call in calls))
        result: dict[str, Any] = {
            "question": question,
            "route": {
                "user": route.user,
                "public": False,
                "knowledge": knowledge_writable,
                "reason": route.reason,
                "knowledge_write_policy": route.knowledge_write_policy,
                "creative": route.creative,
            },
            "domains": {},
        }
        if contextual_result is not None or contextual_utility is not None:
            result["contextual_plasticity"] = {
                "candidates": contextual_result or {
                    "enabled": True,
                    "created": [],
                    "rejected": [],
                    "external_calls": 0,
                },
                "utility": contextual_utility or {
                    "enabled": True,
                    "updated": 0,
                    "external_calls": 0,
                },
            }
            if (contextual_result or {}).get("created") or (
                contextual_utility or {}
            ).get("updated"):
                await self.knowledge.refresh_graph_state()
        for (domain, _call), recalled in zip(calls, values, strict=True):
            raw = recalled.raw_result or {}
            result["domains"][domain] = {
                "error": recalled.error,
                "episode_ids": list(raw.get("episode_ids") or []),
                "association_ids": list(raw.get("association_ids") or []),
                "new_association_ids": list(raw.get("new_association_ids") or []),
                "reinforced_association_ids": list(
                    raw.get("reinforced_association_ids") or []
                ),
                "growth_utility_gate": raw.get("growth_utility_gate") or {},
            }
            changed = bool(
                result["domains"][domain]["new_association_ids"]
                or result["domains"][domain]["reinforced_association_ids"]
            )
            if changed:
                foreground = (
                    await self._user_service(identity)
                    if domain == "user"
                    else self.knowledge
                )
                await foreground.refresh_graph_state()
        return result

    async def import_conversation_file(self, path: Path, source_root: Path) -> dict:
        try:
            identity = PlatformIdentity.from_metadata(path.resolve().parent)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"cannot resolve platform identity for {path}") from exc
        writer = await self._background_user_service(identity)
        result = await writer.import_file(path, source_root)
        await (await self._user_service(identity)).refresh_indexes()
        return result

    async def import_public_file(self, path: Path, source_root: Path) -> dict:
        """Explicit audited/admin entry point; conversations never call this."""
        writer = await self._background_service(
            "public", self.config.public_database_path
        )
        result = await writer.import_file(path, source_root)
        await self.public.refresh_indexes()
        return result

    async def import_knowledge_file(
        self,
        path: Path,
        source_root: Path,
        *,
        refresh_indexes: bool = True,
    ) -> dict:
        writer = await self._background_service(
            "knowledge", self.config.knowledge_database_path
        )
        result = await writer.import_file(path, source_root)
        if refresh_indexes:
            await self.knowledge.refresh_indexes()
        return result

    async def import_private_file(
        self,
        identity: PlatformIdentity,
        path: Path,
        source_root: Path,
    ) -> dict:
        writer = await self._background_user_service(identity)
        result = await writer.import_file(path, source_root)
        await (await self._user_service(identity)).refresh_indexes()
        return result

    async def stats(
        self, identity: PlatformIdentity | None = None
    ) -> dict[str, Any]:
        knowledge, public = await asyncio.gather(
            self.knowledge.stats(), self.public.stats()
        )
        result: dict[str, Any] = {"knowledge": knowledge, "public": public}
        if identity is not None:
            result["user"] = await (await self._user_service(identity)).stats()
        return result
