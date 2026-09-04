from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any

from config.prompt_config import render_domain_memory_section

from .config import MemorySystemConfig
from .contracts import DomainRecallRequest, RetrievedMemory
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
    ) -> RetrievedMemory:
        selected = route or route_memory_query(question)
        requests = domain_requests or {}
        private_request = requests.get("private")
        public_request = requests.get("public")
        knowledge_request = requests.get("knowledge")
        private_question = private_request.query if private_request else question
        public_question = public_request.query if public_request else question
        knowledge_question = knowledge_request.query if knowledge_request else question
        calls: list[tuple[str, Any]] = []
        if selected.user:
            calls.append(
                (
                    "user",
                    (await self._user_service(identity)).recall(
                        private_question,
                        request=private_request,
                        retrieval_intensity=selected.intensity,
                        auto_escalate=False,
                        query_embeddings_override=query_embeddings_override,
                    ),
                )
            )
        if selected.public:
            calls.append(
                (
                    "public",
                    self.public.recall(
                        public_question,
                        request=public_request,
                        retrieval_intensity=selected.intensity,
                        auto_escalate=False,
                        query_embeddings_override=query_embeddings_override,
                    ),
                )
            )
        if selected.knowledge:
            calls.append(
                (
                    "knowledge",
                    self.knowledge.recall(
                        knowledge_question,
                        request=knowledge_request,
                        retrieval_intensity=selected.intensity,
                        auto_escalate=False,
                        query_embeddings_override=query_embeddings_override,
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
            escalation_jobs: list[tuple[int, str, RetrievedMemory, Any]] = []
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
                escalation_jobs.append(
                    (
                        index,
                        domain,
                        value,
                        target_service.recall(
                            knowledge_question
                            if domain == "knowledge"
                            else private_question
                            if domain == "user"
                            else public_question,
                            request=(
                                knowledge_request
                                if domain == "knowledge"
                                else private_request
                                if domain == "user"
                                else public_request
                            ),
                            retrieval_plan=target_service.retrieval_plan("deep"),
                            auto_escalate=False,
                        ),
                    )
                )
            if escalation_jobs:
                deep_values = await asyncio.gather(
                    *(job for _index, _domain, _value, job in escalation_jobs)
                )
                for (
                    index,
                    _domain,
                    initial_value,
                    _job,
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
                        "initial_timings": deepcopy(
                            initial_raw.get("timings") or {}
                        ),
                        "deep_error": deep_value.error,
                    }
                    if deep_value.raw_result and not deep_value.error:
                        deep_raw = deepcopy(deep_value.raw_result)
                        deep_raw["retrieval_escalation"] = escalation
                        values[index] = RetrievedMemory(
                            context=deep_value.context,
                            raw_result=deep_raw,
                            domains=deep_value.domains,
                        )
                    else:
                        fallback_raw = deepcopy(initial_raw)
                        fallback_raw["retrieval_escalation"] = escalation
                        values[index] = RetrievedMemory(
                            context=initial_value.context,
                            raw_result=fallback_raw,
                            error=initial_value.error,
                            domains=initial_value.domains,
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
            },
            error="; ".join(errors),
            domains=tuple(used_domains),
            semantic_vector=semantic_vector,
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
                    ),
                )
            )
        if not calls:
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
