from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
import inspect
import sys
from time import monotonic
from typing import Any

from .config import (
    MemoryServiceConfig,
    _apply_ingestion_env_overrides,
    _resolve_rerank_policy,
)
from .contracts import (
    DomainRecallRequest,
    RetrievedMemory,
    RetrievalPlan,
    _retrieval_answer_shape,
    assess_retrieval_quality,
    build_lore_fact_contract,
    format_memory_context,
)
from .routing import _lightweight_search_queries, _lightweight_target_entities


@asynccontextmanager
async def _concurrent_read_scope():
    """Mark foreground query state as request-local and lock-free."""

    yield


def _consume_background_task(task: asyncio.Task) -> None:
    """Observe a timed-out ``to_thread`` task once its worker eventually exits."""

    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def _accepts_keyword(call: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(call).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        item.kind is inspect.Parameter.VAR_KEYWORD or item.name == name
        for item in parameters
    )


class AssociativeMemoryService:
    """Async facade with concurrent reads and serialized mutable operations.

    Foreground recall uses a query-local QueryEngine and configuration snapshot
    over shared read-only RAM indexes. Imports, audits and background growth
    keep using ``_operation_lock`` because they can change SQLite or indexes.
    """

    def __init__(self, config: MemoryServiceConfig):
        self.config = config
        self._application: Any | None = None
        self._query_engine: Any | None = None
        self._operation_lock = asyncio.Lock()
        self._initialize_lock = asyncio.Lock()
        self._has_memories = False
        self._recall_cache: OrderedDict[
            str, tuple[float, RetrievedMemory]
        ] = OrderedDict()
        self._recall_failure_cache: OrderedDict[
            str, tuple[float, RetrievedMemory]
        ] = OrderedDict()
        self._query_embedding_cache: OrderedDict[str, Any] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self._small_database_fast_path = False

    @property
    def initialized(self) -> bool:
        return self._application is not None and self._query_engine is not None

    def _load_package(self) -> tuple[Any, Any]:
        source_root = str((self.config.engine_root / "src").resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from memory_demo import AppConfig, MemoryApplication

        return MemoryApplication, AppConfig

    def _initialize_sync(self) -> None:
        self.config.validate()
        MemoryApplication, AppConfig = self._load_package()
        app_config = AppConfig()
        app_config.database_path = self.config.database_path
        app_config.log_dir = self.config.log_dir
        app_config.model.api_key = self.config.api_key
        app_config.model.reranker_model = self.config.reranker_model
        if self.config.reasoning_model:
            app_config.model.reasoning_model = self.config.reasoning_model
        if self.config.fallback_model:
            app_config.model.fallback_model = self.config.fallback_model
        if self.config.reasoning_max_tokens > 0:
            app_config.model.reasoning_max_tokens = (
                self.config.reasoning_max_tokens
            )
        if self.config.reasoning_enable_thinking is not None:
            app_config.model.reasoning_enable_thinking = (
                self.config.reasoning_enable_thinking
            )
        app_config.apply_optimization_profile(self.config.optimization_profile)
        _apply_ingestion_env_overrides(app_config)
        app_config.paragraph.enabled = self.config.paragraph_enabled
        app_config.concept_extraction.profile = self.config.concept_profile
        app_config.retrieval.sparse_enabled = self.config.sparse_enabled
        app_config.retrieval.source_key_cohort_enabled = (
            self.config.source_key_cohort_enabled
        )
        app_config.retrieval.followup_planning_mode = (
            self.config.followup_planning_mode
        )
        app_config.retrieval.rerank_enabled = self.config.rerank_enabled
        app_config.retrieval.rerank_backend = self.config.rerank_backend
        app_config.retrieval.rerank_coverage_audit_enabled = (
            self.config.rerank_coverage_audit_enabled
        )
        if self.config.rerank_review_mode:
            app_config.retrieval.rerank_review_mode = (
                self.config.rerank_review_mode
            )
        app_config.retrieval.rerank_atomic_query_limit = (
            self.config.rerank_atomic_query_limit
        )
        app_config.retrieval.rerank_precompression_limit = (
            self.config.rerank_precompression_limit
        )
        app_config.retrieval.growth_persist_only_used = (
            self.config.growth_persist_only_used
        )
        app_config.retrieval.growth_counterfactual_utility_enabled = (
            self.config.growth_counterfactual_utility_enabled
        )
        app_config.retrieval.growth_staging_enabled = (
            self.config.growth_staging_enabled
        )
        if not self.config.growth_enabled:
            app_config.retrieval.growth_max_rounds = 0
        app_config.retrieval.association_cue_enabled = (
            self.config.association_cue_enabled
        )
        app_config.retrieval.association_cue_fast_path_enabled = (
            self.config.association_cue_fast_path_enabled
        )
        app_config.retrieval.association_cue_fast_path_min_similarity = (
            self.config.association_cue_fast_path_min_similarity
        )
        app_config.retrieval.association_cue_fast_path_min_confidence = (
            self.config.association_cue_fast_path_min_confidence
        )
        app_config.retrieval.association_cue_fast_path_min_margin = (
            self.config.association_cue_fast_path_min_margin
        )
        app_config.retrieval.contextual_association_enabled = (
            self.config.contextual_association_enabled
        )
        app_config.retrieval.contextual_association_shadow = (
            self.config.contextual_association_shadow
        )
        app_config.retrieval.contextual_promotion_enabled = (
            self.config.contextual_promotion_enabled
        )
        app_config.retrieval.contextual_context_top_k = (
            self.config.contextual_context_top_k
        )
        app_config.retrieval.contextual_need_top_k = (
            self.config.contextual_need_top_k
        )
        app_config.retrieval.contextual_edge_top_k = (
            self.config.contextual_edge_top_k
        )
        app_config.retrieval.contextual_context_threshold = (
            self.config.contextual_context_threshold
        )
        app_config.retrieval.contextual_need_threshold = (
            self.config.contextual_need_threshold
        )
        app_config.retrieval.contextual_combine_mode = (
            self.config.contextual_combine_mode
        )
        app_config.retrieval.contextual_endpoint_limit_light = (
            self.config.contextual_endpoint_limit_light
        )
        app_config.retrieval.contextual_endpoint_limit_standard = (
            self.config.contextual_endpoint_limit_standard
        )
        app_config.retrieval.contextual_endpoint_limit_deep = (
            self.config.contextual_endpoint_limit_deep
        )
        app_config.retrieval.contextual_max_candidates_per_turn = (
            self.config.contextual_max_candidates_per_turn
        )
        app_config.retrieval.contextual_probation_limit = (
            self.config.contextual_probation_limit
        )
        app_config.retrieval.contextual_probation_ttl = (
            self.config.contextual_probation_ttl
        )
        app_config.retrieval.contextual_min_distinct_successes = (
            self.config.contextual_min_distinct_successes
        )
        app_config.retrieval.contextual_noop_decay = self.config.contextual_noop_decay
        app_config.retrieval.contextual_harm_multiplier = self.config.contextual_harm_multiplier
        app_config.ensure_directories()
        application = MemoryApplication(app_config)
        application.rebuild_indexes()
        episode_count = application.episodes.count()
        self._small_database_fast_path = bool(
            self.config.small_database_fast_path_max_episodes > 0
            and episode_count
            <= self.config.small_database_fast_path_max_episodes
        )
        if self._small_database_fast_path:
            # A complete small personal database already fits inside the
            # deterministic candidate lanes. Spending LLM calls on intent
            # decomposition and reranking only adds latency and stochastic
            # omissions; the final chat model still receives every selected
            # evidence row and the private-memory guard remains active.
            app_config.retrieval.followup_planning_mode = "off"
            app_config.retrieval.rerank_enabled = False
        if self.config.association_cue_enabled:
            application.rebuild_association_cue_index()
        self._application = application
        self._has_memories = episode_count > 0
        self._query_engine = application.query_engine(
            application.new_logger("chatbot-query")
        )

    async def initialize(self) -> None:
        if self.initialized:
            return
        async with self._initialize_lock:
            if not self.initialized:
                await asyncio.to_thread(self._initialize_sync)

    async def embed_query_text(self, text: str):
        """Embed one cache-probe query without invoking a reasoning model."""

        if not text.strip():
            raise ValueError("semantic cache query must not be empty")
        await self.initialize()
        engine, _route = self._request_query_engine(self.retrieval_plan("light"))
        embed = getattr(engine, "embed_query_text", None)
        if not callable(embed):
            raise RuntimeError("memory engine does not expose query embeddings")
        return await asyncio.to_thread(embed, text.strip())

    async def embed_query_bundle(self, texts):
        """Create one request-level vector bundle through the engine."""
        await self.initialize()
        engine, _route = self._request_query_engine(self.retrieval_plan("light"))
        embed_bundle = getattr(engine, "embed_query_bundle", None)
        if not callable(embed_bundle):
            raise RuntimeError("memory engine does not expose query bundles")
        return await asyncio.to_thread(embed_bundle, texts)

    async def match_association_cue(self, text: str) -> dict[str, Any] | None:
        """Check the local audited-edge cue index without any cloud request."""

        if not text.strip() or not self.config.association_cue_fast_path_enabled:
            return None
        await self.initialize()
        matcher = getattr(self._query_engine, "match_association_cue_text", None)
        if not callable(matcher):
            return None
        return await asyncio.to_thread(matcher, text.strip())

    @staticmethod
    def retrieval_plan(preset: str) -> RetrievalPlan:
        """Expand a user-facing preset into independent retrieval controls."""

        normalized = str(preset or "standard").strip().casefold()
        if normalized == "light":
            return RetrievalPlan(
                preset="light",
                query_planner="heuristic",
                graph_hops=1,
                candidate_limit=20,
                reranker="configured",
                evidence_slots=True,
                followup_policy="never",
                verification="local",
                deadline_seconds=8.0,
                answer_episode_limit=8,
                answer_concept_limit=6,
                answer_path_limit=6,
            )
        if normalized == "deep":
            return RetrievalPlan(
                preset="deep",
                query_planner="llm",
                graph_hops=3,
                candidate_limit=40,
                reranker="llm",
                evidence_slots=True,
                followup_policy="always",
                verification="local",
                deadline_seconds=30.0,
                answer_episode_limit=12,
                answer_concept_limit=10,
                answer_path_limit=14,
            )
        return RetrievalPlan(
            preset="standard",
            query_planner="heuristic",
            graph_hops=2,
            candidate_limit=30,
            reranker="configured",
            evidence_slots=True,
            followup_policy="on_insufficient_evidence",
            verification="local",
            deadline_seconds=18.0,
            answer_episode_limit=16,
            answer_concept_limit=12,
            answer_path_limit=12,
        )

    def _request_query_engine(
        self,
        plan: RetrievalPlan,
    ) -> tuple[Any, str]:
        """Build an isolated engine without copying repositories or indexes."""

        template_config = getattr(self._query_engine, "config", None)
        application_factory = getattr(self._application, "query_engine", None)
        if template_config is None or not callable(application_factory):
            return self._query_engine, "unavailable"
        request_config = deepcopy(template_config)
        if hasattr(request_config, "model") and hasattr(
            request_config.model, "timeout_seconds"
        ):
            request_config.model.timeout_seconds = min(
                float(request_config.model.timeout_seconds),
                max(1.0, float(plan.deadline_seconds)),
            )
        retrieval = request_config.retrieval
        retrieval.association_cue_fast_path_enabled = bool(
            getattr(retrieval, "association_cue_fast_path_enabled", False)
            and plan.preset in {"light", "standard"}
        )
        retrieval.graph_max_hops = plan.graph_hops
        retrieval.candidate_limit = plan.candidate_limit
        retrieval.rerank_candidate_limit = plan.candidate_limit
        retrieval.rerank_precompression_limit = (
            plan.answer_episode_limit
            if plan.preset == "deep"
            else plan.candidate_limit
        )
        retrieval.answer_episode_limit = plan.answer_episode_limit
        retrieval.answer_concept_limit = plan.answer_concept_limit
        retrieval.answer_path_limit = plan.answer_path_limit
        retrieval.rerank_atomic_floor_enabled = plan.evidence_slots
        retrieval.followup_planning_mode = (
            "always" if plan.followup_policy == "always" else "off"
        )
        configured_enabled = bool(retrieval.rerank_enabled)
        configured_backend = str(retrieval.rerank_backend)
        if plan.reranker == "disabled" or not configured_enabled:
            retrieval.rerank_enabled = False
            rerank_route = "disabled_not_configured"
        else:
            retrieval.rerank_enabled = True
            retrieval.rerank_backend = (
                "llm" if plan.reranker == "llm" else configured_backend
            )
            if plan.preset == "deep":
                # BGE first reduces the evidence pool; one evidence-bound LLM
                # pass then covers the remaining shortlist. Independent scouts
                # and a third compressor added minutes without creating facts.
                retrieval.rerank_review_mode = "lean"
                retrieval.rerank_atomic_query_limit = 12
                rerank_route = "llm_deep_default"
            else:
                rerank_route = (
                    f"{retrieval.rerank_backend}_{plan.preset}"
                )
        return (
            application_factory(config=request_config),
            rerank_route,
        )

    async def recall(
        self,
        question: str,
        *,
        request: DomainRecallRequest | None = None,
        intent_override: dict | None = None,
        followup_queries_override: list[str] | None = None,
        retrieval_intensity: str = "standard",
        retrieval_plan: RetrievalPlan | None = None,
        auto_escalate: bool = True,
        growth_persist_only_used_override: bool | None = None,
        growth_max_rounds_override: int | None = None,
        query_embeddings_override: dict[str, Any] | None = None,
        query_vector_bundle: Any | None = None,
        deadline_seconds: float | None = None,
    ) -> RetrievedMemory:
        if request is not None:
            question = request.query
        if not question.strip():
            return RetrievedMemory(context="")
        await self.initialize()
        if not self._has_memories:
            return RetrievedMemory(
                context="",
                raw_result={
                    "empty_domain": True,
                    "service_cache_hit": True,
                },
                domains=(self.config.domain,),
            )
        plan = retrieval_plan or self.retrieval_plan(retrieval_intensity)
        if deadline_seconds is not None:
            # A caller-owned request deadline is an upper bound, never an
            # opportunity to grant a deep retry a fresh full budget.
            plan = replace(
                plan,
                deadline_seconds=max(
                    0.1,
                    min(float(plan.deadline_seconds), float(deadline_seconds)),
                ),
            )
        operation_scope = (
            self._operation_lock
            if self.config.growth_enabled
            or (
                growth_max_rounds_override is not None
                and growth_max_rounds_override > 0
            )
            else _concurrent_read_scope()
        )
        async with operation_scope:
            intensity = plan.preset
            lightweight = intensity == "light"
            diagnostic_override = (
                (request is None and intent_override is not None)
                or (request is None and followup_queries_override is not None)
                or growth_persist_only_used_override is not None
                or growth_max_rounds_override is not None
            )
            request_cache_key = (
                request.cache_key
                if request is not None
                else " ".join(question.casefold().split())
            )
            cache_key = plan.cache_key + "\0" + request_cache_key
            cached = (
                None
                if diagnostic_override
                else self._recall_cache.get(cache_key)
            )
            if cached is not None:
                stored_at, value = cached
                if (
                    self.config.recall_cache_ttl_seconds <= 0
                    or monotonic() - stored_at
                    <= self.config.recall_cache_ttl_seconds
                ):
                    self._recall_cache.move_to_end(cache_key)
                    self.cache_hits += 1
                    raw = deepcopy(value.raw_result)
                    raw["service_cache_hit"] = True
                    return RetrievedMemory(
                        context=value.context,
                        raw_result=raw,
                        error=value.error,
                        domains=value.domains,
                        semantic_vector=(
                            value.semantic_vector.copy()
                            if hasattr(value.semantic_vector, "copy")
                            else value.semantic_vector
                        ),
                        query_vector_bundle=value.query_vector_bundle,
                    )
                self._recall_cache.pop(cache_key, None)
            failed = (
                None
                if diagnostic_override or plan.preset != "deep"
                else self._recall_failure_cache.get(cache_key)
            )
            if failed is not None:
                failed_at, value = failed
                if (
                    self.config.recall_failure_ttl_seconds > 0
                    and monotonic() - failed_at
                    <= self.config.recall_failure_ttl_seconds
                ):
                    self._recall_failure_cache.move_to_end(cache_key)
                    self.cache_hits += 1
                    raw = deepcopy(value.raw_result)
                    raw["service_cache_hit"] = True
                    raw["failure_cache_hit"] = True
                    return RetrievedMemory(
                        context=value.context,
                        raw_result=raw,
                        error=value.error,
                        domains=value.domains,
                        semantic_vector=(
                            value.semantic_vector.copy()
                            if hasattr(value.semantic_vector, "copy")
                            else value.semantic_vector
                        ),
                        query_vector_bundle=value.query_vector_bundle,
                    )
                self._recall_failure_cache.pop(cache_key, None)
            self.cache_misses += 1
            try:
                query_options: dict[str, Any] = {"generate_answer": False}
                effective_intent = (
                    request.intent_override
                    if request is not None
                    else intent_override
                )
                effective_followups = (
                    request.followup_queries
                    if request is not None
                    else followup_queries_override
                )
                if effective_intent:
                    query_options["intent_override"] = effective_intent
                elif (
                    self._small_database_fast_path
                    or plan.query_planner == "heuristic"
                ):
                    query_options["intent_override"] = {
                        "language": "zh",
                        "target_entities": _lightweight_target_entities(
                            question
                        ),
                        "search_queries": (
                            _lightweight_search_queries(question)
                            if not self._small_database_fast_path
                            else [question.strip()]
                        ),
                        "requested_relation": "",
                        "temporal_constraint": "",
                        "causal_constraint": "",
                        "answer_shape": _retrieval_answer_shape(
                            self.config.domain,
                            lightweight=lightweight,
                        ),
                        "uncertainty_required": not lightweight,
                    }
                if effective_followups is not None:
                    query_options["followup_queries_override"] = (
                        list(effective_followups)
                    )
                elif (
                    self._small_database_fast_path
                    or plan.followup_policy != "always"
                ):
                    query_options["followup_queries_override"] = []
                request_engine, planned_rerank_route = (
                    self._request_query_engine(plan)
                )
                if (
                    self.config.query_embedding_cache_size > 0
                    and self._query_embedding_cache
                    and _accepts_keyword(
                        request_engine.query, "query_embeddings_override"
                    )
                ):
                    query_options["query_embeddings_override"] = {
                        key: value.copy() if hasattr(value, "copy") else value
                        for key, value in self._query_embedding_cache.items()
                    }
                if query_embeddings_override and _accepts_keyword(
                    request_engine.query, "query_embeddings_override"
                ):
                    existing_overrides = query_options.setdefault(
                        "query_embeddings_override", {}
                    )
                    existing_overrides.update(query_embeddings_override)
                if query_vector_bundle is not None and _accepts_keyword(
                    request_engine.query, "query_vector_bundle"
                ):
                    query_options["query_vector_bundle"] = query_vector_bundle
                if (
                    query_vector_bundle is not None
                    and _accepts_keyword(
                        request_engine.query, "contextual_endpoint_limit"
                    )
                ):
                    endpoint_limits = {
                        "light": self.config.contextual_endpoint_limit_light,
                        "standard": self.config.contextual_endpoint_limit_standard,
                        "deep": self.config.contextual_endpoint_limit_deep,
                    }
                    query_options["contextual_endpoint_limit"] = endpoint_limits[
                        plan.preset
                    ]
                if (
                    not self.config.growth_enabled
                    and _accepts_keyword(request_engine.query, "deadline_seconds")
                ):
                    query_options["deadline_seconds"] = plan.deadline_seconds
                retrieval_config = getattr(
                    getattr(request_engine, "config", None),
                    "retrieval",
                    None,
                )
                original_rerank_enabled: bool | None = None
                original_rerank_backend: str | None = None
                original_deep_rerank_settings: dict[str, Any] = {}
                original_light_budget_settings: dict[str, Any] = {}
                rerank_route = planned_rerank_route
                original_growth_persist_only_used: bool | None = None
                original_growth_max_rounds: int | None = None
                try:
                    if retrieval_config is not None:
                        original_rerank_enabled = bool(
                            retrieval_config.rerank_enabled
                        )
                        original_rerank_backend = str(
                            retrieval_config.rerank_backend
                        )
                        (
                            retrieval_config.rerank_enabled,
                            retrieval_config.rerank_backend,
                            rerank_route,
                        ) = _resolve_rerank_policy(
                            intensity,
                            enabled=original_rerank_enabled,
                            configured_backend=original_rerank_backend,
                        )
                        if (
                            not original_rerank_enabled
                            and self._small_database_fast_path
                        ):
                            rerank_route = "disabled_small_database_fast_path"
                        if rerank_route == "llm_deep_default":
                            for name, value in {
                                "rerank_review_mode": "lean",
                                "followup_planning_mode": "always",
                                "rerank_atomic_query_limit": 12,
                                "rerank_precompression_limit": (
                                    plan.answer_episode_limit
                                ),
                            }.items():
                                original_deep_rerank_settings[name] = getattr(
                                    retrieval_config, name
                                )
                                setattr(retrieval_config, name, value)
                        if lightweight:
                            for name, limit in {
                                "answer_episode_limit": 12,
                                "answer_concept_limit": 10,
                                "answer_path_limit": 10,
                            }.items():
                                if not hasattr(retrieval_config, name):
                                    continue
                                current = getattr(retrieval_config, name)
                                original_light_budget_settings[name] = current
                                setattr(
                                    retrieval_config,
                                    name,
                                    min(int(current), limit),
                                )
                    if (
                        growth_persist_only_used_override is not None
                        and retrieval_config is not None
                    ):
                        original_growth_persist_only_used = bool(
                            retrieval_config.growth_persist_only_used
                        )
                        retrieval_config.growth_persist_only_used = bool(
                            growth_persist_only_used_override
                        )
                    if (
                        growth_max_rounds_override is not None
                        and retrieval_config is not None
                    ):
                        original_growth_max_rounds = int(
                            retrieval_config.growth_max_rounds
                        )
                        retrieval_config.growth_max_rounds = max(
                            0, int(growth_max_rounds_override)
                        )
                    query_task = asyncio.create_task(
                        asyncio.to_thread(
                            partial(
                                request_engine.query,
                                question.strip(),
                                **query_options,
                            )
                        )
                    )
                    if not self.config.growth_enabled:
                        # Foreground engine/config state is request-local.
                        # Cancellation need not wait merely to restore shared
                        # policy, and other reads remain unblocked.
                        try:
                            result = await asyncio.wait_for(
                                query_task,
                                timeout=max(1.0, plan.deadline_seconds),
                            )
                        except TimeoutError as exc:
                            raise TimeoutError(
                                f"memory {plan.preset} deadline exceeded "
                                f"({plan.deadline_seconds:.1f}s)"
                            ) from exc
                    else:
                        # A cancelled to_thread keeps running. Background growth
                        # may still write SQLite, so retain the write lock until
                        # the worker really finishes.
                        try:
                            result = await asyncio.shield(query_task)
                        except asyncio.CancelledError:
                            while not query_task.done():
                                try:
                                    await asyncio.shield(query_task)
                                except asyncio.CancelledError:
                                    continue
                                except Exception:
                                    break
                            raise
                finally:
                    if (
                        retrieval_config is not None
                        and original_rerank_enabled is not None
                    ):
                        retrieval_config.rerank_enabled = (
                            original_rerank_enabled
                        )
                    if (
                        retrieval_config is not None
                        and original_rerank_backend is not None
                    ):
                        retrieval_config.rerank_backend = (
                            original_rerank_backend
                        )
                    if retrieval_config is not None:
                        for name, value in original_deep_rerank_settings.items():
                            setattr(retrieval_config, name, value)
                        for name, value in original_light_budget_settings.items():
                            setattr(retrieval_config, name, value)
                    if (
                        retrieval_config is not None
                        and original_growth_persist_only_used is not None
                    ):
                        retrieval_config.growth_persist_only_used = (
                            original_growth_persist_only_used
                        )
                    if (
                        retrieval_config is not None
                        and original_growth_max_rounds is not None
                    ):
                        retrieval_config.growth_max_rounds = (
                            original_growth_max_rounds
                        )
                result["service_cache_hit"] = False
                learned_query_embeddings = getattr(
                    request_engine, "last_query_embeddings", {}
                )
                if self.config.query_embedding_cache_size > 0:
                    for key, value in learned_query_embeddings.items():
                        self._query_embedding_cache[str(key)] = (
                            value.copy() if hasattr(value, "copy") else value
                        )
                        self._query_embedding_cache.move_to_end(str(key))
                    while (
                        len(self._query_embedding_cache)
                        > self.config.query_embedding_cache_size
                    ):
                        self._query_embedding_cache.popitem(last=False)
                result["small_database_fast_path"] = (
                    self._small_database_fast_path
                )
                result["retrieval_intensity"] = intensity
                result["retrieval_plan"] = asdict(plan)
                result["lightweight_fast_path"] = lightweight
                result["rerank_policy"] = rerank_route
                # Compatibility alias for existing experiment readers.
                result["rerank_route"] = rerank_route
                rerank_trace = result.get("rerank_trace") or {}
                policy_enabled = not rerank_route.startswith("disabled_")
                if not policy_enabled:
                    execution_status = "disabled"
                elif not rerank_trace:
                    execution_status = "not_needed"
                elif rerank_trace.get("error"):
                    execution_status = "degraded"
                else:
                    execution_status = "completed"
                result["rerank_execution"] = {
                    "status": execution_status,
                    "attempted": bool(policy_enabled and rerank_trace),
                    "backend": rerank_trace.get("backend"),
                    "model": rerank_trace.get("model"),
                    "review_level": rerank_trace.get("review_level"),
                    "cache_hit": bool(rerank_trace.get("cache_hit")),
                    "error": str(rerank_trace.get("error") or ""),
                    "fallback_used": False,
                }
                if lightweight:
                    result["evidence_episodes"] = list(
                        result.get("evidence_episodes") or []
                    )[:12]
                    result["evidence_concepts"] = list(
                        result.get("evidence_concepts") or []
                    )[:10]
                    result["association_paths"] = list(
                        result.get("association_paths") or []
                    )[:10]
                    result["chronology_notes"] = list(
                        result.get("chronology_notes") or []
                    )[:6]
                quality = assess_retrieval_quality(
                    question,
                    result,
                )
                result["retrieval_quality"] = quality.to_dict()
                result["coverage_selector"] = {
                    "version": "coverage-selector-v1",
                    "strategy": (
                        "bge_plus_deterministic_slot_floor"
                        if rerank_trace.get("backend") == "cross_encoder"
                        else "audited_association_capsule"
                        if rerank_trace.get("backend") == "association_capsule"
                        else "llm_coverage_plus_deterministic_slot_floor"
                        if rerank_trace.get("backend") == "llm"
                        else "deterministic_slot_floor"
                    ),
                    "candidate_limit": plan.candidate_limit,
                    "evidence_slot_trace": deepcopy(
                        result.get("evidence_slot_trace") or {}
                    ),
                    "quality": quality.to_dict(),
                }
                result["lore_fact_contract"] = build_lore_fact_contract(
                    result,
                    quality,
                )
            except Exception as exc:
                failed_value = RetrievedMemory(
                    context="长期记忆检索失败；本轮只能依据当前对话回答。",
                    raw_result={"service_cache_hit": False},
                    error=f"{type(exc).__name__}: {exc}",
                    domains=(self.config.domain,),
                )
                if (
                    self.config.recall_failure_ttl_seconds > 0
                    and self.config.recall_cache_size > 0
                    and not diagnostic_override
                    and not self.config.growth_enabled
                    and plan.preset == "deep"
                ):
                    self._recall_failure_cache[cache_key] = (
                        monotonic(),
                        failed_value,
                    )
                    self._recall_failure_cache.move_to_end(cache_key)
                    while (
                        len(self._recall_failure_cache)
                        > self.config.recall_cache_size
                    ):
                        self._recall_failure_cache.popitem(last=False)
                return failed_value
            recalled = RetrievedMemory(
                context=format_memory_context(
                    result,
                    min(self.config.context_max_chars, 4_000)
                    if lightweight
                    else self.config.context_max_chars,
                ),
                raw_result=result,
                domains=(self.config.domain,),
                semantic_vector=getattr(
                    request_engine, "last_query_embeddings", {}
                ).get(question.strip()),
                query_vector_bundle=query_vector_bundle,
            )
            self._recall_failure_cache.pop(cache_key, None)
            if self.config.recall_cache_size > 0 and not diagnostic_override:
                cached_value = RetrievedMemory(
                    context=recalled.context,
                    raw_result=deepcopy(result),
                    domains=recalled.domains,
                    semantic_vector=(
                        recalled.semantic_vector.copy()
                        if hasattr(recalled.semantic_vector, "copy")
                        else recalled.semantic_vector
                    ),
                    query_vector_bundle=recalled.query_vector_bundle,
                )
                self._recall_cache[cache_key] = (
                    monotonic(),
                    cached_value,
                )
                self._recall_cache.move_to_end(cache_key)
                while len(self._recall_cache) > self.config.recall_cache_size:
                    self._recall_cache.popitem(last=False)
            quality = result["retrieval_quality"]
            should_escalate = bool(
                plan.preset == "standard"
                and plan.followup_policy == "on_insufficient_evidence"
                and not quality["sufficient"]
                and not diagnostic_override
                and not self._small_database_fast_path
                and not self.config.growth_enabled
                and auto_escalate
            )
            if should_escalate:
                deep_result = await self.recall(
                    question,
                    request=request,
                    retrieval_plan=self.retrieval_plan("deep"),
                    auto_escalate=False,
                    query_vector_bundle=query_vector_bundle,
                )
                escalation = {
                    "triggered": True,
                    "from": "standard",
                    "to": "deep",
                    "reasons": list(quality["reasons"]),
                    "initial_quality": deepcopy(quality),
                    "initial_episode_ids": list(
                        result.get("episode_ids") or []
                    ),
                    "initial_timings": deepcopy(result.get("timings") or {}),
                    "deep_error": deep_result.error,
                }
                if deep_result.raw_result and not deep_result.error:
                    final_raw = deepcopy(deep_result.raw_result)
                    final_raw["retrieval_escalation"] = escalation
                    final = RetrievedMemory(
                        context=deep_result.context,
                        raw_result=final_raw,
                        domains=deep_result.domains,
                        semantic_vector=deep_result.semantic_vector,
                        query_vector_bundle=deep_result.query_vector_bundle,
                    )
                    if self.config.recall_cache_size > 0:
                        self._recall_cache[cache_key] = (
                            monotonic(),
                            RetrievedMemory(
                                context=final.context,
                                raw_result=deepcopy(final_raw),
                                domains=final.domains,
                                semantic_vector=(
                                    final.semantic_vector.copy()
                                    if hasattr(final.semantic_vector, "copy")
                                    else final.semantic_vector
                                ),
                                query_vector_bundle=final.query_vector_bundle,
                            ),
                        )
                    return final
                result["retrieval_escalation"] = escalation
                recalled.raw_result = result
            return recalled

    async def audit_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> RetrievedMemory:
        """Audit fixed, foreground-visible lore edges without re-retrieval."""

        await self.initialize()
        if not self._has_memories or not candidates:
            return RetrievedMemory(
                context="",
                raw_result={
                    "episode_ids": [],
                    "association_ids": [],
                    "new_association_ids": [],
                    "reinforced_association_ids": [],
                    "direct_candidate_audit": True,
                },
                domains=(self.config.domain,),
            )

        async with self._operation_lock:
            try:
                outcome = await asyncio.to_thread(
                    self._query_engine.growth.audit_candidates,
                    candidates,
                )
            except Exception as exc:
                return RetrievedMemory(
                    context="",
                    raw_result={"direct_candidate_audit": True},
                    error=f"{type(exc).__name__}: {exc}",
                    domains=(self.config.domain,),
                )
            changed_ids = outcome.changed_ids
            if changed_ids:
                cache = getattr(
                    self._application.associations,
                    "_concept_reach_cache",
                    None,
                )
                if isinstance(cache, dict):
                    cache.clear()
                self._recall_cache.clear()
                self._recall_failure_cache.clear()
            episode_ids: list[int] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                for value in candidate.get("premise_episode_ids", []):
                    try:
                        episode_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if episode_id not in episode_ids:
                        episode_ids.append(episode_id)
            return RetrievedMemory(
                context="",
                raw_result={
                    "episode_ids": episode_ids,
                    "association_ids": changed_ids,
                    "new_association_ids": list(outcome.created_ids),
                    "reinforced_association_ids": list(
                        outcome.reinforced_ids
                    ),
                    "direct_candidate_audit": True,
                    "growth_utility_gate": {
                        "enabled": False,
                        "mode": "fixed_endpoint_dual_audit",
                        "reason": (
                            "候选端点由前台可见证据固定；通过主审、对抗终审和完整性检查后直接持久化，"
                            "后续以独立检索 A/B 评估实际收益。"
                        ),
                    },
                },
                domains=(self.config.domain,),
            )

    async def apply_contextual_plasticity(
        self,
        candidates: list[dict[str, Any]] | None,
        query_vector_bundle: Any | None,
    ) -> dict[str, Any]:
        """Persist retrieval-only double-key edges from local trace metadata."""

        if not self.config.contextual_association_enabled or not candidates:
            return {"enabled": False, "created": [], "rejected": [], "external_calls": 0}
        if query_vector_bundle is None:
            return {
                "enabled": True,
                "created": [],
                "rejected": [{"reason": "query_vector_bundle_unavailable"}],
                "external_calls": 0,
            }
        await self.initialize()
        from memory_demo.types import ContextualRecallCandidate

        query_items = tuple(getattr(query_vector_bundle, "queries", ()))
        queries = {str(item.query_id): item for item in query_items}
        whole = next((item for item in query_items if item.role == "whole"), None)
        if whole is None:
            whole = type(
                "WholeQuery",
                (),
                {
                    "query_id": "",
                    "text_hash": "",
                    "text": "",
                    "vector": getattr(query_vector_bundle, "whole", None),
                },
            )()
        created: list[int] = []
        rejected: list[dict[str, Any]] = []
        async with self._operation_lock:
            for raw in list(candidates)[: self.config.contextual_max_candidates_per_turn]:
                if not isinstance(raw, dict):
                    continue
                try:
                    candidate = ContextualRecallCandidate(
                        anchor_type=str(raw.get("anchor_type", "episode")),
                        anchor_id=int(raw["anchor_id"]),
                        target_episode_id=int(raw["target_episode_id"]),
                        context_query_id=str(raw["context_query_id"]),
                        need_query_id=str(raw["need_query_id"]),
                        slot_id=str(raw.get("slot_id", "")),
                        source_request_hash=str(raw.get("source_request_hash", "")),
                        reason=str(raw.get("reason", "recovered_missing_evidence")),
                    )
                    need = queries.get(candidate.need_query_id)
                    context = queries.get(candidate.context_query_id, whole)
                    if need is None or getattr(context, "vector", None) is None:
                        raise ValueError("query vector reference is unavailable")
                    association_id = self._application.create_contextual_association(
                        candidate,
                        domain=self.config.domain,
                        model_id=str(getattr(query_vector_bundle, "model_id", "")),
                        context_vector=context.vector,
                        need_vector=need.vector,
                        context_text_hash=str(getattr(context, "text_hash", "")),
                        need_text_hash=str(getattr(need, "text_hash", "")),
                        context_display_text=str(getattr(context, "text", "")),
                        need_display_text=str(getattr(need, "text", "")),
                    )
                    created.append(int(association_id))
                except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                    rejected.append(
                        {
                            "target_episode_id": raw.get("target_episode_id"),
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if created:
                self._recall_cache.clear()
                self._recall_failure_cache.clear()
        return {
            "enabled": True,
            "created": list(dict.fromkeys(created)),
            "rejected": rejected,
            "external_calls": 0,
        }

    async def record_contextual_utility(
        self, observations: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        """Apply local Treatment/Masked observations in one short transaction."""

        if not self.config.contextual_association_enabled or not observations:
            return {"enabled": False, "updated": 0, "external_calls": 0}
        if self.config.contextual_association_shadow:
            return {
                "enabled": True,
                "updated": 0,
                "skipped": "shadow_mode",
                "external_calls": 0,
            }
        await self.initialize()
        from memory_demo.types import ContextualUtilityObservation

        normalized = []
        for raw in observations:
            if not isinstance(raw, dict):
                continue
            try:
                normalized.append(
                    ContextualUtilityObservation(
                        association_id=int(raw["association_id"]),
                        query_hash=str(raw.get("query_hash", "")),
                        outcome=str(raw.get("outcome", "no_op")),
                        delta_slots=int(raw.get("delta_slots", 0)),
                        treatment_episode_ids=tuple(
                            int(value) for value in raw.get("treatment_episode_ids", [])
                        ),
                        masked_episode_ids=tuple(
                            int(value) for value in raw.get("masked_episode_ids", [])
                        ),
                        attribution=str(raw.get("attribution", "batch")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        if not normalized:
            return {"enabled": True, "updated": 0, "external_calls": 0}
        async with self._operation_lock:
            result = self._application.associations.record_utility(normalized)
            self._recall_cache.clear()
            self._recall_failure_cache.clear()
        return {"enabled": True, **dict(result), "external_calls": 0}

    async def import_file(self, path: str | Path, source_root: str | Path) -> dict:
        await self.initialize()
        input_path = Path(path).resolve()
        root = Path(source_root).resolve()
        async with self._operation_lock:
            result = await asyncio.to_thread(
                partial(
                    self._application.import_path,
                    input_path,
                    source_root=root,
                    rebuild_indexes_before_import=False,
                )
            )
            # ImportPipeline updates Episode/Concept/Paragraph indexes in place.
            # Association cue embeddings are RAM-only and need an explicit rebuild.
            if self.config.association_cue_enabled:
                await asyncio.to_thread(
                    self._application.rebuild_association_cue_index
                )
                self._query_engine = self._application.query_engine(
                    self._application.new_logger("chatbot-query")
                )
            self._has_memories = self._application.episodes.count() > 0
            self._recall_cache.clear()
            self._recall_failure_cache.clear()
            return result

    async def refresh_indexes(self) -> None:
        """Refresh RAM indexes after another runtime imports durable nodes."""

        await self.initialize()
        async with self._operation_lock:
            await asyncio.to_thread(self._application.rebuild_indexes)
            if self.config.association_cue_enabled:
                await asyncio.to_thread(
                    self._application.rebuild_association_cue_index
                )
                self._query_engine = self._application.query_engine(
                    self._application.new_logger("chatbot-query")
                )
            self._has_memories = self._application.episodes.count() > 0
            self._recall_cache.clear()
            self._recall_failure_cache.clear()

    async def refresh_graph_state(self) -> None:
        """Expose background Association writes to the foreground runtime."""

        await self.initialize()
        async with self._operation_lock:
            cache = getattr(self._application.associations, "_concept_reach_cache", None)
            if isinstance(cache, dict):
                cache.clear()
            self._recall_cache.clear()
            self._recall_failure_cache.clear()
            if self.config.association_cue_enabled:
                await asyncio.to_thread(
                    self._application.rebuild_association_cue_index
                )
                # Rebuilding replaces the RAM EmbeddingIndex. The long-lived
                # route matcher also caches its lexical view, so keeping the
                # old QueryEngine would hide a newly audited edge until the
                # process restarts.
                self._query_engine = self._application.query_engine(
                    self._application.new_logger("chatbot-query")
                )

    async def stats(self) -> dict[str, Any]:
        await self.initialize()
        async with self._operation_lock:
            result = await asyncio.to_thread(self._application.stats)
            result["small_database_fast_path"] = (
                self._small_database_fast_path
            )
            result["small_database_fast_path_max_episodes"] = (
                self.config.small_database_fast_path_max_episodes
            )
            return result


