from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any


FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() not in FALSE_VALUES


def _apply_ingestion_env_overrides(app_config: Any) -> None:
    """Forward chatbot import controls into the embedded memory engine."""

    app_config.ingestion.task_max_attempts = max(
        1, int(os.getenv("MEMORY_IMPORT_TASK_MAX_ATTEMPTS", "2"))
    )
    for attribute, env_name in (
        ("prepare_workers", "MEMORY_IMPORT_PREPARE_WORKERS"),
        ("relation_workers", "MEMORY_IMPORT_RELATION_WORKERS"),
        ("relation_batch_size", "MEMORY_IMPORT_RELATION_BATCH_SIZE"),
        (
            "episode_factual_audit_batch_size",
            "MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_BATCH_SIZE",
        ),
    ):
        current = getattr(
            app_config.ingestion,
            attribute,
            3 if attribute == "episode_factual_audit_batch_size" else 1,
        )
        setattr(
            app_config.ingestion,
            attribute,
            max(1, int(os.getenv(env_name, str(current)))),
        )
    app_config.ingestion.episode_audit_always = _env_bool(
        "MEMORY_IMPORT_EPISODE_AUDIT_ALWAYS", False
    )
    episode_profile = (
        os.getenv(
            "MEMORY_IMPORT_EPISODE_PROFILE",
            getattr(app_config.ingestion, "episode_extraction_profile", "legacy"),
        )
        .strip()
        .casefold()
    )
    if episode_profile not in {
        "legacy",
        "single_pass_evidence",
        "single_pass_audited",
        "document_map_assisted",
        "document_map_contextual",
        "adaptive_anchor_map",
        "source_scoped_plain",
    }:
        raise ValueError(
            "MEMORY_IMPORT_EPISODE_PROFILE must be legacy, "
            "single_pass_evidence, single_pass_audited, "
            "document_map_assisted, document_map_contextual, or "
            "adaptive_anchor_map, or source_scoped_plain"
        )
    app_config.ingestion.episode_extraction_profile = episode_profile
    if episode_profile == "single_pass_evidence":
        app_config.prompt_version = "v4.1_single_pass_line_spans"
    elif episode_profile == "single_pass_audited":
        app_config.prompt_version = "v4.4_single_pass_entailment_roles"
    elif episode_profile == "document_map_assisted":
        app_config.prompt_version = "v4.5_document_map_planned_audited"
    elif episode_profile == "document_map_contextual":
        app_config.prompt_version = "v4.6_document_map_context_audited"
    elif episode_profile == "adaptive_anchor_map":
        app_config.prompt_version = "v4.7_adaptive_literal_anchor_audited"
    elif episode_profile == "source_scoped_plain":
        app_config.prompt_version = "v5.0_source_scoped_program_owned"
    factual_audit_mode = (
        os.getenv("MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODE", "off").strip().casefold()
    )
    if factual_audit_mode not in {"off", "adaptive", "always"}:
        raise ValueError(
            "MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODE must be off, adaptive, or always"
        )
    app_config.ingestion.episode_factual_audit_mode = factual_audit_mode
    app_config.ingestion.episode_factual_audit_model = os.getenv(
        "MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODEL", ""
    ).strip()
    app_config.ingestion.build_inference_relations = _env_bool(
        "MEMORY_IMPORT_BUILD_INFERENCE_RELATIONS",
        getattr(app_config.ingestion, "build_inference_relations", True),
    )


def _resolve_rerank_policy(
    intensity: str,
    *,
    enabled: bool,
    configured_backend: str,
) -> tuple[bool, str, str]:
    """Choose reranking by workload without making the model a global switch."""

    if not enabled:
        return False, configured_backend, "disabled_not_configured"
    if intensity == "light":
        return (
            True,
            configured_backend,
            (
                "cross_encoder_light"
                if configured_backend == "cross_encoder"
                else "llm_light"
            ),
        )
    if intensity == "deep":
        return True, "llm", "llm_deep_default"
    return (
        True,
        configured_backend,
        (
            "cross_encoder_standard"
            if configured_backend == "cross_encoder"
            else "llm_standard"
        ),
    )


def _portable_path_text(value: str, platform_name: str | None = None) -> str:
    """Translate Windows/WSL absolute paths at the configuration boundary."""

    raw = str(value).strip()
    platform = platform_name or os.name
    windows_path = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if platform != "nt" and windows_path:
        drive = windows_path.group(1).casefold()
        remainder = windows_path.group(2).replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{remainder}"
    wsl_path = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", raw)
    if platform == "nt" and wsl_path:
        drive = wsl_path.group(1).upper()
        remainder = (wsl_path.group(2) or "").replace("/", "\\")
        return f"{drive}:\\{remainder}"
    return raw


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(_portable_path_text(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(slots=True)
class MemoryServiceConfig:
    """Configuration boundary between the chatbot and memory-demo package."""

    engine_root: Path
    database_path: Path
    log_dir: Path
    api_key: str
    reranker_model: str = ""
    reasoning_model: str = ""
    fallback_model: str = ""
    reasoning_max_tokens: int = 0
    reasoning_enable_thinking: bool | None = None
    optimization_profile: str = "balanced"
    paragraph_enabled: bool = False
    concept_profile: str = "conservative"
    sparse_enabled: bool = True
    source_key_cohort_enabled: bool = True
    followup_planning_mode: str = "always"
    rerank_enabled: bool = False
    rerank_backend: str = "llm"
    rerank_coverage_audit_enabled: bool = True
    rerank_review_mode: str = ""
    rerank_atomic_query_limit: int = 40
    rerank_precompression_limit: int = 100
    growth_enabled: bool = True
    growth_persist_only_used: bool = True
    growth_counterfactual_utility_enabled: bool = True
    growth_staging_enabled: bool = True
    association_cue_enabled: bool = False
    association_cue_fast_path_enabled: bool = False
    association_cue_fast_path_min_similarity: float = 0.62
    association_cue_fast_path_min_margin: float = 0.08
    association_cue_fast_path_min_confidence: float = 0.72
    recall_cache_size: int = 0
    query_embedding_cache_size: int = 512
    recall_cache_ttl_seconds: float = 900.0
    recall_failure_ttl_seconds: float = 60.0
    context_max_chars: int = 8_000
    small_database_fast_path_max_episodes: int = 0
    domain: str = "knowledge"
    allow_create_database: bool = False

    @classmethod
    def from_env(cls, project_root: str | Path) -> "MemoryServiceConfig":
        root = Path(project_root).resolve()
        engine_value = os.getenv("MEMORY_ENGINE_ROOT", "").strip()
        if not engine_value:
            raise ValueError(
                "MEMORY_ENGINE_ROOT is required and must point to the associative-memory project"
            )
        db_value = os.getenv("MEMORY_DB_PATH", "data/memory.db")
        log_value = os.getenv("MEMORY_LOG_DIR", "logs/memory")
        reranker_model = os.getenv("MEMORY_RERANKER_MODEL", "").strip()
        return cls(
            engine_root=_resolve_path(engine_value, root),
            database_path=_resolve_path(db_value, root),
            log_dir=_resolve_path(log_value, root),
            api_key=os.getenv("SILICONFLOW_API_KEY", "").strip(),
            reranker_model=reranker_model,
            reasoning_model=os.getenv("MEMORY_REASONING_MODEL", "").strip(),
            fallback_model=os.getenv("MEMORY_FALLBACK_MODEL", "").strip(),
            reasoning_max_tokens=max(
                0, int(os.getenv("MEMORY_REASONING_MAX_TOKENS", "0"))
            ),
            reasoning_enable_thinking=(
                None
                if os.getenv("MEMORY_REASONING_ENABLE_THINKING") is None
                else _env_bool("MEMORY_REASONING_ENABLE_THINKING", False)
            ),
            optimization_profile=os.getenv(
                "MEMORY_OPTIMIZATION_PROFILE", "balanced"
            ).strip(),
            paragraph_enabled=_env_bool("MEMORY_PARAGRAPH_ENABLED", False),
            concept_profile=os.getenv("MEMORY_CONCEPT_PROFILE", "conservative").strip(),
            sparse_enabled=_env_bool("MEMORY_SPARSE_ENABLED", True),
            source_key_cohort_enabled=_env_bool(
                "MEMORY_SOURCE_KEY_COHORT_ENABLED", True
            ),
            followup_planning_mode=os.getenv(
                "MEMORY_FOLLOWUP_PLANNING_MODE", "always"
            ).strip(),
            rerank_enabled=bool(reranker_model),
            rerank_backend=("cross_encoder" if reranker_model else "llm"),
            rerank_coverage_audit_enabled=_env_bool(
                "MEMORY_RERANK_COVERAGE_AUDIT_ENABLED", True
            ),
            rerank_review_mode=os.getenv("MEMORY_RERANK_REVIEW_MODE", "").strip(),
            rerank_atomic_query_limit=max(
                1,
                int(os.getenv("MEMORY_RERANK_ATOMIC_QUERY_LIMIT", "40")),
            ),
            rerank_precompression_limit=max(
                1,
                int(os.getenv("MEMORY_RERANK_PRECOMPRESSION_LIMIT", "100")),
            ),
            growth_enabled=_env_bool("MEMORY_GROWTH_ENABLED", True),
            growth_persist_only_used=_env_bool("MEMORY_GROWTH_PERSIST_ONLY_USED", True),
            growth_counterfactual_utility_enabled=_env_bool(
                "MEMORY_GROWTH_COUNTERFACTUAL_UTILITY_ENABLED", True
            ),
            growth_staging_enabled=_env_bool("MEMORY_GROWTH_STAGING_ENABLED", True),
            association_cue_enabled=_env_bool("MEMORY_ASSOCIATION_CUE_ENABLED", False),
            association_cue_fast_path_enabled=_env_bool(
                "MEMORY_ASSOCIATION_CUE_FAST_PATH_ENABLED", False
            ),
            association_cue_fast_path_min_similarity=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "MEMORY_ASSOCIATION_CUE_FAST_PATH_MIN_SIMILARITY",
                            "0.62",
                        )
                    ),
                ),
            ),
            association_cue_fast_path_min_confidence=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "MEMORY_ASSOCIATION_CUE_FAST_PATH_MIN_CONFIDENCE",
                            "0.72",
                        )
                    ),
                ),
            ),
            association_cue_fast_path_min_margin=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "MEMORY_ASSOCIATION_CUE_FAST_PATH_MIN_MARGIN",
                            "0.08",
                        )
                    ),
                ),
            ),
            recall_cache_size=max(0, int(os.getenv("MEMORY_RECALL_CACHE_SIZE", "0"))),
            query_embedding_cache_size=max(
                0, int(os.getenv("MEMORY_QUERY_EMBEDDING_CACHE_SIZE", "512"))
            ),
            recall_cache_ttl_seconds=max(
                0.0,
                float(os.getenv("MEMORY_RECALL_CACHE_TTL_SECONDS", "900")),
            ),
            context_max_chars=max(
                2_000, int(os.getenv("MEMORY_CONTEXT_MAX_CHARS", "8000"))
            ),
            recall_failure_ttl_seconds=max(
                0.0,
                float(os.getenv("MEMORY_RECALL_FAILURE_TTL_SECONDS", "60")),
            ),
        )

    def validate(self) -> None:
        package_dir = self.engine_root / "src" / "memory_demo"
        if not package_dir.is_dir():
            raise FileNotFoundError(
                f"associative-memory package not found under {package_dir}"
            )
        if not self.database_path.is_file() and not self.allow_create_database:
            raise FileNotFoundError(
                f"runtime memory database does not exist: {self.database_path}"
            )
        if not self.api_key:
            raise ValueError("SILICONFLOW_API_KEY is required for memory retrieval")
        if self.rerank_backend not in {"llm", "cross_encoder"}:
            raise ValueError("rerank_backend must be llm or cross_encoder")
        if (
            self.rerank_enabled
            and self.rerank_backend == "cross_encoder"
            and not self.reranker_model
        ):
            raise ValueError(
                "reranker_model is required when cross_encoder rerank is enabled"
            )


@dataclass(slots=True)
class MemorySystemConfig:
    engine_root: Path
    knowledge_database_path: Path
    public_database_path: Path
    user_database_dir: Path
    log_dir: Path
    api_key: str
    reranker_model: str = ""
    optimization_profile: str = "balanced"
    user_background_profile: str = "lean"
    user_reasoning_model: str = "Qwen/Qwen3.5-35B-A3B"
    user_fallback_model: str = "deepseek-ai/DeepSeek-V3.2"
    user_reasoning_max_tokens: int = 2400
    user_reasoning_enable_thinking: bool = False
    knowledge_growth_reasoning_max_tokens: int = 3000
    knowledge_growth_max_rounds: int = 1
    knowledge_growth_reasoning_model: str = "deepseek-ai/DeepSeek-V3.2"
    knowledge_growth_fallback_model: str = "Qwen/Qwen3.5-35B-A3B"
    knowledge_growth_enable_thinking: bool = False
    # Foreground retrieval disables Association growth. Its fast-adaptive
    # evidence review trusts a complete first-pass coverage map and spends one
    # compressor call only when that pass reports a visible gap.
    foreground_profile: str = "balanced"
    foreground_reasoning_model: str = "Qwen/Qwen3.5-9B"
    foreground_reasoning_max_tokens: int = 1200
    foreground_reasoning_enable_thinking: bool = False
    foreground_review_mode: str = "fast_adaptive"
    foreground_rerank_backend: str = "llm"
    foreground_rerank_enabled: bool = False
    foreground_followup_planning_mode: str = "entity_resolved"
    foreground_atomic_query_limit: int = 12
    foreground_rerank_precompression_limit: int = 24
    foreground_recall_cache_size: int = 64
    foreground_recall_cache_ttl_seconds: float = 900.0
    foreground_recall_failure_ttl_seconds: float = 60.0
    paragraph_enabled: bool = False
    concept_profile: str = "conservative"
    context_max_chars: int = 8_000
    knowledge_growth_enabled: bool = True
    public_growth_enabled: bool = True
    user_growth_enabled: bool = True

    @classmethod
    def from_env(cls, project_root: str | Path) -> "MemorySystemConfig":
        root = Path(project_root).resolve()
        engine_value = os.getenv("MEMORY_ENGINE_ROOT", "").strip()
        if not engine_value:
            raise ValueError("MEMORY_ENGINE_ROOT is required")
        reranker_model = os.getenv("MEMORY_RERANKER_MODEL", "").strip()
        return cls(
            engine_root=_resolve_path(engine_value, root),
            knowledge_database_path=_resolve_path(
                os.getenv("KNOWLEDGE_MEMORY_DB_PATH", "data/knowledge/blue_archive.db"),
                root,
            ),
            public_database_path=_resolve_path(
                os.getenv("PUBLIC_MEMORY_DB_PATH", "data/public/memory.db"), root
            ),
            user_database_dir=_resolve_path(
                os.getenv("USER_MEMORY_DB_DIR", "data/users"), root
            ),
            log_dir=_resolve_path(os.getenv("MEMORY_LOG_DIR", "logs/memory"), root),
            api_key=os.getenv("SILICONFLOW_API_KEY", "").strip(),
            reranker_model=reranker_model,
            optimization_profile=os.getenv(
                "MEMORY_OPTIMIZATION_PROFILE", "balanced"
            ).strip(),
            user_background_profile=os.getenv(
                "USER_MEMORY_OPTIMIZATION_PROFILE", "lean"
            ).strip(),
            user_reasoning_model=os.getenv(
                "USER_MEMORY_REASONING_MODEL", "Qwen/Qwen3.5-35B-A3B"
            ).strip(),
            user_fallback_model=os.getenv(
                "USER_MEMORY_FALLBACK_MODEL", "deepseek-ai/DeepSeek-V3.2"
            ).strip(),
            user_reasoning_max_tokens=max(
                256,
                int(os.getenv("USER_MEMORY_REASONING_MAX_TOKENS", "2400")),
            ),
            user_reasoning_enable_thinking=_env_bool(
                "USER_MEMORY_REASONING_ENABLE_THINKING", False
            ),
            knowledge_growth_reasoning_max_tokens=max(
                512,
                int(os.getenv("KNOWLEDGE_GROWTH_REASONING_MAX_TOKENS", "3000")),
            ),
            knowledge_growth_max_rounds=max(
                1, int(os.getenv("KNOWLEDGE_GROWTH_MAX_ROUNDS", "1"))
            ),
            knowledge_growth_reasoning_model=os.getenv(
                "KNOWLEDGE_GROWTH_REASONING_MODEL",
                "deepseek-ai/DeepSeek-V3.2",
            ).strip(),
            knowledge_growth_fallback_model=os.getenv(
                "KNOWLEDGE_GROWTH_FALLBACK_MODEL",
                "Qwen/Qwen3.5-35B-A3B",
            ).strip(),
            knowledge_growth_enable_thinking=_env_bool(
                "KNOWLEDGE_GROWTH_ENABLE_THINKING", False
            ),
            foreground_profile=os.getenv(
                "MEMORY_FOREGROUND_PROFILE", "balanced"
            ).strip(),
            foreground_reasoning_model=os.getenv(
                "MEMORY_FOREGROUND_REASONING_MODEL", "Qwen/Qwen3.5-9B"
            ).strip(),
            foreground_reasoning_max_tokens=max(
                256,
                int(os.getenv("MEMORY_FOREGROUND_REASONING_MAX_TOKENS", "1200")),
            ),
            foreground_reasoning_enable_thinking=_env_bool(
                "MEMORY_FOREGROUND_REASONING_ENABLE_THINKING", False
            ),
            foreground_review_mode=os.getenv(
                "MEMORY_FOREGROUND_REVIEW_MODE", "fast_adaptive"
            ).strip(),
            foreground_rerank_backend=("cross_encoder" if reranker_model else "llm"),
            foreground_rerank_enabled=bool(reranker_model),
            foreground_followup_planning_mode=os.getenv(
                "MEMORY_FOREGROUND_FOLLOWUP_PLANNING_MODE",
                "entity_resolved",
            ).strip(),
            foreground_atomic_query_limit=max(
                1,
                int(os.getenv("MEMORY_FOREGROUND_ATOMIC_QUERY_LIMIT", "12")),
            ),
            foreground_rerank_precompression_limit=max(
                1,
                int(
                    os.getenv(
                        "MEMORY_FOREGROUND_RERANK_PRECOMPRESSION_LIMIT",
                        "24",
                    )
                ),
            ),
            foreground_recall_cache_size=max(
                0, int(os.getenv("MEMORY_RECALL_CACHE_SIZE", "64"))
            ),
            foreground_recall_cache_ttl_seconds=max(
                0.0,
                float(os.getenv("MEMORY_RECALL_CACHE_TTL_SECONDS", "900")),
            ),
            paragraph_enabled=_env_bool("MEMORY_PARAGRAPH_ENABLED", False),
            concept_profile=os.getenv("MEMORY_CONCEPT_PROFILE", "conservative").strip(),
            context_max_chars=max(
                2_000, int(os.getenv("MEMORY_CONTEXT_MAX_CHARS", "8000"))
            ),
            foreground_recall_failure_ttl_seconds=max(
                0.0,
                float(os.getenv("MEMORY_RECALL_FAILURE_TTL_SECONDS", "60")),
            ),
            knowledge_growth_enabled=_env_bool("KNOWLEDGE_MEMORY_GROWTH_ENABLED", True),
            public_growth_enabled=_env_bool("PUBLIC_MEMORY_GROWTH_ENABLED", True),
            user_growth_enabled=_env_bool("USER_MEMORY_GROWTH_ENABLED", True),
        )

    def domain_config(
        self,
        domain: str,
        database_path: Path,
        *,
        background: bool = False,
    ) -> MemoryServiceConfig:
        is_user = domain.startswith("user:")
        is_public = domain == "public"
        return MemoryServiceConfig(
            engine_root=self.engine_root,
            database_path=database_path,
            log_dir=(
                self.log_dir
                / ("background" if background else "foreground")
                / domain.replace(":", "_")
            ),
            api_key=self.api_key,
            reranker_model=self.reranker_model,
            reasoning_model=(
                self.user_reasoning_model
                if background and is_user
                else self.knowledge_growth_reasoning_model
                if background and domain == "knowledge"
                else self.foreground_reasoning_model
            ),
            fallback_model=(
                self.user_fallback_model
                if background and is_user
                else self.knowledge_growth_fallback_model
                if background and domain == "knowledge"
                else ""
            ),
            reasoning_max_tokens=(
                self.user_reasoning_max_tokens
                if background and is_user
                else self.knowledge_growth_reasoning_max_tokens
                if background and domain == "knowledge"
                else self.foreground_reasoning_max_tokens
            ),
            reasoning_enable_thinking=(
                self.user_reasoning_enable_thinking
                if background and is_user
                else self.knowledge_growth_enable_thinking
                if background and domain == "knowledge"
                else self.foreground_reasoning_enable_thinking
            ),
            optimization_profile=(
                self.user_background_profile
                if background and is_user
                else self.optimization_profile
                if background
                else self.foreground_profile
            ),
            paragraph_enabled=self.paragraph_enabled,
            concept_profile=self.concept_profile,
            sparse_enabled=_env_bool("MEMORY_SPARSE_ENABLED", True),
            source_key_cohort_enabled=_env_bool(
                "MEMORY_SOURCE_KEY_COHORT_ENABLED", True
            ),
            followup_planning_mode=(
                "always" if background else self.foreground_followup_planning_mode
            ),
            rerank_enabled=self.foreground_rerank_enabled,
            rerank_backend=("llm" if background else self.foreground_rerank_backend),
            rerank_coverage_audit_enabled=_env_bool(
                "MEMORY_RERANK_COVERAGE_AUDIT_ENABLED", True
            ),
            rerank_review_mode=("" if background else self.foreground_review_mode),
            rerank_atomic_query_limit=(
                40 if background else self.foreground_atomic_query_limit
            ),
            rerank_precompression_limit=(
                100 if background else self.foreground_rerank_precompression_limit
            ),
            growth_enabled=(
                (
                    self.user_growth_enabled
                    if is_user
                    else self.public_growth_enabled
                    if is_public
                    else self.knowledge_growth_enabled
                )
                if background
                else False
            ),
            growth_persist_only_used=_env_bool("MEMORY_GROWTH_PERSIST_ONLY_USED", True),
            growth_counterfactual_utility_enabled=_env_bool(
                "MEMORY_GROWTH_COUNTERFACTUAL_UTILITY_ENABLED", True
            ),
            growth_staging_enabled=_env_bool("MEMORY_GROWTH_STAGING_ENABLED", True),
            association_cue_enabled=_env_bool("MEMORY_ASSOCIATION_CUE_ENABLED", False),
            association_cue_fast_path_enabled=_env_bool(
                "MEMORY_ASSOCIATION_CUE_FAST_PATH_ENABLED", False
            ),
            association_cue_fast_path_min_similarity=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "MEMORY_ASSOCIATION_CUE_FAST_PATH_MIN_SIMILARITY",
                            "0.62",
                        )
                    ),
                ),
            ),
            association_cue_fast_path_min_confidence=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "MEMORY_ASSOCIATION_CUE_FAST_PATH_MIN_CONFIDENCE",
                            "0.72",
                        )
                    ),
                ),
            ),
            association_cue_fast_path_min_margin=max(
                0.0,
                min(
                    1.0,
                    float(
                        os.getenv(
                            "MEMORY_ASSOCIATION_CUE_FAST_PATH_MIN_MARGIN",
                            "0.08",
                        )
                    ),
                ),
            ),
            recall_cache_size=(0 if background else self.foreground_recall_cache_size),
            query_embedding_cache_size=(
                0
                if background
                else max(
                    0,
                    int(os.getenv("MEMORY_QUERY_EMBEDDING_CACHE_SIZE", "512")),
                )
            ),
            recall_cache_ttl_seconds=(self.foreground_recall_cache_ttl_seconds),
            recall_failure_ttl_seconds=(self.foreground_recall_failure_ttl_seconds),
            context_max_chars=max(2_000, self.context_max_chars // 2),
            small_database_fast_path_max_episodes=(
                0
                if background
                else max(
                    0,
                    int(
                        os.getenv(
                            "MEMORY_SMALL_DB_FAST_PATH_MAX_EPISODES",
                            "32",
                        )
                    ),
                )
            ),
            domain=domain,
            allow_create_database=is_user or is_public,
        )
