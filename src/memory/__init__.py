"""Chatbot integration for the associative Source/Episode/Concept graph."""

from .service import (
    AssociativeMemoryService,
    DomainRecallRequest,
    MemoryIntentPlan,
    MemoryIntentPlanner,
    MemoryRoute,
    MemoryServiceConfig,
    MemorySystem,
    MemorySystemConfig,
    RetrievalPlan,
    RetrievalQuality,
    RetrievedMemory,
    assess_retrieval_quality,
    build_lore_fact_contract,
    format_memory_context,
    route_memory_query,
)
from .identity import PlatformIdentity
from .growth import BackgroundGrowthWorker, GrowthJob
from .answer_consolidator import (
    AnswerMemoryConsolidator,
    ContextualRecallCandidate,
    ConsolidationDecision,
    derive_contextual_recall_candidates,
    derive_contextual_utility_observations,
    inline_memory_prompt,
    split_inline_memory_response,
)

__all__ = [
    "AssociativeMemoryService",
    "DomainRecallRequest",
    "MemoryIntentPlan",
    "MemoryIntentPlanner",
    "MemoryRoute",
    "MemoryServiceConfig",
    "MemorySystem",
    "MemorySystemConfig",
    "RetrievalPlan",
    "RetrievalQuality",
    "RetrievedMemory",
    "PlatformIdentity",
    "BackgroundGrowthWorker",
    "GrowthJob",
    "AnswerMemoryConsolidator",
    "ContextualRecallCandidate",
    "ConsolidationDecision",
    "derive_contextual_recall_candidates",
    "derive_contextual_utility_observations",
    "inline_memory_prompt",
    "split_inline_memory_response",
    "format_memory_context",
    "assess_retrieval_quality",
    "build_lore_fact_contract",
    "route_memory_query",
]
