"""Public memory-service API.

Implementation is split by responsibility:

``config`` -> environment and domain configuration
``contracts`` -> immutable retrieval policy and evidence quality
``intent_planner`` -> model-produced semantic retrieval request
``routing`` -> domain-agnostic failure fallback
``domain`` -> one physical memory database
``system`` -> private/public/knowledge orchestration

Importing from this module remains the supported integration boundary.
"""

from .config import MemoryServiceConfig, MemorySystemConfig
from .contracts import (
    DomainRecallRequest,
    RetrievedMemory,
    RetrievalPlan,
    RetrievalQuality,
    assess_retrieval_quality,
    build_lore_fact_contract,
    format_memory_context,
)
from .domain import AssociativeMemoryService
from .intent_planner import MemoryIntentPlan, MemoryIntentPlanner
from .routing import MemoryRoute, route_memory_query
from .system import MemorySystem

__all__ = [
    "AssociativeMemoryService",
    "DomainRecallRequest",
    "MemoryRoute",
    "MemoryIntentPlan",
    "MemoryIntentPlanner",
    "MemoryServiceConfig",
    "MemorySystem",
    "MemorySystemConfig",
    "RetrievalPlan",
    "RetrievalQuality",
    "RetrievedMemory",
    "assess_retrieval_quality",
    "build_lore_fact_contract",
    "format_memory_context",
    "route_memory_query",
]
