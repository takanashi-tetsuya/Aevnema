"""Small, domain-agnostic fallback used when model planning is unavailable."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(slots=True, frozen=True)
class MemoryRoute:
    user: bool
    public: bool
    knowledge: bool
    reason: str
    intensity: str = "standard"
    knowledge_write_policy: str = "query_and_answer"
    creative: bool = False


_CASUAL_UTTERANCES = {
    "hi",
    "hello",
    "hey",
    "你好",
    "你好呀",
    "早上好",
    "中午好",
    "晚上好",
    "晚安",
    "谢谢",
    "多谢",
    "哈哈",
    "哈哈哈",
}

_EXPLICIT_DEEP_MARKERS = (
    "综合分析",
    "深入分析",
    "彻底分析",
    "完整分析",
    "完整推导",
    "逐步推导",
    "完整论证",
    "深度检索",
    "deep search",
    "comprehensive analysis",
)


def _lightweight_target_entities(question: str) -> list[str]:
    """Extract only explicit generic anchors for the emergency fast path."""

    normalized = " ".join(question.strip().split())
    candidates: list[str] = []
    candidates.extend(
        match.strip()
        for match in re.findall(r"[“\"'「『](.{1,40}?)[”\"'」』]", normalized)
        if match.strip()
    )
    relation = re.search(
        r"([A-Za-z0-9_·\u3400-\u9fff]{1,20})"
        r"(?:和|与|跟|及|and)"
        r"([A-Za-z0-9_·\u3400-\u9fff]{1,20})"
        r"(?:是什么关系|有何关系|的关系|之间|relationship)",
        normalized,
        re.IGNORECASE,
    )
    if relation:
        candidates.extend(value.strip() for value in relation.groups())
    return list(dict.fromkeys(candidates))[:6]


def _knowledge_retrieval_question(question: str) -> str:
    """Fallback keeps user semantics intact; model planning owns rewriting."""

    return " ".join(question.strip().split())


def _lightweight_search_queries(question: str) -> list[str]:
    """Split explicit clauses without adding domain-specific interpretations."""

    normalized = " ".join(question.strip().split())
    queries = [normalized]
    queries.extend(
        part.strip()
        for part in re.split(r"[，,。；;？?！!\n]", normalized)
        if len(part.strip()) >= 4
    )
    return list(dict.fromkeys(value for value in queries if value))[:7]


def route_memory_query(question: str) -> MemoryRoute:
    """Fail open to all domains without pretending to understand semantics.

    The production path uses ``MemoryIntentPlanner``. If that model is
    disabled or fails, a non-trivial message searches all isolated domains and
    relies on evidence selection to discard irrelevant results. This avoids a
    second, brittle expert system hidden in fallback code.
    """

    normalized = " ".join(question.casefold().split()).strip("。.!！?？~～ ")
    if not normalized:
        return MemoryRoute(False, False, False, "empty")
    if normalized in _CASUAL_UTTERANCES:
        return MemoryRoute(False, False, False, "casual")
    intensity = (
        "deep"
        if any(marker in normalized for marker in _EXPLICIT_DEEP_MARKERS)
        else "standard"
    )
    return MemoryRoute(
        True,
        True,
        True,
        "model_planner_fallback",
        intensity,
        "evidence_gated",
        False,
    )
