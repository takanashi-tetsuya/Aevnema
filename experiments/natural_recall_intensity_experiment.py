from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from src.memory import MemorySystem, MemorySystemConfig, route_memory_query
from src.memory.service import _knowledge_retrieval_question


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "shiroko_profile",
        "question": "那你记得白子的哪些事情呢？",
        "expected_episode_ids": [26, 38, 63, 150, 381],
    },
    {
        "name": "shiroko_terror_difference",
        "question": "你还记得白子Terror和现在的白子有什么不同吗？",
        "expected_episode_ids": [80],
    },
    {
        "name": "hoshino_yume_guilt",
        "question": "你还记得星野为什么一直对梦前辈的事情自责吗？",
        "expected_episode_ids": [25, 37, 154, 198],
    },
    {
        "name": "nagisa_makeup_club",
        "question": "你记得渚为什么要组织补课部吗？",
        "expected_episode_ids": [101, 134, 163, 310],
    },
    {
        "name": "kaiser_abydos",
        "question": "你还记得凯撒集团和阿拜多斯有什么纠葛吗？",
        "expected_episode_ids": [156, 195, 286],
    },
    {
        "name": "missing_shiroko_encounter",
        "question": "你还记得为师当时怎么和白子相遇的吗？",
        "expected_episode_ids": [],
        "expected_missing": True,
    },
    {
        "name": "private_only_password",
        "question": "你还记得我们的暗号吗？",
        "expected_episode_ids": [],
        "route_only": True,
    },
)


def _route_dict(route: Any) -> dict[str, Any]:
    return {
        "user": route.user,
        "public": route.public,
        "knowledge": route.knowledge,
        "reason": route.reason,
        "intensity": route.intensity,
        "knowledge_write_policy": route.knowledge_write_policy,
        "creative": route.creative,
    }


async def _run_case(
    memory: MemorySystem,
    case: dict[str, Any],
    intensity: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    route = route_memory_query(case["question"])
    base = {
        "name": case["name"],
        "question": case["question"],
        "route": _route_dict(route),
        "expected_episode_ids": list(case.get("expected_episode_ids") or []),
        "expected_missing": bool(case.get("expected_missing")),
        "intensity": intensity,
    }
    if case.get("route_only"):
        base.update(
            {
                "skipped": True,
                "route_passed": route.user and not route.knowledge,
            }
        )
        return base
    if not route.knowledge:
        base.update({"error": "route did not select knowledge", "hit_at_20": False})
        return base

    knowledge_question = _knowledge_retrieval_question(case["question"])
    async with semaphore:
        started = perf_counter()
        recalled = await memory.knowledge.recall(
            knowledge_question,
            retrieval_intensity=intensity,
            auto_escalate=False,
        )
        elapsed = perf_counter() - started
    raw = recalled.raw_result or {}
    episode_ids = [int(value) for value in raw.get("episode_ids") or []]
    expected = {int(value) for value in case.get("expected_episode_ids") or []}
    top20 = episode_ids[:20]
    hit = bool(expected.intersection(top20)) if expected else None
    evidence = list(raw.get("evidence_episodes") or [])[:8]
    base.update(
        {
            "knowledge_question": knowledge_question,
            "elapsed_seconds": round(elapsed, 6),
            "error": recalled.error,
            "hit_at_20": hit,
            "matched_expected_ids": sorted(expected.intersection(top20)),
            "top20_episode_ids": top20,
            "retrieval_quality": raw.get("retrieval_quality") or {},
            "stage_timings": raw.get("timings") or {},
            "rerank_policy": raw.get("rerank_policy"),
            "rerank_execution": raw.get("rerank_execution") or {},
            "context_chars": len(recalled.context),
            "top_evidence": [
                {
                    "id": item.get("id"),
                    "source_key": item.get("source_key"),
                    "text": " ".join(str(item.get("text", "")).split())[:360],
                }
                for item in evidence
            ],
        }
    )
    return base


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_intensity: dict[str, Any] = {}
    for intensity in ("light", "standard"):
        selected = [
            row
            for row in rows
            if row.get("intensity") == intensity and not row.get("skipped")
        ]
        scored = [row for row in selected if row.get("hit_at_20") is not None]
        latencies = [float(row["elapsed_seconds"]) for row in selected if not row.get("error")]
        by_intensity[intensity] = {
            "case_count": len(selected),
            "scored_case_count": len(scored),
            "recall_at_20": (
                round(sum(bool(row.get("hit_at_20")) for row in scored) / len(scored), 6)
                if scored
                else None
            ),
            "mean_seconds": (
                round(sum(latencies) / len(latencies), 6) if latencies else None
            ),
            "max_seconds": round(max(latencies), 6) if latencies else None,
            "under_five_seconds_rate": (
                round(sum(value <= 5.0 for value in latencies) / len(latencies), 6)
                if latencies
                else None
            ),
        }
    return by_intensity


async def run(output: Path, concurrency: int) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
    await memory.initialize()
    semaphore = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(
        *(
            _run_case(memory, case, intensity, semaphore)
            for intensity in ("light", "standard")
            for case in CASES
        )
    )
    payload = {
        "experiment": "natural-recall-intensity-v1",
        "plans": {
            intensity: asdict(memory.knowledge.retrieval_plan(intensity))
            for intensity in ("light", "standard")
        },
        "summary": _summary(rows),
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    payload = asyncio.run(run(args.output, args.concurrency))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
