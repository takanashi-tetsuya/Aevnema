from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from src.memory import MemorySystem, MemorySystemConfig, RetrievalPlan


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES: tuple[dict[str, Any], ...] = (
    {"name": "shiroko_profile", "question": "那你记得白子的哪些事情呢？", "expected": [26, 38, 63, 150, 381]},
    {"name": "shiroko_terror_difference", "question": "你还记得白子Terror和现在的白子有什么不同吗？", "expected": [80]},
    {"name": "hoshino_yume_guilt", "question": "你还记得星野为什么一直对梦前辈的事情自责吗？", "expected": [25, 37, 154, 198]},
    {"name": "nagisa_makeup_club", "question": "你记得渚为什么要组织补课部吗？", "expected": [101, 134, 163, 310]},
    {"name": "kaiser_abydos", "question": "你还记得凯撒集团和阿拜多斯有什么纠葛吗？", "expected": [156, 195, 286]},
    {"name": "missing_shiroko_encounter", "question": "你还记得为师当时怎么和白子相遇的吗？", "expected": [], "expected_missing": True},
    {"name": "private_password", "question": "你还记得我们的暗号吗？", "expected": [], "non_knowledge": True},
    {"name": "private_preference", "question": "我以前说过自己喜欢什么饮料？", "expected": [], "non_knowledge": True},
    {"name": "public_convention", "question": "所有用户都通用的安全约定是什么？", "expected": [], "non_knowledge": True},
    {"name": "casual_greeting", "question": "早上好。", "expected": [], "non_knowledge": True},
    {"name": "creative_messages", "question": "有哪些学生刚刚给为师发消息了？", "expected": [], "creative": True},
)


def probe_plan(with_reranker: bool) -> RetrievalPlan:
    return RetrievalPlan(
        preset="light",
        query_planner="heuristic",
        graph_hops=0,
        candidate_limit=20,
        reranker="configured" if with_reranker else "disabled",
        evidence_slots=False,
        followup_policy="never",
        verification="local",
        deadline_seconds=3.0,
        answer_episode_limit=12,
        answer_concept_limit=8,
        answer_path_limit=0,
    )


async def run(
    output: Path, concurrency: int, with_reranker: bool
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
    await memory.initialize()
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        intent = {
            "language": "auto",
            "target_entities": [],
            "search_queries": [case["question"]],
            "requested_relation": "",
            "temporal_constraint": "",
            "causal_constraint": "",
            "answer_shape": "cross_domain_probe",
            "uncertainty_required": True,
        }
        async with semaphore:
            started = perf_counter()
            recalled = await memory.knowledge.recall(
                case["question"],
                intent_override=intent,
                followup_queries_override=[],
                retrieval_plan=probe_plan(with_reranker),
                auto_escalate=False,
            )
            elapsed = perf_counter() - started
        raw = recalled.raw_result or {}
        episode_ids = [int(value) for value in raw.get("episode_ids") or []]
        expected = {int(value) for value in case.get("expected") or []}
        return {
            **case,
            "seconds": round(elapsed, 6),
            "episode_ids": episode_ids[:20],
            "hit_at_20": bool(expected.intersection(episode_ids[:20])) if expected else None,
            "matched": sorted(expected.intersection(episode_ids[:20])),
            "quality": raw.get("retrieval_quality") or {},
            "timings": raw.get("timings") or {},
            "top_evidence": [
                {
                    "id": item.get("id"),
                    "source_key": item.get("source_key"),
                    "text": " ".join(str(item.get("text", "")).split())[:260],
                }
                for item in (raw.get("evidence_episodes") or [])[:6]
            ],
        }

    rows = await asyncio.gather(*(evaluate(case) for case in CASES))
    scored = [row for row in rows if row["hit_at_20"] is not None]
    payload = {
        "experiment": "probe-recall-bge-v1" if with_reranker else "probe-recall-v1",
        "summary": {
            "recall_at_20": sum(row["hit_at_20"] for row in scored) / len(scored),
            "mean_seconds": mean(row["seconds"] for row in rows),
            "max_seconds": max(row["seconds"] for row in rows),
        },
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--with-reranker", action="store_true")
    args = parser.parse_args()
    payload = asyncio.run(
        run(args.output, max(1, args.concurrency), args.with_reranker)
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    for row in payload["cases"]:
        print(row["name"], row["seconds"], row["hit_at_20"], row["episode_ids"][:5])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
