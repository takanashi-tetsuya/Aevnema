from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from src.memory import MemorySystem, MemorySystemConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABEL_DOCUMENTS: dict[str, str] = {
    "private": "该请求的答案可能来自当前用户过去的对话、偏好、经历、约定、印象，或该用户专属的角色扮演连续性，因此私人记忆可能提供有用证据。",
    "public": "该请求的答案可能来自所有用户共同可用的共享长期记忆、跨用户公共约定或系统保存的普适信息，因此公共记忆可能提供有用证据。",
    "knowledge": "该请求的答案可能来自外部导入的剧情、角色、事件、文档、百科、资料库或专业语料，因此知识库可能提供有用证据。",
    "none": "该请求无需过去对话、共享记忆或外部资料库，直接依据当前消息即可妥善回答。",
    "creative": "该请求要求创造当前的新任务、新消息、新日程或角色扮演事件，同时可能需要已有用户连续性和世界观作为约束。",
}

CASES: tuple[dict[str, Any], ...] = (
    {"name": "mixed_character_encounter", "text": "你还记得为师当时怎么和某位角色相遇的吗？", "required": ["private", "public", "knowledge"], "forbidden": ["none"]},
    {"name": "mixed_character_profile", "text": "那你记得某位角色的哪些事情呢？", "required": ["private", "public", "knowledge"], "forbidden": ["none"]},
    {"name": "private_password", "text": "你还记得我们的暗号吗？", "required": ["private"], "forbidden": ["knowledge", "none"]},
    {"name": "private_preference", "text": "我以前说过自己喜欢什么饮料？", "required": ["private"], "forbidden": ["knowledge", "none"]},
    {"name": "direct_lore_cause", "text": "补课部真正是为什么成立的？", "required": ["knowledge"], "forbidden": ["private", "none"]},
    {"name": "creative_messages", "text": "有哪些学生刚刚给我发消息了？", "required": ["private", "knowledge", "creative"], "forbidden": ["none"]},
    {"name": "public_convention", "text": "所有用户都通用的安全约定是什么？", "required": ["public"], "forbidden": ["private", "knowledge", "none"]},
    {"name": "casual_no_memory", "text": "早上好。", "required": ["none"], "forbidden": ["private", "knowledge"]},
    {"name": "explicit_deep", "text": "请综合全部资料完整推导这场政治危机的多阶段因果链。", "required": ["knowledge"], "forbidden": ["private", "none"]},
    {"name": "english_mixed_lore", "text": "Do you remember what kind of person that character is?", "required": ["private", "public", "knowledge"], "forbidden": ["none"]},
    {"name": "japanese_lore_relation", "text": "あの二人が初めて出会った経緯を覚えていますか？", "required": ["private", "public", "knowledge"], "forbidden": ["none"]},
    {"name": "elliptical_with_history", "text": "上文：我们刚刚在讨论某位高层暗中帮助外部势力进入学园。\n当前问题：那她为什么这么做？", "required": ["knowledge"], "forbidden": ["none"]},
)


async def run(output: Path, concurrency: int) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
    await memory.initialize()
    model = memory.knowledge._query_engine.model
    labels = list(LABEL_DOCUMENTS)
    documents = [LABEL_DOCUMENTS[label] for label in labels]
    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            started = perf_counter()
            ranked = await asyncio.to_thread(
                model.rerank, case["text"], documents, top_n=len(documents)
            )
            elapsed = perf_counter() - started
        scores = {label: 0.0 for label in labels}
        for item in ranked:
            scores[labels[int(item["index"])]] = float(item["relevance_score"])
        order = sorted(scores, key=scores.get, reverse=True)
        return {
            **case,
            "seconds": round(elapsed, 6),
            "scores": {key: round(value, 8) for key, value in scores.items()},
            "ranked": order,
            "required_ranks": {
                label: order.index(label) + 1 for label in case.get("required", [])
            },
        }

    rows = await asyncio.gather(*(evaluate(case) for case in CASES))
    payload = {
        "experiment": "reranker-memory-router-v1",
        "label_documents": LABEL_DOCUMENTS,
        "summary": {
            "cases": len(rows),
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
    args = parser.parse_args()
    payload = asyncio.run(run(args.output, max(1, args.concurrency)))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    for row in payload["cases"]:
        print(row["name"], row["scores"], row["required_ranks"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
