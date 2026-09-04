from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from dotenv import load_dotenv

from src.memory import MemorySystem, MemorySystemConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES: tuple[dict[str, Any], ...] = (
    {"name": "shiroko_encounter", "text": "你还记得为师当时怎么和白子相遇的吗？", "knowledge_expected": True},
    {"name": "shiroko_profile", "text": "那你记得白子的哪些事情呢？", "knowledge_expected": True},
    {"name": "private_password", "text": "你还记得我们的暗号吗？", "knowledge_expected": False},
    {"name": "private_preference", "text": "我以前说过自己喜欢什么饮料？", "knowledge_expected": False},
    {"name": "makeup_club_cause", "text": "补课部真正是为什么成立的？", "knowledge_expected": True},
    {"name": "creative_messages", "text": "有哪些学生刚刚给为师发消息了？", "knowledge_expected": True},
    {"name": "public_convention", "text": "所有用户都通用的安全约定是什么？", "knowledge_expected": False},
    {"name": "casual_greeting", "text": "早上好。", "knowledge_expected": False},
    {"name": "deep_lore", "text": "请综合全部资料完整推导这场政治危机的多阶段因果链。", "knowledge_expected": True},
    {"name": "hoshino_english", "text": "Do you remember what kind of person Hoshino is?", "knowledge_expected": True},
    {"name": "mika_nagisa_japanese", "text": "ミカとナギサが初めて出会った経緯を覚えていますか？", "knowledge_expected": True},
    {"name": "elliptical_history", "text": "上文：我们刚刚在讨论未花暗中帮助阿里乌斯进入圣三一。\n当前问题：那她为什么这么做？", "knowledge_expected": True},
)


async def run(output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
    await memory.initialize()
    engine = memory.knowledge._query_engine
    texts = [str(item["text"]) for item in CASES]
    started = perf_counter()
    matrix = await asyncio.to_thread(engine.model.embed, texts)
    embed_seconds = perf_counter() - started
    rows: list[dict[str, Any]] = []
    for case, vector in zip(CASES, np.asarray(matrix, dtype=np.float32), strict=True):
        episode_hits = engine.episode_index.search(vector, 5)
        concept_hits = engine.concept_index.search(vector, 5)
        rows.append(
            {
                **case,
                "episode_top_score": max(
                    (float(item[1]) for item in episode_hits), default=0.0
                ),
                "concept_top_score": max(
                    (float(item[1]) for item in concept_hits), default=0.0
                ),
                "episode_ids": [int(item[0]) for item in episode_hits],
                "concept_ids": [int(item[0]) for item in concept_hits],
            }
        )
    payload = {
        "experiment": "semantic-evidence-probe-v1",
        "embedding_batch_seconds": round(embed_seconds, 6),
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = asyncio.run(run(args.output))
    print("embedding_batch_seconds", payload["embedding_batch_seconds"])
    for row in payload["cases"]:
        print(
            row["name"],
            row["knowledge_expected"],
            round(row["episode_top_score"], 6),
            round(row["concept_top_score"], 6),
        )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
