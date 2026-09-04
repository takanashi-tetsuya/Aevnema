from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from src.memory import MemorySystem, MemorySystemConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOMAIN_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "private": (
        "检索当前用户过去亲口说过的内容、个人偏好、经历、约定、对话和只属于该用户的长期印象。",
        "检索当前用户与助手共同经历或共同创造的角色扮演事件，以及只属于该用户的连续性。",
        "Recall user-specific conversation history, preferences, promises, experiences, and private roleplay continuity.",
    ),
    "public": (
        "检索所有用户共同可用、与特定用户无关的共享长期记忆、公共约定和普适信息。",
        "Recall cross-user shared memory, common conventions, and durable information intentionally available to every user.",
    ),
    "knowledge": (
        "检索从剧情、小说、文档、百科、资料库或专业语料导入的角色、实体、事件、关系和背景知识。",
        "Retrieve facts, entities, events, relations, and timelines from imported stories, documents, encyclopedias, or domain corpora.",
        "物語、キャラクター、出来事、関係、資料や外部文書から取り込まれた知識を検索する。",
    ),
    "none": (
        "普通寒暄、即时闲聊或无需任何历史记录和外部资料就能回答的请求。",
        "A casual message that requires no stored conversation, shared memory, or imported knowledge.",
    ),
    "creative": (
        "根据已有角色、世界观和用户连续性创造当前的新任务、新消息、新日程或角色扮演事件。",
        "Invent a new current roleplay event while preserving user continuity and established fictional-world constraints.",
    ),
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


async def run(output: Path) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
    await memory.initialize()
    labels: list[str] = []
    prototype_texts: list[str] = []
    for label, values in DOMAIN_PROTOTYPES.items():
        for value in values:
            labels.append(label)
            prototype_texts.append(value)
    case_texts = [str(case["text"]) for case in CASES]
    matrix = await asyncio.to_thread(
        memory.knowledge._query_engine.model.embed,
        [*prototype_texts, *case_texts],
    )
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    prototypes = matrix[: len(prototype_texts)]
    queries = matrix[len(prototype_texts) :]
    rows: list[dict[str, Any]] = []
    for case, query in zip(CASES, queries, strict=True):
        similarities = prototypes @ query
        scores = {
            label: max(
                float(score)
                for score, candidate_label in zip(
                    similarities, labels, strict=True
                )
                if candidate_label == label
            )
            for label in DOMAIN_PROTOTYPES
        }
        ranked = sorted(scores, key=scores.get, reverse=True)
        rows.append(
            {
                **case,
                "scores": {key: round(value, 8) for key, value in scores.items()},
                "ranked": ranked,
                "required_ranks": {
                    label: ranked.index(label) + 1
                    for label in case.get("required", [])
                },
            }
        )
    payload = {
        "experiment": "semantic-memory-router-v1",
        "prototypes": DOMAIN_PROTOTYPES,
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
    for row in payload["cases"]:
        print(row["name"], row["scores"], row["required_ranks"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
