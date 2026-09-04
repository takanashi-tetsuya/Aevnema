from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.association_capsule_benchmark import FIXTURES
from src.memory import AssociativeMemoryService, MemorySystemConfig


HARD_NEGATIVES = (
    "渚平时喜欢喝什么茶、吃什么点心？",
    "梓使用什么武器，她的战斗方式有什么特点？",
    "未花最喜欢的甜点和课外活动是什么？",
    "星野使用的盾牌叫什么，她擅长什么战术？",
    "梦前辈的生日和身高是多少？",
    "阿鲁误点情侣专用菜单后发生了什么？",
    "日奈为什么要阻止星野进入沙漠？",
    "圣亚提出的乐园悖论是什么意思？",
    "白子被前辈救助的回忆为何突然出现？",
    "山海经的月饼宴会与圣三一会议有什么差异？",
)


async def run(output: Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    system = MemorySystemConfig.from_env(PROJECT_ROOT)
    config = replace(
        system.domain_config("knowledge", system.knowledge_database_path),
        growth_enabled=False,
        association_cue_enabled=False,
        association_cue_fast_path_enabled=False,
    )
    service = AssociativeMemoryService(config)
    await service.initialize()
    relation_texts = [str(item["relation_text"]) for item in FIXTURES]
    positives = [
        (index, question)
        for index, fixture in enumerate(FIXTURES)
        for question in fixture["questions"]
    ]
    all_texts = [
        *relation_texts,
        *(question for _, question in positives),
        *HARD_NEGATIVES,
    ]
    matrix = np.asarray(
        await asyncio.to_thread(service._query_engine.model.embed, all_texts),
        dtype=np.float32,
    )
    matrix /= np.maximum(
        np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12
    )
    relation_matrix = matrix[: len(relation_texts)]
    positive_start = len(relation_texts)
    positive_rows = []
    for offset, (relation_index, question) in enumerate(positives):
        scores = relation_matrix @ matrix[positive_start + offset]
        positive_rows.append(
            {
                "question": question,
                "expected_relation": FIXTURES[relation_index]["name"],
                "expected_score": float(scores[relation_index]),
                "best_relation": FIXTURES[int(np.argmax(scores))]["name"],
                "best_score": float(np.max(scores)),
                "scores": [float(value) for value in scores],
            }
        )
    negative_start = positive_start + len(positives)
    negative_rows = []
    for offset, question in enumerate(HARD_NEGATIVES):
        scores = relation_matrix @ matrix[negative_start + offset]
        negative_rows.append(
            {
                "question": question,
                "best_relation": FIXTURES[int(np.argmax(scores))]["name"],
                "best_score": float(np.max(scores)),
                "scores": [float(value) for value in scores],
            }
        )
    semantic_plan_pairs = []
    for relation_index, fixture in enumerate(FIXTURES):
        first_positive_index = relation_index * 2
        left = matrix[positive_start + first_positive_index]
        right = matrix[positive_start + first_positive_index + 1]
        semantic_plan_pairs.append(
            {
                "fixture": fixture["name"],
                "cosine": float(left @ right),
                "should_reuse": True,
            }
        )
    positive_scores = [row["expected_score"] for row in positive_rows]
    negative_scores = [row["best_score"] for row in negative_rows]
    thresholds = []
    for threshold in (0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82):
        thresholds.append(
            {
                "threshold": threshold,
                "positive_recall": sum(
                    value >= threshold for value in positive_scores
                )
                / len(positive_scores),
                "hard_negative_activation_rate": sum(
                    value >= threshold for value in negative_scores
                )
                / len(negative_scores),
            }
        )
    report = {
        "version": "association-cue-similarity-calibration-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "positive_rows": positive_rows,
        "hard_negative_rows": negative_rows,
        "semantic_plan_pairs": semantic_plan_pairs,
        "thresholds": thresholds,
        "positive_min": min(positive_scores),
        "positive_mean": sum(positive_scores) / len(positive_scores),
        "hard_negative_max": max(negative_scores),
        "hard_negative_mean": sum(negative_scores) / len(negative_scores),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    output = (
        PROJECT_ROOT
        / "logs"
        / "experiments"
        / f"association-cue-calibration-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    report = asyncio.run(run(output))
    print(
        json.dumps(
            {
                "positive_min": report["positive_min"],
                "positive_mean": report["positive_mean"],
                "hard_negative_max": report["hard_negative_max"],
                "hard_negative_mean": report["hard_negative_mean"],
                "thresholds": report["thresholds"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
