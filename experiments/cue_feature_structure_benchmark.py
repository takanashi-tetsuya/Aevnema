from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import re
import unicodedata


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RELATIONS = (
    (
        "makeup_screening",
        "一段证据记录渚以成绩和退学为补课部的表面安排并寻找潜在内鬼，另一段记录梓来自阿里乌斯并承担破坏任务；只用于共同检索两项观察，不断言梓必然就是渚寻找的内鬼。",
    ),
    (
        "treaty_attack_bridge",
        "一段证据记录未花与阿里乌斯学园的合作，另一段记录阿里乌斯学园袭击伊甸园条约现场；只用于共同检索两项事实，不断言合作导致或直接指挥了袭击。",
    ),
    (
        "hoshino_trauma_bridge",
        "一段证据记录星野后来试图独自牺牲保护后辈，另一段记录梦前辈死亡后星野的创伤和自责；只用于共同检索两项观察，不断言创伤是该行为的唯一原因。",
    ),
)

POSITIVES = (
    ("makeup_screening", "补课部为何不只是差生社团，渚的内鬼排查与梓的阿里乌斯任务如何一起理解？"),
    ("makeup_screening", "渚成立补课部的政治目的是什么，梓的真实来历为什么值得同时查看？"),
    ("makeup_screening", "把补课部表面上的退学安排和梓承担的破坏任务放在一起说明。"),
    ("treaty_attack_bridge", "是谁袭击伊甸园条约现场，茶会内部哪项合作为危机提供了背景？"),
    ("treaty_attack_bridge", "把未花与阿里乌斯的合作以及条约签署现场遭袭这两项事实一起找出来。"),
    ("treaty_attack_bridge", "条约危机中阿里乌斯做了什么，未花此前又和哪个势力合作？"),
    ("hoshino_trauma_bridge", "星野为何总想独自牺牲保护后辈，这与梦前辈死后的创伤需要怎样结合理解？"),
    ("hoshino_trauma_bridge", "同时找出星野一人承担危机的行为和她对梦前辈之死的自责。"),
    ("hoshino_trauma_bridge", "梦的死亡创伤与星野保护后辈时的自我牺牲倾向有哪些可见证据？"),
)

HARD_NEGATIVES = (
    "补课部成员喜欢什么食物？",
    "渚平时喝哪一种茶？",
    "梓使用的武器是什么？",
    "阿里乌斯学园有哪些学生？",
    "伊甸园条约在哪里签署？",
    "未花最喜欢什么甜点？",
    "茶会成员的任期有多久？",
    "星野使用的盾牌叫什么？",
    "梦前辈的生日和身高是多少？",
    "阿拜多斯有哪些日常委托？",
    "老师第一次在哪里遇到白子？",
    "日奈为什么前往沙漠？",
    "乐园悖论是谁提出的？",
    "山海经为客人准备了哪些月饼？",
)

_CJK_RUN_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_WORD_RE = re.compile(r"[a-z0-9_]{2,}")


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").casefold()


def _features(value: str, variant: str) -> set[str]:
    normalized = _normalize(value)
    result = {f"w:{word}" for word in _WORD_RE.findall(normalized)}
    for run in _CJK_RUN_RE.findall(normalized):
        if variant in {"character", "hybrid"}:
            result.update(f"c:{char}" for char in run)
        sizes = {
            "bigram": (2,),
            "bigram_trigram": (2, 3),
            "hybrid": (2,),
            "character": (),
        }[variant]
        for size in sizes:
            result.update(
                f"n{size}:{run[index:index + size]}"
                for index in range(max(0, len(run) - size + 1))
            )
    return result


def _rank(variant: str) -> tuple[list[dict], list[dict]]:
    names = [name for name, _text in RELATIONS]
    documents = [_features(text, variant) for _name, text in RELATIONS]
    frequencies: Counter[str] = Counter()
    for features in documents:
        frequencies.update(features)
    count = len(documents)
    idf = {
        feature: math.log((count + 1.0) / (frequency + 0.5))
        for feature, frequency in frequencies.items()
    }

    def scores(query: str) -> list[float]:
        query_features = _features(query, variant)
        values: list[float] = []
        for document in documents:
            denominator = sum(idf.get(feature, 0.0) for feature in document)
            overlap = sum(
                idf.get(feature, 0.0)
                for feature in document.intersection(query_features)
            )
            values.append(overlap / denominator if denominator else 0.0)
        return values

    positives: list[dict] = []
    for expected, query in POSITIVES:
        values = scores(query)
        order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
        positives.append(
            {
                "query": query,
                "expected": expected,
                "best": names[order[0]],
                "score": values[order[0]],
                "expected_score": values[names.index(expected)],
                "margin": values[order[0]] - values[order[1]],
            }
        )
    negatives: list[dict] = []
    for query in HARD_NEGATIVES:
        values = scores(query)
        order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
        negatives.append(
            {
                "query": query,
                "best": names[order[0]],
                "score": values[order[0]],
                "margin": values[order[0]] - values[order[1]],
            }
        )
    return positives, negatives


def main() -> int:
    variants: dict[str, dict] = {}
    for variant in ("character", "bigram", "bigram_trigram", "hybrid"):
        positives, negatives = _rank(variant)
        thresholds: list[dict] = []
        for coverage in (0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26):
            for margin in (0.0, 0.02, 0.04, 0.06):
                correct = sum(
                    row["best"] == row["expected"]
                    and row["score"] >= coverage
                    and row["margin"] >= margin
                    for row in positives
                )
                false_activations = sum(
                    row["score"] >= coverage and row["margin"] >= margin
                    for row in negatives
                )
                thresholds.append(
                    {
                        "coverage": coverage,
                        "margin": margin,
                        "positive_recall": correct / len(positives),
                        "hard_negative_activation": false_activations / len(negatives),
                    }
                )
        zero_false_positive = [
            row for row in thresholds if row["hard_negative_activation"] == 0.0
        ]
        best = max(
            zero_false_positive or thresholds,
            key=lambda row: (
                row["positive_recall"],
                -row["hard_negative_activation"],
                -row["coverage"],
                -row["margin"],
            ),
        )
        current = next(
            row
            for row in thresholds
            if row["coverage"] == 0.18 and row["margin"] == 0.04
        )
        variants[variant] = {
            "best_zero_false_positive": best,
            "current_threshold_result": current,
            "positives": positives,
            "hard_negatives": negatives,
        }
    report = {
        "version": "cue-feature-structure-benchmark-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "positive_count": len(POSITIVES),
        "hard_negative_count": len(HARD_NEGATIVES),
        "variants": variants,
    }
    output = (
        PROJECT_ROOT
        / "logs"
        / "experiments"
        / f"cue-feature-structure-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": {
                    name: {
                        "best_zero_false_positive": row["best_zero_false_positive"],
                        "current_threshold_result": row["current_threshold_result"],
                    }
                    for name, row in variants.items()
                },
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
