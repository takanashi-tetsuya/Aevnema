from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from dotenv import load_dotenv

from experiments.memory_foreground_benchmark import _score_groups
from src.memory import MemorySystem, MemorySystemConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    """Return an observed percentile using the nearest-rank definition."""
    if not values:
        return 0.0
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _distribution(values: list[float]) -> dict:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90_nearest_rank": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    normalized = [float(value) for value in values]
    return {
        "mean": round(mean(normalized), 6),
        "median": round(median(normalized), 6),
        "p90_nearest_rank": round(
            _nearest_rank_percentile(normalized, 0.9), 6
        ),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }


def _variant_summary(rows: list[dict]) -> dict:
    group_count = sum(len(row["evidence_groups"]) for row in rows)
    matched_group_count = sum(
        int(group["passed"])
        for row in rows
        for group in row["evidence_groups"]
    )
    phase_names = sorted(
        {
            str(name)
            for row in rows
            for name in (
                (row.get("timings") or {}).get("phases_seconds") or {}
            )
        }
    )
    return {
        "runs": len(rows),
        "passed_runs": sum(int(row["passed"]) for row in rows),
        "matched_fact_slots": matched_group_count,
        "required_fact_slots": group_count,
        "fact_slot_recall": (
            matched_group_count / group_count if group_count else 0.0
        ),
        "elapsed_seconds": _distribution(
            [float(row["elapsed_seconds"]) for row in rows]
        ),
        "pipeline_seconds": _distribution(
            [
                float((row.get("timings") or {}).get("total_seconds") or 0.0)
                for row in rows
            ]
        ),
        "phase_seconds": {
            name: _distribution(
                [
                    float(
                        (
                            (row.get("timings") or {}).get("phases_seconds")
                            or {}
                        ).get(name, 0.0)
                    )
                    for row in rows
                ]
            )
            for name in phase_names
        },
    }


def _paired_summary(rows: list[dict]) -> dict:
    pairs: dict[int, dict[str, dict]] = {}
    for row in rows:
        pairs.setdefault(int(row["pair"]), {})[str(row["variant"])] = row

    comparisons = []
    for pair_number in sorted(pairs):
        pair = pairs[pair_number]
        if "atomic40" not in pair or "atomic12" not in pair:
            continue
        large = pair["atomic40"]
        small = pair["atomic12"]
        large_elapsed = float(large["elapsed_seconds"])
        small_elapsed = float(small["elapsed_seconds"])
        large_rerank = float(
            ((large.get("timings") or {}).get("phases_seconds") or {}).get(
                "evidence_rerank", 0.0
            )
        )
        small_rerank = float(
            ((small.get("timings") or {}).get("phases_seconds") or {}).get(
                "evidence_rerank", 0.0
            )
        )
        comparisons.append(
            {
                "pair": pair_number,
                "order": [large["position"], small["position"]],
                "candidate_sequence_equal": (
                    large["candidate_episode_ids"]
                    == small["candidate_episode_ids"]
                ),
                "candidate_set_equal": (
                    set(large["candidate_episode_ids"])
                    == set(small["candidate_episode_ids"])
                ),
                "atomic40_passed": large["passed"],
                "atomic12_passed": small["passed"],
                "atomic40_fact_slots": large["matched_fact_slots"],
                "atomic12_fact_slots": small["matched_fact_slots"],
                "elapsed_seconds_saved_by_12": round(
                    large_elapsed - small_elapsed, 6
                ),
                "elapsed_relative_reduction": round(
                    (large_elapsed - small_elapsed) / large_elapsed
                    if large_elapsed
                    else 0.0,
                    6,
                ),
                "rerank_seconds_saved_by_12": round(
                    large_rerank - small_rerank, 6
                ),
            }
        )
    return {
        "complete_pairs": len(comparisons),
        "candidate_sequence_equal_pairs": sum(
            int(item["candidate_sequence_equal"]) for item in comparisons
        ),
        "candidate_set_equal_pairs": sum(
            int(item["candidate_set_equal"]) for item in comparisons
        ),
        "elapsed_seconds_saved_by_12": _distribution(
            [item["elapsed_seconds_saved_by_12"] for item in comparisons]
        ),
        "elapsed_relative_reduction": _distribution(
            [item["elapsed_relative_reduction"] for item in comparisons]
        ),
        "rerank_seconds_saved_by_12": _distribution(
            [item["rerank_seconds_saved_by_12"] for item in comparisons]
        ),
        "pairs": comparisons,
    }


def _write_report(report: dict, output_root: Path) -> dict:
    target = output_root / report["run_id"] / "report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(target.resolve())
    return report


async def _run_once(
    memory: MemorySystem,
    item: dict,
    *,
    pair_number: int,
    position: int,
    variant: str,
    atomic_limit: int,
) -> dict:
    started = perf_counter()
    recalled = await memory.knowledge.recall(
        item["question"],
        intent_override=item.get("intent_override"),
        followup_queries_override=item.get("followup_queries_override"),
    )
    elapsed = perf_counter() - started
    raw = recalled.raw_result
    final_ids = {int(value) for value in raw.get("episode_ids") or []}
    groups = _score_groups(item, final_ids)
    trace = raw.get("rerank_trace") or {}
    matched = sum(int(group["passed"]) for group in groups)
    return {
        "pair": pair_number,
        "position": position,
        "variant": variant,
        "atomic_query_limit": atomic_limit,
        "id": item["id"],
        "question": item["question"],
        "elapsed_seconds": round(elapsed, 6),
        "error": recalled.error,
        "candidate_episode_ids": [
            int(value) for value in raw.get("candidate_episode_ids") or []
        ],
        "episode_ids": sorted(final_ids),
        "expanded_atomic_query_count": int(
            trace.get("expanded_atomic_query_count") or 0
        ),
        "retained_atomic_query_count": len(trace.get("atomic_queries") or []),
        "atomic_queries_truncated": bool(
            trace.get("atomic_queries_truncated")
        ),
        "review_mode": trace.get("review_mode"),
        "review_level": trace.get("review_level"),
        "coverage_audit_performed": bool(
            trace.get("coverage_audit_performed")
        ),
        "compressor_performed": bool(trace.get("compressor_performed")),
        "timings": raw.get("timings") or {},
        "matched_fact_slots": matched,
        "required_fact_slots": len(groups),
        "evidence_groups": groups,
        "passed": not recalled.error and matched == len(groups),
    }


async def run(
    manifest_path: Path,
    output_root: Path,
    pairs: int,
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = manifest.get("questions") or []
    if len(questions) != 1:
        raise ValueError("paired fixed-plan benchmark requires exactly one question")
    item = questions[0]

    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    common = replace(base, foreground_recall_cache_size=0)
    configs = {
        "atomic40": replace(common, foreground_atomic_query_limit=40),
        "atomic12": replace(common, foreground_atomic_query_limit=12),
    }
    systems = {name: MemorySystem(config) for name, config in configs.items()}
    await asyncio.gather(*(system.initialize() for system in systems.values()))

    rows = []
    for pair_number in range(1, pairs + 1):
        order = (
            ["atomic40", "atomic12"]
            if pair_number % 2
            else ["atomic12", "atomic40"]
        )
        for position, variant in enumerate(order, start=1):
            rows.append(
                await _run_once(
                    systems[variant],
                    item,
                    pair_number=pair_number,
                    position=position,
                    variant=variant,
                    atomic_limit=configs[variant].foreground_atomic_query_limit,
                )
            )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    report = {
        "version": "foreground-atomic-ab-report-v1",
        "run_id": run_id,
        "manifest": str(manifest_path.resolve()),
        "manifest_version": manifest.get("version"),
        "pairs": pairs,
        "order_policy": "odd: atomic40->atomic12; even: atomic12->atomic40",
        "percentile_method": "nearest-rank",
        "variants": {
            name: _variant_summary(
                [row for row in rows if row["variant"] == name]
            )
            for name in configs
        },
        "paired": _paired_summary(rows),
        "rows": rows,
    }
    return _write_report(report, output_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run alternating fixed-plan atomic evidence budget A/B pairs"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "manifests"
            / "foreground_latency_fixed_plan_v1.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "foreground-atomic-ab",
    )
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    report = asyncio.run(run(args.manifest, args.output_root, args.pairs))
    compact = {
        "version": report["version"],
        "run_id": report["run_id"],
        "pairs": report["pairs"],
        "variants": report["variants"],
        "paired": report["paired"],
        "report_path": report["report_path"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
