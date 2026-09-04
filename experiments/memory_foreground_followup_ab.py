from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from dotenv import load_dotenv

from experiments.memory_foreground_benchmark import _load_manifest, _score_groups
from src.memory.service import (
    AssociativeMemoryService,
    MemorySystemConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_frozen_plans(report_path: Path, query_log_path: Path) -> dict[str, dict]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in query_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = [
        event["result"]
        for event in events
        if event.get("event") == "retrieval_completed"
    ]
    rows = report.get("rows") or []
    if len(rows) != len(results):
        raise ValueError("report rows and retrieval_completed events do not align")
    plans = {}
    for row, result in zip(rows, results):
        if row["question"] != result["question"]:
            raise ValueError(f"question mismatch for {row['id']}")
        plans[str(row["id"])] = {
            "intent": result["intent"],
            "followups": list(result.get("followup_search_queries") or []),
            "baseline_followup_planning_seconds": float(
                (
                    (row.get("timings") or {}).get("phases_seconds") or {}
                ).get("followup_planning", 0.0)
            ),
        }
    return plans


def _summary(rows: list[dict], group_field: str, passed_field: str) -> dict:
    slots = sum(len(row[group_field]) for row in rows)
    matches = sum(
        int(group["passed"])
        for row in rows
        for group in row[group_field]
    )
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    return {
        "runs": len(rows),
        "complete_questions": sum(int(row[passed_field]) for row in rows),
        "matched_fact_slots": matches,
        "required_fact_slots": slots,
        "fact_slot_recall": matches / slots if slots else 0.0,
        "mean_elapsed_seconds": mean(elapsed) if elapsed else 0.0,
    }


async def run(
    manifest_path: Path,
    report_path: Path,
    query_log_path: Path,
    output_root: Path,
    question_ids: set[str],
    rerank_enabled: bool,
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    manifest = _load_manifest(manifest_path, base.engine_root)
    questions = {str(item["id"]): item for item in manifest["questions"]}
    plans = _load_frozen_plans(report_path, query_log_path)
    selected = [
        question_id
        for question_id in questions
        if question_id in plans
        and plans[question_id]["followups"]
        and (not question_ids or question_id in question_ids)
    ]
    if question_ids:
        missing = question_ids.difference(selected)
        if missing:
            raise ValueError(
                "selected ids must exist and have non-empty baseline followups: "
                f"{sorted(missing)}"
            )

    service_config = replace(
        base.domain_config(
            "knowledge",
            base.knowledge_database_path,
        ),
        rerank_enabled=rerank_enabled,
        growth_enabled=False,
        recall_cache_size=0,
        association_cue_enabled=False,
    )
    service = AssociativeMemoryService(service_config)
    await service.initialize()

    rows = []
    for pair, question_id in enumerate(selected, start=1):
        item = questions[question_id]
        plan = plans[question_id]
        order = (
            ("with_followup", "without_followup")
            if pair % 2
            else ("without_followup", "with_followup")
        )
        for position, variant in enumerate(order, start=1):
            followups = plan["followups"] if variant == "with_followup" else []
            started = perf_counter()
            recalled = await service.recall(
                item["question"],
                intent_override=plan["intent"],
                followup_queries_override=followups,
            )
            elapsed = perf_counter() - started
            raw = recalled.raw_result
            candidate_ids = {
                int(value) for value in raw.get("candidate_episode_ids") or []
            }
            final_ids = {int(value) for value in raw.get("episode_ids") or []}
            candidate_groups = _score_groups(item, candidate_ids)
            final_groups = _score_groups(item, final_ids)
            rows.append(
                {
                    "pair": pair,
                    "position": position,
                    "variant": variant,
                    "id": question_id,
                    "question": item["question"],
                    "followup_query_count": len(followups),
                    "baseline_followup_planning_seconds": plan[
                        "baseline_followup_planning_seconds"
                    ],
                    "elapsed_seconds": round(elapsed, 6),
                    "error": recalled.error,
                    "candidate_episode_ids": sorted(candidate_ids),
                    "episode_ids": sorted(final_ids),
                    "candidate_evidence_groups": candidate_groups,
                    "final_evidence_groups": final_groups,
                    "candidate_passed": not recalled.error
                    and all(group["passed"] for group in candidate_groups),
                    "final_passed": not recalled.error
                    and all(group["passed"] for group in final_groups),
                }
            )

    pairs = {}
    for row in rows:
        pairs.setdefault(row["id"], {})[row["variant"]] = row
    comparisons = []
    for question_id in selected:
        pair = pairs[question_id]
        with_followup = pair["with_followup"]
        without_followup = pair["without_followup"]
        comparisons.append(
            {
                "id": question_id,
                "candidate_set_equal": set(
                    with_followup["candidate_episode_ids"]
                )
                == set(without_followup["candidate_episode_ids"]),
                "with_followup_candidate_passed": with_followup[
                    "candidate_passed"
                ],
                "without_followup_candidate_passed": without_followup[
                    "candidate_passed"
                ],
                "with_followup_final_passed": with_followup["final_passed"],
                "without_followup_final_passed": without_followup[
                    "final_passed"
                ],
                "with_followup_candidate_fact_slots": sum(
                    int(group["passed"])
                    for group in with_followup["candidate_evidence_groups"]
                ),
                "without_followup_candidate_fact_slots": sum(
                    int(group["passed"])
                    for group in without_followup["candidate_evidence_groups"]
                ),
                "with_followup_final_fact_slots": sum(
                    int(group["passed"])
                    for group in with_followup["final_evidence_groups"]
                ),
                "without_followup_final_fact_slots": sum(
                    int(group["passed"])
                    for group in without_followup["final_evidence_groups"]
                ),
                "baseline_followup_planning_seconds": with_followup[
                    "baseline_followup_planning_seconds"
                ],
            }
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    report = {
        "version": "foreground-followup-candidate-ab-v1",
        "run_id": run_id,
        "manifest": str(manifest_path.resolve()),
        "frozen_report": str(report_path.resolve()),
        "frozen_query_log": str(query_log_path.resolve()),
        "method": (
            "Frozen intent; growth disabled; compare the observed followup "
            "queries versus an empty followup list."
        ),
        "rerank_enabled": rerank_enabled,
        "variants": {
            variant: {
                "candidate": _summary(
                    [row for row in rows if row["variant"] == variant],
                    "candidate_evidence_groups",
                    "candidate_passed",
                ),
                "final": _summary(
                    [row for row in rows if row["variant"] == variant],
                    "final_evidence_groups",
                    "final_passed",
                ),
            }
            for variant in ("with_followup", "without_followup")
        },
        "planner_returned_empty_questions": [
            question_id
            for question_id, plan in plans.items()
            if not plan["followups"]
        ],
        "comparisons": comparisons,
        "rows": rows,
    }
    target = output_root / run_id / "report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["report_path"] = str(target.resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Candidate@100 A/B for entity-resolved followup queries."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-report", type=Path, required=True)
    parser.add_argument("--frozen-query-log", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "foreground-followup-ab",
    )
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable the final LLM evidence reranker in addition to Candidate@100.",
    )
    args = parser.parse_args()
    report = asyncio.run(
        run(
            args.manifest,
            args.frozen_report,
            args.frozen_query_log,
            args.output_root,
            set(args.question_id),
            args.rerank,
        )
    )
    print(
        json.dumps(
            {
                "version": report["version"],
                "run_id": report["run_id"],
                "variants": report["variants"],
                "planner_returned_empty_questions": report[
                    "planner_returned_empty_questions"
                ],
                "comparisons": report["comparisons"],
                "report_path": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
