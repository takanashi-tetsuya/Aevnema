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
from experiments.memory_foreground_followup_ab import _load_frozen_plans
from src.memory.service import AssociativeMemoryService, MemorySystemConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _layer_summary(rows: list[dict], groups_field: str) -> dict:
    group_count = sum(len(row[groups_field]) for row in rows)
    matched = sum(
        int(group["passed"])
        for row in rows
        for group in row[groups_field]
    )
    return {
        "complete_questions": sum(
            all(group["passed"] for group in row[groups_field])
            for row in rows
        ),
        "matched_fact_slots": matched,
        "required_fact_slots": group_count,
        "fact_slot_recall": matched / group_count if group_count else 0.0,
    }


def _budget_summary(rows: list[dict]) -> dict:
    rerank_seconds = [float(row["evidence_rerank_seconds"]) for row in rows]
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    return {
        "runs": len(rows),
        "candidate_pool": _layer_summary(rows, "candidate_pool_groups"),
        "rerank_input": _layer_summary(rows, "rerank_input_groups"),
        "final": _layer_summary(rows, "final_groups"),
        "mean_elapsed_seconds": mean(elapsed) if elapsed else 0.0,
        "mean_evidence_rerank_seconds": (
            mean(rerank_seconds) if rerank_seconds else 0.0
        ),
        "mean_rerank_input_count": mean(
            len(row["rerank_input_episode_ids"]) for row in rows
        ) if rows else 0.0,
        "mean_rerank_candidate_text_chars": mean(
            int(row["rerank_candidate_text_chars"]) for row in rows
        ) if rows else 0.0,
        "mean_initial_prompt_chars": mean(
            int(row["initial_prompt_chars"]) for row in rows
        ) if rows else 0.0,
    }


async def run(
    manifest_path: Path,
    frozen_report: Path,
    frozen_query_log: Path,
    output_root: Path,
    budgets: list[int],
    question_ids: set[str],
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    manifest = _load_manifest(manifest_path, base.engine_root)
    questions = {str(item["id"]): item for item in manifest["questions"]}
    plans = _load_frozen_plans(frozen_report, frozen_query_log)
    selected = [
        question_id
        for question_id in questions
        if question_id in plans and (not question_ids or question_id in question_ids)
    ]
    if question_ids:
        missing = question_ids.difference(selected)
        if missing:
            raise ValueError(f"unknown or unfrozen question ids: {sorted(missing)}")

    services: dict[int, AssociativeMemoryService] = {}
    for budget in budgets:
        service_config = replace(
            base.domain_config("knowledge", base.knowledge_database_path),
            log_dir=(
                output_root / "query-logs" / f"budget-{budget}"
            ),
            rerank_precompression_limit=budget,
            growth_enabled=False,
            recall_cache_size=0,
            association_cue_enabled=False,
        )
        service = AssociativeMemoryService(service_config)
        await service.initialize()
        services[budget] = service

    rows = []
    for question_index, question_id in enumerate(selected):
        item = questions[question_id]
        plan = plans[question_id]
        ordered_budgets = [
            *budgets[question_index % len(budgets):],
            *budgets[: question_index % len(budgets)],
        ]
        for position, budget in enumerate(ordered_budgets, start=1):
            started = perf_counter()
            recalled = await services[budget].recall(
                item["question"],
                intent_override=plan["intent"],
                followup_queries_override=plan["followups"],
            )
            elapsed = perf_counter() - started
            raw = recalled.raw_result
            trace = raw.get("rerank_trace") or {}
            timings = raw.get("timings") or {}
            phases = timings.get("phases_seconds") or {}
            pool_ids = [int(value) for value in raw.get("candidate_episode_ids") or []]
            rerank_input_ids = [
                int(value)
                for value in trace.get("rerank_input_episode_ids") or pool_ids
            ]
            final_ids = [int(value) for value in raw.get("episode_ids") or []]
            rows.append(
                {
                    "id": question_id,
                    "question": item["question"],
                    "budget": budget,
                    "position": position,
                    "elapsed_seconds": round(elapsed, 6),
                    "evidence_rerank_seconds": float(
                        phases.get("evidence_rerank", 0.0)
                    ),
                    "error": recalled.error,
                    "candidate_pool_episode_ids": pool_ids,
                    "rerank_input_episode_ids": rerank_input_ids,
                    "final_episode_ids": final_ids,
                    "candidate_pool_groups": _score_groups(item, set(pool_ids)),
                    "rerank_input_groups": _score_groups(
                        item, set(rerank_input_ids)
                    ),
                    "final_groups": _score_groups(item, set(final_ids)),
                    "precompression": trace.get("precompression") or {},
                    "rerank_candidate_text_chars": int(
                        trace.get("rerank_candidate_text_chars") or 0
                    ),
                    "initial_prompt_chars": int(
                        trace.get("initial_prompt_chars") or 0
                    ),
                }
            )

    comparisons = []
    for question_id in selected:
        question_rows = {
            int(row["budget"]): row for row in rows if row["id"] == question_id
        }
        pool_sequences = {
            tuple(row["candidate_pool_episode_ids"])
            for row in question_rows.values()
        }
        pool_sets = {
            frozenset(row["candidate_pool_episode_ids"])
            for row in question_rows.values()
        }
        comparisons.append(
            {
                "id": question_id,
                "candidate_pool_sequence_equal": len(pool_sequences) == 1,
                "candidate_pool_set_equal": len(pool_sets) == 1,
                "budgets": {
                    str(budget): {
                        "rerank_input_fact_slots": sum(
                            int(group["passed"])
                            for group in question_rows[budget]["rerank_input_groups"]
                        ),
                        "final_fact_slots": sum(
                            int(group["passed"])
                            for group in question_rows[budget]["final_groups"]
                        ),
                        "required_fact_slots": len(
                            question_rows[budget]["final_groups"]
                        ),
                    }
                    for budget in budgets
                },
            }
        )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    report = {
        "version": "foreground-precompression-ab-v1",
        "run_id": run_id,
        "manifest": str(manifest_path.resolve()),
        "frozen_report": str(frozen_report.resolve()),
        "frozen_query_log": str(frozen_query_log.resolve()),
        "method": (
            "Freeze intent and follow-up queries; preserve Candidate@100; "
            "vary only the deterministic pre-LLM rerank input budget."
        ),
        "budgets": budgets,
        "question_ids": selected,
        "summaries": {
            str(budget): _budget_summary(
                [row for row in rows if int(row["budget"]) == budget]
            )
            for budget in budgets
        },
        "comparisons": comparisons,
        "rows": rows,
    }
    target = output_root / run_id / "report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_path"] = str(target.resolve())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Controlled Candidate@100 -> rerank-input budget A/B."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frozen-report", type=Path, required=True)
    parser.add_argument("--frozen-query-log", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "foreground-precompression-ab",
    )
    parser.add_argument("--budget", type=int, action="append", default=[])
    parser.add_argument("--question-id", action="append", default=[])
    args = parser.parse_args()
    budgets = list(dict.fromkeys(args.budget or [100, 50, 30]))
    if any(value < 1 for value in budgets):
        parser.error("all --budget values must be at least 1")
    report = asyncio.run(
        run(
            args.manifest,
            args.frozen_report,
            args.frozen_query_log,
            args.output_root,
            budgets,
            set(args.question_id),
        )
    )
    print(
        json.dumps(
            {
                "version": report["version"],
                "run_id": report["run_id"],
                "summaries": report["summaries"],
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
