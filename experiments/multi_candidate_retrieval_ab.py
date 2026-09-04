from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.memory import AssociativeMemoryService, MemorySystemConfig


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(
        sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    ) as source_connection:
        with closing(sqlite3.connect(target)) as target_connection:
            source_connection.backup(target_connection)


def _database_signature(path: Path) -> dict[str, Any]:
    with closing(
        sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    ) as connection:
        count, maximum, total_use_count, latest_use = connection.execute(
            "SELECT COUNT(*), MAX(id), "
            "COALESCE(SUM(use_count), 0), MAX(last_used) FROM association"
        ).fetchone()
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "association_count": int(count),
        "association_max_id": int(maximum or 0),
        "association_total_use_count": int(total_use_count or 0),
        "association_latest_use": latest_use,
    }


def _apply_association_rows(database: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with closing(sqlite3.connect(database)) as connection:
        table_columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(association)")
        }
        for row in rows:
            columns = [name for name in row if name in table_columns]
            placeholders = ",".join("?" for _ in columns)
            connection.execute(
                f"INSERT OR REPLACE INTO association "
                f"({','.join(columns)}) VALUES ({placeholders})",
                [row[name] for name in columns],
            )
        connection.commit()


def _rank(ids: list[int], target: int) -> int | None:
    try:
        return ids.index(target) + 1
    except ValueError:
        return None


async def _recall_case(
    service: AssociativeMemoryService,
    question: str,
    edge_id: int,
    premise_ids: tuple[int, int],
) -> dict:
    started = perf_counter()
    result = await service.recall(
        question,
        retrieval_intensity="light",
        followup_queries_override=[],
    )
    elapsed = perf_counter() - started
    raw = result.raw_result or {}
    episode_ids = [int(value) for value in raw.get("episode_ids") or []]
    association_ids = [
        int(value) for value in raw.get("association_ids") or []
    ]
    edge_path = next(
        (
            row
            for row in raw.get("association_paths", [])
            if int(row.get("association_id", -1)) == edge_id
        ),
        None,
    )
    return {
        "elapsed_seconds": round(elapsed, 3),
        "error": result.error,
        "episode_ids": episode_ids,
        "association_ids": association_ids,
        "premise_ranks": {
            str(value): _rank(episode_ids, value) for value in premise_ids
        },
        "premise_recall_count": sum(
            value in episode_ids for value in premise_ids
        ),
        "candidate_edge_retrieved": edge_id in association_ids,
        "candidate_edge_path": edge_path,
    }


async def run(
    growth_report_path: Path,
    output_path: Path | None,
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    growth_report = json.loads(
        growth_report_path.read_text(encoding="utf-8")
    )
    fixture_path = Path(growth_report["fixture_path"])
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture_by_id = {
        row["candidate_id"]: row for row in fixture["candidates"]
    }
    targets = [
        row
        for row in growth_report["candidate_results"]
        if row.get("accepted")
        and "created" in row.get("actions", [])
        and row.get("association_ids")
    ]
    if not targets:
        raise ValueError("growth report contains no newly created candidate edge")

    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    formal_database = base.knowledge_database_path.resolve()
    formal_before = _database_signature(formal_database)
    baseline_database = growth_report_path.parent / "retrieval-baseline.db"
    treatment_database = growth_report_path.parent / "retrieval-treatment.db"
    _sqlite_backup(formal_database, baseline_database)
    _sqlite_backup(formal_database, treatment_database)
    target_edge_ids = {
        int(row["association_ids"][0]) for row in targets
    }
    _apply_association_rows(
        treatment_database,
        [
            row
            for row in growth_report.get("changed_associations", [])
            if int(row["id"]) in target_edge_ids
        ],
    )
    baseline_config = replace(
        base,
        knowledge_database_path=baseline_database,
        log_dir=growth_report_path.parent / "retrieval-ab-baseline-logs",
    )
    baseline = AssociativeMemoryService(
        baseline_config.domain_config("knowledge", baseline_database)
    )
    treatment_config = replace(
        base,
        knowledge_database_path=treatment_database,
        log_dir=growth_report_path.parent / "retrieval-ab-logs",
    )
    treatment = AssociativeMemoryService(
        treatment_config.domain_config("knowledge", treatment_database)
    )
    await asyncio.gather(baseline.initialize(), treatment.initialize())

    candidate_reports: list[dict] = []
    all_cases: list[dict] = []
    for target in targets:
        fixture_row = fixture_by_id[target["candidate_id"]]
        edge_id = int(target["association_ids"][0])
        premise_ids = tuple(
            int(value) for value in fixture_row["premise_episode_ids"]
        )
        cases: list[dict] = []
        for question in fixture_row["questions"]:
            baseline_result, treatment_result = await asyncio.gather(
                _recall_case(
                    baseline,
                    question,
                    edge_id,
                    premise_ids,
                ),
                _recall_case(
                    treatment,
                    question,
                    edge_id,
                    premise_ids,
                ),
            )
            case = {
                "candidate_id": target["candidate_id"],
                "question": question,
                "baseline": baseline_result,
                "treatment": treatment_result,
                "premise_recall_gain": (
                    treatment_result["premise_recall_count"]
                    - baseline_result["premise_recall_count"]
                ),
                "edge_path_gain": (
                    treatment_result["candidate_edge_retrieved"]
                    and not baseline_result["candidate_edge_retrieved"]
                ),
            }
            cases.append(case)
            all_cases.append(case)
        candidate_reports.append(
            {
                "candidate_id": target["candidate_id"],
                "label": target["label"],
                "inference_type": target["inference_type"],
                "association_id": edge_id,
                "premise_episode_ids": list(premise_ids),
                "question_count": len(cases),
                "edge_path_benefit_questions": sum(
                    bool(row["edge_path_gain"]) for row in cases
                ),
                "recall_coverage_gain_questions": sum(
                    row["premise_recall_gain"] > 0 for row in cases
                ),
                "total_premise_recall_gain": sum(
                    row["premise_recall_gain"] for row in cases
                ),
                "premise_recall_regression_questions": sum(
                    row["premise_recall_gain"] < 0 for row in cases
                ),
                "cases": cases,
            }
        )

    baseline_seconds = sum(
        row["baseline"]["elapsed_seconds"] for row in all_cases
    )
    treatment_seconds = sum(
        row["treatment"]["elapsed_seconds"] for row in all_cases
    )
    candidates_with_path = sum(
        row["edge_path_benefit_questions"] > 0 for row in candidate_reports
    )
    candidates_with_recall_gain = sum(
        row["recall_coverage_gain_questions"] > 0
        for row in candidate_reports
    )
    metrics = {
        "candidate_count": len(candidate_reports),
        "query_count": len(all_cases),
        "candidates_with_edge_path_benefit": candidates_with_path,
        "candidates_with_recall_coverage_gain": candidates_with_recall_gain,
        "edge_path_benefit_queries": sum(
            bool(row["edge_path_gain"]) for row in all_cases
        ),
        "recall_coverage_gain_queries": sum(
            row["premise_recall_gain"] > 0 for row in all_cases
        ),
        "premise_recall_regression_queries": sum(
            row["premise_recall_gain"] < 0 for row in all_cases
        ),
        "total_premise_recall_gain": sum(
            row["premise_recall_gain"] for row in all_cases
        ),
        "mean_baseline_seconds": round(
            baseline_seconds / len(all_cases), 3
        ),
        "mean_treatment_seconds": round(
            treatment_seconds / len(all_cases), 3
        ),
        "mean_latency_delta_seconds": round(
            (treatment_seconds - baseline_seconds) / len(all_cases), 3
        ),
    }
    formal_after = _database_signature(formal_database)
    checks = {
        "at_least_three_new_candidates_evaluated": len(candidate_reports) >= 3,
        "all_queries_completed": all(
            not row[side]["error"]
            for row in all_cases
            for side in ("baseline", "treatment")
        ),
        "no_target_premise_recall_regression": metrics[
            "premise_recall_regression_queries"
        ] == 0,
        "majority_of_candidates_have_edge_path_benefit": (
            candidates_with_path / len(candidate_reports) >= 0.5
        ),
        "at_least_two_candidates_gain_recall_coverage": (
            candidates_with_recall_gain >= 2
        ),
        "total_target_premise_recall_improves": (
            metrics["total_premise_recall_gain"] > 0
        ),
        "formal_database_unchanged": formal_before == formal_after,
    }
    report = {
        "version": "multi-candidate-retrieval-ab-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "growth_report": str(growth_report_path),
        "fixture_path": str(fixture_path),
        "fixture_sha256": growth_report["fixture_sha256"],
        "formal_database": str(formal_database),
        "formal_database_before": formal_before,
        "formal_database_after": formal_after,
        "baseline_database": str(baseline_database),
        "treatment_database": str(treatment_database),
        "audited_database": str(
            Path(growth_report["knowledge_database"]).resolve()
        ),
        "candidate_reports": candidate_reports,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }
    destination = output_path or growth_report_path.parent / "retrieval-ab.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(destination)
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Evaluate multiple frozen candidate edges against baseline"
    )
    parser.add_argument("growth_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = asyncio.run(
        run(
            args.growth_report.resolve(),
            args.output.resolve() if args.output else None,
        )
    )
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "passed": report["passed"],
                "metrics": report["metrics"],
                "checks": report["checks"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
