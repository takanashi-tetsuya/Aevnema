from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from src.memory import AssociativeMemoryService, MemorySystemConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = (
    "未花把阿里乌斯视为合作对象后，为什么说这段合作实际上超出了她的控制？",
    "哪两段剧情共同表现出未花的合作设想与阿里乌斯小队实际目标之间的反差？",
    "未花收到圣娅死亡报告后的崩溃，与她此前主动支援阿里乌斯有什么关系？",
)


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
    episode_ids = list(raw.get("episode_ids") or [])
    association_ids = list(raw.get("association_ids") or [])
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
        "candidate_edge_path": next(
            (
                row
                for row in raw.get("association_paths", [])
                if int(row.get("association_id", -1)) == edge_id
            ),
            None,
        ),
    }


async def run(growth_report_path: Path, output_path: Path | None = None) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    growth_report = json.loads(growth_report_path.read_text(encoding="utf-8"))
    changed = growth_report.get("changed_associations") or []
    if len(changed) != 1:
        raise ValueError("growth report must contain exactly one changed edge")
    edge = changed[0]
    edge_id = int(edge["id"])
    premise_ids = (int(edge["from_id"]), int(edge["to_id"]))

    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    treatment_database = Path(growth_report["knowledge_database"]).resolve()
    baseline = AssociativeMemoryService(
        base.domain_config("knowledge", base.knowledge_database_path)
    )
    treatment_system_config = replace(
        base,
        knowledge_database_path=treatment_database,
        log_dir=growth_report_path.parent / "retrieval-ab-logs",
    )
    treatment = AssociativeMemoryService(
        treatment_system_config.domain_config(
            "knowledge", treatment_database
        )
    )
    await asyncio.gather(baseline.initialize(), treatment.initialize())

    cases = []
    for question in QUESTIONS:
        baseline_result, treatment_result = await asyncio.gather(
            _recall_case(baseline, question, edge_id, premise_ids),
            _recall_case(treatment, question, edge_id, premise_ids),
        )
        cases.append(
            {
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
        )

    checks = {
        "all_queries_completed": all(
            not row[side]["error"]
            for row in cases
            for side in ("baseline", "treatment")
        ),
        "candidate_edge_recovers_missing_premise": any(
            row["premise_recall_gain"] > 0 for row in cases
        ),
        "candidate_edge_used_by_at_least_one_query": any(
            row["edge_path_gain"] for row in cases
        ),
        "candidate_edge_used_by_all_queries": all(
            row["edge_path_gain"] for row in cases
        ),
        "candidate_edge_does_not_reduce_premise_recall": all(
            row["premise_recall_gain"] >= 0 for row in cases
        ),
    }
    baseline_seconds = sum(row["baseline"]["elapsed_seconds"] for row in cases)
    treatment_seconds = sum(row["treatment"]["elapsed_seconds"] for row in cases)
    metrics = {
        "query_count": len(cases),
        "semantic_path_benefit_queries": sum(
            bool(row["edge_path_gain"]) for row in cases
        ),
        "recall_coverage_gain_queries": sum(
            row["premise_recall_gain"] > 0 for row in cases
        ),
        "total_premise_recall_gain": sum(
            row["premise_recall_gain"] for row in cases
        ),
        "mean_baseline_seconds": round(baseline_seconds / len(cases), 3),
        "mean_treatment_seconds": round(treatment_seconds / len(cases), 3),
        "mean_latency_delta_seconds": round(
            (treatment_seconds - baseline_seconds) / len(cases), 3
        ),
    }
    report = {
        "version": "candidate-edge-retrieval-ab-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "growth_report": str(growth_report_path),
        "baseline_database": str(base.knowledge_database_path),
        "treatment_database": str(treatment_database),
        "candidate_edge": edge,
        "cases": cases,
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
    parser = argparse.ArgumentParser(
        description="Compare retrieval before and after one candidate edge"
    )
    parser.add_argument("growth_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(run(args.growth_report.resolve(), args.output))
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
