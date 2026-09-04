from __future__ import annotations

import argparse
from contextlib import closing
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "experiments" / "manifests" / "foreground_benchmark_v1.json"
)
DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "logs"
    / "foreground-benchmark"
    / "20260830T044236.550995Z"
    / "report.json"
)
MULTILINGUAL_PROBES = [
    ("zh", "未花想让渚下台并由自己成为茶会话事人。哪段剧情直接说明了这件事？"),
    (
        "en",
        "Which episode says Mika wanted Nagisa removed and planned to become the Tea Party host herself?",
    ),
    (
        "ja",
        "ミカがナギサを失脚させ、自分がティーパーティーのホストになろうとした話はどれですか？",
    ),
    (
        "ko",
        "미카가 나기사를 물러나게 하고 자신이 티파티의 호스트가 되려 한 내용은 어느 에피소드인가요?",
    ),
]
MULTILINGUAL_PROBE_EPISODE_IDS = [1168, 303, 13, 1428]


def _load_model_client():
    engine_root = Path(os.environ["MEMORY_ENGINE_ROOT"]).resolve()
    source_root = engine_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from memory_demo.config import ModelConfig
    from memory_demo.llm.client import ModelClient

    config = ModelConfig(
        api_key=os.environ.get("SILICONFLOW_API_KEY", "").strip(),
        reranker_model=os.environ.get(
            "MEMORY_RERANKER_MODEL", "Pro/BAAI/bge-reranker-v2-m3"
        ).strip(),
        timeout_seconds=120.0,
        max_retries=2,
    )
    return ModelClient(config), config


def _episode_catalog(database: Path, ids: list[int]) -> dict[int, dict]:
    placeholders = ",".join("?" for _ in ids)
    uri = f"file:{database.as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, source_key, text, participants_json, event_type, "
            "story_time_text, timeline_scope FROM episode "
            f"WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    return {int(row["id"]): dict(row) for row in rows}


def _document(row: dict) -> str:
    try:
        participants = ", ".join(json.loads(row["participants_json"] or "[]"))
    except (json.JSONDecodeError, TypeError):
        participants = ""
    fields = [str(row["text"])]
    if participants:
        fields.append(f"人物：{participants}")
    if row.get("event_type"):
        fields.append(f"事件：{row['event_type']}")
    if row.get("story_time_text"):
        fields.append(f"故事时间：{row['story_time_text']}")
    if row.get("timeline_scope"):
        fields.append(f"时间线：{row['timeline_scope']}")
    return "\n".join(fields)


def _slot_ranks(groups: list[dict], ranked_ids: list[int]) -> list[int | None]:
    ranks = {episode_id: rank for rank, episode_id in enumerate(ranked_ids, 1)}
    return [
        min(
            (ranks[value] for value in group["alternatives"] if value in ranks),
            default=None,
        )
        for group in groups
    ]


def _recall_at(slot_ranks: list[int | None], cutoff: int) -> float:
    if not slot_ranks:
        return 0.0
    return sum(rank is not None and rank <= cutoff for rank in slot_ranks) / len(
        slot_ranks
    )


def _mean_reciprocal_slot_rank(slot_ranks: list[int | None]) -> float:
    if not slot_ranks:
        return 0.0
    return sum(1.0 / rank for rank in slot_ranks if rank is not None) / len(
        slot_ranks
    )


def _enforce_floor(
    selected_ids: list[int], required_ids: list[int], limit: int
) -> list[int]:
    required = list(dict.fromkeys(required_ids))[:limit]
    required_set = set(required)
    result = list(dict.fromkeys(selected_ids))[:limit]
    for add_id in required:
        if add_id in result:
            continue
        remove_id = next(
            (value for value in reversed(result) if value not in required_set),
            None,
        )
        if remove_id is None:
            break
        result[result.index(remove_id)] = add_id
    return result


def run(
    manifest_path: Path,
    baseline_path: Path,
    database: Path,
    output_root: Path,
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    model, model_config = _load_model_client()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    questions = {row["id"]: row for row in manifest["questions"]}
    all_candidate_ids = sorted(
        {
            int(value)
            for row in baseline["rows"]
            for value in row["rerank_input_episode_ids"]
        }.union(MULTILINGUAL_PROBE_EPISODE_IDS)
    )
    catalog = _episode_catalog(database, all_candidate_ids)
    if len(catalog) != len(all_candidate_ids):
        missing = sorted(set(all_candidate_ids).difference(catalog))
        raise RuntimeError(f"candidate Episodes missing from database: {missing}")

    cases: list[dict] = []
    for baseline_row in baseline["rows"]:
        item = questions[baseline_row["id"]]
        candidate_ids = [
            int(value) for value in baseline_row["rerank_input_episode_ids"]
        ]
        documents = [_document(catalog[value]) for value in candidate_ids]
        started = perf_counter()
        results = model.rerank(
            item["question"], documents, top_n=len(documents)
        )
        elapsed = perf_counter() - started
        ranked_ids = [candidate_ids[int(row["index"])] for row in results]
        slot_ranks = _slot_ranks(item["evidence_groups"], ranked_ids)
        required_ids = [
            int(decision["episode_id"])
            for decision in baseline_row.get("precompression", {}).get(
                "decisions", []
            )
            if decision.get("kept")
            and decision.get("reason") == "required_evidence_floor"
        ]
        hybrid_by_cutoff = {
            cutoff: _enforce_floor(ranked_ids[:cutoff], required_ids, cutoff)
            for cutoff in (10, 20, 30)
        }
        hybrid_ranks_by_cutoff = {
            cutoff: _slot_ranks(
                item["evidence_groups"], hybrid_by_cutoff[cutoff]
            )
            for cutoff in (10, 20, 30)
        }
        candidate_slot_ranks = _slot_ranks(
            item["evidence_groups"], candidate_ids
        )
        baseline_rerank_seconds = float(
            baseline_row.get("timings", {})
            .get("phases_seconds", {})
            .get("evidence_rerank", 0.0)
        )
        cases.append(
            {
                "id": baseline_row["id"],
                "repetition": int(baseline_row["repetition"]),
                "question": item["question"],
                "candidate_count": len(candidate_ids),
                "candidate_pool_fact_slot_recall": _recall_at(
                    candidate_slot_ranks, len(candidate_ids)
                ),
                "original_order_fact_slot_recall_at_10": _recall_at(
                    candidate_slot_ranks, 10
                ),
                "original_order_fact_slot_recall_at_20": _recall_at(
                    candidate_slot_ranks, 20
                ),
                "original_order_fact_slot_recall_at_30": _recall_at(
                    candidate_slot_ranks, 30
                ),
                "baseline_llm_fact_slot_recall": sum(
                    bool(row["passed"])
                    for row in baseline_row["evidence_groups"]
                )
                / len(baseline_row["evidence_groups"]),
                "baseline_llm_rerank_seconds": round(
                    baseline_rerank_seconds, 6
                ),
                "bge_rerank_seconds": round(elapsed, 6),
                "bge_ranked_episode_ids": ranked_ids,
                "bge_ranked_scores": [
                    round(float(row["relevance_score"]), 8) for row in results
                ],
                "fact_slot_ranks": slot_ranks,
                "fact_slot_recall_at_10": _recall_at(slot_ranks, 10),
                "fact_slot_recall_at_20": _recall_at(slot_ranks, 20),
                "fact_slot_recall_at_30": _recall_at(slot_ranks, 30),
                "required_evidence_floor_ids": required_ids,
                "hybrid_episode_ids_at_10": hybrid_by_cutoff[10],
                "hybrid_episode_ids_at_20": hybrid_by_cutoff[20],
                "hybrid_episode_ids_at_30": hybrid_by_cutoff[30],
                "hybrid_fact_slot_ranks_at_10": hybrid_ranks_by_cutoff[10],
                "hybrid_fact_slot_ranks_at_20": hybrid_ranks_by_cutoff[20],
                "hybrid_fact_slot_ranks_at_30": hybrid_ranks_by_cutoff[30],
                "hybrid_fact_slot_recall_at_10": _recall_at(
                    hybrid_ranks_by_cutoff[10], 10
                ),
                "hybrid_fact_slot_recall_at_20": _recall_at(
                    hybrid_ranks_by_cutoff[20], 20
                ),
                "hybrid_fact_slot_recall_at_30": _recall_at(
                    hybrid_ranks_by_cutoff[30], 30
                ),
                "mean_reciprocal_fact_slot_rank": (
                    _mean_reciprocal_slot_rank(slot_ranks)
                ),
            }
        )

    multilingual_documents = [
        _document(catalog[value])
        for value in MULTILINGUAL_PROBE_EPISODE_IDS
    ]
    multilingual_probes: list[dict] = []
    for language, query in MULTILINGUAL_PROBES:
        started = perf_counter()
        ranked = model.rerank(
            query,
            multilingual_documents,
            top_n=len(multilingual_documents),
        )
        elapsed = perf_counter() - started
        ranked_ids = [
            MULTILINGUAL_PROBE_EPISODE_IDS[int(item["index"])]
            for item in ranked
        ]
        multilingual_probes.append(
            {
                "language": language,
                "query": query,
                "ranked_episode_ids": ranked_ids,
                "expected_episode_id": 1168,
                "expected_rank": ranked_ids.index(1168) + 1,
                "elapsed_seconds": round(elapsed, 6),
                "passed": ranked_ids[0] == 1168,
            }
        )

    slot_count = sum(
        len(questions[row["id"]]["evidence_groups"]) for row in cases
    )
    mean = lambda values: sum(values) / len(values) if values else 0.0
    baseline_seconds = [row["baseline_llm_rerank_seconds"] for row in cases]
    bge_seconds = [row["bge_rerank_seconds"] for row in cases]
    summary = {
        "case_count": len(cases),
        "fact_slot_count": slot_count,
        "candidate_pool_fact_slot_recall": mean(
            [row["candidate_pool_fact_slot_recall"] for row in cases]
        ),
        "baseline_llm_fact_slot_recall": mean(
            [row["baseline_llm_fact_slot_recall"] for row in cases]
        ),
        "original_order_fact_slot_recall_at_10": mean(
            [row["original_order_fact_slot_recall_at_10"] for row in cases]
        ),
        "original_order_fact_slot_recall_at_20": mean(
            [row["original_order_fact_slot_recall_at_20"] for row in cases]
        ),
        "original_order_fact_slot_recall_at_30": mean(
            [row["original_order_fact_slot_recall_at_30"] for row in cases]
        ),
        "bge_fact_slot_recall_at_10": mean(
            [row["fact_slot_recall_at_10"] for row in cases]
        ),
        "bge_fact_slot_recall_at_20": mean(
            [row["fact_slot_recall_at_20"] for row in cases]
        ),
        "bge_fact_slot_recall_at_30": mean(
            [row["fact_slot_recall_at_30"] for row in cases]
        ),
        "bge_with_evidence_floor_fact_slot_recall_at_30": mean(
            [row["hybrid_fact_slot_recall_at_30"] for row in cases]
        ),
        "bge_with_evidence_floor_fact_slot_recall_at_10": mean(
            [row["hybrid_fact_slot_recall_at_10"] for row in cases]
        ),
        "bge_with_evidence_floor_fact_slot_recall_at_20": mean(
            [row["hybrid_fact_slot_recall_at_20"] for row in cases]
        ),
        "bge_mean_reciprocal_fact_slot_rank": mean(
            [row["mean_reciprocal_fact_slot_rank"] for row in cases]
        ),
        "baseline_llm_mean_rerank_seconds": mean(baseline_seconds),
        "bge_mean_rerank_seconds": mean(bge_seconds),
        "bge_median_rerank_seconds": sorted(bge_seconds)[
            len(bge_seconds) // 2
        ],
        "estimated_mean_seconds_saved": mean(baseline_seconds)
        - mean(bge_seconds),
        "estimated_rerank_speedup": (
            mean(baseline_seconds) / mean(bge_seconds)
            if mean(bge_seconds) > 0
            else math.inf
        ),
        "failed_cases": sum(
            row["fact_slot_recall_at_30"] < 1.0 for row in cases
        ),
        "multilingual_probe_passed": sum(
            row["passed"] for row in multilingual_probes
        ),
        "multilingual_probe_count": len(multilingual_probes),
        "multilingual_probe_mean_seconds": mean(
            [row["elapsed_seconds"] for row in multilingual_probes]
        ),
    }
    checks = {
        "all_candidate_pools_cover_every_fact_slot": summary[
            "candidate_pool_fact_slot_recall"
        ]
        == 1.0,
        "bge_recall_at_20_at_least_95_percent": summary[
            "bge_fact_slot_recall_at_20"
        ]
        >= 0.95,
        "bge_with_floor_recall_at_30_matches_baseline": summary[
            "bge_with_evidence_floor_fact_slot_recall_at_30"
        ]
        >= summary["baseline_llm_fact_slot_recall"],
        "bge_rerank_is_faster": summary["bge_mean_rerank_seconds"]
        < summary["baseline_llm_mean_rerank_seconds"],
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    target = output_root / run_id / "candidate-pool-ab.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    pure_checks = {
        "recall_at_20_at_least_95_percent": summary[
            "bge_fact_slot_recall_at_20"
        ]
        >= 0.95,
        "recall_at_30_matches_baseline": summary[
            "bge_fact_slot_recall_at_30"
        ]
        >= summary["baseline_llm_fact_slot_recall"],
    }
    hybrid_checks = {
        "all_candidate_pools_cover_every_fact_slot": checks[
            "all_candidate_pools_cover_every_fact_slot"
        ],
        "floor_recall_at_30_matches_baseline": checks[
            "bge_with_floor_recall_at_30_matches_baseline"
        ],
        "floor_recall_at_20_at_least_95_percent": summary[
            "bge_with_evidence_floor_fact_slot_recall_at_20"
        ]
        >= 0.95,
        "rerank_is_faster": checks["bge_rerank_is_faster"],
        "all_multilingual_probes_rank_expected_episode_first": all(
            row["passed"] for row in multilingual_probes
        ),
    }
    report = {
        "version": "bge-reranker-candidate-pool-ab-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": model_config.reranker_model,
        "manifest": str(manifest_path),
        "baseline_report": str(baseline_path),
        "database": str(database),
        "summary": summary,
        "checks": checks,
        "pure_replacement_checks": pure_checks,
        "pure_replacement_passed": all(pure_checks.values()),
        "hybrid_checks": hybrid_checks,
        "hybrid_passed": all(hybrid_checks.values()),
        "passed": all(hybrid_checks.values()),
        "multilingual_probes": multilingual_probes,
        "cases": cases,
    }
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_path"] = str(target)
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Compare BGE reranker against frozen LLM rerank pools"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "data" / "knowledge" / "blue_archive.db",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "bge-reranker-ab",
    )
    args = parser.parse_args()
    report = run(
        args.manifest.resolve(),
        args.baseline.resolve(),
        args.database.resolve(),
        args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "passed": report["passed"],
                "summary": report["summary"],
                "checks": report["checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
