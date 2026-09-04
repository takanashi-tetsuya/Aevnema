from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from dotenv import load_dotenv

from src.memory import MemorySystem, MemorySystemConfig, route_memory_query


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(
        sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    ) as source_connection:
        with closing(sqlite3.connect(target)) as target_connection:
            source_connection.backup(target_connection)


def _database_signature(path: Path) -> dict:
    with closing(
        sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    ) as connection:
        count, maximum, total_use_count, latest_use = connection.execute(
            "SELECT COUNT(*), MAX(id), COALESCE(SUM(use_count), 0), "
            "MAX(last_used) FROM association"
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


def _resolve_manifest_path(
    value: str,
    manifest_dir: Path,
    engine_root: Path,
) -> Path:
    if value.startswith("engine://"):
        return engine_root / value.removeprefix("engine://")
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def _load_manifest(manifest_path: Path, engine_root: Path) -> dict:
    """Resolve direct questions plus provenance-preserving catalog imports."""
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = list(payload.get("questions") or [])
    resolved_assets = []

    for value in payload.get("include_manifests") or []:
        included_path = _resolve_manifest_path(
            str(value), manifest_path.parent, engine_root
        ).resolve()
        included = _load_manifest(included_path, engine_root)
        questions.extend(included.get("questions") or [])
        resolved_assets.extend(included.get("resolved_assets") or [])
        resolved_assets.append(str(included_path))

    for imported in payload.get("catalog_imports") or []:
        if not isinstance(imported, dict):
            raise ValueError("catalog_imports entries must be objects")
        question_path = _resolve_manifest_path(
            str(imported["questions_path"]),
            manifest_path.parent,
            engine_root,
        ).resolve()
        evidence_path = _resolve_manifest_path(
            str(imported["evidence_path"]),
            manifest_path.parent,
            engine_root,
        ).resolve()
        catalog = json.loads(question_path.read_text(encoding="utf-8"))
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        catalog_by_id = {str(item["id"]): item for item in catalog}
        evidence_by_id = evidence.get("questions") or {}
        labels_by_id = imported.get("group_labels") or {}
        additions_by_id = imported.get("group_alternative_additions") or {}
        overrides_by_id = imported.get("group_alternative_overrides") or {}
        source_additions_by_id = (
            imported.get("group_required_source_additions") or {}
        )
        for question_id in imported.get("include_ids") or []:
            question_id = str(question_id)
            if question_id not in catalog_by_id:
                raise ValueError(
                    f"question {question_id!r} missing from {question_path}"
                )
            evidence_item = evidence_by_id.get(question_id)
            if not isinstance(evidence_item, dict):
                raise ValueError(
                    f"evidence {question_id!r} missing from {evidence_path}"
                )
            episode_groups = evidence_item.get("required_episode_groups") or []
            labels = labels_by_id.get(question_id) or []
            additions = additions_by_id.get(question_id) or []
            overrides = overrides_by_id.get(question_id) or []
            source_additions = source_additions_by_id.get(question_id) or []
            groups = []
            for index, alternatives in enumerate(episode_groups, start=1):
                original_alternatives = [int(value) for value in alternatives]
                override = (
                    overrides[index - 1]
                    if index <= len(overrides)
                    else None
                )
                added = (
                    additions[index - 1]
                    if index <= len(additions)
                    else []
                )
                added_sources = (
                    source_additions[index - 1]
                    if index <= len(source_additions)
                    else []
                )
                active_alternatives = (
                    [int(value) for value in override]
                    if override is not None
                    else original_alternatives
                )
                active_alternatives = list(
                    dict.fromkeys(
                        [*active_alternatives, *(int(value) for value in added)]
                    )
                )
                label = (
                    str(labels[index - 1])
                    if index <= len(labels)
                    else f"legacy evidence group {index}"
                )
                group = {
                    "id": f"{question_id}_slot_{index:02d}",
                    "description": label,
                    "alternatives": active_alternatives,
                    "required_sources": list(
                        dict.fromkeys(
                            [
                                *evidence_item.get("required_sources", []),
                                *(str(value) for value in added_sources),
                            ]
                        )
                    ),
                }
                if override is not None or added or added_sources:
                    group["oracle_adjustment"] = {
                        "original_alternatives": original_alternatives,
                        "override_applied": override is not None,
                        "added_alternatives": [int(value) for value in added],
                        "added_required_sources": [
                            str(value) for value in added_sources
                        ],
                    }
                groups.append(group)
            questions.append(
                {
                    **catalog_by_id[question_id],
                    "evidence_groups": groups,
                    "catalog_provenance": {
                        "questions_path": str(question_path),
                        "evidence_path": str(evidence_path),
                    },
                }
            )
        resolved_assets.extend([str(question_path), str(evidence_path)])

    duplicate_ids = sorted(
        {
            question_id
            for question_id in (str(item["id"]) for item in questions)
            if sum(str(item["id"]) == question_id for item in questions) > 1
        }
    )
    if duplicate_ids:
        raise ValueError(f"duplicate question ids: {duplicate_ids}")
    return {
        **payload,
        "questions": questions,
        "resolved_assets": list(dict.fromkeys(resolved_assets)),
    }


def _score_groups(item: dict, final_ids: set[int]) -> list[dict]:
    groups = []
    for group in item["evidence_groups"]:
        alternatives = {int(value) for value in group["alternatives"]}
        matches = sorted(final_ids & alternatives)
        groups.append(
            {
                **group,
                "matches": matches,
                "passed": bool(matches),
            }
        )
    return groups


def validate_manifest(manifest_path: Path, project_root: Path = PROJECT_ROOT) -> dict:
    load_dotenv(project_root / ".env")
    config = MemorySystemConfig.from_env(project_root)
    manifest = _load_manifest(manifest_path, config.engine_root)
    expected_ids = {
        int(value)
        for item in manifest["questions"]
        for group in item["evidence_groups"]
        for value in group["alternatives"]
    }
    connection = sqlite3.connect(
        f"file:{config.knowledge_database_path}?mode=ro",
        uri=True,
    )
    try:
        placeholders = ",".join("?" for _ in expected_ids)
        rows = (
            connection.execute(
                f"SELECT id, source_key FROM episode WHERE id IN ({placeholders})",
                sorted(expected_ids),
            ).fetchall()
            if expected_ids
            else []
        )
    finally:
        connection.close()
    source_by_id = {int(node_id): str(source_key) for node_id, source_key in rows}
    missing_ids = sorted(expected_ids.difference(source_by_id))
    source_violations = []
    empty_groups = []
    for item in manifest["questions"]:
        for group in item["evidence_groups"]:
            alternatives = [int(value) for value in group["alternatives"]]
            if not alternatives:
                empty_groups.append(
                    {"question_id": item["id"], "group_id": group["id"]}
                )
            allowed_sources = {
                str(value) for value in group.get("required_sources") or []
            }
            for episode_id in alternatives:
                actual_source = source_by_id.get(episode_id)
                if (
                    actual_source is not None
                    and allowed_sources
                    and actual_source not in allowed_sources
                ):
                    source_violations.append(
                        {
                            "question_id": item["id"],
                            "group_id": group["id"],
                            "episode_id": episode_id,
                            "actual_source": actual_source,
                            "allowed_sources": sorted(allowed_sources),
                        }
                    )
    return {
        "version": "foreground-manifest-validation-v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_version": manifest.get("version"),
        "questions": len(manifest["questions"]),
        "evidence_groups": sum(
            len(item["evidence_groups"]) for item in manifest["questions"]
        ),
        "unique_episode_ids": len(expected_ids),
        "missing_episode_ids": missing_ids,
        "empty_groups": empty_groups,
        "source_violations": source_violations,
        "valid": not missing_ids and not empty_groups and not source_violations,
        "resolved_assets": manifest.get("resolved_assets") or [],
    }


def _summary(rows: list[dict]) -> dict:
    elapsed_values = [float(row["elapsed_seconds"]) for row in rows]
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
    total_pipeline_seconds = sum(
        float((row.get("timings") or {}).get("total_seconds") or 0.0)
        for row in rows
    )
    phase_summary = {}
    for name in phase_names:
        values = [
            float(
                ((row.get("timings") or {}).get("phases_seconds") or {}).get(
                    name, 0.0
                )
            )
            for row in rows
        ]
        phase_total = sum(values)
        phase_summary[name] = {
            "mean_seconds": round(phase_total / len(values), 6),
            "minimum_seconds": min(values),
            "maximum_seconds": max(values),
            "share_of_pipeline": round(
                phase_total / total_pipeline_seconds
                if total_pipeline_seconds
                else 0.0,
                6,
            ),
        }
    return {
        "runs": len(rows),
        "passed_runs": sum(int(row["passed"]) for row in rows),
        "fact_slot_recall": (
            matched_group_count / group_count if group_count else 0.0
        ),
        "matched_fact_slots": matched_group_count,
        "required_fact_slots": group_count,
        "mean_seconds": (
            round(sum(elapsed_values) / len(elapsed_values), 3)
            if elapsed_values
            else 0.0
        ),
        "minimum_seconds": min(elapsed_values, default=0.0),
        "maximum_seconds": max(elapsed_values, default=0.0),
        "phase_timings": phase_summary,
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


async def run(
    manifest_path: Path,
    output_root: Path,
    repeat: int,
    question_ids: set[str] | None = None,
    rerank_input_limit: int | None = None,
    rerank_backend: str | None = None,
    reranker_model: str | None = None,
    isolate_database: bool = False,
    disable_rerank: bool = False,
    retrieval_intensity: str = "auto",
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    config = MemorySystemConfig.from_env(PROJECT_ROOT)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    formal_database = config.knowledge_database_path.resolve()
    formal_before = (
        _database_signature(formal_database) if isolate_database else None
    )
    if isolate_database:
        isolated_database = output_root / run_id / "knowledge.db"
        _sqlite_backup(formal_database, isolated_database)
        config.knowledge_database_path = isolated_database
    if rerank_input_limit is not None:
        config.foreground_rerank_precompression_limit = max(
            1, int(rerank_input_limit)
        )
    if reranker_model is not None:
        config.reranker_model = reranker_model.strip()
        config.foreground_rerank_enabled = bool(config.reranker_model)
        config.foreground_rerank_backend = (
            "cross_encoder" if config.reranker_model else "llm"
        )
    if rerank_backend is not None:
        if rerank_backend == "cross_encoder" and not config.reranker_model:
            raise ValueError(
                "cross_encoder requires MEMORY_RERANKER_MODEL or "
                "--reranker-model"
            )
        config.foreground_rerank_backend = rerank_backend
        config.foreground_rerank_enabled = True
    if disable_rerank:
        config.foreground_rerank_enabled = False
    manifest = _load_manifest(manifest_path, config.engine_root)
    # Stability repeats must execute the model path rather than measuring the
    # exact-query LRU cache.
    config.foreground_recall_cache_size = 0
    memory = MemorySystem(config)
    await memory.initialize()

    selected_questions = [
        item
        for item in manifest["questions"]
        if not question_ids or item["id"] in question_ids
    ]
    if question_ids:
        missing = question_ids.difference(
            item["id"] for item in selected_questions
        )
        if missing:
            raise ValueError(f"unknown question ids: {sorted(missing)}")
    rows = []
    for repetition in range(1, repeat + 1):
        for item in selected_questions:
            item_intensity = (
                route_memory_query(item["question"]).intensity
                if retrieval_intensity == "auto"
                else retrieval_intensity
            )
            started = perf_counter()
            recalled = await memory.knowledge.recall(
                item["question"],
                intent_override=item.get("intent_override"),
                followup_queries_override=item.get(
                    "followup_queries_override"
                ),
                retrieval_intensity=item_intensity,
            )
            elapsed = perf_counter() - started
            final_ids = set(recalled.raw_result.get("episode_ids") or [])
            groups = _score_groups(item, final_ids)
            trace = recalled.raw_result.get("rerank_trace") or {}
            precompression = trace.get("precompression") or {}
            timings = recalled.raw_result.get("timings") or {}
            rows.append(
                {
                    "id": item["id"],
                    "repetition": repetition,
                    "question": item["question"],
                    "retrieval_intensity": item_intensity,
                    "rerank_policy": recalled.raw_result.get(
                        "rerank_policy"
                    ),
                    "rerank_execution": recalled.raw_result.get(
                        "rerank_execution"
                    ),
                    "elapsed_seconds": round(elapsed, 3),
                    "error": recalled.error,
                    "review_mode": trace.get("review_mode"),
                    "review_level": trace.get("review_level"),
                    "coverage_audit_performed": bool(
                        trace.get("coverage_audit_performed")
                    ),
                    "compressor_performed": bool(
                        trace.get("compressor_performed")
                    ),
                    "precompression": precompression,
                    "rerank_input_episode_ids": list(
                        trace.get("rerank_input_episode_ids") or []
                    ),
                    "rerank_candidate_text_chars": int(
                        trace.get("rerank_candidate_text_chars") or 0
                    ),
                    "initial_prompt_chars": int(
                        trace.get("initial_prompt_chars") or 0
                    ),
                    "timings": timings,
                    "episode_ids": sorted(final_ids),
                    "evidence_groups": groups,
                    "passed": not recalled.error
                    and all(group["passed"] for group in groups),
                }
            )

    formal_after = (
        _database_signature(formal_database) if isolate_database else None
    )
    report = {
        "version": "foreground-benchmark-report-v1",
        "run_id": run_id,
        "manifest": str(manifest_path.resolve()),
        "manifest_version": manifest.get("version"),
        "resolved_assets": manifest.get("resolved_assets") or [],
        "review_mode": config.foreground_review_mode,
        "rerank_backend": config.foreground_rerank_backend,
        "rerank_enabled": config.foreground_rerank_enabled,
        "reranker_model": config.reranker_model,
        "requested_retrieval_intensity": retrieval_intensity,
        "knowledge_database": str(config.knowledge_database_path),
        "formal_database": str(formal_database),
        "formal_database_before": formal_before,
        "formal_database_after": formal_after,
        "formal_database_unchanged": (
            formal_before == formal_after if isolate_database else None
        ),
        "rerank_precompression_limit": (
            config.foreground_rerank_precompression_limit
        ),
        "repeat": repeat,
        "summary": _summary(rows),
        "rows": rows,
    }
    return _write_report(report, output_root)


def rescore(manifest_path: Path, replay_path: Path, output_root: Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    config = MemorySystemConfig.from_env(PROJECT_ROOT)
    manifest = _load_manifest(manifest_path, config.engine_root)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    questions = {item["id"]: item for item in manifest["questions"]}
    rows = []
    for previous in replay["rows"]:
        item = questions.get(previous["id"])
        if item is None:
            raise ValueError(f"unknown question id in replay report: {previous['id']}")
        final_ids = {int(value) for value in previous.get("episode_ids") or []}
        groups = _score_groups(item, final_ids)
        rows.append(
            {
                **previous,
                "question": item["question"],
                "episode_ids": sorted(final_ids),
                "evidence_groups": groups,
                "passed": not previous.get("error")
                and all(group["passed"] for group in groups),
            }
        )
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    report = {
        **replay,
        "run_id": run_id,
        "manifest": str(manifest_path.resolve()),
        "manifest_version": manifest.get("version"),
        "resolved_assets": manifest.get("resolved_assets") or [],
        "rescored_from": str(replay_path.resolve()),
        "summary": _summary(rows),
        "rows": rows,
    }
    report.pop("report_path", None)
    return _write_report(report, output_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed foreground fact-slot benchmark"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "manifests"
            / "foreground_benchmark_v1.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "foreground-benchmark",
    )
    parser.add_argument(
        "--replay-report",
        type=Path,
        help="Rescore an existing report with the current manifest without API calls",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen Episode ids/source boundaries without API calls",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--rerank-input-limit",
        type=int,
        help=(
            "Deterministically compress Candidate@100 to this many Episodes "
            "before the selected reranker."
        ),
    )
    parser.add_argument(
        "--rerank-backend",
        choices=("llm", "cross_encoder"),
        help="Select the evidence rerank implementation for this run.",
    )
    parser.add_argument(
        "--reranker-model",
        help=(
            "Override MEMORY_RERANKER_MODEL for this benchmark. An empty "
            "value disables reranking."
        ),
    )
    parser.add_argument(
        "--isolated-database",
        action="store_true",
        help="Run against a SQLite backup and verify the formal DB signature.",
    )
    parser.add_argument(
        "--disable-rerank",
        action="store_true",
        help="Skip reranking to measure whether it changes final membership.",
    )
    parser.add_argument(
        "--retrieval-intensity",
        choices=("auto", "light", "standard", "deep"),
        default="auto",
        help=(
            "Use the real chatbot router (auto) or force one retrieval "
            "intensity for a component benchmark."
        ),
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Run only the selected manifest question id; may be repeated",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.rerank_input_limit is not None and args.rerank_input_limit < 1:
        parser.error("--rerank-input-limit must be at least 1")
    if args.disable_rerank and (
        args.rerank_backend is not None or args.reranker_model is not None
    ):
        parser.error(
            "--disable-rerank cannot be combined with reranker overrides"
        )
    if (
        args.reranker_model is not None
        and not args.reranker_model.strip()
        and args.rerank_backend is not None
    ):
        parser.error(
            "an empty --reranker-model cannot be combined with --rerank-backend"
        )
    if args.validate_only:
        report = validate_manifest(args.manifest)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        return 0 if report["valid"] else 1
    if args.replay_report:
        report = rescore(args.manifest, args.replay_report, args.output_root)
    else:
        report = asyncio.run(
            run(
                args.manifest,
                args.output_root,
                args.repeat,
                set(args.question_id),
                args.rerank_input_limit,
                args.rerank_backend,
                args.reranker_model,
                args.isolated_database,
                args.disable_rerank,
                args.retrieval_intensity,
            )
        )
    print(
        json.dumps(
            {
                "report_path": report.get("report_path"),
                "rerank_backend": report.get("rerank_backend"),
                "rerank_enabled": report.get("rerank_enabled"),
                "reranker_model": report.get("reranker_model"),
                "requested_retrieval_intensity": report.get(
                    "requested_retrieval_intensity"
                ),
                "formal_database_unchanged": report.get(
                    "formal_database_unchanged"
                ),
                "summary": report["summary"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0 if report["summary"]["passed_runs"] == report["summary"]["runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
