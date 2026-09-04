from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
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


DEFAULT_FIXTURE = Path(__file__).with_name("multi_candidate_growth_fixture.json")


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
        count, maximum = connection.execute(
            "SELECT COUNT(*), MAX(id) FROM association"
        ).fetchone()
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "association_count": int(count),
        "association_max_id": int(maximum or 0),
    }


def _association_rows(database_path: Path, ids: list[int]) -> list[dict]:
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT * FROM association WHERE id IN ({placeholders}) ORDER BY id",
            ids,
        ).fetchall()
    return [dict(row) for row in rows]


def _episode_rows(database_path: Path, ids: list[int]) -> list[dict]:
    placeholders = ",".join("?" for _ in ids)
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            f"SELECT id, source_key, text FROM episode "
            f"WHERE id IN ({placeholders}) ORDER BY id",
            ids,
        ).fetchall()
    return [
        {"id": int(row[0]), "source_key": row[1], "text": row[2]}
        for row in rows
    ]


def _last_audit_event(log_root: Path) -> dict:
    events: list[dict] = []
    for path in sorted(log_root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "growth_relationship_audit":
                events.append(event)
    return events[-1] if events else {}


def _candidate_payload(candidate: dict) -> dict:
    return {
        "claim": str(candidate["claim"]),
        "premise_episode_ids": [
            int(value) for value in candidate["premise_episode_ids"]
        ],
        "confidence": float(candidate["confidence"]),
        "inference_type": str(candidate["inference_type"]),
    }


def _matches_candidate(row: dict, candidate: dict) -> bool:
    premise_ids = tuple(int(value) for value in candidate["premise_episode_ids"])
    return (
        str(row["from_type"]) == "episode"
        and str(row["to_type"]) == "episode"
        and int(row["from_id"]) == premise_ids[0]
        and int(row["to_id"]) == premise_ids[1]
    )


async def run(
    fixture_path: Path,
    output_root: Path,
) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    fixture_bytes = fixture_path.read_bytes()
    fixture = json.loads(fixture_bytes.decode("utf-8"))
    candidates = list(fixture.get("candidates") or [])
    if not fixture.get("frozen_before_audit") or not candidates:
        raise ValueError("fixture must be frozen and contain candidates")

    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    formal_before = _database_signature(base.knowledge_database_path)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_root = (output_root / run_id).resolve()
    knowledge_copy = run_root / "knowledge" / "blue_archive.db"
    await asyncio.to_thread(
        _sqlite_backup,
        base.knowledge_database_path,
        knowledge_copy,
    )
    config = replace(
        base,
        knowledge_database_path=knowledge_copy,
        log_dir=run_root / "memory-logs",
    )
    service = AssociativeMemoryService(
        config.domain_config("knowledge", knowledge_copy, background=True)
    )
    await service.initialize()

    premise_ids = list(
        dict.fromkeys(
            int(value)
            for candidate in candidates
            for value in candidate["premise_episode_ids"]
        )
    )
    premise_rows = _episode_rows(knowledge_copy, premise_ids)
    if {row["id"] for row in premise_rows} != set(premise_ids):
        raise RuntimeError("one or more frozen premise Episodes are missing")

    started = perf_counter()
    result = await service.audit_candidates(
        [_candidate_payload(candidate) for candidate in candidates]
    )
    elapsed = perf_counter() - started
    raw = result.raw_result or {}
    changed_ids = list(
        dict.fromkeys(
            [
                *raw.get("new_association_ids", []),
                *raw.get("reinforced_association_ids", []),
            ]
        )
    )
    new_ids = {int(value) for value in raw.get("new_association_ids", [])}
    reinforced_ids = {
        int(value) for value in raw.get("reinforced_association_ids", [])
    }
    changed_rows = _association_rows(knowledge_copy, changed_ids)
    audit_event = _last_audit_event(run_root / "memory-logs")
    decisions = list(audit_event.get("decisions") or [])

    candidate_results: list[dict] = []
    for index, candidate in enumerate(candidates):
        matching = [
            row for row in changed_rows if _matches_candidate(row, candidate)
        ]
        decision = decisions[index] if index < len(decisions) else {}
        candidate_results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "label": candidate["label"],
                "inference_type": candidate["inference_type"],
                "premise_episode_ids": candidate["premise_episode_ids"],
                "accepted": bool(matching),
                "association_ids": [int(row["id"]) for row in matching],
                "actions": [
                    "created"
                    if int(row["id"]) in new_ids
                    else "reinforced"
                    if int(row["id"]) in reinforced_ids
                    else "unknown"
                    for row in matching
                ],
                "audit": decision,
            }
        )

    formal_after = _database_signature(base.knowledge_database_path)
    accepted = [row for row in candidate_results if row["accepted"]]
    accepted_types = sorted(
        {str(row["inference_type"]) for row in accepted}
    )
    frozen_endpoint_pairs = {
        tuple(int(value) for value in candidate["premise_episode_ids"])
        for candidate in candidates
    }
    changed_endpoint_pairs = {
        (int(row["from_id"]), int(row["to_id"])) for row in changed_rows
    }
    checks = {
        "fixture_was_frozen": bool(fixture.get("frozen_before_audit")),
        "all_premises_exist": len(premise_rows) == len(premise_ids),
        "audit_completed_without_error": not result.error,
        "fixed_endpoint_dual_audit_used": (
            (raw.get("growth_utility_gate") or {}).get("mode")
            == "fixed_endpoint_dual_audit"
        ),
        "at_least_four_candidates_accepted": len(accepted) >= 4,
        # Relation-type diversity is a coverage sanity check, not a target
        # acceptance quota.  A strict auditor must be allowed to reject an
        # invalid type-specific fixture without making a sound run fail.
        "multiple_relation_types_accepted": len(accepted_types) >= 2,
        "no_unfrozen_endpoint_pair_written": changed_endpoint_pairs.issubset(
            frozen_endpoint_pairs
        ),
        "formal_database_unchanged": formal_before == formal_after,
    }
    report = {
        "version": "multi-candidate-growth-acceptance-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "fixture_path": str(fixture_path),
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "knowledge_database": str(knowledge_copy),
        "formal_database": str(base.knowledge_database_path),
        "formal_database_before": formal_before,
        "formal_database_after": formal_after,
        "candidate_count": len(candidates),
        "accepted_candidate_count": len(accepted),
        "new_association_ids": sorted(new_ids),
        "reinforced_association_ids": sorted(reinforced_ids),
        "accepted_relation_types": accepted_types,
        "elapsed_seconds": round(elapsed, 3),
        "premise_episodes": premise_rows,
        "candidate_results": candidate_results,
        "changed_associations": changed_rows,
        "audit_event": audit_event,
        "checks": checks,
        "passed": all(checks.values()),
        "error": result.error,
    }
    report_path = run_root / "growth-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Audit a frozen set of fixed-endpoint lore candidates"
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "multi-candidate-growth",
    )
    args = parser.parse_args()
    report = asyncio.run(run(args.fixture.resolve(), args.output_root.resolve()))
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "passed": report["passed"],
                "candidate_count": report["candidate_count"],
                "accepted_candidate_count": report["accepted_candidate_count"],
                "accepted_relation_types": report["accepted_relation_types"],
                "elapsed_seconds": report["elapsed_seconds"],
                "checks": report["checks"],
                "error": report["error"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
