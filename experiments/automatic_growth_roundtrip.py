from __future__ import annotations

import asyncio
from contextlib import closing
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sqlite3
from time import perf_counter

from dotenv import load_dotenv

from src.bot.runtime import AdapterRuntime
from src.memory import MemorySystemConfig, PlatformIdentity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTION = (
    "是谁袭击伊甸园条约现场，茶会内部哪项合作为这场危机提供了背景？"
)
FOLLOWUP = (
    "把条约现场的袭击者和茶会成员此前与该势力的合作一起说明。"
)
PREMISE_GROUPS = ({291, 312}, {311})


def _clone(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source)) as source_connection:
        with closing(sqlite3.connect(target)) as target_connection:
            source_connection.backup(target_connection)


def _association_signature(path: Path) -> tuple[int, int]:
    with closing(sqlite3.connect(path)) as connection:
        count, maximum = connection.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM association"
        ).fetchone()
    return int(count), int(maximum)


def _new_edges(path: Path, previous_maximum: int) -> list[dict]:
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM association WHERE id > ? ORDER BY id",
            (int(previous_maximum),),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "from_id": int(row["from_id"]),
            "to_id": int(row["to_id"]),
            "relation_key": str(row["relation_key"]),
            "relation_text": str(row["relation_text"]),
            "confidence": float(row["confidence"]),
            "generation": int(row["generation"]),
            "audit_status": str(row["audit_status"]),
            "evidence_count": int(row["evidence_count"]),
        }
        for row in rows
    ]


async def run() -> dict:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    baseline = MemorySystemConfig.from_env(PROJECT_ROOT)
    formal_before = _association_signature(baseline.knowledge_database_path)
    root = (
        PROJECT_ROOT
        / "logs"
        / "experiments"
        / f"automatic-growth-roundtrip-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    database = root / "knowledge.db"
    await asyncio.to_thread(_clone, baseline.knowledge_database_path, database)
    clone_before = _association_signature(database)

    os.environ["KNOWLEDGE_MEMORY_DB_PATH"] = str(database)
    os.environ["MEMORY_LOG_DIR"] = str(root / "memory-logs")
    os.environ["MEMORY_GROWTH_QUEUE_DIR"] = str(root / "growth-queue")
    os.environ["CONVERSATION_INBOX_DIR"] = str(root / "conversation-inbox")
    os.environ["ADAPTER_EVENT_DB_PATH"] = str(root / "events.db")
    os.environ["USER_MODEL_SETTINGS_PATH"] = str(root / "model-settings.json")

    runtime = await AdapterRuntime.create()
    identity = PlatformIdentity("synthetic", "automatic-growth")
    try:
        first_started = perf_counter()
        first = await runtime.coordinator.handle(
            identity=identity,
            text=QUESTION,
            conversation_key="automatic-growth-roundtrip",
            defer_commit=True,
        )
        first_seconds = perf_counter() - first_started
        pending = await runtime.growth_worker.enqueue(
            identity,
            QUESTION,
            first.route,
            assistant_response=first.text,
            memory=first.memory,
            consolidation_decision=first.memory_consolidation,
        )
        if pending is not None:
            await asyncio.wait_for(runtime.growth_worker._queue.join(), timeout=90.0)

        edges = await asyncio.to_thread(_new_edges, database, clone_before[1])
        second_started = perf_counter()
        second = await runtime.coordinator.handle(
            identity=identity,
            text=FOLLOWUP,
            conversation_key="automatic-growth-roundtrip",
            defer_commit=True,
        )
        second_seconds = perf_counter() - second_started
    finally:
        await runtime.close()

    candidates = list(
        first.memory_consolidation.get("knowledge_candidates") or []
    )
    knowledge = (
        (second.memory.raw_result.get("domains") or {}).get("knowledge") or {}
    )
    rerank = knowledge.get("rerank_trace") or {}
    episode_ids = {int(value) for value in knowledge.get("episode_ids") or []}
    formal_after = _association_signature(baseline.knowledge_database_path)
    checks = {
        "first_turn_saw_each_answer_group": all(
            group.intersection(
                {
                int(row["id"])
                for row in (
                    (
                        (first.memory.raw_result.get("domains") or {}).get(
                            "knowledge"
                        )
                        or {}
                    ).get("evidence_episodes")
                    or []
                )
                if isinstance(row, dict) and row.get("id") is not None
                }
            )
            for group in PREMISE_GROUPS
        ),
        "foreground_created_candidate": bool(candidates),
        "foreground_selected_evidence_bridge": any(
            str(row.get("inference_type")) == "evidence_bridge"
            for row in candidates
            if isinstance(row, dict)
        ),
        "background_created_dual_audited_edge": any(
            row["audit_status"] == "dual_accepted"
            and row["relation_key"] == "evidence_bridge"
            and row["generation"] == 1
            for row in edges
        ),
        "second_turn_used_association_route": (
            (second.memory.raw_result.get("intent_planning") or {}).get("planner")
            == "association_cache"
        ),
        "second_turn_used_capsule": rerank.get("backend") == "association_capsule",
        "second_turn_recalled_grown_edge_endpoints": any(
            {row["from_id"], row["to_id"]}.issubset(episode_ids)
            for row in edges
            if row["relation_key"] == "evidence_bridge"
        ),
        "second_turn_zero_query_embedding_calls": (
            (knowledge.get("query_embedding_cache") or {}).get("miss_count", 0)
            == 0
        ),
        "second_turn_under_five_seconds": second_seconds < 5.0,
        "formal_database_unchanged": formal_before == formal_after,
    }
    report = {
        "version": "automatic-growth-roundtrip-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "database": str(database),
        "question": QUESTION,
        "first_seconds": round(first_seconds, 6),
        "first_reply": first.text,
        "first_route": {
            "knowledge": first.route.knowledge,
            "intensity": first.route.intensity,
            "knowledge_write_policy": first.route.knowledge_write_policy,
        },
        "first_intent_planning": first.memory.raw_result.get("intent_planning"),
        "first_consolidation": first.memory_consolidation,
        "growth_job_enqueued": pending is not None,
        "growth_worker": {
            "completed": runtime.growth_worker.completed_jobs,
            "failed": runtime.growth_worker.failed_jobs,
            "last_error": runtime.growth_worker.last_error,
        },
        "new_edges": edges,
        "followup": FOLLOWUP,
        "second_seconds": round(second_seconds, 6),
        "second_reply": second.text,
        "second_intent_planning": second.memory.raw_result.get("intent_planning"),
        "second_episode_ids": sorted(episode_ids),
        "second_rerank": rerank,
        "checks": checks,
        "passed": all(checks.values()),
    }
    root.mkdir(parents=True, exist_ok=True)
    output = root / "report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(output)
    return report


def main() -> int:
    report = asyncio.run(run())
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": report["checks"],
                "first_seconds": report["first_seconds"],
                "second_seconds": report["second_seconds"],
                "candidate_types": [
                    row.get("inference_type")
                    for row in report["first_consolidation"].get(
                        "knowledge_candidates", []
                    )
                    if isinstance(row, dict)
                ],
                "new_edge_keys": [
                    row["relation_key"] for row in report["new_edges"]
                ],
                "report_path": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
