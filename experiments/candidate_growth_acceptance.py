from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

from dotenv import load_dotenv

from src.memory import (
    BackgroundGrowthWorker,
    MemoryRoute,
    MemorySystem,
    MemorySystemConfig,
    PlatformIdentity,
    RetrievedMemory,
)
from src.memory.answer_consolidator import (
    ConsolidationDecision,
    KnowledgeCandidate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREMISE_IDS = (1166, 1361)
CANDIDATE = KnowledgeCandidate(
    claim=(
        "未花把阿里乌斯视为基于共同敌人的合作对象，但阿里乌斯小队从一开始就准备"
        "破坏圣娅的光环，超出最初的绑架计划，显示合作设想与实际执行目标存在失控性反差。"
    ),
    premise_episode_ids=PREMISE_IDS,
    confidence=0.86,
    reason="两个不同剧情文件分别提供前置合作与后续执行证据",
    inference_type="contrast",
)


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{source.as_posix()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


def _episode_rows(database_path: Path) -> list[dict]:
    placeholders = ",".join("?" for _ in PREMISE_IDS)
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            f"SELECT id, source_key, text FROM episode WHERE id IN ({placeholders})",
            PREMISE_IDS,
        ).fetchall()
    return [
        {"id": row[0], "source_key": row[1], "text": row[2]}
        for row in rows
    ]


def _association_rows(database_path: Path, association_ids: list[int]) -> list[dict]:
    if not association_ids:
        return []
    placeholders = ",".join("?" for _ in association_ids)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT * FROM association WHERE id IN ({placeholders}) ORDER BY id",
            association_ids,
        ).fetchall()
    return [dict(row) for row in rows]


async def run(output_root: Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_root = (output_root / run_id).resolve()
    knowledge_copy = run_root / "knowledge" / "blue_archive.db"
    await asyncio.to_thread(
        _sqlite_backup, base.knowledge_database_path, knowledge_copy
    )
    config = replace(
        base,
        knowledge_database_path=knowledge_copy,
        public_database_path=run_root / "public" / "memory.db",
        user_database_dir=run_root / "users",
        log_dir=run_root / "memory-logs",
        knowledge_growth_enabled=True,
        public_growth_enabled=False,
        user_growth_enabled=False,
    )
    memory_system = MemorySystem(config)
    await memory_system.initialize()

    evidence_rows = _episode_rows(knowledge_copy)
    if {row["id"] for row in evidence_rows} != set(PREMISE_IDS):
        raise RuntimeError("candidate premise Episodes are missing from the cloned DB")
    memory = RetrievedMemory(
        context="candidate acceptance evidence",
        raw_result={
            "domains": {
                "knowledge": {
                    "evidence_episodes": evidence_rows,
                    "association_paths": [],
                }
            }
        },
        domains=("knowledge",),
    )
    decision = ConsolidationDecision(
        knowledge_candidates=[CANDIDATE]
    ).as_dict()
    identity = PlatformIdentity("diagnostic", "candidate-growth")
    route = MemoryRoute(
        False,
        False,
        True,
        "candidate_growth_acceptance",
        "deep",
        "evidence_gated",
        False,
    )

    worker = BackgroundGrowthWorker(
        run_root / "queue",
        memory_system.grow,
        min_chars=8,
    )
    await worker.start()
    started = perf_counter()
    pending = await worker.enqueue(
        identity,
        "未花的合作设想与阿里乌斯小队的实际目标之间有什么反差？",
        route,
        assistant_response="前台回答已提出一条等待审计的跨剧情因果候选。",
        memory=memory,
        consolidation_decision=decision,
    )
    if pending is None:
        raise RuntimeError("valid candidate was unexpectedly rejected locally")
    await worker._queue.join()
    elapsed = perf_counter() - started
    await worker.close()

    receipt_path = next((run_root / "queue").glob("*.completed.json"), None)
    receipt = (
        json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt_path is not None
        else {}
    )
    knowledge_result = (
        receipt.get("result", {}).get("domains", {}).get("knowledge", {})
    )
    changed_ids = list(
        dict.fromkeys(
            [
                *knowledge_result.get("new_association_ids", []),
                *knowledge_result.get("reinforced_association_ids", []),
            ]
        )
    )

    # Verify that the durable receipt, not process-local state, suppresses the
    # same candidate after a restart and a completely different chat wording.
    restarted = BackgroundGrowthWorker(
        run_root / "queue",
        memory_system.grow,
        min_chars=8,
    )
    await restarted.start()
    duplicate = await restarted.enqueue(
        identity,
        "换一种问法：未花对阿里乌斯的控制为什么会失效？",
        route,
        assistant_response="回答措辞不同，但候选边与证据没有改变。",
        memory=memory,
        consolidation_decision=decision,
    )
    await restarted.close()

    utility_gate = knowledge_result.get("growth_utility_gate") or {}
    checks = {
        "candidate_admitted_locally": pending is not None,
        "background_job_completed": worker.completed_jobs == 1,
        "background_job_did_not_fail": worker.failed_jobs == 0,
        "fixed_endpoint_dual_audit_used": (
            utility_gate.get("mode") == "fixed_endpoint_dual_audit"
        ),
        "candidate_produced_durable_value": bool(changed_ids),
        "duplicate_blocked_after_restart": duplicate is None,
    }
    report = {
        "version": "candidate-growth-acceptance-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "run_root": str(run_root),
        "knowledge_database": str(knowledge_copy),
        "candidate": decision,
        "premise_episodes": evidence_rows,
        "elapsed_seconds": round(elapsed, 3),
        "growth_result": knowledge_result,
        "changed_associations": _association_rows(knowledge_copy, changed_ids),
        "checks": checks,
        "passed": all(checks.values()),
        "worker_error": worker.last_error,
    }
    report_path = run_root / "report.json"
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
        description="Audit one genuine multi-Episode lore-growth candidate"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "candidate-growth-acceptance",
    )
    args = parser.parse_args()
    result = asyncio.run(run(args.output_root))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
