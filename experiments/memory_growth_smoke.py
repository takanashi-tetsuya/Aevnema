from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from dotenv import load_dotenv

from src.memory import (
    BackgroundGrowthWorker,
    MemoryRoute,
    MemorySystem,
    MemorySystemConfig,
    PlatformIdentity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION = (
    "伊甸园条约签订仪式遭到袭击时，古圣堂的大爆炸由哪个分校势力直接执行？"
    "他们能够参与这场危机，与哪位茶会高层此前暗中支援阿里乌斯有什么关系？"
)

# A retrieval test must score supported facts, not one arbitrarily preferred
# row. Several independently extracted Episodes can prove the same hop. In
# particular #1421 names both the order giver and the Arius squad, while #1595
# describes the missile launch with a pronoun whose referent lives nearby.
KEY_EVIDENCE = {
    "explosion_scene": {1481, 1528},
    "arius_direct_execution": {1421, 1518, 1545, 1595},
    "mika_support": {1166, 1167, 1360, 1421},
}


def _sqlite_backup(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{source.as_posix()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source_connection:
        with sqlite3.connect(target) as target_connection:
            source_connection.backup(target_connection)


async def run(output_root: Path, question: str) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_root = (output_root / run_id).resolve()
    knowledge_copy = run_root / "knowledge" / "memory.db"
    await asyncio.to_thread(
        _sqlite_backup, base.knowledge_database_path, knowledge_copy
    )

    config = MemorySystemConfig(
        engine_root=base.engine_root,
        knowledge_database_path=knowledge_copy,
        public_database_path=run_root / "public" / "memory.db",
        user_database_dir=run_root / "users",
        log_dir=run_root / "logs",
        api_key=base.api_key,
        optimization_profile="balanced",
        foreground_profile="balanced",
        foreground_review_mode="fast_adaptive",
        paragraph_enabled=base.paragraph_enabled,
        concept_profile=base.concept_profile,
        context_max_chars=base.context_max_chars,
        knowledge_growth_enabled=True,
        public_growth_enabled=False,
        user_growth_enabled=False,
    )
    memory = MemorySystem(config)
    await memory.initialize()
    worker = BackgroundGrowthWorker(
        run_root / "queue",
        memory.grow,
        min_chars=8,
    )
    await worker.start()
    identity = PlatformIdentity("diagnostic", "background-growth-smoke")
    route = MemoryRoute(False, False, True, "background_growth_smoke")

    total_started = perf_counter()
    pending = await worker.enqueue(identity, question, route)
    if pending is None:
        raise RuntimeError("growth smoke question was not admitted to the queue")
    for _ in range(300):
        if worker.active_job_id:
            break
        await asyncio.sleep(0.1)
    if not worker.active_job_id:
        raise RuntimeError("background growth job did not start")

    foreground_started = perf_counter()
    foreground = await memory.recall(identity, question, route=route)
    foreground_seconds = perf_counter() - foreground_started
    background_still_running = bool(worker.active_job_id)
    await worker._queue.join()
    total_seconds = perf_counter() - total_started
    await worker.close()

    completed_files = sorted((run_root / "queue").glob("*.completed.json"))
    completed_payload = (
        json.loads(completed_files[-1].read_text(encoding="utf-8"))
        if completed_files
        else {}
    )
    raw_knowledge = (foreground.raw_result.get("domains") or {}).get(
        "knowledge", {}
    )
    final_ids = list(raw_knowledge.get("episode_ids") or [])
    final_id_set = set(final_ids)
    key_chain = {
        name: sorted(final_id_set & accepted_ids)
        for name, accepted_ids in KEY_EVIDENCE.items()
    }
    growth_domains = (
        completed_payload.get("result", {}).get("domains", {})
        if completed_payload
        else {}
    )
    checks = {
        "foreground_has_complete_key_chain": all(key_chain.values()),
        "foreground_completed_before_background": background_still_running,
        "background_job_completed": worker.completed_jobs == 1,
        "background_job_did_not_fail": worker.failed_jobs == 0,
        "public_domain_was_not_written": "public" not in growth_domains,
    }
    result = {
        "run_id": run_id,
        "assessment_version": "v2_equivalent_direct_evidence",
        "run_root": str(run_root),
        "question": question,
        "foreground_seconds": round(foreground_seconds, 3),
        "foreground_completed_before_background": background_still_running,
        "total_seconds": round(total_seconds, 3),
        "foreground_episode_ids": final_ids,
        "foreground_key_chain": key_chain,
        "background_result": completed_payload.get("result", {}),
        "checks": checks,
        "passed": all(checks.values()),
        "errors": {
            "foreground": foreground.error,
            "background": worker.last_error,
        },
    }
    report_path = run_root / "report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    result["report_path"] = str(report_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that foreground recall is not blocked by deep growth"
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "growth-smoke",
    )
    args = parser.parse_args()
    result = asyncio.run(run(args.output_root, args.question))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
