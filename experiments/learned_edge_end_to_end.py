from __future__ import annotations

import asyncio
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from time import perf_counter

from dotenv import load_dotenv

from src.memory import AssociativeMemoryService, MemorySystemConfig
from src.memory.contracts import DomainRecallRequest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREMISES = (311, 312)
CLAIM = (
    "一段证据记录未花与阿里乌斯学园的合作，另一段记录阿里乌斯学园袭击"
    "伊甸园条约现场；两项观察都涉及阿里乌斯与同一条约危机，此关联只用于"
    "共同检索两项事实，不断言合作导致或直接指挥了袭击。"
)
FOLLOWUP = (
    "是谁袭击伊甸园条约现场，茶会内部哪项合作为这场危机提供了前置背景？"
)


def clone_database(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source)) as source_connection:
        with closing(sqlite3.connect(target)) as target_connection:
            source_connection.backup(target_connection)


def signature(path: Path) -> tuple[int, int]:
    with closing(sqlite3.connect(path)) as connection:
        count, maximum = connection.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM association"
        ).fetchone()
    return int(count), int(maximum)


async def run() -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    formal_before = signature(base.knowledge_database_path)
    root = (
        PROJECT_ROOT
        / "logs"
        / "experiments"
        / f"learned-edge-e2e-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    database = root / "knowledge.db"
    await asyncio.to_thread(clone_database, base.knowledge_database_path, database)

    background_config = base.domain_config(
        "knowledge", database, background=True
    )
    background_config.growth_enabled = True
    background_config.association_cue_enabled = False
    background = AssociativeMemoryService(background_config)
    await background.initialize()
    audit_started = perf_counter()
    audited = await background.audit_candidates(
        [
            {
                "claim": CLAIM,
                "premise_episode_ids": list(PREMISES),
                "confidence": 0.9,
                "inference_type": "evidence_bridge",
            }
        ]
    )
    audit_seconds = perf_counter() - audit_started
    raw_audit = audited.raw_result or {}
    changed_ids = list(
        dict.fromkeys(
            [
                *raw_audit.get("new_association_ids", []),
                *raw_audit.get("reinforced_association_ids", []),
            ]
        )
    )

    foreground_config = base.domain_config(
        "knowledge", database, background=False
    )
    foreground_config.growth_enabled = False
    foreground_config.association_cue_enabled = True
    foreground_config.association_cue_fast_path_enabled = True
    foreground = AssociativeMemoryService(foreground_config)
    await foreground.initialize()
    recall_started = perf_counter()
    recalled = await foreground.recall(
        FOLLOWUP,
        request=DomainRecallRequest(
            query=FOLLOWUP,
            intent_override={"search_queries": [FOLLOWUP]},
            followup_queries=(),
        ),
        retrieval_intensity="standard",
        auto_escalate=False,
    )
    recall_seconds = perf_counter() - recall_started
    raw_recall = recalled.raw_result or {}
    episode_ids = [int(value) for value in raw_recall.get("episode_ids", [])]
    rerank = raw_recall.get("rerank_trace") or {}
    formal_after = signature(base.knowledge_database_path)
    checks = {
        "audit_succeeded": not audited.error,
        "edge_was_written": bool(changed_ids),
        "edge_is_dual_accepted": bool(
            rerank.get("cache_kind") == "audited_association_cue"
        ),
        "followup_used_association_capsule": (
            rerank.get("backend") == "association_capsule"
        ),
        "both_direct_premises_recalled": set(PREMISES).issubset(episode_ids),
        "followup_query_embedding_calls": (
            (raw_recall.get("query_embedding_cache") or {}).get("miss_count", 0)
            == 0
        ),
        "followup_recall_under_five_seconds": recall_seconds < 5.0,
        "formal_database_unchanged": formal_before == formal_after,
    }
    report = {
        "version": "learned-edge-e2e-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "database": str(database),
        "claim": CLAIM,
        "premises": list(PREMISES),
        "changed_association_ids": changed_ids,
        "audit_seconds": round(audit_seconds, 6),
        "audit_error": audited.error,
        "followup": FOLLOWUP,
        "followup_recall_seconds": round(recall_seconds, 6),
        "followup_episode_ids": episode_ids,
        "rerank": rerank,
        "checks": checks,
        "passed": all(checks.values()),
    }
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    report = asyncio.run(run())
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": report["checks"],
                "audit_seconds": report["audit_seconds"],
                "followup_recall_seconds": report["followup_recall_seconds"],
                "changed_association_ids": report["changed_association_ids"],
                "report_path": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
