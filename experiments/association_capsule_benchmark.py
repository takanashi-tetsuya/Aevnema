from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from time import perf_counter

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.memory import AssociativeMemoryService, MemorySystemConfig


FIXTURES = (
    {
        "name": "makeup_club_political_screening",
        "premise_ids": (310, 311),
        "relation_text": (
            "渚以成绩与退学机制为补课部的表面安排，把潜在内鬼集中起来排查；"
            "梓的阿里乌斯出身和破坏任务构成这一政治筛查的关键风险。"
        ),
        "questions": (
            "渚建立补课部真正想排查什么，梓的身份为何让这项安排具有政治意义？",
            "为什么说补课部并非单纯帮助差生，梓的阿里乌斯背景与渚寻找内鬼有什么联系？",
        ),
    },
    {
        "name": "treaty_attack_political_prelude",
        "premise_ids": (311, 312),
        "relation_text": (
            "伊甸园条约袭击由阿里乌斯执行，而未花与阿里乌斯的合作构成茶会内部"
            "政治选择与外部袭击之间的前置联系。"
        ),
        "questions": (
            "条约仪式的阿里乌斯袭击和未花此前的政治合作之间是什么关系？",
            "是谁袭击伊甸园条约现场，茶会内部哪项合作为这场危机提供了前置条件？",
        ),
    },
    {
        "name": "hoshino_trauma_self_sacrifice",
        "premise_ids": (196, 198),
        "relation_text": (
            "星野后来试图独自牺牲以保护后辈的倾向，与梦前辈死亡后形成的创伤、"
            "自责和独自承担责任的模式相连。"
        ),
        "questions": (
            "星野为何总想独自牺牲保护后辈，这与梦前辈的死亡创伤有何联系？",
            "梦的事故怎样影响了星野后来一人承担阿拜多斯危机的选择？",
        ),
    },
)


def _clone_database(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(source)) as source_connection:
        with closing(sqlite3.connect(target)) as target_connection:
            source_connection.backup(target_connection)


def _insert_fixture_edges(database: Path) -> dict[str, int]:
    now = datetime.now(UTC).isoformat()
    result: dict[str, int] = {}
    with closing(sqlite3.connect(database)) as connection:
        next_id = int(
            connection.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM association"
            ).fetchone()[0]
        )
        for offset, fixture in enumerate(FIXTURES):
            edge_id = next_id + offset
            left, right = fixture["premise_ids"]
            evidence = json.dumps(
                [
                    {"node_type": "episode", "node_id": left},
                    {"node_type": "episode", "node_id": right},
                ],
                ensure_ascii=False,
            )
            connection.execute(
                """
                INSERT INTO association(
                    id, from_type, from_id, to_type, to_id,
                    relation_type, relation_key, relation_text, polarity,
                    weight, confidence, generation, evidence_count,
                    claim_level, audit_status, evidence_json, audit_json,
                    created_reason, last_used, use_count, created_at, updated_at
                ) VALUES(?, 'episode', ?, 'episode', ?, ?, ?, ?, 1,
                         0.9, 0.91, 1, 2, 'supported_inference', 'dual_accepted', ?, ?,
                         ?, NULL, 0, ?, ?)
                """,
                (
                    edge_id,
                    left,
                    right,
                    "supports_inference",
                    "cross_episode_inference",
                    fixture["relation_text"],
                    evidence,
                    json.dumps(
                        {"fixture": True, "verdicts": ["accepted", "accepted"]},
                        ensure_ascii=False,
                    ),
                    "查询中自主增长：受控 Association 缓存实验",
                    now,
                    now,
                ),
            )
            result[str(fixture["name"])] = edge_id
        connection.commit()
    return result


def _rank(ids: list[int], target: int) -> int | None:
    try:
        return ids.index(target) + 1
    except ValueError:
        return None


async def _case(
    service: AssociativeMemoryService,
    fixture: dict,
    question: str,
    expected_edge_id: int | None,
) -> dict:
    started = perf_counter()
    recalled = await service.recall(
        question,
        retrieval_intensity="light",
        auto_escalate=False,
        followup_queries_override=[],
    )
    elapsed = perf_counter() - started
    raw = recalled.raw_result or {}
    episode_ids = [int(value) for value in raw.get("episode_ids") or []]
    cue_ids = [int(value) for value in raw.get("association_cue_ids") or []]
    premise_ids = [int(value) for value in fixture["premise_ids"]]
    trace = raw.get("rerank_trace") or {}
    return {
        "question": question,
        "elapsed_seconds": round(elapsed, 3),
        "error": recalled.error,
        "episode_ids": episode_ids,
        "premise_ranks": {
            str(value): _rank(episode_ids, value) for value in premise_ids
        },
        "premise_recall": sum(value in episode_ids for value in premise_ids),
        "expected_edge_id": expected_edge_id,
        "expected_cue_hit": bool(
            expected_edge_id is not None and expected_edge_id in cue_ids
        ),
        "cue_ids": cue_ids,
        "rerank_backend": trace.get("backend"),
        "association_cache_hit": bool(
            trace.get("cache_kind") == "audited_association_cue"
        ),
        "cloud_requests_avoided": int(
            trace.get("cloud_requests_avoided", 0) or 0
        ),
        "retrieval_quality": raw.get("retrieval_quality") or {},
        "timings": raw.get("timings") or {},
    }


async def run(output_dir: Path, *, capsule_only: bool = False) -> dict:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    system = MemorySystemConfig.from_env(PROJECT_ROOT)
    formal = system.knowledge_database_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_db = output_dir / "baseline.db"
    treatment_db = output_dir / "treatment.db"
    _clone_database(formal, baseline_db)
    _clone_database(formal, treatment_db)
    edge_ids = _insert_fixture_edges(treatment_db)

    base_config = system.domain_config("knowledge", baseline_db)
    base_config = replace(
        base_config,
        database_path=baseline_db,
        log_dir=output_dir / "baseline-logs",
        growth_enabled=False,
        association_cue_enabled=False,
        association_cue_fast_path_enabled=False,
        recall_cache_size=0,
    )
    graph_config = replace(
        base_config,
        database_path=treatment_db,
        log_dir=output_dir / "graph-logs",
    )
    cue_config = replace(
        graph_config,
        log_dir=output_dir / "cue-logs",
        association_cue_enabled=True,
        association_cue_fast_path_enabled=False,
    )
    capsule_config = replace(
        cue_config,
        log_dir=output_dir / "capsule-logs",
        association_cue_fast_path_enabled=True,
    )
    services = {
        "baseline": AssociativeMemoryService(base_config),
        "graph_only": AssociativeMemoryService(graph_config),
        "cue_bge": AssociativeMemoryService(cue_config),
        "cue_capsule": AssociativeMemoryService(capsule_config),
    }
    if capsule_only:
        services = {"cue_capsule": services["cue_capsule"]}
    for service in services.values():
        await service.initialize()

    cases: list[dict] = []
    for fixture in FIXTURES:
        edge_id = edge_ids[str(fixture["name"])]
        for question in fixture["questions"]:
            arms = {}
            for name, service in services.items():
                arms[name] = await _case(
                    service,
                    fixture,
                    question,
                    edge_id if name in {"cue_bge", "cue_capsule"} else None,
                )
            cases.append(
                {
                    "fixture": fixture["name"],
                    "premise_ids": fixture["premise_ids"],
                    "edge_id": edge_id,
                    "arms": arms,
                }
            )

    def arm_values(name: str, key: str) -> list[float]:
        return [float(item["arms"][name][key]) for item in cases]

    metrics = {}
    for name in services:
        elapsed = arm_values(name, "elapsed_seconds")
        recalls = arm_values(name, "premise_recall")
        metrics[name] = {
            "mean_seconds": round(sum(elapsed) / len(elapsed), 3),
            "max_seconds": round(max(elapsed), 3),
            "mean_premise_recall": round(sum(recalls) / len(recalls), 3),
            "full_premise_recall_rate": round(
                sum(value == 2 for value in recalls) / len(recalls), 3
            ),
            "association_cache_hit_rate": round(
                sum(
                    bool(item["arms"][name]["association_cache_hit"])
                    for item in cases
                )
                / len(cases),
                3,
            ),
        }
    capsule_completed = all(
        not case["arms"]["cue_capsule"]["error"] for case in cases
    )
    checks = {
        "capsule_all_completed": capsule_completed,
        "capsule_used": all(
            case["arms"]["cue_capsule"]["association_cache_hit"]
            for case in cases
        ),
        "capsule_full_premise_recall": all(
            case["arms"]["cue_capsule"]["premise_recall"] == 2
            for case in cases
        ),
        "capsule_under_five_seconds": all(
            case["arms"]["cue_capsule"]["elapsed_seconds"] < 5.0
            for case in cases
        ),
    }
    if not capsule_only:
        checks.update(
            {
                "all_completed": all(
                    not arm["error"]
                    for case in cases
                    for arm in case["arms"].values()
                ),
                "capsule_never_reduces_premise_recall_vs_bge": all(
                    case["arms"]["cue_capsule"]["premise_recall"]
                    >= case["arms"]["cue_bge"]["premise_recall"]
                    for case in cases
                ),
                "capsule_faster_than_cue_bge": (
                    metrics["cue_capsule"]["mean_seconds"]
                    < metrics["cue_bge"]["mean_seconds"]
                ),
            }
        )
    report = {
        "version": "association-capsule-benchmark-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "formal_database_untouched": str(formal),
        "edge_ids": edge_ids,
        "cases": cases,
        "metrics": metrics,
        "checks": checks,
    }
    report["passed"] = all(report["checks"].values())
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "logs"
            / "experiments"
            / f"association-capsule-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ),
    )
    parser.add_argument("--capsule-only", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        run(args.output_dir.resolve(), capsule_only=args.capsule_only)
    )
    print(json.dumps({"metrics": result["metrics"], "checks": result["checks"], "report_path": result["report_path"]}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
