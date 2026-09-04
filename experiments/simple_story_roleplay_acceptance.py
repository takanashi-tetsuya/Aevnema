from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import sys
from time import perf_counter

from dotenv import load_dotenv

from src.bot.chat_service import ConversationCoordinator
from src.bot.memory_guard import PrivateMemoryResponseGuard
from src.bot.telegram_adapter import (
    get_dynamic_system_prompt,
    get_private_memory_emergency_reply,
)
from src.llm.engine import EngineFactory, Message
from src.memory import (
    BackgroundGrowthWorker,
    MemoryRoute,
    MemorySystem,
    MemorySystemConfig,
    PlatformIdentity,
)
from src.memory.conversation import (
    ConversationIngestionWorker,
    ConversationJournal,
    ConversationSessionBuffer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _route_dict(route: MemoryRoute) -> dict:
    return {
        "user": route.user,
        "public": route.public,
        "knowledge": route.knowledge,
        "reason": route.reason,
        "intensity": route.intensity,
        "knowledge_write_policy": route.knowledge_write_policy,
        "creative": route.creative,
    }


def _memory_summary(memory) -> dict:
    domains: dict[str, dict] = {}
    for domain, result in (memory.raw_result.get("domains") or {}).items():
        rows = list(result.get("evidence_episodes") or [])
        domains[domain] = {
            "episode_ids": list(result.get("episode_ids") or []),
            "concept_ids": list(result.get("concept_ids") or []),
            "association_ids": list(result.get("association_ids") or []),
            "retrieval_intensity": result.get("retrieval_intensity"),
            "lightweight_fast_path": bool(
                result.get("lightweight_fast_path")
            ),
            "service_cache_hit": bool(result.get("service_cache_hit")),
            "timings": result.get("timings") or {},
            "evidence_episodes": [
                {
                    "id": row.get("id"),
                    "source_key": row.get("source_key"),
                    "text": " ".join(str(row.get("text", "")).split())[:700],
                    "origin": row.get("evidence_origin"),
                    "status": row.get("epistemic_status"),
                    "generation": row.get("generation"),
                }
                for row in rows[:10]
            ],
        }
    return {
        "available": memory.available,
        "error": memory.error,
        "domains": domains,
    }


def _count_snapshot(stats: dict) -> dict:
    associations = stats.get("associations") or {}
    return {
        "sources": int(stats.get("sources", 0)),
        "episodes": int(stats.get("episodes", 0)),
        "concepts": int(stats.get("concepts", 0)),
        "paragraphs": int(stats.get("paragraphs", 0)),
        "association_edges": int(associations.get("edges", 0)),
        "negative_associations": int(associations.get("negative", 0)),
    }


async def _turn(
    coordinator: ConversationCoordinator,
    identity: PlatformIdentity,
    *,
    label: str,
    question: str,
    kind: str,
    deferred_growth: list[dict] | None = None,
) -> dict:
    started = perf_counter()
    reply = await coordinator.handle(identity=identity, text=question)
    elapsed = perf_counter() - started
    if deferred_growth is not None:
        deferred_growth.append(
            {
                "identity": identity,
                "question": question,
                "route": reply.route,
                "assistant_response": reply.text,
                "memory": reply.memory,
                "consolidation_decision": reply.memory_consolidation,
            }
        )
    return {
        "label": label,
        "kind": kind,
        "question": question,
        "assistant": reply.text,
        "route": _route_dict(reply.route),
        "latency_seconds": round(elapsed, 3),
        "under_five_seconds": elapsed < 5.0,
        "timings": reply.timings,
        "memory": _memory_summary(reply.memory),
        "memory_guard": reply.memory_guard,
        "memory_consolidation": reply.memory_consolidation,
    }


def _source_rows(database_path: Path) -> list[str]:
    with sqlite3.connect(database_path) as connection:
        return [
            str(row[0])
            for row in connection.execute("SELECT raw_text FROM source ORDER BY id")
        ]


def _conversation_contents(source_rows: list[str]) -> list[str]:
    values: list[str] = []
    marker = "conversation_record: "
    for source in source_rows:
        for line in source.splitlines():
            if marker not in line:
                continue
            try:
                payload = json.loads(line.split(marker, 1)[1])
            except (json.JSONDecodeError, IndexError):
                continue
            content = payload.get("content") if isinstance(payload, dict) else None
            if isinstance(content, str):
                values.append(content)
    return values


def _association_max_id(database_path: Path) -> int:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT COALESCE(MAX(id), 0) FROM association").fetchone()
    return int(row[0] if row else 0)


def _association_rows(database_path: Path, association_ids: list[int]) -> list[dict]:
    if not association_ids:
        return []
    placeholders = ",".join("?" for _value in association_ids)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, from_type, from_id, to_type, to_id, relation_type, "
            "relation_key, relation_text, weight, confidence, generation, "
            "claim_level, audit_status, evidence_json, created_reason "
            f"FROM association WHERE id IN ({placeholders}) ORDER BY id",
            association_ids,
        ).fetchall()
    return [dict(row) for row in rows]


def _deterministic_checks(
    turns: list[dict],
    *,
    knowledge_before: dict,
    knowledge_after: dict,
    private_sources: list[str],
    private_recall: dict,
    growth_summary: dict,
) -> dict[str, dict]:
    by_label = {item["label"]: item for item in turns}
    all_answers = "\n".join(item["assistant"] for item in turns)
    stored_text = "\n".join(private_sources)
    stored_contents = _conversation_contents(private_sources)
    direct = by_label["direct_story_qa"]
    opinion = by_label["indirect_character_opinion"]
    trip = by_label["implicit_trip_context"]
    tasks = by_label["creative_tasks"]
    messages = by_label["creative_messages"]
    creative_answers = (tasks["assistant"], messages["assistant"])
    local_knowledge_candidates = sum(
        len(item.get("memory_consolidation", {}).get("knowledge_candidates", []))
        for item in turns
    )
    local_private_candidates = sum(
        len(item.get("memory_consolidation", {}).get("private_candidates", []))
        for item in turns
    )
    private_evidence = (
        private_recall.get("domains", {})
        .get("user", {})
        .get("evidence_episodes", [])
    )
    evidence_text = "\n".join(
        str(row.get("text", "")) for row in private_evidence
    )
    return {
        "all_turns_use_light_route": {
            "passed": all(
                item["route"]["intensity"] == "light" for item in turns
            ),
            "detail": "五个简单会话均由轻量检索路径处理",
        },
        "light_path_reached_knowledge": {
            "passed": all(
                item["memory"]["domains"]
                .get("knowledge", {})
                .get("lightweight_fast_path")
                for item in turns
            ),
            "detail": "剧情域实际执行了无规划、无精排的快速召回",
        },
        "direct_story_answer": {
            "passed": "白子" in direct["assistant"]
            and any(
                marker in direct["assistant"]
                for marker in ("骑车", "自行车", "公路车")
            )
            and any(marker in direct["assistant"] for marker in ("支持", "感动", "触动")),
            "detail": "直接问题回答白子的兴趣，以及老师成为支持者后的反应",
        },
        "indirect_character_answer": {
            "passed": "白子" in opinion["assistant"]
            and len(opinion["assistant"].strip()) >= 35,
            "detail": "没有显式说‘剧情检索’时仍结合背景评价白子",
        },
        "implicit_trip_uses_world_context": {
            "passed": "阿拜多斯" in trip["assistant"]
            and any(
                marker in trip["assistant"]
                for marker in ("沙漠", "炎热", "高温", "对策委员会", "白子", "债")
            ),
            "detail": "出差陈述触发阿拜多斯背景而非纯寒暄",
        },
        "creative_answers_are_substantive": {
            "passed": all(len(answer.strip()) >= 45 for answer in creative_answers)
            and any(
                marker in messages["assistant"]
                for marker in ("白子", "星野", "日奈", "日富美", "学生")
            ),
            "detail": "任务与学生消息都给出可继续互动的新内容",
        },
        "creative_routes_use_evidence_gated_growth": {
            "passed": all(
                item["route"]["creative"]
                and item["route"]["knowledge_write_policy"]
                == "evidence_gated"
                for item in (trip, tasks, messages)
            ),
            "detail": "创造性回合不锁死剧情增长，但共享写入必须经过命题级证据门",
        },
        "knowledge_nodes_not_polluted": {
            "passed": all(
                knowledge_before[name] == knowledge_after[name]
                for name in ("sources", "episodes", "concepts", "paragraphs")
            ),
            "detail": "角色扮演没有向剧情库写入 Source/Episode/Concept；只允许审计后的 Association 变化",
        },
        "answer_claims_were_classified": {
            "passed": growth_summary["completed_jobs"]
            == growth_summary["enqueued_jobs"]
            and growth_summary["failed_jobs"] == 0
            and local_private_candidates >= 3,
            "detail": "同一次回答已完成本地命题分流；临时任务/消息被识别为私人候选",
        },
        "knowledge_growth_is_inferred_and_audited": {
            "passed": local_knowledge_candidates >= 1
            and all(
                int(row.get("generation", 0)) >= 1
                and row.get("audit_status") == "dual_accepted"
                and bool(str(row.get("evidence_json", "")).strip())
                for row in growth_summary["changed_knowledge_associations"]
            ),
            "detail": "回答内产生了有前景的剧情推论；若核心接受写入，则必须具备 generation、双重审计和前提证据",
        },
        "transient_roleplay_not_written_as_lore": {
            "passed": not any(
                marker in str(row.get("relation_text", ""))
                for row in growth_summary["changed_knowledge_associations"]
                for marker in (
                    "今天发来消息",
                    "刚刚发来消息",
                    "当前出差任务",
                    "老师今天",
                )
            ),
            "detail": "新剧情关系文本没有把本轮临时消息或任务冒充原作事实",
        },
        "creative_dialogue_saved_privately": {
            "passed": bool(private_sources)
            and all(answer in stored_contents for answer in creative_answers)
            and "[evidence_origin: system]" in stored_text
            and "[evidence_generation: 1]" in stored_text,
            "detail": "创造内容原样进入该虚拟用户的私人 Source，并标为系统生成、generation=1",
        },
        "private_memory_retrievable_after_import": {
            "passed": bool(private_evidence)
            and any(
                marker in evidence_text
                for marker in ("出差", "任务", "消息", "学生")
            ),
            "detail": "导入后用仅私人域检索能重新命中刚才的角色扮演内容",
        },
        "no_internal_structure_leak": {
            "passed": not any(
                marker in all_answers
                for marker in (
                    "Episode #",
                    "Association #",
                    "source_key",
                    "knowledge_write_policy",
                    "数据库 ID",
                )
            ),
            "detail": "用户可见回答不暴露内部检索与存储字段",
        },
    }


async def _judge(turns: list[dict]) -> dict:
    cases = []
    for item in turns:
        evidence = (
            item["memory"]["domains"]
            .get("knowledge", {})
            .get("evidence_episodes", [])
        )
        cases.append(
            {
                "label": item["label"],
                "kind": item["kind"],
                "question": item["question"],
                "assistant": item["assistant"],
                "retrieved_evidence": evidence,
            }
        )
    schema = {
        "case_scores": [
            {
                "label": "",
                "background_use": 0,
                "answer_relevance": 0,
                "roleplay_naturalness": 0,
                "fact_boundary": 0,
                "problems": [],
            }
        ],
        "overall_score": 0,
        "overall_pass": False,
        "summary": "",
    }
    prompt = (
        "你是严格的蔚蓝档案角色扮演问答验收员。对每个案例按0到5分评价："
        "background_use（是否真正使用所给背景）、answer_relevance、roleplay_naturalness、"
        "fact_boundary。直接/间接知识回答必须受 retrieved_evidence 支持；创造性任务或消息"
        "允许新增当前角色扮演内容，只要不把新增内容冒充原作既定剧情，且与检索到的世界观"
        "不明显矛盾。回答者固定扮演阿洛娜，称呼用户为老师；这是系统人格，不需要由本轮"
        "retrieved_evidence 再次证明，不得因此扣分。不要因为合理创造内容不在证据中而扣分。"
        "overall_pass 要求每个案例四项均>=3，"
        "整体平均>=4，且没有明显剧情事实错误。只输出合法 JSON。\n"
        f"期望结构：{json.dumps(schema, ensure_ascii=False)}\n"
        f"案例：{json.dumps(cases, ensure_ascii=False)}"
    )
    response = await EngineFactory.create("extract").generate_response(
        [Message(role="user", content=prompt)],
        system_prompt="只能依据提供的问题、回答和检索证据评分，不得自行补剧情。",
        task_context="simple-story-roleplay:judge",
    )
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {"parse_error": response}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"parse_error": response}


async def run(output_root: Path, *, foreground_only: bool = False) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    output_root.mkdir(parents=True, exist_ok=True)

    config = MemorySystemConfig.from_env(PROJECT_ROOT)
    source_knowledge_database = config.knowledge_database_path
    experimental_knowledge_database = (
        output_root / "knowledge" / "blue_archive.db"
    )
    experimental_knowledge_database.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_knowledge_database, experimental_knowledge_database)
    config.knowledge_database_path = experimental_knowledge_database
    config.public_database_path = output_root / "public" / "memory.db"
    config.user_database_dir = output_root / "users"
    config.log_dir = output_root / "memory-logs"
    config.foreground_recall_cache_size = 0
    config.knowledge_growth_enabled = True
    config.public_growth_enabled = False
    config.user_growth_enabled = False

    memory = MemorySystem(config)
    initialization_started = perf_counter()
    await memory.initialize()
    initialization_seconds = round(perf_counter() - initialization_started, 3)
    knowledge_before = _count_snapshot((await memory.stats())["knowledge"])
    association_max_before = _association_max_id(
        config.knowledge_database_path
    )

    identity = PlatformIdentity("synthetic-chat", "simple-teacher-01", "老师")
    journal = ConversationJournal(
        output_root / "conversation-inbox",
        batch_exchanges=20,
        batch_chars=100_000,
    )
    ingestion = ConversationIngestionWorker(
        journal, memory.import_conversation_file
    )
    await ingestion.start()
    coordinator = ConversationCoordinator(
        chat_engine=EngineFactory.create("chat"),
        fast_chat_engine=EngineFactory.create("chat_fast"),
        memory_system=memory,
        system_prompt_factory=get_dynamic_system_prompt,
        private_memory_guard=PrivateMemoryResponseGuard(
            audit_engine=EngineFactory.create("memory_audit"),
            rewrite_engine=EngineFactory.create("memory_rewrite"),
            emergency_reply_factory=get_private_memory_emergency_reply,
        ),
        sessions=ConversationSessionBuffer(20),
        journal=journal,
        ingestion_worker=ingestion,
    )
    specifications = (
        (
            "direct_story_qa",
            "direct_knowledge",
            "白子喜欢什么活动？当老师说自己是她的支持者后，她有什么反应？",
        ),
        (
            "indirect_character_opinion",
            "indirect_knowledge",
            "你觉得白子是一个什么样的人？",
        ),
        (
            "implicit_trip_context",
            "implicit_creative",
            "我现在打算去阿拜多斯出差了。",
        ),
        (
            "creative_tasks",
            "creative_roleplay",
            "现在有哪些出差任务？",
        ),
        (
            "creative_messages",
            "creative_roleplay",
            "有哪些学生给为师发消息了？",
        ),
    )
    turns: list[dict] = []
    deferred_growth: list[dict] = []
    for label, kind, question in specifications:
        turns.append(
            await _turn(
                coordinator,
                identity,
                label=label,
                question=question,
                kind=kind,
                deferred_growth=deferred_growth,
            )
        )

    latencies = [float(item["latency_seconds"]) for item in turns]
    latency = {
        "target_seconds": 5.0,
        "count": len(latencies),
        "under_target_count": sum(value < 5.0 for value in latencies),
        "under_target_rate": round(
            sum(value < 5.0 for value in latencies) / len(latencies), 3
        ),
        "minimum_seconds": min(latencies),
        "median_seconds": round(statistics.median(latencies), 3),
        "maximum_seconds": max(latencies),
        "target_met": all(value < 5.0 for value in latencies),
    }
    foreground_checkpoint = {
        "version": "simple-story-roleplay-foreground-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "initialization_seconds": initialization_seconds,
        "turns": turns,
        "latency": latency,
    }
    (output_root / "foreground-checkpoint.json").write_text(
        json.dumps(foreground_checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if foreground_only:
        await ingestion.close(flush_pending=False)
        return {
            **foreground_checkpoint,
            "foreground_only": True,
            "functional_pass": False,
            "checks": {},
            "llm_judge": {"skipped": "foreground-only run"},
        }

    ingestion_started = perf_counter()
    await ingestion.close(flush_pending=True)
    ingestion_seconds = round(perf_counter() - ingestion_started, 3)

    growth_worker = BackgroundGrowthWorker(
        output_root / "growth-queue",
        memory.grow,
        min_chars=1,
    )
    await growth_worker.start()
    growth_started = perf_counter()
    enqueued_jobs = 0
    for item in deferred_growth:
        queued = await growth_worker.enqueue(
            item["identity"],
            item["question"],
            item["route"],
            memory=item["memory"],
            consolidation_decision=item["consolidation_decision"],
        )
        if queued is not None:
            enqueued_jobs += 1
    await growth_worker._queue.join()
    await growth_worker.close()
    growth_seconds = round(perf_counter() - growth_started, 3)

    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "growth-queue").glob("*.completed.json"))
    ]
    changed_ids = sorted(
        {
            int(value)
            for receipt in receipts
            for value in (
                receipt.get("result", {})
                .get("domains", {})
                .get("knowledge", {})
                .get("new_association_ids", [])
                + receipt.get("result", {})
                .get("domains", {})
                .get("knowledge", {})
                .get("reinforced_association_ids", [])
            )
        }
    )
    changed_rows = _association_rows(config.knowledge_database_path, changed_ids)
    growth_summary = {
        "enqueued_jobs": enqueued_jobs,
        "completed_jobs": growth_worker.completed_jobs,
        "failed_jobs": growth_worker.failed_jobs,
        "last_error": growth_worker.last_error,
        "knowledge_candidate_count": sum(
            len(receipt.get("consolidation", {}).get("knowledge_candidates", []))
            for receipt in receipts
        ),
        "private_candidate_count": sum(
            len(receipt.get("consolidation", {}).get("private_candidates", []))
            for receipt in receipts
        ),
        "new_knowledge_association_ids": sorted(
            value for value in changed_ids if value > association_max_before
        ),
        "changed_knowledge_association_ids": changed_ids,
        "changed_knowledge_associations": changed_rows,
        "receipts": receipts,
    }

    stats_after = await memory.stats(identity)
    knowledge_after = _count_snapshot(stats_after["knowledge"])
    user_database = (
        config.user_database_dir
        / identity.platform
        / identity.storage_key
        / "memory.db"
    )
    private_sources = _source_rows(user_database)
    private_memory = await memory.recall(
        identity,
        "刚才为老师安排了哪些任务，又有哪些学生发来了消息？",
        route=MemoryRoute(
            True,
            False,
            False,
            "private_post_import_verification",
            "light",
        ),
    )
    private_recall = _memory_summary(private_memory)

    checks = _deterministic_checks(
        turns,
        knowledge_before=knowledge_before,
        knowledge_after=knowledge_after,
        private_sources=private_sources,
        private_recall=private_recall,
        growth_summary=growth_summary,
    )
    judge = await _judge(turns)
    deterministic_pass = all(item["passed"] for item in checks.values())
    return {
        "version": "simple-story-roleplay-acceptance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "identity": {
            "key": identity.key,
            "storage_key": identity.storage_key,
        },
        "knowledge_database": str(config.knowledge_database_path),
        "source_knowledge_database": str(source_knowledge_database),
        "isolated_user_database": str(user_database),
        "initialization_seconds": initialization_seconds,
        "conversation_ingestion_seconds": ingestion_seconds,
        "background_consolidation_and_growth_seconds": growth_seconds,
        "turns": turns,
        "latency": latency,
        "knowledge_counts_before": knowledge_before,
        "knowledge_counts_after": knowledge_after,
        "private_stats": _count_snapshot(stats_after["user"]),
        "private_source_count": len(private_sources),
        "private_recall_after_import": private_recall,
        "growth": growth_summary,
        "checks": checks,
        "deterministic_pass": deterministic_pass,
        "llm_judge": judge,
        "functional_pass": bool(
            deterministic_pass and judge.get("overall_pass") is True
        ),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--foreground-only", action="store_true")
    args = parser.parse_args()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else PROJECT_ROOT / "logs" / "simple-story-roleplay" / _utc_stamp()
    )
    result = asyncio.run(run(output_root, foreground_only=args.foreground_only))
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "functional_pass": result["functional_pass"],
                "latency": result["latency"],
                "checks": result["checks"],
                "llm_judge": result["llm_judge"],
                "turns": [
                    {
                        "label": item["label"],
                        "latency_seconds": item["latency_seconds"],
                        "timings": item["timings"],
                        "route": item["route"],
                        "assistant": item["assistant"],
                    }
                    for item in result["turns"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
