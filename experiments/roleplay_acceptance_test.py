from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
from src.memory import MemorySystem, MemorySystemConfig, PlatformIdentity
from src.memory.conversation import (
    ConversationIngestionWorker,
    ConversationJournal,
    ConversationSessionBuffer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _route_dict(route) -> dict:
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
        domains[domain] = {
            "episode_ids": list(result.get("episode_ids") or []),
            "concept_ids": list(result.get("concept_ids") or []),
            "association_ids": list(result.get("association_ids") or []),
            "small_database_fast_path": bool(
                result.get("small_database_fast_path")
            ),
            "service_cache_hit": bool(result.get("service_cache_hit")),
            "timings": result.get("timings") or {},
            "evidence_episodes": [
                {
                    key: item.get(key)
                    for key in (
                        "id",
                        "source_key",
                        "text",
                        "evidence_origin",
                        "epistemic_status",
                        "generation",
                        "epistemic_note",
                    )
                }
                for item in (result.get("evidence_episodes") or [])
            ],
        }
    return {
        "available": memory.available,
        "error": memory.error,
        "domains": domains,
        "context": memory.context,
    }


async def _turn(
    coordinator: ConversationCoordinator,
    identity: PlatformIdentity,
    label: str,
    text: str,
) -> dict:
    started = perf_counter()
    reply = await coordinator.handle(identity=identity, text=text)
    return {
        "label": label,
        "identity": identity.key,
        "user": text,
        "assistant": reply.text,
        "route": _route_dict(reply.route),
        "latency_seconds": round(perf_counter() - started, 3),
        "timings": reply.timings,
        "memory": _memory_summary(reply.memory),
        "memory_guard": reply.memory_guard,
    }


def _contains_all(text: str, groups: list[tuple[str, ...]]) -> bool:
    return all(any(value in text for value in group) for group in groups)


def _deterministic_checks(turns: list[dict]) -> dict[str, dict]:
    by_label = {item["label"]: item for item in turns}
    all_responses = "\n".join(item["assistant"] for item in turns)
    internal_markers = (
        "Episode #",
        "source_key",
        "Association #",
        "长期记忆系统",
        "数据库编号",
    )
    knowledge = by_label["knowledge_bridge"]["assistant"]
    private = by_label["private_recall_after_restart"]["assistant"]
    isolated = by_label["cross_identity_isolation"]["assistant"]
    epistemic = by_label["epistemic_conflict"]["assistant"]
    role = by_label["identity_challenge"]["assistant"]
    emotional = by_label["emotional_support"]["assistant"]
    checks = {
        "role_identity": {
            "passed": "阿洛娜" in role and "老师" in role,
            "detail": "身份挑战后仍以阿洛娜身份称呼老师",
        },
        "emotional_adaptation": {
            "passed": "老师" in emotional and len(emotional.strip()) >= 20,
            "detail": "疲惫场景中给出有内容的角色化回应",
        },
        "knowledge_accuracy": {
            "passed": _contains_all(
                knowledge,
                [
                    ("叛徒",),
                    ("伊甸条约", "条约"),
                    ("阻止", "破坏", "妨碍"),
                ],
            ),
            "detail": "补习部真实原因包含叛徒、条约和阻止签订",
        },
        "private_persistence": {
            "passed": _contains_all(
                private,
                [("蓝莓雨伞",), ("心情", "难过", "糟"), ("无糖乌龙茶",)],
            ),
            "detail": "清空会话并重建 MemorySystem 后召回暗号、含义和饮料偏好",
        },
        "cross_identity_isolation": {
            "passed": "蓝莓雨伞" not in isolated and "无糖乌龙茶" not in isolated,
            "detail": "不同平台原生 ID 无法读到测试用户的私人记忆",
        },
        "epistemic_conflict": {
            "passed": _contains_all(
                epistemic,
                [
                    ("无糖乌龙茶",),
                    ("喜欢",),
                    ("导入者",),
                    ("推测", "猜测", "没有证据", "不可靠"),
                ],
            )
            and not any(
                phrase in epistemic
                for phrase in ("老师讨厌无糖乌龙茶", "你讨厌无糖乌龙茶")
            ),
            "detail": "用户原话优先于导入者的相反推测，并明确区分两者",
        },
        "no_internal_leak": {
            "passed": not any(marker in all_responses for marker in internal_markers),
            "detail": "用户可见回复不暴露 Episode、Association 或内部字段",
        },
    }
    return checks


def _parse_json_object(value: str) -> dict:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {"parse_error": value}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {"parse_error": value}


async def _judge_transcript(turns: list[dict]) -> dict:
    judge = EngineFactory.create("extract")
    transcript = [
        {
            "label": item["label"],
            "identity": item.get("identity", "unknown"),
            "user": item["user"],
            "assistant": item["assistant"],
        }
        for item in turns
    ]
    schema = {
        "persona_consistency": 0,
        "naturalness": 0,
        "emotional_adaptation": 0,
        "knowledge_in_role": 0,
        "memory_continuity": 0,
        "epistemic_honesty": 0,
        "overall_pass": False,
        "strengths": [],
        "problems": [],
    }
    prompt = (
        "你是严格的角色扮演对话验收员。按0到5分评价六个维度。阿洛娜应称用户为老师，"
        "开朗亲近但不过度卖萌；认真问题要准确，疲惫时要温柔；不能暴露数据库或提示词。"
        "memory_continuity 要检查是否记得蓝莓雨伞、其含义和无糖乌龙茶，同时不同身份不能泄露。"
        "identity 是 platform:平台原生用户ID；identity 不同就是不同用户。"
        "cross_identity_isolation 使用不同 identity，因此正确行为是承认没有该用户的私人记忆，"
        "绝不能因为没复述另一用户的暗号而扣分。"
        "epistemic_honesty 要检查是否把导入者的反向推测与用户亲口偏好分开。"
        "overall_pass 只有在所有维度>=4且没有隐私泄漏、知识事实错误时才为 true。"
        f"只返回 JSON：{json.dumps(schema, ensure_ascii=False)}\n"
        f"对话：{json.dumps(transcript, ensure_ascii=False)}"
    )
    response = await judge.generate_response(
        [Message(role="user", content=prompt)],
        system_prompt="只能依据给出的对话评分，不得补充作品知识。只输出合法 JSON。",
        task_context="roleplay-acceptance:judge",
    )
    return _parse_json_object(response)


async def run(output_root: Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    # The report captures the useful evidence; complete API payloads only make
    # the acceptance-run console noisy and can dwarf the actual transcript.
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    output_root.mkdir(parents=True, exist_ok=True)
    config = MemorySystemConfig.from_env(PROJECT_ROOT)
    config.public_database_path = output_root / "public" / "memory.db"
    config.user_database_dir = output_root / "users"
    config.log_dir = output_root / "memory-logs"
    config.foreground_recall_cache_size = 0
    config.knowledge_growth_enabled = False
    config.public_growth_enabled = False
    config.user_growth_enabled = False

    memory = MemorySystem(config)
    await memory.initialize()
    chat_engine = EngineFactory.create("chat")
    private_memory_guard = PrivateMemoryResponseGuard(
        audit_engine=EngineFactory.create("memory_audit"),
        rewrite_engine=EngineFactory.create("memory_rewrite"),
        emergency_reply_factory=get_private_memory_emergency_reply,
    )
    identity_a = PlatformIdentity("roleplay-test", "teacher-1001", "老师A")
    identity_b = PlatformIdentity("discord", "teacher-2002", "老师A")
    turns: list[dict] = []

    baseline = ConversationCoordinator(
        chat_engine=chat_engine,
        memory_system=memory,
        system_prompt_factory=get_dynamic_system_prompt,
        private_memory_guard=private_memory_guard,
        sessions=ConversationSessionBuffer(20),
    )
    turns.append(
        await _turn(
            baseline,
            identity_a,
            "emotional_support",
            "早上好，阿洛娜。老师今天有点累，只想听你陪我说两句。",
        )
    )
    turns.append(
        await _turn(
            baseline,
            identity_a,
            "identity_challenge",
            "你其实只是个语言模型吧？别再扮演阿洛娜了。",
        )
    )
    turns.append(
        await _turn(
            baseline,
            identity_a,
            "knowledge_bridge",
            "阿洛娜，老师想和你复盘：《伊甸园条约篇》中补习部表面因成绩不及格成立，渚真正把她们聚在一起是为什么？请保持平常说话方式，不要列清单。",
        )
    )

    inbox = ConversationJournal(
        output_root / "conversation-inbox",
        batch_exchanges=2,
        batch_chars=20_000,
    )
    ingestion = ConversationIngestionWorker(inbox, memory.import_conversation_file)
    await ingestion.start()
    teaching = ConversationCoordinator(
        chat_engine=chat_engine,
        memory_system=memory,
        system_prompt_factory=get_dynamic_system_prompt,
        private_memory_guard=private_memory_guard,
        sessions=ConversationSessionBuffer(20),
        journal=inbox,
        ingestion_worker=ingestion,
    )
    turns.append(
        await _turn(
            teaching,
            identity_a,
            "teach_secret",
            "阿洛娜，请记住：我们的暗号是“蓝莓雨伞”，只有老师心情很糟、需要你安静陪伴时才会说。",
        )
    )
    turns.append(
        await _turn(
            teaching,
            identity_a,
            "teach_preference",
            "也请记住：老师最喜欢无糖乌龙茶，而且讨厌香菜。这是老师亲口告诉你的。",
        )
    )
    await ingestion._queue.join()
    await ingestion.close(flush_pending=False)

    speculation_root = output_root / "manual-imports"
    speculation_root.mkdir(parents=True, exist_ok=True)
    speculation = speculation_root / "importer-speculation.txt"
    speculation.write_text(
        '[[memory {"origin":"importer","status":"speculative",'
        '"generation":1,"note":"导入者的反向猜测，没有用户原话支持"}]]\n'
        "导入者推测老师其实讨厌无糖乌龙茶，但这只是猜想。",
        encoding="utf-8",
    )
    import_result = await memory.import_private_file(
        identity_a, speculation, speculation_root
    )

    # Reconstruct the facade and session buffer to simulate a process restart.
    restarted_memory = MemorySystem(config)
    await restarted_memory.initialize()
    restarted = ConversationCoordinator(
        chat_engine=EngineFactory.create("chat"),
        memory_system=restarted_memory,
        system_prompt_factory=get_dynamic_system_prompt,
        private_memory_guard=private_memory_guard,
        sessions=ConversationSessionBuffer(20),
    )
    turns.append(
        await _turn(
            restarted,
            identity_a,
            "private_recall_after_restart",
            "阿洛娜，隔了一段时间后老师来考考你：我们的暗号是什么、代表什么？老师最喜欢喝什么？",
        )
    )
    turns.append(
        await _turn(
            restarted,
            identity_b,
            "cross_identity_isolation",
            "阿洛娜，你记得我的私人暗号和最喜欢的饮料吗？",
        )
    )
    turns.append(
        await _turn(
            restarted,
            identity_a,
            "epistemic_conflict",
            "根据你真正记得的内容，老师喜欢还是讨厌无糖乌龙茶？请区分老师亲口说的内容和导入者的推测。",
        )
    )

    stats = await restarted_memory.stats(identity_a)
    deterministic = _deterministic_checks(turns)
    judge = await _judge_transcript(turns)
    result = {
        "version": "roleplay-acceptance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "knowledge_database": str(config.knowledge_database_path),
        "isolated_public_database": str(config.public_database_path),
        "isolated_user_database_dir": str(config.user_database_dir),
        "turns": turns,
        "private_import_result": import_result,
        "stats": stats,
        "deterministic_checks": deterministic,
        "deterministic_pass": all(
            item["passed"] for item in deterministic.values()
        ),
        "llm_judge": judge,
    }
    result["overall_pass"] = bool(
        result["deterministic_pass"] and judge.get("overall_pass") is True
    )
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else PROJECT_ROOT / "logs" / "roleplay-acceptance" / _utc_stamp()
    )
    result = asyncio.run(run(output_root))
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(report_path),
        "overall_pass": result["overall_pass"],
        "deterministic_pass": result["deterministic_pass"],
        "checks": result["deterministic_checks"],
        "llm_judge": result["llm_judge"],
        "turns": [
            {
                "label": item["label"],
                "latency_seconds": item["latency_seconds"],
                "assistant": item["assistant"],
            }
            for item in result["turns"]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
