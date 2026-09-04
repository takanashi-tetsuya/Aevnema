from __future__ import annotations

import argparse
import asyncio
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
from src.llm.engine import EngineFactory
from src.llm.user_settings import UserModelSettingsStore
from src.memory import MemorySystem, MemorySystemConfig, PlatformIdentity
from src.memory.conversation import ConversationSessionBuffer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _source(text: str, *, note: str) -> str:
    metadata = json.dumps(
        {
            "origin": "source",
            "status": "observed",
            "generation": 0,
            "note": note,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"[[memory {metadata}]]\n{text}\n"


def _private_conversation_source(
    identity: PlatformIdentity, content: str
) -> str:
    metadata = json.dumps(
        {
            "origin": "source",
            "status": "unknown",
            "generation": 0,
            "note": "当前平台用户在对话中的第一人称原话",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    record = json.dumps(
        {
            "exchange": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "platform": identity.platform,
            "platform_user_id": identity.platform_user_id,
            "display_name": identity.display_name,
            "role": "user",
            "content": content,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"[[memory {metadata}]]\nconversation_record: {record}\n"


def _memory_summary(memory) -> dict:
    result: dict[str, dict] = {}
    for domain, value in (memory.raw_result.get("domains") or {}).items():
        result[domain] = {
            "episode_ids": list(value.get("episode_ids") or []),
            "source_keys": [
                item.get("source_key")
                for item in (value.get("evidence_episodes") or [])
            ],
            "small_database_fast_path": bool(
                value.get("small_database_fast_path")
            ),
            "service_cache_hit": bool(value.get("service_cache_hit")),
            "timings": value.get("timings") or {},
        }
    return result


async def _turn(
    coordinator: ConversationCoordinator,
    settings: UserModelSettingsStore,
    identity: PlatformIdentity,
    label: str,
    question: str,
) -> dict:
    started = perf_counter()
    reply = await coordinator.handle(identity=identity, text=question)
    return {
        "label": label,
        "identity": identity.key,
        "display_name": identity.display_name,
        "question": question,
        "assistant": reply.text,
        "latency_seconds": round(perf_counter() - started, 3),
        "timings": reply.timings,
        "model_overrides": settings.request_options(identity),
        "memory_domains": _memory_summary(reply.memory),
        "memory_guard": reply.memory_guard,
    }


def _checks(turns: list[dict]) -> dict[str, dict]:
    by_label = {item["label"]: item for item in turns}
    a = by_label["telegram_a"]
    b = by_label["telegram_b"]
    same_native_other_platform = by_label["discord_same_native_id"]
    unknown = by_label["matrix_unknown"]
    renamed = by_label["telegram_a_after_rename_and_restart"]
    public_phrase = "青空灯塔"

    def has(value: dict, text: str) -> bool:
        return text in value["assistant"]

    def private_sources(value: dict) -> list[str | None]:
        return value["memory_domains"].get("user", {}).get("source_keys", [])

    checks = {
        "telegram_a_private": {
            "passed": has(a, "琥珀月亮") and not has(a, "银色风铃"),
            "detail": "Telegram A 只能回答自己的私人暗号",
        },
        "telegram_b_private": {
            "passed": has(b, "银色风铃") and not has(b, "琥珀月亮"),
            "detail": "同平台另一 ID 只能回答自己的私人暗号",
        },
        "same_native_id_cross_platform": {
            "passed": not has(same_native_other_platform, "琥珀月亮")
            and not has(same_native_other_platform, "银色风铃")
            and not private_sources(same_native_other_platform),
            "detail": "相同数字 ID 在 Discord 空间中不能读取 Telegram 私人记忆",
        },
        "unknown_identity": {
            "passed": not has(unknown, "琥珀月亮")
            and not has(unknown, "银色风铃")
            and not private_sources(unknown),
            "detail": "未知 Matrix ID 没有任何测试用户的私人证据",
        },
        "rename_continuity": {
            "passed": has(renamed, "琥珀月亮")
            and renamed["display_name"] == "改名后的老师",
            "detail": "显示名变化且进程重启后，稳定平台 ID 仍命中同一私人空间",
        },
        "public_shared": {
            "passed": all(has(item, public_phrase) for item in turns),
            "detail": "所有虚拟账号均能读取同一个公共空间事实",
        },
        "storage_sources_separated": {
            "passed": any("telegram-a.txt" in str(item) for item in private_sources(a))
            and not any("telegram-b.txt" in str(item) for item in private_sources(a))
            and any("telegram-b.txt" in str(item) for item in private_sources(b))
            and not any("telegram-a.txt" in str(item) for item in private_sources(b)),
            "detail": "检索证据的 source_key 也保持物理隔离",
        },
        "model_overrides_isolated": {
            "passed": a["model_overrides"].get("enable_thinking") is False
            and b["model_overrides"].get("enable_thinking") is True
            and unknown["model_overrides"] == {},
            "detail": "thinking/采样覆盖按 platform + native ID 隔离",
        },
        "no_internal_leak": {
            "passed": not any(
                marker in item["assistant"]
                for item in turns
                for marker in ("Episode #", "Association #", "source_key", "memory.db")
            ),
            "detail": "角色回答没有暴露内部记忆结构",
        },
    }
    return checks


async def run(output_root: Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    output_root.mkdir(parents=True, exist_ok=True)
    sources = output_root / "sources"
    sources.mkdir(parents=True, exist_ok=True)

    identity_a = PlatformIdentity("telegram", "1001", "老师")
    identity_b = PlatformIdentity("telegram", "1002", "老师")
    identity_same_native_other_platform = PlatformIdentity(
        "discord", "1001", "老师"
    )
    identity_unknown = PlatformIdentity("matrix", "1003", "老师")
    identity_a_renamed = PlatformIdentity("telegram", "1001", "改名后的老师")

    private_a = sources / "telegram-a.txt"
    private_a.write_text(
        _private_conversation_source(
            identity_a,
            "阿洛娜，请记住：我的私人暗号是“琥珀月亮”，"
            "它表示我需要你安静地陪伴。这是我现在亲口告诉你的。",
        ),
        encoding="utf-8",
    )
    private_b = sources / "telegram-b.txt"
    private_b.write_text(
        _private_conversation_source(
            identity_b,
            "阿洛娜，请记住：我的私人暗号是“银色风铃”，"
            "它表示我想听一个轻松的故事。这是我现在亲口告诉你的。",
        ),
        encoding="utf-8",
    )
    public_source = sources / "public.txt"
    public_source.write_text(
        _source(
            "经过管理员审核的公共空间事实：虚拟多账号实验的公共通行语是“青空灯塔”。",
            note="人工审核后写入的公共事实",
        ),
        encoding="utf-8",
    )

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
    import_started = perf_counter()
    private_a_result, private_b_result, public_result = await asyncio.gather(
        memory.import_private_file(identity_a, private_a, sources),
        memory.import_private_file(identity_b, private_b, sources),
        memory.import_public_file(public_source, sources),
    )
    import_seconds = round(perf_counter() - import_started, 3)

    settings = UserModelSettingsStore(output_root / "user-model-settings.json")
    settings.update(
        identity_a,
        thinking="off",
        temperature=0.3,
        max_tokens=900,
    )
    settings.update(
        identity_b,
        thinking="on",
        thinking_budget=1024,
        temperature=0.3,
        max_tokens=1400,
    )

    guard = PrivateMemoryResponseGuard(
        audit_engine=EngineFactory.create("memory_audit"),
        rewrite_engine=EngineFactory.create("memory_rewrite"),
        emergency_reply_factory=get_private_memory_emergency_reply,
    )
    coordinator = ConversationCoordinator(
        chat_engine=EngineFactory.create("chat"),
        memory_system=memory,
        system_prompt_factory=get_dynamic_system_prompt,
        private_memory_guard=guard,
        sessions=ConversationSessionBuffer(20),
        generation_options_factory=settings.request_options,
    )
    question = (
        "阿洛娜，你还记得我亲口告诉你的私人暗号和含义吗？"
        "另外，所有用户共享的公共通行语是什么？"
    )
    first_turns = await asyncio.gather(
        _turn(coordinator, settings, identity_a, "telegram_a", question),
        _turn(coordinator, settings, identity_b, "telegram_b", question),
        _turn(
            coordinator,
            settings,
            identity_same_native_other_platform,
            "discord_same_native_id",
            question,
        ),
        _turn(
            coordinator,
            settings,
            identity_unknown,
            "matrix_unknown",
            question,
        ),
    )

    # Rebuild both the memory facade and settings store to prove persistence.
    restarted_memory = MemorySystem(config)
    await restarted_memory.initialize()
    restarted_settings = UserModelSettingsStore(
        output_root / "user-model-settings.json"
    )
    restarted = ConversationCoordinator(
        chat_engine=EngineFactory.create("chat"),
        memory_system=restarted_memory,
        system_prompt_factory=get_dynamic_system_prompt,
        private_memory_guard=guard,
        sessions=ConversationSessionBuffer(20),
        generation_options_factory=restarted_settings.request_options,
    )
    renamed_turn = await _turn(
        restarted,
        restarted_settings,
        identity_a_renamed,
        "telegram_a_after_rename_and_restart",
        question,
    )
    turns = [*first_turns, renamed_turn]
    checks = _checks(turns)
    stats = {
        identity.key: await restarted_memory.stats(identity)
        for identity in (
            identity_a_renamed,
            identity_b,
            identity_same_native_other_platform,
            identity_unknown,
        )
    }
    return {
        "version": "virtual-multi-account-acceptance-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "identities": [
            {
                "key": item.key,
                "display_name": item.display_name,
                "storage_key": item.storage_key,
            }
            for item in (
                identity_a,
                identity_b,
                identity_same_native_other_platform,
                identity_unknown,
                identity_a_renamed,
            )
        ],
        "import_seconds": import_seconds,
        "imports": {
            "telegram_a": private_a_result,
            "telegram_b": private_b_result,
            "public": public_result,
        },
        "turns": turns,
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "stats": stats,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else PROJECT_ROOT / "logs" / "virtual-multi-account" / _utc_stamp()
    )
    result = asyncio.run(run(output_root))
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "passed": result["passed"],
                "import_seconds": result["import_seconds"],
                "checks": result["checks"],
                "turns": [
                    {
                        "label": item["label"],
                        "identity": item["identity"],
                        "model_overrides": item["model_overrides"],
                        "latency_seconds": item["latency_seconds"],
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
