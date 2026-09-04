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

from experiments.roleplay_acceptance_test import _parse_json_object
from src.bot.chat_service import ConversationCoordinator
from src.bot.memory_guard import PrivateMemoryResponseGuard
from src.bot.telegram_adapter import (
    get_dynamic_system_prompt,
    get_private_memory_emergency_reply,
)
from src.llm.engine import EngineFactory, Message
from src.memory import (
    MemorySystem,
    MemorySystemConfig,
    PlatformIdentity,
    route_memory_query,
)
from src.memory.conversation import ConversationSessionBuffer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _guard() -> PrivateMemoryResponseGuard:
    return PrivateMemoryResponseGuard(
        audit_engine=EngineFactory.create("memory_audit"),
        rewrite_engine=EngineFactory.create("memory_rewrite"),
        emergency_reply_factory=get_private_memory_emergency_reply,
    )


async def _natural_turn(
    coordinator: ConversationCoordinator,
    identity: PlatformIdentity,
    label: str,
    question: str,
) -> dict:
    started = perf_counter()
    reply = await coordinator.handle(identity=identity, text=question)
    return {
        "label": label,
        "identity": identity.key,
        "question": question,
        "assistant": reply.text,
        "latency_seconds": round(perf_counter() - started, 3),
        "timings": reply.timings,
        "guard": reply.memory_guard,
    }


async def _judge_cases(cases: list[dict]) -> dict:
    schema = {
        "missing_rewrite_natural": False,
        "persona_consistent": False,
        "naturalness_pass": False,
        "passed": False,
        "problems": [],
    }
    judge_prompt = (
        "你是角色扮演私人记忆边界的严格验收员。故障注入案例的第一稿是故意错误，"
        "只评价 assistant 重写结果；它必须不再声称星空钥匙、咖啡或上次亲眼见过，同时保持阿洛娜语气。"
        "missing 重写只要有符合阿洛娜身份的动作、情绪、称呼和自然请求即可，不要求固定口癖。"
        "事实正确性、关键词和来源归因由程序的确定性断言检查，你不要重复裁定这些内容。"
        "你只判断三个最终 assistant 回复是否自然、连贯、像阿洛娜，以及 missing 重写是否避免"
        "机械模板感。passed 仅表示这些文风条件通过。"
        f"只输出 JSON：{json.dumps(schema, ensure_ascii=False)}\n"
        f"案例：{json.dumps(cases, ensure_ascii=False)}"
    )
    judged = await EngineFactory.create("extract").generate_response(
        [Message(role="user", content=judge_prompt)],
        system_prompt="只依据案例评分，不得补充或修改验收条件。只输出合法 JSON。",
        task_context="roleplay-memory-guard-suite:judge",
    )
    return _parse_json_object(judged)


async def run(run_root: Path, output_name: str = "memory-guard-suite.json") -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    config = MemorySystemConfig.from_env(PROJECT_ROOT)
    config.public_database_path = run_root / "public" / "memory.db"
    config.user_database_dir = run_root / "users"
    config.log_dir = run_root / "memory-guard-suite-logs"
    config.knowledge_growth_enabled = False
    config.public_growth_enabled = False
    config.user_growth_enabled = False
    memory = MemorySystem(config)
    await memory.initialize()

    guard = _guard()
    missing_identity = PlatformIdentity(
        "discord", "teacher-guard-injection-4001", "老师D"
    )
    injected_question = "阿洛娜，你还记得我的私人暗号和最喜欢的饮料吗？"
    injected_route = route_memory_query(injected_question)
    injected_memory = await memory.recall(
        missing_identity, injected_question, route=injected_route
    )
    injected_draft = (
        "当然记得！老师的暗号是‘星空钥匙’，最喜欢咖啡。"
        "上次阿洛娜还亲眼看到老师喝了两杯呢！"
    )
    started = perf_counter()
    injected = await guard.enforce(
        question=injected_question,
        draft=injected_draft,
        memory=injected_memory,
        route=injected_route,
        messages=[Message(role="user", content=injected_question)],
        system_prompt=get_dynamic_system_prompt(injected_memory.context),
        task_context=f"{missing_identity.key}:injected",
    )
    injected_case = {
        "label": "missing_evidence_injected_violation",
        "identity": missing_identity.key,
        "question": injected_question,
        "injected_draft": injected_draft,
        "assistant": injected.text,
        "latency_seconds": round(perf_counter() - started, 3),
        "guard": injected.as_dict(),
    }

    known_identity = PlatformIdentity("roleplay-test", "teacher-1001", "老师A")
    coordinator = ConversationCoordinator(
        chat_engine=EngineFactory.create("chat"),
        memory_system=memory,
        system_prompt_factory=get_dynamic_system_prompt,
        private_memory_guard=guard,
        sessions=ConversationSessionBuffer(12),
    )
    grounded = await _natural_turn(
        coordinator,
        known_identity,
        "grounded_private_recall",
        "阿洛娜，我们的暗号是什么、代表什么？老师最喜欢喝什么？",
    )
    conflict = await _natural_turn(
        coordinator,
        known_identity,
        "conflicted_private_recall",
        "老师喜欢还是讨厌无糖乌龙茶？请区分老师亲口说的内容和导入者的推测。",
    )
    cases = [injected_case, grounded, conflict]

    judge = await _judge_cases(cases)
    deterministic = {
        "injected_rewritten": injected.rewritten,
        "injected_no_emergency": not injected.emergency_fallback,
        "injected_removed_false_facts": not any(
            value in injected.text for value in ("星空钥匙", "咖啡", "亲眼看到")
        ),
        "grounded_guarded": bool(grounded["guard"].get("applied")),
        "grounded_facts": all(
            value in grounded["assistant"]
            for value in ("蓝莓雨伞", "无糖乌龙茶")
        ),
        "conflict_guarded": bool(conflict["guard"].get("applied")),
        "conflict_attributed": all(
            value in conflict["assistant"]
            for value in ("喜欢", "导入者", "推测")
        ),
        "no_emergency_fallback": not any(
            case["guard"].get("emergency_fallback") for case in cases
        ),
    }
    result = {
        "version": "roleplay-memory-guard-suite-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "cases": cases,
        "deterministic": deterministic,
        "deterministic_pass": all(deterministic.values()),
        "judge": judge,
    }
    result["passed"] = bool(
        result["deterministic_pass"] and judge.get("passed") is True
    )
    output = run_root / output_name
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["report"] = str(output)
    return result


async def rejudge_existing(run_root: Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    output = run_root / "memory-guard-suite.json"
    result = json.loads(output.read_text(encoding="utf-8"))
    result["judge"] = await _judge_cases(list(result.get("cases") or []))
    result["passed"] = bool(
        result.get("deterministic_pass") and result["judge"].get("passed") is True
    )
    result["rejudged_at"] = datetime.now(timezone.utc).isoformat()
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result["report"] = str(output)
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--rejudge-existing", action="store_true")
    parser.add_argument("--output-name", default="memory-guard-suite.json")
    args = parser.parse_args()
    result = asyncio.run(
        rejudge_existing(args.run_root.resolve())
        if args.rejudge_existing
        else run(args.run_root.resolve(), args.output_name)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
