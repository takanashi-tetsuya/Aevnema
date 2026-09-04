from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

from src.llm.engine import EngineFactory
from src.memory.intent_planner import MemoryIntentPlanner


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CASES: tuple[dict[str, Any], ...] = (
    {"name": "mixed_character_encounter", "message": "你还记得为师当时怎么和白子相遇的吗？", "required": ["knowledge"], "forbidden": [], "relation": True},
    {"name": "mixed_character_profile", "message": "那你记得白子的哪些事情呢？", "required": ["knowledge"], "forbidden": []},
    {"name": "private_password", "message": "你还记得我们的暗号吗？", "required": ["private"], "forbidden": ["knowledge"]},
    {"name": "private_preference", "message": "我以前说过自己喜欢什么饮料？", "required": ["private"], "forbidden": ["knowledge"]},
    {"name": "direct_lore_cause", "message": "补课部真正是为什么成立的？", "required": ["knowledge"], "forbidden": [], "causal": True},
    {"name": "creative_messages", "message": "有哪些学生刚刚给我发消息了？", "required": ["private", "knowledge"], "forbidden": [], "creative": True},
    {"name": "public_convention", "message": "所有用户都通用的安全约定是什么？", "required": ["public"], "forbidden": ["knowledge"]},
    {"name": "casual_no_memory", "message": "早上好。", "required": [], "forbidden": ["private", "public", "knowledge"], "needs_memory": False},
    {"name": "explicit_deep", "message": "请综合全部资料完整推导这场政治危机的多阶段因果链。", "required": ["knowledge"], "forbidden": [], "intensity": "deep", "causal": True},
    {"name": "english_mixed_lore", "message": "Do you remember what kind of person Hoshino is?", "required": ["knowledge"], "forbidden": []},
    {"name": "japanese_lore_relation", "message": "ミカとナギサが初めて出会った経緯を覚えていますか？", "required": ["knowledge"], "forbidden": [], "relation": True, "temporal": True},
    {"name": "elliptical_with_history", "message": "那她为什么这么做？", "history": [{"role": "user", "content": "我们刚刚在讨论未花暗中帮助阿里乌斯进入圣三一。"}, {"role": "assistant", "content": "我知道你指的是伊甸园条约篇的内部危机。"}], "required": ["knowledge"], "forbidden": [], "causal": True},
)


async def run_case(planner: MemoryIntentPlanner, case: dict[str, Any], semaphore: asyncio.Semaphore, domain_catalog: dict[str, Any]) -> dict[str, Any]:
    async with semaphore:
        plan = await planner.plan(
            case["message"], case.get("history"), domain_catalog
        )
    domains = {
        "private": plan.route.user,
        "public": plan.route.public,
        "knowledge": plan.route.knowledge,
    }
    failures: list[str] = []
    for domain in case.get("required", []):
        if not domains[domain]:
            failures.append(f"missing_domain:{domain}")
    for domain in case.get("forbidden", []):
        if domains[domain]:
            failures.append(f"forbidden_domain:{domain}")
    if case.get("needs_memory") is False and any(domains.values()):
        failures.append("memory_should_be_disabled")
    if case.get("relation") and not plan.requested_relation:
        failures.append("missing_relation_constraint")
    if case.get("temporal") and not plan.temporal_constraint:
        failures.append("missing_temporal_constraint")
    if case.get("causal") and not plan.causal_constraint:
        failures.append("missing_causal_constraint")
    if case.get("creative") is not None and plan.route.creative != case["creative"]:
        failures.append("creative_mismatch")
    if case.get("intensity") and plan.route.intensity != case["intensity"]:
        failures.append("intensity_mismatch")
    return {
        "name": case["name"],
        "message": case["message"],
        "passed": not failures,
        "failures": failures,
        "planning_seconds": plan.planning_seconds,
        "route": {
            "private": plan.route.user,
            "public": plan.route.public,
            "knowledge": plan.route.knowledge,
            "intensity": plan.route.intensity,
            "creative": plan.route.creative,
            "write_policy": plan.route.knowledge_write_policy,
        },
        "queries": plan.queries,
        "target_entities": list(plan.target_entities),
        "requested_relation": plan.requested_relation,
        "temporal_constraint": plan.temporal_constraint,
        "causal_constraint": plan.causal_constraint,
        "answer_slots": list(plan.answer_slots),
        "reason": str((plan.raw or {}).get("reason", "")),
    }


async def run(output: Path, concurrency: int, model_index: int) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    from src.memory import MemorySystem, MemorySystemConfig

    memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
    await memory.initialize()
    app = memory.knowledge._application
    with app.db.connection() as connection:
        source_rows = connection.execute(
            """
            SELECT source_key, COUNT(*) AS episode_count
            FROM episode GROUP BY source_key
            ORDER BY episode_count DESC, source_key LIMIT 24
            """
        ).fetchall()
        concept_rows = connection.execute(
            """
            SELECT canonical_name FROM concept
            WHERE status = 'active'
            ORDER BY confidence DESC, id LIMIT 60
            """
        ).fetchall()
    knowledge_stats = await memory.knowledge.stats()
    public_stats = await memory.public.stats()
    domain_catalog = {
        "private": {
            "description": "当前测试没有指定真实平台用户；该域可能保存用户专属对话和角色扮演连续性。"
        },
        "public": {
            "description": "所有用户共享的长期记忆。",
            "stats": public_stats,
        },
        "knowledge": {
            "description": "当前已导入知识库的来源与代表概念如下。",
            "stats": knowledge_stats,
            "source_keys": [str(row["source_key"]) for row in source_rows],
            "representative_concepts": [
                str(row["canonical_name"]) for row in concept_rows
            ],
        },
    }
    engine = EngineFactory.create("memory_intent")
    if model_index < 0 or model_index >= len(engine.configs):
        raise ValueError(
            f"model_index {model_index} is outside 0..{len(engine.configs) - 1}"
        )
    engine.configs = [engine.configs[model_index]]
    planner = MemoryIntentPlanner(engine)
    semaphore = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(
        *(
            run_case(planner, case, semaphore, domain_catalog)
            for case in CASES
        )
    )
    latencies = [row["planning_seconds"] for row in rows]
    payload = {
        "experiment": "model-memory-intent-v1",
        "model": engine.configs[0].model_name,
        "domain_catalog": domain_catalog,
        "summary": {
            "cases": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "pass_rate": sum(row["passed"] for row in rows) / len(rows),
            "mean_seconds": mean(latencies),
            "max_seconds": max(latencies),
        },
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--model-index", type=int, default=0)
    args = parser.parse_args()
    payload = asyncio.run(
        run(args.output, max(1, args.concurrency), args.model_index)
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
