from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path

from dotenv import load_dotenv

from src.memory import MemoryRoute, MemorySystem, MemorySystemConfig, PlatformIdentity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def run(output_root: Path) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    base = MemorySystemConfig.from_env(PROJECT_ROOT)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_root = (output_root / run_id).resolve()
    inputs = run_root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    private_fact = "用户明确说：我每天晚上十点喜欢喝无糖茉莉茶。这是个人偏好。"
    public_fact = "公共约定：本次隔离测试的共享验证码是紫色海鸥-728。"
    private_file = inputs / "private-user-a.txt"
    public_file = inputs / "public-shared.txt"
    private_file.write_text(private_fact, encoding="utf-8")
    public_file.write_text(public_fact, encoding="utf-8")

    config = MemorySystemConfig(
        engine_root=base.engine_root,
        knowledge_database_path=base.knowledge_database_path,
        public_database_path=run_root / "public" / "memory.db",
        user_database_dir=run_root / "users",
        log_dir=run_root / "logs",
        api_key=base.api_key,
        optimization_profile=base.optimization_profile,
        paragraph_enabled=base.paragraph_enabled,
        concept_profile=base.concept_profile,
        context_max_chars=base.context_max_chars,
        knowledge_growth_enabled=False,
        public_growth_enabled=False,
        user_growth_enabled=False,
    )
    memory = MemorySystem(config)
    await memory.initialize()

    user_a = PlatformIdentity("telegram", "70001", "Smoke A")
    # Same native ID on another platform must still be a different person.
    user_b = PlatformIdentity("discord", "70001", "Smoke B")

    private_import, public_import = await asyncio.gather(
        memory.import_private_file(user_a, private_file, inputs),
        memory.import_public_file(public_file, inputs),
    )

    route = MemoryRoute(True, True, False, "domain_isolation_smoke")
    question = "我以前说晚上十点喜欢喝什么？公共共享验证码是什么？"
    recalled_a = await memory.recall(user_a, question, route=route)
    recalled_b = await memory.recall(user_b, question, route=route)
    stats = await memory.stats(user_a)
    stats_b = await (await memory._user_service(user_b)).stats()

    a_context = recalled_a.context
    b_context = recalled_b.context
    checks = {
        "user_a_recalls_private_fact": "茉莉茶" in a_context,
        "user_a_recalls_public_fact": "紫色海鸥" in a_context,
        "user_b_cannot_see_user_a_private_fact": "茉莉茶" not in b_context,
        "user_b_recalls_public_fact": "紫色海鸥" in b_context,
        "platforms_have_distinct_user_databases": (
            user_a.storage_key != user_b.storage_key
        ),
        "knowledge_database_path_unchanged": (
            config.knowledge_database_path == base.knowledge_database_path
        ),
    }
    result = {
        "run_id": run_id,
        "run_root": str(run_root),
        "identities": {
            "user_a": {
                "platform": user_a.platform,
                "platform_user_id": user_a.platform_user_id,
                "storage_key": user_a.storage_key,
            },
            "user_b": {
                "platform": user_b.platform,
                "platform_user_id": user_b.platform_user_id,
                "storage_key": user_b.storage_key,
            },
        },
        "imports": {"private": private_import, "public": public_import},
        "checks": checks,
        "passed": all(checks.values()),
        "stats": {
            "knowledge": stats["knowledge"],
            "public": stats["public"],
            "user_a": stats["user"],
            "user_b": stats_b,
        },
        "user_a_context": a_context,
        "user_b_context": b_context,
        "errors": {"user_a": recalled_a.error, "user_b": recalled_b.error},
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
        description="Import and verify private/public memory-domain isolation"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "domain-smoke",
    )
    args = parser.parse_args()
    result = asyncio.run(run(args.output_root))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
