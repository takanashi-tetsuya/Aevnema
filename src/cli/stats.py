from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src.memory import MemorySystem, MemorySystemConfig, PlatformIdentity


PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def run(platform: str | None, user_id: str | None) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
    await memory.initialize()
    identity = (
        PlatformIdentity(platform, user_id)
        if platform is not None and user_id is not None
        else None
    )
    return await memory.stats(identity)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show associative memory statistics")
    parser.add_argument("--platform")
    parser.add_argument("--user-id")
    args = parser.parse_args()
    if (args.platform is None) != (args.user_id is None):
        parser.error("--platform and --user-id must be provided together")
    result = asyncio.run(run(args.platform, args.user_id))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
