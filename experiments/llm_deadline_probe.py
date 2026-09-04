from __future__ import annotations

import argparse
import asyncio
from time import perf_counter

from dotenv import load_dotenv

from src.llm.engine import EngineFactory, Message


async def run(deadline: float) -> None:
    engine = EngineFactory.create("chat_fast")
    started = perf_counter()
    try:
        reply = await engine.generate_response(
            [Message(role="user", content="请只回答：测试")],
            system_prompt="简短回答。",
            task_context="deadline-probe",
            deadline_seconds=deadline,
        )
        print({"status": "ok", "seconds": perf_counter() - started, "reply": reply})
    except Exception as exc:
        print(
            {
                "status": "error",
                "seconds": perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            },
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=float, default=2.0)
    args = parser.parse_args()
    load_dotenv()
    asyncio.run(run(args.deadline))


if __name__ == "__main__":
    main()
