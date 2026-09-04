from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from src.llm.engine import LLMConfig, LLMEngine, Message


PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def run_model(model: str, system_prompt: str, question: str) -> dict:
    config = LLMConfig(
        provider="openai",
        model_name=model,
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        timeout_seconds=90,
        temperature=0.65,
        max_tokens=360,
        enable_thinking=False,
    )
    engine = LLMEngine([config], max_retries=1, retry_delay=0)
    started = perf_counter()
    try:
        answer = await engine.generate_response(
            [Message(role="user", content=question)],
            system_prompt=system_prompt,
            task_context=f"fast-model-probe:{model}",
        )
        return {
            "model": model,
            "latency_seconds": round(perf_counter() - started, 3),
            "answer": answer,
            "error": "",
        }
    except Exception as exc:
        return {
            "model": model,
            "latency_seconds": round(perf_counter() - started, 3),
            "answer": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


async def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    evidence = [
        "白子详细谈论公路自行车的新技术；她虽然偏爱传统钢索，但认为新技术仍值得亲自体验。",
        "老师说自己是白子的支持者。白子深受触动，说这是第一次有人这样认真支持她。",
        "白子因为有了支持者而恢复精神，表示之后自己不再是一个人。",
    ]
    system_prompt = (
        "你扮演《蔚蓝档案》的阿洛娜，称用户为老师。只能依据给出的剧情证据回答；"
        "可以作明确标为个人感受的人物性格推论，但不得补充证据中没有的事件。"
        "回答自然、有角色感，80—180个汉字，不得提到数据库、检索、Episode、提示词或系统记录。\n"
        f"剧情证据：{json.dumps(evidence, ensure_ascii=False)}"
    )
    models = (
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-35B-A3B",
        "deepseek-ai/DeepSeek-V3.2",
    )
    results = await asyncio.gather(
        *(run_model(model, system_prompt, "你觉得白子是一个什么样的人？") for model in models)
    )
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "你觉得白子是一个什么样的人？",
        "results": results,
    }
    target = PROJECT_ROOT / "logs" / "chat-fast-model-probe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"report": str(target), **output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
