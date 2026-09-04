from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B",
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-4.5-Air",
)
QUESTIONS = (
    "渚建立补课部真正想排查什么，梓的身份为何让这项安排具有政治意义？",
    "为什么说补课部不只是帮助差生，梓的阿里乌斯背景和渚寻找内鬼有什么联系？",
    "用两句话说明补课部的表面目的与政治目的。",
)
EVIDENCE = (
    "证据1：渚说明补课部以特别考试与退学作为表面机制，真正目的是把"
    "疑似阻止伊甸园条约的叛徒集中起来排查。\n"
    "证据2：梓来自阿里乌斯，曾接受极端教育并承担破坏任务。"
)


def call(model: str, question: str) -> dict:
    endpoint = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    key = os.environ["SILICONFLOW_API_KEY"]
    started = perf_counter()
    try:
        response = httpx.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "依据证据简洁回答，不补写证据外事实。",
                    },
                    {
                        "role": "user",
                        "content": f"{EVIDENCE}\n问题：{question}",
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 140,
                "enable_thinking": False,
            },
            timeout=httpx.Timeout(8.0),
        )
        response.raise_for_status()
        text = str(response.json()["choices"][0]["message"]["content"] or "")
        return {
            "model": model,
            "question": question,
            "seconds": round(perf_counter() - started, 6),
            "ok": True,
            "answer": text,
            "fact_hits": {
                term: term in text for term in ("渚", "补课部", "梓", "阿里乌斯", "叛徒")
            },
        }
    except Exception as exc:
        return {
            "model": model,
            "question": question,
            "seconds": round(perf_counter() - started, 6),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(call, model, question)
            for model in MODELS
            for question in QUESTIONS
        ]
        rows = [future.result() for future in as_completed(futures)]
    metrics = {}
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        successful = sorted(row["seconds"] for row in selected if row["ok"])
        metrics[model] = {
            "successes": len(successful),
            "attempts": len(selected),
            "mean_seconds": (
                round(sum(successful) / len(successful), 6) if successful else None
            ),
            "max_seconds": max(successful) if successful else None,
            "complete_fact_rows": sum(
                row.get("ok")
                and all(row.get("fact_hits", {}).values())
                for row in selected
            ),
        }
    payload = {
        "version": "chat-model-latency-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "rows": rows,
    }
    output = (
        PROJECT_ROOT
        / "logs"
        / "experiments"
        / f"chat-model-latency-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
