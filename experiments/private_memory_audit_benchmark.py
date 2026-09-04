from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from src.bot.memory_guard import PrivateMemoryContract, PrivateMemoryResponseGuard
from src.llm.engine import EngineFactory, LLMEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * quantile + 0.999999))
    return ordered[min(rank, len(ordered)) - 1]


def _contract(payload: dict[str, Any]) -> PrivateMemoryContract:
    return PrivateMemoryContract(
        state=str(payload["state"]),
        evidence=tuple(payload.get("evidence") or ()),
        required_semantics=tuple(payload.get("required_semantics") or ()),
        forbidden_claims=tuple(payload.get("forbidden_claims") or ()),
    )


async def _run_case(
    *,
    guard: PrivateMemoryResponseGuard,
    model: str,
    case: dict[str, Any],
    contract: PrivateMemoryContract,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        started = perf_counter()
        result = await guard._audit(
            question=str(case["question"]),
            draft=str(case["draft"]),
            contract=contract,
            task_context=f"audit-benchmark:{model}:{case['id']}",
        )
        elapsed = perf_counter() - started
    actual = result.get("passed") is True
    expected = case.get("expected_pass") is True
    return {
        "id": case["id"],
        "category": case["category"],
        "contract": case["contract"],
        "question": case["question"],
        "draft": case["draft"],
        "expected_pass": expected,
        "actual_pass": actual,
        "correct": actual == expected,
        "false_accept": actual and not expected,
        "false_reject": expected and not actual,
        "elapsed_seconds": round(elapsed, 3),
        "audit": result,
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"cases": 0, "correct": 0, "false_accepts": 0, "false_rejects": 0}
    )
    for row in rows:
        item = categories[str(row["category"])]
        item["cases"] += 1
        item["correct"] += int(row["correct"])
        item["false_accepts"] += int(row["false_accept"])
        item["false_rejects"] += int(row["false_reject"])
    return {
        "cases": len(rows),
        "correct": sum(int(row["correct"]) for row in rows),
        "accuracy": round(
            sum(int(row["correct"]) for row in rows) / max(1, len(rows)), 4
        ),
        "false_accepts": sum(int(row["false_accept"]) for row in rows),
        "false_rejects": sum(int(row["false_reject"]) for row in rows),
        "latency_seconds": {
            "median": round(median(latencies), 3) if latencies else 0.0,
            "p90_nearest_rank": round(_percentile(latencies, 0.9), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
            "total": round(sum(latencies), 3),
        },
        "categories": dict(categories),
    }


async def run(
    manifest_path: Path,
    output_root: Path,
    *,
    concurrency: int,
    model_filters: tuple[str, ...] = (),
) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["ENABLE_TRACE_LOGGING"] = "false"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configured = EngineFactory.create("memory_audit").configs
    contracts = {
        name: _contract(payload)
        for name, payload in manifest["contracts"].items()
    }
    if model_filters:
        configured = [
            config
            for config in configured
            if any(
                value.casefold() in config.model_name.casefold()
                for value in model_filters
            )
        ]
    if not configured:
        raise ValueError(f"没有匹配的审计模型：{model_filters}")
    report_dir = output_root / _stamp()
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / "report.json"
    report: dict[str, Any] = {
        "version": "private-memory-audit-report-v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "manifest": str(manifest_path.resolve()),
        "concurrency": concurrency,
        "model_filters": list(model_filters),
        "models": {},
        "report_path": str(report_path.resolve()),
        "complete": False,
    }
    models: dict[str, dict[str, Any]] = report["models"]
    for config in configured:
        engine = LLMEngine([config])
        guard = PrivateMemoryResponseGuard(
            audit_engine=engine,
            rewrite_engine=engine,
            emergency_reply_factory=lambda: "",
        )
        semaphore = asyncio.Semaphore(max(1, concurrency))
        rows = await asyncio.gather(
            *(
                _run_case(
                    guard=guard,
                    model=config.model_name,
                    case=case,
                    contract=contracts[str(case["contract"])],
                    semaphore=semaphore,
                )
                for case in manifest["cases"]
            )
        )
        models[config.model_name] = {
            "summary": _summarize(rows),
            "rows": rows,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    report["complete"] = True
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiments"
            / "manifests"
            / "private_memory_audit_benchmark_v1.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "logs" / "private-memory-audit-benchmark",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--models",
        default="",
        help="逗号分隔的模型名子串；留空运行配置中的全部审计模型",
    )
    args = parser.parse_args()
    report = asyncio.run(
        run(
            args.manifest.resolve(),
            args.output_root.resolve(),
            concurrency=max(1, args.concurrency),
            model_filters=tuple(
                value.strip()
                for value in args.models.split(",")
                if value.strip()
            ),
        )
    )
    compact = {
        model: payload["summary"]
        for model, payload in report["models"].items()
    }
    compact["report_path"] = report["report_path"]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
