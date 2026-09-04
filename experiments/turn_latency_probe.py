from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from src.bot.runtime import AdapterRuntime
from src.memory import PlatformIdentity


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _domain_trace(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    rerank = raw.get("rerank_trace") or {}
    escalation = raw.get("retrieval_escalation") or {}
    return {
        "cache_hit": bool(raw.get("service_cache_hit")),
        "intensity": raw.get("retrieval_intensity"),
        "quality": raw.get("retrieval_quality") or {},
        "episode_ids": list(raw.get("episode_ids") or [])[:20],
        "timings": raw.get("timings") or {},
        "escalation": {
            "triggered": bool(escalation.get("triggered")),
            "from": escalation.get("from"),
            "to": escalation.get("to"),
            "reasons": list(escalation.get("reasons") or []),
        },
        "rerank": {
            "backend": rerank.get("backend"),
            "review_level": rerank.get("review_level"),
            "input_count": len(rerank.get("rerank_input_episode_ids") or []),
            "bge_prefilter": rerank.get("bge_prefilter") or {},
            "initial_prompt_chars": rerank.get("initial_prompt_chars"),
            "coverage_prompt_chars": rerank.get("coverage_prompt_chars"),
            "coverage_audit_performed": rerank.get("coverage_audit_performed"),
            "compressor_performed": rerank.get("compressor_performed"),
            "cache_kind": rerank.get("cache_kind"),
            "cloud_requests_avoided": rerank.get("cloud_requests_avoided", 0),
        },
        "query_embedding_cache": raw.get("query_embedding_cache") or {},
        "association_capsule_fast_lane": bool(
            raw.get("association_capsule_fast_lane")
        ),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    runtime = await AdapterRuntime.create()
    identity = PlatformIdentity(args.platform, args.user_id, args.display_name)
    rows: list[dict[str, Any]] = []
    try:
        questions = [args.question]
        if args.followup_question:
            questions.extend(args.followup_question)
        elif args.repeat > 1:
            questions.extend([args.question] * (args.repeat - 1))
        conversation_key = f"latency-probe:{identity.key}"
        for index, question in enumerate(questions):
            started = perf_counter()
            reply = await runtime.coordinator.handle(
                identity=identity,
                text=question,
                conversation_key=conversation_key,
                defer_commit=args.defer_side_effects,
            )
            if args.defer_side_effects:
                runtime.coordinator.sessions.add(
                    conversation_key, "user", question
                )
                runtime.coordinator.sessions.add(
                    conversation_key, "assistant", reply.text
                )
            elapsed = perf_counter() - started
            domains = reply.memory.raw_result.get("domains", {})
            rows.append(
                {
                    "run": index + 1,
                    "question": question,
                    "elapsed_seconds": round(elapsed, 6),
                    "reply": reply.text,
                    "route": {
                        "user": reply.route.user,
                        "public": reply.route.public,
                        "knowledge": reply.route.knowledge,
                        "intensity": reply.route.intensity,
                    },
                    "timings": reply.timings,
                    "guard": reply.memory_guard,
                    "intent_planning": (
                        reply.memory.raw_result.get("intent_planning") or {}
                    ),
                    "domains": {
                        name: _domain_trace(value)
                        for name, value in domains.items()
                    },
                }
            )
    finally:
        await runtime.close()
    return {
        "version": "turn-latency-probe-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "identity": identity.key,
        "question": args.question,
        "runs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--platform", default="synthetic")
    parser.add_argument("--user-id", default="latency-probe")
    parser.add_argument("--display-name", default="Latency Probe")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--followup-question",
        action="append",
        default=[],
        help="Ask a different follow-up in the same conversation/runtime.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--defer-side-effects",
        action="store_true",
        help="Measure foreground latency without importing probe dialogue.",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    payload = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
