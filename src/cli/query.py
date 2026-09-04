from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from src.memory import MemoryRoute, MemorySystem, MemorySystemConfig, PlatformIdentity


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _compact_result(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep the fields that explain why a retrieval succeeded or failed."""

    return {
        "question": raw.get("question"),
        "intent": raw.get("intent"),
        "followup_search_queries": raw.get("followup_search_queries", []),
        "atomic_anchor_episode_ids": raw.get("atomic_anchor_episode_ids", []),
        "candidate_episode_ids": raw.get("candidate_episode_ids", []),
        "reranked_episode_ids": raw.get("reranked_episode_ids", []),
        "episode_ids": raw.get("episode_ids", []),
        "concept_ids": raw.get("concept_ids", []),
        "association_ids": raw.get("association_ids", []),
        "new_association_ids": raw.get("new_association_ids", []),
        "reinforced_association_ids": raw.get("reinforced_association_ids", []),
        "growth_utility_gate": raw.get("growth_utility_gate", {}),
        "growth_counterfactual_utility": raw.get(
            "growth_counterfactual_utility", {}
        ),
        "evidence_episodes": raw.get("evidence_episodes", []),
        "evidence_concepts": raw.get("evidence_concepts", []),
        "association_paths": raw.get("association_paths", []),
        "chronology_notes": raw.get("chronology_notes", []),
        "source_key_cohort": raw.get("source_key_cohort", {}),
        "rerank_trace": raw.get("rerank_trace", {}),
        "retrieval_intensity": raw.get("retrieval_intensity"),
        "rerank_policy": raw.get("rerank_policy"),
        "rerank_execution": raw.get("rerank_execution", {}),
        "evidence_slot_trace": raw.get("evidence_slot_trace", {}),
        "timings": raw.get("timings", {}),
        "service_cache_hit": bool(raw.get("service_cache_hit", False)),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
    await memory.initialize()

    identity = PlatformIdentity(args.platform, args.user_id, args.display_name)
    if args.domain == "auto":
        async def recall_once():
            return await memory.recall(identity, args.question)
    elif args.domain in {"knowledge", "public", "user"}:
        if args.domain == "knowledge":
            service = (
                memory.knowledge
                if args.mode == "fast"
                else await memory._background_service(
                    "knowledge", memory.config.knowledge_database_path
                )
            )
        elif args.domain == "public":
            service = (
                memory.public
                if args.mode == "fast"
                else await memory._background_service(
                    "public", memory.config.public_database_path
                )
            )
        else:
            service = (
                await memory._user_service(identity)
                if args.mode == "fast"
                else await memory._background_user_service(identity)
            )

        async def recall_once():
            return await service.recall(args.question)
    else:
        async def recall_once():
            return await memory.recall(
                identity,
                args.question,
                route=MemoryRoute(True, True, True, "diagnostic_all_domains"),
            )

    timings: list[float] = []
    recalled = None
    for _ in range(args.repeat):
        started = perf_counter()
        recalled = await recall_once()
        timings.append(round(perf_counter() - started, 6))
    assert recalled is not None
    raw = recalled.raw_result

    result: dict[str, Any] = {
        "domain": args.domain,
        "mode": args.mode,
        "identity": {
            "platform": identity.platform,
            "platform_user_id": identity.platform_user_id,
        },
        "error": recalled.error,
        "domains": list(recalled.domains),
        "repeat": args.repeat,
        "timings_seconds": timings,
        "context": recalled.context,
    }
    if args.domain in {"all", "auto"}:
        result["route"] = raw.get("route", {})
        result["results"] = {
            domain: _compact_result(domain_raw or {})
            for domain, domain_raw in (raw.get("domains") or {}).items()
        }
    else:
        result["result"] = _compact_result(raw)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one associative-memory retrieval and save its evidence trace"
    )
    parser.add_argument("question")
    parser.add_argument(
        "--domain",
        choices=("auto", "knowledge", "public", "user", "all"),
        default="knowledge",
    )
    parser.add_argument("--mode", choices=("fast", "deep"), default="fast")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--platform", default="diagnostic")
    parser.add_argument("--user-id", default="local-test")
    parser.add_argument("--display-name", default="Local diagnostic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.domain in {"all", "auto"} and args.mode == "deep":
        parser.error("--mode deep must target one explicit domain")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    result = asyncio.run(run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output.resolve())
    else:
        print(rendered)
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
