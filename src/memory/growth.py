from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
import inspect
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from .identity import PlatformIdentity
from .service import MemoryRoute


GROWTH_MARKERS = (
    "为什么",
    "为何",
    "真正原因",
    "真实身份",
    "第一次",
    "关系",
    "联系",
    "导致",
    "使得",
    "背后",
    "暗中",
    "想起",
    "回忆",
    "比较",
    "对比",
    "是否",
    "有没有",
    "why",
    "relationship",
    "remember",
    "cause",
)


@dataclass(slots=True, frozen=True)
class GrowthJob:
    job_id: str
    created_at: str
    fingerprint: str
    platform: str
    platform_user_id: str
    display_name: str
    question: str
    user: bool
    knowledge: bool
    reason: str
    intensity: str = "standard"
    knowledge_write_policy: str = "query_and_answer"
    creative: bool = False
    assistant_response: str = ""
    evidence_snapshot: dict[str, Any] = field(default_factory=dict)
    consolidation_decision: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> PlatformIdentity:
        return PlatformIdentity(
            self.platform,
            self.platform_user_id,
            self.display_name,
        )

    @property
    def route(self) -> MemoryRoute:
        return MemoryRoute(
            self.user,
            False,
            self.knowledge,
            self.reason,
            self.intensity,
            self.knowledge_write_policy,
            self.creative,
        )

    @classmethod
    def from_dict(cls, value: dict) -> "GrowthJob":
        return cls(
            job_id=str(value["job_id"]),
            created_at=str(value["created_at"]),
            fingerprint=str(value["fingerprint"]),
            platform=str(value["platform"]),
            platform_user_id=str(value["platform_user_id"]),
            display_name=str(value.get("display_name", "")),
            question=str(value["question"]),
            user=bool(value.get("user", False)),
            knowledge=bool(value.get("knowledge", False)),
            reason=str(value.get("reason", "background_growth")),
            intensity=str(value.get("intensity", "standard")),
            knowledge_write_policy=str(
                value.get(
                    "knowledge_write_policy",
                    "none"
                    if value.get("knowledge_read_only", False)
                    else "query_and_answer",
                )
            ),
            creative=bool(value.get("creative", False)),
            assistant_response=str(value.get("assistant_response", "")),
            evidence_snapshot=dict(value.get("evidence_snapshot") or {}),
            consolidation_decision=dict(
                value.get("consolidation_decision") or {}
            ),
        )


class BackgroundGrowthWorker:
    """Persistent, deduplicated queue for expensive Association growth."""

    def __init__(
        self,
        root: str | Path,
        grow: Callable[[PlatformIdentity, str, MemoryRoute], Awaitable[dict]],
        *,
        consolidate: Callable[..., Awaitable[dict]] | None = None,
        max_queue: int = 100,
        min_chars: int = 18,
    ):
        self.root = Path(root).resolve()
        self.grow = grow
        self.consolidate = consolidate
        self.max_queue = max(1, int(max_queue))
        self.min_chars = max(1, int(min_chars))
        self._queue: asyncio.Queue[Path | None] = asyncio.Queue()
        self._fingerprints: set[str] = set()
        self._active_candidate_fingerprints: set[str] = set()
        self._completed_candidate_fingerprints: set[str] = set()
        self._task: asyncio.Task | None = None
        self._stop_after_current = False
        self._accepting = False
        self.active_job_id = ""
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.dropped_jobs = 0
        self.last_error = ""

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @staticmethod
    def _fingerprint(
        identity: PlatformIdentity,
        question: str,
        route: MemoryRoute,
        assistant_response: str = "",
    ) -> str:
        payload = "\n".join(
            (
                identity.key,
                " ".join(question.casefold().split()),
                str(int(route.user)),
                str(int(route.knowledge)),
                " ".join(assistant_response.casefold().split()),
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _candidate_fingerprint(
        consolidation_decision: dict[str, Any] | None,
    ) -> str:
        """Identify the proposed lore relation independently of chat wording."""

        rows: list[dict[str, Any]] = []
        decision = consolidation_decision or {}
        for raw in decision.get("knowledge_candidates", []):
            if not isinstance(raw, dict):
                continue
            claim = " ".join(str(raw.get("claim", "")).casefold().split())
            try:
                premise_ids = sorted(
                    {
                        int(value)
                        for value in raw.get("premise_episode_ids", [])
                    }
                )
            except (TypeError, ValueError):
                continue
            inference_type = " ".join(
                str(raw.get("inference_type", "")).casefold().split()
            )
            if claim and premise_ids:
                rows.append(
                    {
                        "claim": claim,
                        "premise_episode_ids": premise_ids,
                        "inference_type": inference_type,
                    }
                )
        if not rows:
            return ""
        rows.sort(
            key=lambda row: (
                row["inference_type"],
                row["premise_episode_ids"],
                row["claim"],
            )
        )
        payload = json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def should_enqueue(
        self,
        question: str,
        route: MemoryRoute,
        assistant_response: str = "",
        consolidation_decision: dict[str, Any] | None = None,
    ) -> bool:
        normalized = " ".join(question.casefold().split())
        if not normalized or not (route.user or route.knowledge):
            return False
        decision_supplied = consolidation_decision is not None
        decision = consolidation_decision or {}
        if decision.get("knowledge_candidates"):
            return True
        if str(decision.get("retrieval_growth_query", "")).strip():
            return True
        # Creative content is durably captured by the per-user conversation
        # journal. A second query-growth pass before that source is imported
        # cannot connect the new event and only creates expensive queue churn.
        if route.creative:
            return False
        if decision.get("private_candidates"):
            return True
        # The foreground call already classified this answer and found no
        # durable candidate. Do not enqueue a second no-op model pipeline.
        if decision_supplied:
            return False
        if assistant_response.strip() and route.knowledge:
            return True
        if len(normalized) < self.min_chars:
            return False
        return len(normalized) >= 48 or any(
            marker in normalized for marker in GROWTH_MARKERS
        )

    @staticmethod
    def _retrieval_growth_query(
        memory: Any | None,
        question: str,
        route: MemoryRoute,
    ) -> str:
        """Reuse a costly deep evidence set for durable graph growth.

        This authorizes only another evidence-bound background query. It does
        not treat the assistant answer as lore and does not invent a relation.
        """

        if not route.knowledge or route.knowledge_write_policy == "none":
            return ""
        raw = getattr(memory, "raw_result", {}) if memory is not None else {}
        domains = raw.get("domains", {}) if isinstance(raw, dict) else {}
        knowledge = domains.get("knowledge", {}) if isinstance(domains, dict) else {}
        if not isinstance(knowledge, dict):
            return ""
        escalation = knowledge.get("retrieval_escalation") or {}
        deep_used = bool(escalation.get("triggered")) or (
            str(knowledge.get("retrieval_intensity", "")).casefold() == "deep"
        )
        evidence = [
            row
            for row in list(knowledge.get("evidence_episodes") or [])
            if isinstance(row, dict)
        ]
        if not deep_used or len(evidence) < 2:
            return ""
        return " ".join(question.strip().split())

    @staticmethod
    def _compact_evidence(memory: Any | None) -> dict[str, Any]:
        """Persist only shared-lore evidence; never copy private evidence."""

        raw = getattr(memory, "raw_result", {}) if memory is not None else {}
        domains = raw.get("domains", {}) if isinstance(raw, dict) else {}
        knowledge = (
            domains.get("knowledge", {}) if isinstance(domains, dict) else {}
        )
        if not isinstance(knowledge, dict):
            return {"knowledge_episodes": [], "knowledge_associations": []}
        episodes = [
            {
                "id": row.get("id"),
                "source_key": row.get("source_key"),
                "text": " ".join(str(row.get("text", "")).split())[:420],
                "evidence_origin": row.get("evidence_origin"),
                "epistemic_status": row.get("epistemic_status"),
                "generation": row.get("generation"),
            }
            for row in list(knowledge.get("evidence_episodes") or [])[:6]
            if isinstance(row, dict)
        ]
        return {
            "knowledge_episodes": episodes,
            "knowledge_associations": [],
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    async def start(self) -> None:
        if self._task is not None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        # Successful lore audits remain deduplicated after a restart. A later
        # candidate with new premise IDs gets a different fingerprint and can
        # therefore be reconsidered when genuinely new evidence arrives.
        for path in sorted(self.root.glob("*.completed.json")):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                job_payload = receipt.get("job", {})
                candidate_fingerprint = self._candidate_fingerprint(
                    job_payload.get("consolidation_decision")
                )
            except (OSError, ValueError, json.JSONDecodeError, AttributeError):
                continue
            if candidate_fingerprint:
                self._completed_candidate_fingerprints.add(
                    candidate_fingerprint
                )
        # A process exit during an active job leaves a recoverable request.
        for path in sorted(self.root.glob("*.running.json")):
            pending = path.with_name(path.name.replace(".running.json", ".pending.json"))
            await asyncio.to_thread(path.replace, pending)
        self._accepting = True
        self._stop_after_current = False
        self._task = asyncio.create_task(self._run(), name="memory-growth-worker")
        for path in sorted(self.root.glob("*.pending.json")):
            try:
                job = GrowthJob.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if job.fingerprint in self._fingerprints:
                continue
            candidate_fingerprint = self._candidate_fingerprint(
                job.consolidation_decision
            )
            if candidate_fingerprint and (
                candidate_fingerprint
                in self._completed_candidate_fingerprints
                or candidate_fingerprint
                in self._active_candidate_fingerprints
            ):
                await asyncio.to_thread(path.unlink, missing_ok=True)
                continue
            self._fingerprints.add(job.fingerprint)
            if candidate_fingerprint:
                self._active_candidate_fingerprints.add(
                    candidate_fingerprint
                )
            await self._queue.put(path)

    async def enqueue(
        self,
        identity: PlatformIdentity,
        question: str,
        route: MemoryRoute,
        *,
        assistant_response: str = "",
        memory: Any | None = None,
        consolidation_decision: dict[str, Any] | None = None,
    ) -> Path | None:
        decision = dict(consolidation_decision or {})
        decision_supplied = consolidation_decision is not None
        if (
            not decision.get("knowledge_candidates")
            and not decision_supplied
        ):
            growth_query = self._retrieval_growth_query(memory, question, route)
            if growth_query:
                decision["retrieval_growth_query"] = growth_query
                decision_supplied = True
        if not self._accepting or not self.should_enqueue(
            question,
            route,
            assistant_response,
            decision if decision_supplied else None,
        ):
            return None
        if self._queue.qsize() >= self.max_queue:
            self.dropped_jobs += 1
            return None
        safe_route = MemoryRoute(
            route.user or route.creative,
            False,
            route.knowledge,
            route.reason,
            route.intensity,
            route.knowledge_write_policy,
            route.creative,
        )
        candidate_fingerprint = self._candidate_fingerprint(decision)
        if candidate_fingerprint and (
            candidate_fingerprint in self._active_candidate_fingerprints
            or candidate_fingerprint in self._completed_candidate_fingerprints
        ):
            return None
        fingerprint = self._fingerprint(
            identity, question, safe_route, assistant_response
        )
        if fingerprint in self._fingerprints:
            return None
        now = datetime.now(UTC)
        job_id = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid4().hex[:8]}"
        job = GrowthJob(
            job_id=job_id,
            created_at=now.isoformat(),
            fingerprint=fingerprint,
            platform=identity.platform,
            platform_user_id=identity.platform_user_id,
            display_name=identity.display_name,
            question=question.strip(),
            user=safe_route.user,
            knowledge=safe_route.knowledge,
            reason=safe_route.reason,
            intensity=safe_route.intensity,
            knowledge_write_policy=safe_route.knowledge_write_policy,
            creative=safe_route.creative,
            assistant_response=assistant_response.strip(),
            evidence_snapshot=self._compact_evidence(memory),
            consolidation_decision=decision,
        )
        path = self.root / f"{job_id}.pending.json"
        await asyncio.to_thread(self._write_json, path, asdict(job))
        self._fingerprints.add(fingerprint)
        if candidate_fingerprint:
            self._active_candidate_fingerprints.add(candidate_fingerprint)
        await self._queue.put(path)
        return path

    @staticmethod
    def _accepts_keyword(call: Any, name: str) -> bool:
        try:
            parameters = inspect.signature(call).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            item.kind is inspect.Parameter.VAR_KEYWORD or item.name == name
            for item in parameters
        )

    async def _grow_job(
        self, job: GrowthJob
    ) -> tuple[dict[str, Any], dict[str, Any], MemoryRoute]:
        """Consolidate answer-aware jobs with a fail-closed knowledge gate."""

        consolidation: dict[str, Any] = {}
        knowledge_question: str | None = None
        precomputed = dict(job.consolidation_decision or {})
        answer_aware = bool(job.assistant_response.strip() or precomputed)
        requires_consolidation = (
            answer_aware or job.knowledge_write_policy == "evidence_gated"
        )
        # Legacy question-only jobs preserve the existing audited query-growth
        # behavior. An answer-aware job is never allowed to write shared lore
        # until the consolidator returns at least one evidence-bound candidate.
        knowledge_enabled = job.knowledge and not requires_consolidation
        if requires_consolidation and job.knowledge:
            if precomputed:
                consolidation = precomputed
            elif not answer_aware:
                consolidation = {
                    "error": "evidence-gated job has no assistant response",
                    "knowledge_candidates": [],
                    "private_candidates": [],
                    "rejected_claims": [],
                    "knowledge_query": "",
                }
            elif self.consolidate is None:
                consolidation = {
                    "error": "answer-aware consolidation is not configured",
                    "knowledge_candidates": [],
                    "private_candidates": [],
                    "rejected_claims": [],
                    "knowledge_query": "",
                }
            else:
                try:
                    consolidation = await self.consolidate(
                        job.identity,
                        job.question,
                        job.assistant_response,
                        job.route,
                        job.evidence_snapshot,
                    )
                except Exception as exc:
                    consolidation = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "knowledge_candidates": [],
                        "private_candidates": [],
                        "rejected_claims": [],
                        "knowledge_query": "",
                    }
            candidate_query = str(
                consolidation.get("knowledge_query", "")
            ).strip()
            if not candidate_query:
                candidate_query = str(
                    consolidation.get("retrieval_growth_query", "")
                ).strip()
            knowledge_enabled = bool(
                job.knowledge_write_policy != "none" and candidate_query
            )
            knowledge_question = candidate_query or None

        effective_route = MemoryRoute(
            job.user,
            False,
            knowledge_enabled,
            job.reason,
            job.intensity,
            job.knowledge_write_policy,
            job.creative,
        )
        kwargs: dict[str, Any] = {}
        if self._accepts_keyword(self.grow, "user_question"):
            kwargs["user_question"] = job.question
        if self._accepts_keyword(self.grow, "knowledge_question"):
            kwargs["knowledge_question"] = knowledge_question
        if self._accepts_keyword(self.grow, "knowledge_candidates"):
            kwargs["knowledge_candidates"] = list(
                consolidation.get("knowledge_candidates") or []
            )
        result = await self.grow(
            job.identity, job.question, effective_route, **kwargs
        )
        return result, consolidation, effective_route

    async def _run(self) -> None:
        while True:
            path = await self._queue.get()
            if path is None:
                self._queue.task_done()
                return
            running = path.with_name(path.name.replace(".pending.json", ".running.json"))
            job: GrowthJob | None = None
            try:
                await asyncio.to_thread(path.replace, running)
                job = GrowthJob.from_dict(
                    json.loads(running.read_text(encoding="utf-8"))
                )
                self.active_job_id = job.job_id
                result, consolidation, effective_route = await self._grow_job(job)
                completed = running.with_name(
                    running.name.replace(".running.json", ".completed.json")
                )
                await asyncio.to_thread(
                    self._write_json,
                    completed,
                    {
                        "job": asdict(job),
                        "consolidation": consolidation,
                        "effective_route": asdict(effective_route),
                        "result": result,
                    },
                )
                await asyncio.to_thread(running.unlink, missing_ok=True)
                candidate_fingerprint = self._candidate_fingerprint(
                    job.consolidation_decision
                )
                if candidate_fingerprint:
                    self._completed_candidate_fingerprints.add(
                        candidate_fingerprint
                    )
                self.completed_jobs += 1
                self.last_error = ""
            except Exception as exc:
                self.failed_jobs += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                failed = running.with_name(
                    running.name.replace(".running.json", ".failed.json")
                )
                payload = {
                    "job": asdict(job) if job is not None else {},
                    "error": self.last_error,
                }
                try:
                    await asyncio.to_thread(self._write_json, failed, payload)
                    await asyncio.to_thread(running.unlink, missing_ok=True)
                except OSError:
                    pass
            finally:
                if job is not None:
                    self._fingerprints.discard(job.fingerprint)
                    candidate_fingerprint = self._candidate_fingerprint(
                        job.consolidation_decision
                    )
                    if candidate_fingerprint:
                        self._active_candidate_fingerprints.discard(
                            candidate_fingerprint
                        )
                self.active_job_id = ""
                self._queue.task_done()
            if self._stop_after_current:
                return

    async def close(self) -> None:
        """Stop after the active job; untouched pending files resume next boot."""

        self._accepting = False
        self._stop_after_current = True
        if self._task is None:
            return
        if not self.active_job_id:
            await self._queue.put(None)
        await self._task
        self._task = None
