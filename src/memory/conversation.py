from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Awaitable, Callable
from uuid import uuid4

from .identity import PlatformIdentity


@dataclass(slots=True)
class ChatTurn:
    role: str
    content: str


class ConversationSessionBuffer:
    """Bounded, process-local context; durable memory lives in Source/Episode."""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max(2, int(max_messages))
        self._messages: dict[str, deque[ChatTurn]] = defaultdict(
            lambda: deque(maxlen=self.max_messages)
        )

    def add(self, identity_key: str, role: str, content: str) -> None:
        if content.strip():
            self._messages[str(identity_key)].append(ChatTurn(role, content.strip()))

    def get(self, identity_key: str) -> list[ChatTurn]:
        return list(self._messages.get(str(identity_key), ()))

    def clear(self, identity_key: str) -> None:
        self._messages.pop(str(identity_key), None)


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z_-]+", "_", value.strip())
    return normalized.strip("_")[:80] or "unknown"


class ConversationJournal:
    """Crash-recoverable ingestion spool for completed user/assistant exchanges."""

    def __init__(
        self,
        root: str | Path,
        *,
        batch_exchanges: int = 4,
        batch_chars: int = 3_000,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.batch_exchanges = max(1, int(batch_exchanges))
        self.batch_chars = max(400, int(batch_chars))
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _user_dir(self, identity: PlatformIdentity) -> Path:
        path = self.root / identity.platform / identity.storage_key
        identity.write_metadata(path)
        return path

    def _pending_path(self, identity: PlatformIdentity) -> Path:
        return self._user_dir(identity) / "pending.txt"

    @staticmethod
    def _exchange_count(text: str) -> int:
        return text.count('"role":"user"')

    def _finalize_pending(self, identity: PlatformIdentity) -> Path | None:
        pending = self._pending_path(identity)
        if not pending.exists() or not pending.read_text(encoding="utf-8").strip():
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        target = pending.with_name(f"{timestamp}-{uuid4().hex[:8]}.ready.txt")
        pending.replace(target)
        return target

    async def record_exchange(
        self,
        identity: PlatformIdentity,
        user_text: str,
        assistant_text: str,
    ) -> Path | None:
        async with self._locks[identity.key]:
            path = self._pending_path(identity)
            existing = (
                await asyncio.to_thread(path.read_text, encoding="utf-8")
                if path.exists()
                else ""
            )
            now = datetime.now(timezone.utc).isoformat()
            exchange_number = self._exchange_count(existing) + 1
            user_evidence = json.dumps(
                {
                    "origin": "source",
                    "status": "unknown",
                    "generation": 0,
                    "note": "用户在对话中的原始陈述；具体命题状态由提取器判断",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            assistant_evidence = json.dumps(
                {
                    "origin": "system",
                    "status": "mixed",
                    "generation": 1,
                    "note": "助手生成的回答；可能包含总结或推论，不等同于用户亲历事实",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            user_record = json.dumps(
                {
                    "exchange": exchange_number,
                    "timestamp_utc": now,
                    "platform": identity.platform,
                    "platform_user_id": identity.platform_user_id,
                    "display_name": identity.display_name or "unknown",
                    "role": "user",
                    "content": user_text.strip(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            assistant_record = json.dumps(
                {
                    "exchange": exchange_number,
                    "timestamp_utc": now,
                    "platform": identity.platform,
                    "platform_user_id": identity.platform_user_id,
                    "role": "assistant",
                    "content": assistant_text.strip(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            block = (
                f"[[memory {user_evidence}]]\n"
                f"conversation_record: {user_record}\n\n"
                f"[[memory {assistant_evidence}]]\n"
                f"conversation_record: {assistant_record}\n\n"
            )
            await asyncio.to_thread(
                path.write_text, existing + block, encoding="utf-8"
            )
            new_size = len(existing) + len(block)
            if exchange_number >= self.batch_exchanges or new_size >= self.batch_chars:
                return await asyncio.to_thread(self._finalize_pending, identity)
            return None

    async def flush_all(self) -> list[Path]:
        ready: list[Path] = []
        user_dirs = await asyncio.to_thread(
            lambda: [path.parent for path in self.root.glob("*/*/identity.json")]
        )
        for directory in user_dirs:
            try:
                identity = PlatformIdentity.from_metadata(directory)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            async with self._locks[identity.key]:
                path = await asyncio.to_thread(self._finalize_pending, identity)
                if path is not None:
                    ready.append(path)
        return ready

    def ready_files(self) -> list[Path]:
        return sorted(self.root.glob("*/*/*.ready.txt"))


class ConversationIngestionWorker:
    """Imports ready transcript batches through the same memory pipeline."""

    def __init__(
        self,
        journal: ConversationJournal,
        import_file: Callable[[Path, Path], Awaitable[dict]],
    ):
        self.journal = journal
        self.import_file = import_file
        self._queue: asyncio.Queue[Path | None] = asyncio.Queue()
        self._queued: set[Path] = set()
        self._task: asyncio.Task | None = None
        self.last_error = ""
        self.imported_batches = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="memory-ingestion-worker")
        for path in self.journal.ready_files():
            await self.enqueue(path)

    async def enqueue(self, path: Path | None) -> None:
        if path is None:
            return
        resolved = path.resolve()
        if resolved in self._queued:
            return
        self._queued.add(resolved)
        await self._queue.put(resolved)

    async def _run(self) -> None:
        while True:
            path = await self._queue.get()
            if path is None:
                self._queue.task_done()
                return
            try:
                result = await self.import_file(path, self.journal.root)
                failed = int(result.get("failed_tasks", 0)) + int(
                    result.get("failed_paragraph_sources", 0)
                )
                suffix = ".imported.json" if failed == 0 else ".partial.json"
                receipt = path.with_name(path.name.removesuffix(".ready.txt") + suffix)
                receipt.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )
                path.replace(path.with_suffix(".source.txt"))
                self.imported_batches += 1
                self.last_error = "" if failed == 0 else f"partial import: {failed} failures"
            except Exception as exc:
                # Keep .ready.txt untouched so the next startup retries it.
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queued.discard(path)
                self._queue.task_done()

    async def close(self, *, flush_pending: bool = True) -> None:
        if flush_pending:
            for path in await self.journal.flush_all():
                await self.enqueue(path)
        await self._queue.join()
        if self._task is not None:
            await self._queue.put(None)
            await self._task
            self._task = None
