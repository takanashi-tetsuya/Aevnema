from __future__ import annotations

import asyncio
import os

from src.bot.event_dedup import EventClaim
from src.bot.runtime import AdapterRuntime, SendText, SetTyping
from src.memory.identity import PlatformIdentity
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DebouncedDispatcher:
    """Merge rapid messages per conversation while serializing each identity."""

    def __init__(self, runtime: AdapterRuntime, wait_seconds: float | None = None):
        self.runtime = runtime
        self.wait_seconds = (
            float(os.getenv("BOT_DEBOUNCE_WAIT", "2.0"))
            if wait_seconds is None
            else wait_seconds
        )
        self._buffers: dict[str, list[str]] = {}
        self._images: dict[str, list[tuple[bytes, str]]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._versions: dict[str, int] = {}
        self._claims: dict[str, list[EventClaim]] = {}

    def submit(
        self,
        *,
        conversation_key: str,
        identity: PlatformIdentity,
        text: str,
        send_text: SendText,
        set_typing: SetTyping | None = None,
        images: list[tuple[bytes, str]] | None = None,
        event_id: str | None = None,
        event_scope: str | None = None,
    ) -> bool:
        if event_id:
            claim = self.runtime.event_deduplicator.claim(
                identity.platform,
                event_scope or conversation_key,
                event_id,
            )
            if claim is None:
                logger.info(
                    "忽略持久化去重命中的 %s 事件：%s",
                    identity.platform,
                    event_id,
                )
                return False
            self._claims.setdefault(conversation_key, []).append(claim)
        if text.strip():
            self._buffers.setdefault(conversation_key, []).append(text.strip())
        if images:
            self._images.setdefault(conversation_key, []).extend(images)
        self._versions[conversation_key] = self._versions.get(conversation_key, 0) + 1
        if conversation_key in self._tasks:
            return

        async def worker() -> None:
            active_claims: list[EventClaim] = []
            try:
                while self._buffers.get(conversation_key) or self._images.get(
                    conversation_key
                ):
                    version = self._versions[conversation_key]
                    await asyncio.sleep(self.wait_seconds)
                    if self._versions.get(conversation_key) != version:
                        continue
                    texts = self._buffers.pop(conversation_key, [])
                    image_rows = self._images.pop(conversation_key, [])
                    active_claims = self._claims.pop(conversation_key, [])
                    lock = self._locks.setdefault(conversation_key, asyncio.Lock())
                    async with lock:
                        completed = await self.runtime.process_message(
                            identity=identity,
                            conversation_key=conversation_key,
                            text="\n".join(texts),
                            images=image_rows,
                            send_text=send_text,
                            set_typing=set_typing,
                        )
                    for claim in active_claims:
                        if completed:
                            self.runtime.event_deduplicator.complete(claim)
                        else:
                            self.runtime.event_deduplicator.release(claim)
                    active_claims = []
            except BaseException:
                for claim in active_claims:
                    self.runtime.event_deduplicator.release(claim)
                for claim in self._claims.pop(conversation_key, []):
                    self.runtime.event_deduplicator.release(claim)
                raise
            finally:
                if self._tasks.get(conversation_key) is asyncio.current_task():
                    self._tasks.pop(conversation_key, None)
                    self._versions.pop(conversation_key, None)

        self._tasks[conversation_key] = asyncio.create_task(worker())
        return True

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._buffers.clear()
        self._images.clear()
        for claims in self._claims.values():
            for claim in claims:
                self.runtime.event_deduplicator.release(claim)
        self._claims.clear()
        self._versions.clear()
