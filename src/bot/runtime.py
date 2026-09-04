from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from config.prompt_config import VISION_SYSTEM, VISION_USER
from src.bot.access import _boolean_env
from src.bot.chat_service import ConversationCoordinator
from src.bot.event_dedup import PersistentEventDeduplicator
from src.bot.memory_guard import PrivateMemoryResponseGuard
from src.bot.persona import (
    apply_model_setting,
    get_dynamic_system_prompt,
    get_private_memory_emergency_reply,
)
from src.bot.request_planning import ChatRequestPlanner
from src.llm.engine import EngineFactory, Message
from src.llm.user_settings import UserModelSettingsStore
from src.memory import (
    BackgroundGrowthWorker,
    MemoryIntentPlanner,
    MemorySystem,
    MemorySystemConfig,
    PlatformIdentity,
)
from src.memory.conversation import (
    ConversationIngestionWorker,
    ConversationJournal,
    ConversationSessionBuffer,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SendText = Callable[[str], Awaitable[None]]
SetTyping = Callable[[bool], Awaitable[None]]


def _project_path(env_name: str, default: str) -> Path:
    configured = Path(os.getenv(env_name, default))
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


@dataclass(slots=True)
class AdapterRuntime:
    """Process-wide services injected into every enabled transport adapter."""

    memory: MemorySystem
    journal: ConversationJournal
    ingestion_worker: ConversationIngestionWorker
    growth_worker: BackgroundGrowthWorker
    model_settings_store: UserModelSettingsStore
    coordinator: ConversationCoordinator
    vision_engine: Any
    semaphore: asyncio.Semaphore
    event_deduplicator: PersistentEventDeduplicator
    _closed: bool = False

    @classmethod
    async def create(cls) -> "AdapterRuntime":
        batch_exchanges = int(os.getenv("MEMORY_CONVERSATION_BATCH_EXCHANGES", "4"))
        batch_chars = int(os.getenv("MEMORY_CONVERSATION_BATCH_CHARS", "3000"))
        growth_max_queue = int(os.getenv("MEMORY_GROWTH_QUEUE_MAX", "100"))
        growth_min_chars = int(os.getenv("MEMORY_GROWTH_MIN_CHARS", "18"))
        working_memory_window = int(os.getenv("WORKING_MEMORY_WINDOW", "20"))
        memory_intent_enabled = _boolean_env("MEMORY_INTENT_ENABLED", True)
        memory_intent_context_messages = int(
            os.getenv("MEMORY_INTENT_CONTEXT_MESSAGES", "6")
        )
        memory_intent_context_chars = int(
            os.getenv("MEMORY_INTENT_CONTEXT_CHARS", "4000")
        )
        max_concurrency = int(os.getenv("BOT_MAX_CONCURRENCY", "4"))
        event_ttl = float(os.getenv("ADAPTER_EVENT_TTL_SECONDS", "604800"))
        event_lease = float(os.getenv("ADAPTER_EVENT_LEASE_SECONDS", "900"))
        if max_concurrency < 1:
            raise ValueError("BOT_MAX_CONCURRENCY must be positive")

        memory = MemorySystem(MemorySystemConfig.from_env(PROJECT_ROOT))
        logger.info("正在为平台适配器加载公共记忆与共享剧情索引……")
        await memory.initialize()
        ingestion_worker: ConversationIngestionWorker | None = None
        growth_worker: BackgroundGrowthWorker | None = None
        event_deduplicator: PersistentEventDeduplicator | None = None
        try:
            journal = ConversationJournal(
                _project_path("CONVERSATION_INBOX_DIR", "data/conversation_inbox"),
                batch_exchanges=batch_exchanges,
                batch_chars=batch_chars,
            )
            ingestion_worker = ConversationIngestionWorker(
                journal, memory.import_conversation_file
            )
            await ingestion_worker.start()

            growth_worker = BackgroundGrowthWorker(
                _project_path("MEMORY_GROWTH_QUEUE_DIR", "data/growth_queue"),
                memory.grow,
                max_queue=growth_max_queue,
                min_chars=growth_min_chars,
            )
            await growth_worker.start()

            model_settings_store = UserModelSettingsStore(
                _project_path(
                    "USER_MODEL_SETTINGS_PATH", "data/user_model_settings.json"
                )
            )
            sessions = ConversationSessionBuffer(working_memory_window)
            request_planner = ChatRequestPlanner(
                sessions=sessions,
                intent_planner=(
                    MemoryIntentPlanner(EngineFactory.create("memory_intent"))
                    if memory_intent_enabled
                    else None
                ),
                enabled=memory_intent_enabled,
                context_messages=memory_intent_context_messages,
                context_chars=memory_intent_context_chars,
                semantic_embedder=memory.embed_query_text,
                association_route_matcher=memory.match_association_cue,
                semantic_cache_size=max(
                    0, int(os.getenv("MEMORY_SEMANTIC_PLAN_CACHE_SIZE", "64"))
                ),
                semantic_cache_ttl_seconds=max(
                    0.0,
                    float(
                        os.getenv(
                            "MEMORY_SEMANTIC_PLAN_CACHE_TTL_SECONDS", "900"
                        )
                    ),
                ),
                semantic_cache_similarity=float(
                    os.getenv("MEMORY_SEMANTIC_PLAN_CACHE_SIMILARITY", "0.60")
                ),
            )
            coordinator = ConversationCoordinator(
                chat_engine=EngineFactory.create("chat"),
                fast_chat_engine=EngineFactory.create("chat_fast"),
                memory_system=memory,
                system_prompt_factory=get_dynamic_system_prompt,
                private_memory_guard=PrivateMemoryResponseGuard(
                    audit_engine=EngineFactory.create("memory_audit"),
                    rewrite_engine=EngineFactory.create("memory_rewrite"),
                    emergency_reply_factory=get_private_memory_emergency_reply,
                ),
                sessions=sessions,
                journal=journal,
                ingestion_worker=ingestion_worker,
                growth_worker=growth_worker,
                request_planner=request_planner,
                generation_options_factory=model_settings_store.request_options,
            )
            event_deduplicator = PersistentEventDeduplicator(
                _project_path("ADAPTER_EVENT_DB_PATH", "data/adapter_events.db"),
                ttl_seconds=event_ttl,
                lease_seconds=event_lease,
            )
            return cls(
                memory=memory,
                journal=journal,
                ingestion_worker=ingestion_worker,
                growth_worker=growth_worker,
                model_settings_store=model_settings_store,
                coordinator=coordinator,
                vision_engine=EngineFactory.create("vision"),
                semaphore=asyncio.Semaphore(max_concurrency),
                event_deduplicator=event_deduplicator,
            )
        except BaseException:
            cleanup = []
            if ingestion_worker is not None:
                cleanup.append(ingestion_worker.close(flush_pending=True))
            if growth_worker is not None:
                cleanup.append(growth_worker.close())
            if cleanup:
                results = await asyncio.gather(*cleanup, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        logger.error("共享运行时初始化回滚失败：%s", result)
            if event_deduplicator is not None:
                with suppress(Exception):
                    event_deduplicator.close()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        results = await asyncio.gather(
            self.ingestion_worker.close(flush_pending=True),
            self.growth_worker.close(),
            return_exceptions=True,
        )
        try:
            self.event_deduplicator.close()
        except Exception as exc:
            results.append(exc)
        errors = [item for item in results if isinstance(item, Exception)]
        if errors:
            raise ExceptionGroup("failed to close adapter runtime", errors)

    async def memory_status(self, identity: PlatformIdentity) -> str:
        try:
            stats = await self.memory.stats(identity)

            def summary(label: str, value: dict[str, Any]) -> str:
                return (
                    f"{label}：{value['sources']} Source / "
                    f"{value['episodes']} Episode / {value['concepts']} Concept / "
                    f"{value['associations']['edges']} Association"
                )

            return "\n".join(
                (
                    f"身份：{identity.key}",
                    summary("剧情知识库", stats["knowledge"]),
                    summary("公共记忆库", stats["public"]),
                    summary("私人记忆库", stats["user"]),
                    f"本进程已导入对话批次：{self.ingestion_worker.imported_batches}",
                    f"最近导入错误：{self.ingestion_worker.last_error or '无'}",
                    f"后台增长队列：{self.growth_worker.queue_size}",
                    f"正在增长：{self.growth_worker.active_job_id or '无'}",
                    (
                        "已完成/失败："
                        f"{self.growth_worker.completed_jobs}/"
                        f"{self.growth_worker.failed_jobs}"
                    ),
                    f"最近增长错误：{self.growth_worker.last_error or '无'}",
                )
            )
        except Exception as exc:
            return f"读取长期记忆状态失败：{type(exc).__name__}: {exc}"

    async def command_reply(
        self,
        identity: PlatformIdentity,
        command_line: str,
        *,
        conversation_key: str | None = None,
    ) -> str | None:
        parts = command_line.strip().split()
        if not parts or not parts[0].startswith("/"):
            return None
        command = parts[0][1:].split("@", 1)[0].casefold()
        args = parts[1:]
        if command == "start":
            return (
                "你好，老师。你的稳定身份是 "
                f"{identity.platform}:{identity.platform_user_id}。私人记忆、公共记忆"
                "和剧情知识会分库存放。"
            )
        if command == "identity":
            return (
                f"平台：{identity.platform}\n"
                f"平台用户 ID：{identity.platform_user_id}\n"
                f"显示名：{identity.display_name}"
            )
        if command == "clear":
            if conversation_key is None:
                self.coordinator.clear_recent_context(identity)
            else:
                self.coordinator.clear_recent_context(
                    identity, conversation_key=conversation_key
                )
            return (
                "当前会话上下文已清空；已经形成的长期 Episode、Concept 和 "
                "Association 不会被删除。"
            )
        if command == "memory_status":
            return await self.memory_status(identity)
        if command == "model":
            return apply_model_setting(self.model_settings_store, identity, args)
        return (
            "未知命令。可用命令：/start、/identity、/clear、"
            "/memory_status、/model。"
        )

    async def describe_images(self, images: list[tuple[bytes, str]]) -> str:
        if not images:
            return ""
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": VISION_USER,
            }
        ]
        for payload, mime_type in images:
            encoded = base64.b64encode(payload).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
        return await self.vision_engine.generate_response(
            [Message(role="user", content=content)],
            system_prompt=VISION_SYSTEM,
            task_context="adapter:vision",
        )

    async def process_message(
        self,
        *,
        identity: PlatformIdentity,
        text: str,
        send_text: SendText,
        conversation_key: str | None = None,
        set_typing: SetTyping | None = None,
        images: list[tuple[bytes, str]] | None = None,
    ) -> bool:
        clean_text = text.strip()
        command = await self.command_reply(
            identity, clean_text, conversation_key=conversation_key
        )
        if command is not None:
            try:
                await send_text(command)
                return True
            except Exception:
                logger.exception("发送 %s 命令回复失败", identity.platform)
                return False

        image_rows = images or []
        if image_rows:
            try:
                description = await self.describe_images(image_rows)
                clean_text = (
                    f"{clean_text}\n\n[用户发送的图片内容：{description}]"
                    if clean_text
                    else f"[用户发送的图片内容：{description}]"
                )
            except Exception as exc:
                logger.exception("平台图片理解失败")
                clean_text = (
                    f"{clean_text}\n\n[用户发送了图片，但视觉解析失败：{exc}]"
                    if clean_text
                    else "[用户发送了图片，但视觉解析失败]"
                )
        if not clean_text:
            return True

        if set_typing is not None:
            with suppress(Exception):
                await set_typing(True)
        answer_send_started = False
        try:
            async with self.semaphore:
                reply = await self.coordinator.handle(
                    identity=identity,
                    text=clean_text,
                    conversation_key=conversation_key,
                    defer_commit=True,
                )
                if reply.memory.error:
                    logger.warning(
                        "memory retrieval degraded for %s: %s",
                        identity.key,
                        reply.memory.error,
                    )
                answer_send_started = True
                await send_text(reply.text)
                finalize = getattr(reply, "finalize", None)
                if finalize is not None:
                    await finalize()
                return True
        except Exception as exc:
            logger.exception("处理 %s 消息失败", identity.platform)
            if not answer_send_started:
                with suppress(Exception):
                    await send_text(f"这次处理失败了：{type(exc).__name__}: {exc}")
            return False
        finally:
            if set_typing is not None:
                with suppress(Exception):
                    await set_typing(False)


