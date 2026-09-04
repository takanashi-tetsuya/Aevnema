from __future__ import annotations

import asyncio
import base64
from collections import defaultdict
from contextlib import suppress
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.bot.adapter_support import (
    AdapterRuntime,
    require_access_policy,
    split_message,
    whitelist_enabled,
)
from src.bot.event_dedup import EventClaim
from src.bot.persona import (
    apply_model_setting,
)
from src.bot.chat_service import ConversationCoordinator
from src.llm.user_settings import (
    UserModelSettingsStore,
)
from src.memory import (
    BackgroundGrowthWorker,
    MemorySystem,
    PlatformIdentity,
)
from src.memory.conversation import (
    ConversationIngestionWorker,
)
from src.utils.logger import setup_logger


logger = setup_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

user_message_buffers: dict[str, list[str]] = defaultdict(list)
user_image_buffers: dict[str, list[tuple[str, str]]] = defaultdict(list)
user_debounce_tasks: dict[str, asyncio.Task] = {}
user_processing_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
user_event_claims: dict[str, list[EventClaim]] = defaultdict(list)

def telegram_identity(update: Update) -> PlatformIdentity:
    if update.effective_user is None:
        raise ValueError("Telegram update has no effective user")
    display_name = (
        update.effective_user.username
        or update.effective_user.full_name
        or str(update.effective_user.id)
    )
    return PlatformIdentity("telegram", str(update.effective_user.id), display_name)


def telegram_conversation_key(update: Update, identity: PlatformIdentity) -> str:
    if update.effective_chat is None:
        raise ValueError("Telegram update has no effective chat")
    thread_id = getattr(update.effective_message, "message_thread_id", None)
    return (
        f"telegram:{update.effective_chat.id}:"
        f"{thread_id if thread_id is not None else 'root'}:{identity.platform_user_id}"
    )


def _csv_env(name: str) -> set[str]:
    return {part.strip() for part in os.getenv(name, "").split(",") if part.strip()}


def _telegram_allowed(update: Update) -> bool:
    if not whitelist_enabled("telegram"):
        return True
    if update.effective_user is None or update.effective_chat is None:
        return False
    allowed_users = _csv_env("TELEGRAM_ALLOWED_USERS")
    allowed_chats = _csv_env("TELEGRAM_ALLOWED_CHATS")
    user_allowed = bool(allowed_users) and str(update.effective_user.id) in allowed_users
    chat_allowed = bool(allowed_chats) and str(update.effective_chat.id) in allowed_chats
    return user_allowed or chat_allowed


def _telegram_group_allowed(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> bool:
    if update.effective_chat is None or update.effective_message is None:
        return False
    if str(getattr(update.effective_chat, "type", "private")) == "private":
        return True
    mode = os.getenv("TELEGRAM_GROUP_MODE", "mentions").strip().casefold()
    if mode == "disabled":
        return False
    if mode == "all":
        return True
    if mode != "mentions":
        raise ValueError("TELEGRAM_GROUP_MODE must be mentions, all, or disabled")
    message = update.effective_message
    if str(message.text or "").startswith("/"):
        return True
    username = str(getattr(context.bot, "username", "") or "").strip()
    visible_text = f"{message.text or ''}\n{message.caption or ''}".casefold()
    if username and f"@{username}".casefold() in visible_text:
        return True
    reply = getattr(message, "reply_to_message", None)
    reply_user = getattr(reply, "from_user", None)
    return str(getattr(reply_user, "id", "")) == str(getattr(context.bot, "id", ""))


async def _send_text(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str) -> None:
    for chunk in split_message(text, 3_900):
        await context.bot.send_message(chat_id=chat_id, text=chunk)


async def _typing_keepalive(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("typing keepalive stopped: %s", exc)


def _coordinator(context: ContextTypes.DEFAULT_TYPE) -> ConversationCoordinator:
    return context.application.bot_data["coordinator"]


def _model_settings_store(
    context: ContextTypes.DEFAULT_TYPE,
) -> UserModelSettingsStore:
    return context.application.bot_data["model_settings_store"]


async def model_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not _telegram_allowed(update):
        return
    identity = telegram_identity(update)
    text = apply_model_setting(
        _model_settings_store(context), identity, list(context.args)
    )
    await _send_text(context, int(update.effective_chat.id), text)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _telegram_allowed(update):
        return
    identity = telegram_identity(update)
    await _send_text(
        context,
        int(update.effective_chat.id),
        (
            "你好，老师。你的稳定身份是 "
            f"{identity.platform}:{identity.platform_user_id}。私人记忆、公共记忆和剧情知识"
            "会分库存放。"
        ),
    )


async def identity_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not _telegram_allowed(update):
        return
    identity = telegram_identity(update)
    await _send_text(
        context,
        int(update.effective_chat.id),
        f"平台：{identity.platform}\n平台用户 ID：{identity.platform_user_id}\n显示名：{identity.display_name}",
    )


async def clear_recent_context(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if not _telegram_allowed(update):
        return
    identity = telegram_identity(update)
    _coordinator(context).clear_recent_context(
        identity,
        conversation_key=telegram_conversation_key(update, identity),
    )
    await _send_text(
        context,
        int(update.effective_chat.id),
        "当前会话上下文已清空；已经形成的长期 Episode、Concept 和 Association 不会被删除。",
    )


async def memory_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _telegram_allowed(update):
        return
    memory: MemorySystem = context.application.bot_data["memory_system"]
    worker: ConversationIngestionWorker = context.application.bot_data[
        "ingestion_worker"
    ]
    growth_worker: BackgroundGrowthWorker = context.application.bot_data[
        "growth_worker"
    ]
    identity = telegram_identity(update)
    try:
        stats = await memory.stats(identity)
        knowledge = stats["knowledge"]
        public = stats["public"]
        user = stats["user"]

        def summary(label: str, value: dict) -> str:
            return (
                f"{label}：{value['sources']} Source / {value['episodes']} Episode / "
                f"{value['concepts']} Concept / {value['associations']['edges']} Association"
            )

        text = "\n".join(
            (
                f"身份：{identity.key}",
                summary("剧情知识库", knowledge),
                summary("公共记忆库", public),
                summary("私人记忆库", user),
                f"本进程已导入对话批次：{worker.imported_batches}",
                f"最近导入错误：{worker.last_error or '无'}",
                f"后台增长队列：{growth_worker.queue_size}",
                f"正在增长：{growth_worker.active_job_id or '无'}",
                f"已完成/失败：{growth_worker.completed_jobs}/{growth_worker.failed_jobs}",
                f"最近增长错误：{growth_worker.last_error or '无'}",
            )
        )
    except Exception as exc:
        text = f"读取长期记忆状态失败：{type(exc).__name__}: {exc}"
    await _send_text(context, int(update.effective_chat.id), text)


async def handle_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Route every Telegram command through the platform-neutral runtime."""

    if (
        update.effective_message is None
        or update.effective_user is None
        or update.effective_chat is None
        or not _telegram_allowed(update)
        or not _telegram_group_allowed(update, context)
    ):
        return
    runtime: AdapterRuntime = context.application.bot_data["adapter_runtime"]
    identity = telegram_identity(update)
    conversation_key = telegram_conversation_key(update, identity)
    event_id = str(update.update_id)
    event_scope = str(getattr(context.bot, "id", "") or "telegram-bot")
    claim = runtime.event_deduplicator.claim("telegram", event_scope, event_id)
    if claim is None:
        return
    completed = False
    try:
        reply = await runtime.command_reply(
            identity,
            str(update.effective_message.text or ""),
            conversation_key=conversation_key,
        )
        if reply is not None:
            await _send_text(context, int(update.effective_chat.id), reply)
        completed = True
    finally:
        if completed:
            runtime.event_deduplicator.complete(claim)
        else:
            runtime.event_deduplicator.release(claim)


async def _process_merged_message(
    *,
    chat_id: int,
    identity: PlatformIdentity,
    user_text: str,
    images: list[tuple[str, str]],
    conversation_key: str,
    context: ContextTypes.DEFAULT_TYPE,
) -> bool:
    runtime: AdapterRuntime = context.application.bot_data["adapter_runtime"]
    decoded_images = [
        (base64.b64decode(encoded, validate=True), mime_type)
        for encoded, mime_type in images
    ]
    async with user_processing_locks[conversation_key]:
        typing_task = asyncio.create_task(_typing_keepalive(context, chat_id))
        try:
            return await runtime.process_message(
                identity=identity,
                conversation_key=conversation_key,
                text=user_text,
                images=decoded_images,
                send_text=lambda value: _send_text(context, chat_id, value),
            )
        finally:
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task


async def handle_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    if update.message is None or update.effective_user is None:
        return
    if not _telegram_allowed(update) or not _telegram_group_allowed(update, context):
        logger.warning("忽略未通过 Telegram 访问策略的消息")
        return
    chat_id = int(update.effective_chat.id)
    identity = telegram_identity(update)
    buffer_key = telegram_conversation_key(update, identity)

    pending_texts: list[str] = []
    pending_images: list[tuple[str, str]] = []
    max_image_bytes = int(
        os.getenv("TELEGRAM_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))
    )
    if update.message.text:
        pending_texts.append(update.message.text)
    if update.message.caption:
        pending_texts.append(update.message.caption)

    if update.message.photo:
        photo = update.message.photo[-1]
        if getattr(photo, "file_size", None) and photo.file_size > max_image_bytes:
            await _send_text(context, chat_id, "图片过大，无法处理。")
            return
        remote = await context.bot.get_file(photo.file_id)
        image_bytes = await remote.download_as_bytearray()
        if len(image_bytes) > max_image_bytes:
            await _send_text(context, chat_id, "图片过大，无法处理。")
            return
        pending_images.append(
            (base64.b64encode(image_bytes).decode("ascii"), "image/jpeg")
        )
    elif (
        update.message.sticker
        and not update.message.sticker.is_animated
        and not update.message.sticker.is_video
    ):
        if (
            getattr(update.message.sticker, "file_size", None)
            and update.message.sticker.file_size > max_image_bytes
        ):
            await _send_text(context, chat_id, "图片过大，无法处理。")
            return
        remote = await context.bot.get_file(update.message.sticker.file_id)
        image_bytes = await remote.download_as_bytearray()
        if len(image_bytes) > max_image_bytes:
            await _send_text(context, chat_id, "图片过大，无法处理。")
            return
        pending_images.append(
            (base64.b64encode(image_bytes).decode("ascii"), "image/webp")
        )

    runtime: AdapterRuntime = context.application.bot_data["adapter_runtime"]
    event_id = str(update.update_id)
    event_scope = str(getattr(context.bot, "id", "") or "telegram-bot")
    claim = runtime.event_deduplicator.claim("telegram", event_scope, event_id)
    if claim is None:
        logger.info("忽略持久化去重命中的 Telegram update：%s", event_id)
        return
    user_message_buffers[buffer_key].extend(pending_texts)
    user_image_buffers[buffer_key].extend(pending_images)
    user_event_claims[buffer_key].append(claim)

    existing = user_debounce_tasks.get(buffer_key)
    if existing is not None:
        existing.cancel()

    async def debounce_worker() -> None:
        claims: list[EventClaim] = []
        completed = False
        try:
            await asyncio.sleep(float(os.getenv("BOT_DEBOUNCE_WAIT", "2.0")))
            texts = user_message_buffers.pop(buffer_key, [])
            images = user_image_buffers.pop(buffer_key, [])
            claims = user_event_claims.pop(buffer_key, [])
            if not texts and not images:
                return
            merged = "\n".join(texts).strip()
            if not merged and images:
                merged = "[用户发送了图片或表情包]"
            completed = await _process_merged_message(
                chat_id=chat_id,
                identity=identity,
                user_text=merged,
                images=images,
                conversation_key=buffer_key,
                context=context,
            )
        except asyncio.CancelledError:
            raise
        finally:
            for event_claim in claims:
                if completed:
                    runtime.event_deduplicator.complete(event_claim)
                else:
                    runtime.event_deduplicator.release(event_claim)
            if user_debounce_tasks.get(buffer_key) is asyncio.current_task():
                user_debounce_tasks.pop(buffer_key, None)

    user_debounce_tasks[buffer_key] = asyncio.create_task(debounce_worker())


async def start_telegram_adapter(runtime: AdapterRuntime | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is not configured")
    require_access_policy(
        "telegram", "TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOWED_CHATS"
    )
    group_mode = os.getenv("TELEGRAM_GROUP_MODE", "mentions").strip().casefold()
    if group_mode not in {"mentions", "all", "disabled"}:
        raise ValueError("TELEGRAM_GROUP_MODE must be mentions, all, or disabled")

    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    application = ApplicationBuilder().token(token).build()
    application.bot_data.update(
        {
            "coordinator": active_runtime.coordinator,
            "memory_system": active_runtime.memory,
            "ingestion_worker": active_runtime.ingestion_worker,
            "growth_worker": active_runtime.growth_worker,
            "vision_engine": active_runtime.vision_engine,
            "model_settings_store": active_runtime.model_settings_store,
            "chat_semaphore": active_runtime.semaphore,
            "adapter_runtime": active_runtime,
        }
    )
    application.add_handler(MessageHandler(filters.COMMAND, handle_command))
    application.add_handler(MessageHandler(~filters.COMMAND, handle_message))

    logger.info("Telegram Adapter 正在连接……")
    initialized = False
    started = False
    polling = False
    try:
        await application.initialize()
        initialized = True
        await application.start()
        started = True
        if application.updater is None:
            raise RuntimeError("Telegram updater is unavailable")
        await application.updater.start_polling()
        polling = True
        logger.info("Telegram Adapter 已启动")
        await asyncio.Event().wait()
    finally:
        logger.info("正在停止 Telegram Adapter 并提交待处理对话批次……")
        pending_tasks = list(user_debounce_tasks.values())
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)
        for claims in user_event_claims.values():
            for claim in claims:
                active_runtime.event_deduplicator.release(claim)
        user_event_claims.clear()
        try:
            if polling and application.updater is not None:
                await application.updater.stop()
            if started:
                await application.stop()
            if initialized:
                await application.shutdown()
        finally:
            if owns_runtime:
                await active_runtime.close()
