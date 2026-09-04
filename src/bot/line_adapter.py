from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
import hashlib
import hmac
import os
from typing import Any

from dotenv import load_dotenv

from src.bot.adapter_support import (
    AdapterRuntime,
    DebouncedDispatcher,
    PROJECT_ROOT,
    require_access_policy,
    split_message,
    whitelist_values,
)
from src.memory import PlatformIdentity
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

try:
    from aiohttp import web
    from linebot.v3 import WebhookParser
    from linebot.v3.messaging import (
        AsyncApiClient,
        AsyncMessagingApi,
        AsyncMessagingApiBlob,
        Configuration,
        PushMessageRequest,
        ReplyMessageRequest,
        TextMessage,
    )

    _LINE_AVAILABLE = True
except ModuleNotFoundError:  # Signature and policy helpers remain importable.
    web = Any  # type: ignore[assignment]
    WebhookParser = AsyncApiClient = AsyncMessagingApi = Any  # type: ignore[assignment,misc]
    AsyncMessagingApiBlob = Configuration = Any  # type: ignore[assignment,misc]
    PushMessageRequest = ReplyMessageRequest = TextMessage = Any  # type: ignore[assignment,misc]
    _LINE_AVAILABLE = False


def _csv_env(name: str) -> set[str]:
    return {item.strip() for item in os.getenv(name, "").split(",") if item.strip()}


def verify_line_signature(body: bytes, signature: str, channel_secret: str) -> bool:
    """Verify LINE's base64(HMAC-SHA256(raw request body)) signature."""

    if not signature or not channel_secret:
        return False
    expected = base64.b64encode(
        hmac.new(channel_secret.encode("utf-8"), body, hashlib.sha256).digest()
    ).decode("ascii")
    return hmac.compare_digest(expected, signature.strip())


def line_identity(user_id: str, display_name: str = "") -> PlatformIdentity:
    """LINE userId is stable within a provider and must not be replaced by a name."""

    stable_id = user_id.strip()
    return PlatformIdentity("line", stable_id, display_name.strip() or stable_id)


def line_message_allows(
    *, source_type: str, mentioned_bot: bool, group_mode: str
) -> bool:
    mode = group_mode.strip().casefold()
    if mode not in {"mentions", "all", "disabled"}:
        raise ValueError("LINE_GROUP_MODE must be mentions, all, or disabled")
    if source_type.strip().casefold() == "user":
        return True
    if mode == "all":
        return True
    if mode == "disabled":
        return False
    return bool(mentioned_bot)


def _line_target(source: Any) -> str:
    source_type = str(getattr(source, "type", "") or "").casefold()
    if source_type == "group":
        return str(getattr(source, "group_id", "") or "")
    if source_type == "room":
        return str(getattr(source, "room_id", "") or "")
    return str(getattr(source, "user_id", "") or "")


def _line_bot_mentioned(message: Any) -> bool:
    mention = getattr(message, "mention", None)
    mentionees = getattr(mention, "mentionees", []) or []
    return any(bool(getattr(row, "is_self", False)) for row in mentionees)


def _strip_line_bot_mentions(text: str, message: Any) -> str:
    mention = getattr(message, "mention", None)
    rows = [
        row
        for row in (getattr(mention, "mentionees", []) or [])
        if bool(getattr(row, "is_self", False))
    ]
    # LINE indexes are offsets into the message string. Remove backwards so
    # multiple mentions do not shift later ranges.
    for row in sorted(rows, key=lambda item: int(getattr(item, "index", 0)), reverse=True):
        start = int(getattr(row, "index", 0) or 0)
        end = start + int(getattr(row, "length", 0) or 0)
        if 0 <= start <= end <= len(text):
            text = text[:start] + text[end:]
    return text.lstrip(" :,-\t")


@dataclass(slots=True)
class _LineReplyContext:
    target: str
    reply_token: str
    reply_used: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LineAdapter:
    """LINE Messaging API webhook adapter using the official asynchronous SDK."""

    def __init__(
        self,
        channel_secret: str,
        messaging_api: Any,
        blob_api: Any,
        runtime: AdapterRuntime,
        *,
        parser: Any | None = None,
    ):
        self.channel_secret = channel_secret
        self.messaging_api = messaging_api
        self.blob_api = blob_api
        self.runtime = runtime
        self.parser = parser or WebhookParser(channel_secret)
        self.dispatcher = DebouncedDispatcher(runtime)
        self.group_mode = os.getenv("LINE_GROUP_MODE", "mentions").casefold()
        line_message_allows(
            source_type="user", mentioned_bot=False, group_mode=self.group_mode
        )
        self.allowed_users = whitelist_values("line", "LINE_ALLOWED_USERS")
        self.allowed_chats = whitelist_values("line", "LINE_ALLOWED_CHATS")
        self._tasks: set[asyncio.Task[None]] = set()

    def _accept(self, event: Any) -> bool:
        source = getattr(event, "source", None)
        message = getattr(event, "message", None)
        source_type = str(getattr(source, "type", "") or "")
        user_id = str(getattr(source, "user_id", "") or "")
        target = _line_target(source)
        if not user_id or not target:
            return False
        if self.allowed_users and user_id not in self.allowed_users:
            return False
        if self.allowed_chats and target not in self.allowed_chats:
            return False
        if not line_message_allows(
            source_type=source_type,
            mentioned_bot=_line_bot_mentioned(message),
            group_mode=self.group_mode,
        ):
            return False
        return True

    def accept_webhook(self, body: bytes, signature: str) -> int:
        """Validate, parse and enqueue without waiting for model work."""

        if not verify_line_signature(body, signature, self.channel_secret):
            raise ValueError("invalid LINE webhook signature")
        events = self.parser.parse(body.decode("utf-8"), signature)
        accepted = 0
        for event in events:
            if str(getattr(event, "type", "")) != "message" or not self._accept(event):
                continue
            message_type = str(getattr(event.message, "type", "") or "")
            if message_type == "text":
                self._dispatch_text(event)
                accepted += 1
            elif message_type == "image":
                task = asyncio.create_task(self._dispatch_image(event))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                accepted += 1
        return accepted

    def _reply_context(self, event: Any) -> _LineReplyContext:
        return _LineReplyContext(
            target=_line_target(event.source),
            reply_token=str(getattr(event, "reply_token", "") or ""),
        )

    def _submit(
        self,
        event: Any,
        *,
        text: str,
        images: list[tuple[bytes, str]] | None = None,
    ) -> None:
        user_id = str(event.source.user_id)
        target = _line_target(event.source)
        identity = line_identity(user_id)
        context = self._reply_context(event)
        self.dispatcher.submit(
            conversation_key=f"{target}:{identity.key}",
            identity=identity,
            text=text,
            images=images,
            send_text=lambda value: self._send_text(context, value),
            event_id=str(getattr(event, "webhook_event_id", "") or "") or None,
            event_scope=target,
        )

    def _dispatch_text(self, event: Any) -> None:
        text = str(getattr(event.message, "text", "") or "").strip()
        if str(getattr(event.source, "type", "")) != "user":
            text = _strip_line_bot_mentions(text, event.message)
        if text:
            self._submit(event, text=text)

    async def _dispatch_image(self, event: Any) -> None:
        maximum = int(os.getenv("LINE_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
        try:
            payload = bytes(
                await self.blob_api.get_message_content(str(event.message.id))
            )
        except Exception:
            logger.exception("LINE 图片下载失败")
            return
        if payload and len(payload) <= maximum:
            self._submit(event, text="", images=[(payload, "image/jpeg")])

    async def _push_chunks(self, target: str, chunks: list[str]) -> None:
        for offset in range(0, len(chunks), 5):
            messages = [TextMessage(text=value) for value in chunks[offset : offset + 5]]
            await self.messaging_api.push_message(
                PushMessageRequest(to=target, messages=messages)
            )

    async def _send_text(self, context: _LineReplyContext, text: str) -> None:
        chunks = split_message(text, int(os.getenv("LINE_MESSAGE_LIMIT", "5000")))
        async with context.lock:
            if context.reply_token and not context.reply_used:
                context.reply_used = True
                reply_chunks, chunks = chunks[:5], chunks[5:]
                try:
                    await self.messaging_api.reply_message(
                        ReplyMessageRequest(
                            replyToken=context.reply_token,
                            messages=[TextMessage(text=value) for value in reply_chunks],
                        )
                    )
                except Exception:
                    # Reply tokens are single-use and short-lived. Falling back to
                    # push keeps slow model responses deliverable.
                    logger.exception("LINE reply token 已失效，改用 push message")
                    chunks = reply_chunks + chunks
            if chunks:
                await self._push_chunks(context.target, chunks)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.dispatcher.close()


async def start_line_adapter(runtime: AdapterRuntime | None = None) -> None:
    if not _LINE_AVAILABLE:
        raise RuntimeError("请先安装官方依赖：pip install line-bot-sdk")
    load_dotenv(PROJECT_ROOT / ".env")
    channel_secret = os.getenv("LINE_CHANNEL_SECRET", "").strip()
    access_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "").strip()
    if not channel_secret or not access_token:
        raise RuntimeError(
            "LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN are required"
        )
    require_access_policy("line", "LINE_ALLOWED_USERS", "LINE_ALLOWED_CHATS")

    api_client = AsyncApiClient(Configuration(access_token=access_token))
    messaging_api = AsyncMessagingApi(api_client)
    blob_api = AsyncMessagingApiBlob(api_client)
    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter = LineAdapter(
        channel_secret, messaging_api, blob_api, active_runtime
    )

    async def webhook_handler(request: Any) -> Any:
        maximum = int(os.getenv("LINE_MAX_WEBHOOK_BYTES", str(1024 * 1024)))
        if request.content_length and request.content_length > maximum:
            raise web.HTTPRequestEntityTooLarge(
                max_size=maximum, actual_size=request.content_length
            )
        body = await request.read()
        if len(body) > maximum:
            raise web.HTTPRequestEntityTooLarge(max_size=maximum, actual_size=len(body))
        signature = request.headers.get("x-line-signature", "")
        try:
            adapter.accept_webhook(body, signature)
        except (ValueError, UnicodeDecodeError):
            raise web.HTTPBadRequest(text="invalid LINE webhook")
        return web.Response(text="OK")

    application = web.Application(client_max_size=int(os.getenv("LINE_MAX_WEBHOOK_BYTES", str(1024 * 1024))))
    application.router.add_post(os.getenv("LINE_WEBHOOK_PATH", "/line/webhook"), webhook_handler)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(
        runner,
        os.getenv("LINE_WEBHOOK_HOST", "127.0.0.1"),
        int(os.getenv("LINE_WEBHOOK_PORT", "8081")),
    )
    try:
        await site.start()
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()
        await adapter.close()
        await api_client.close()
        if owns_runtime:
            await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_line_adapter())
