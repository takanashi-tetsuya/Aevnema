from __future__ import annotations

import asyncio
from dataclasses import dataclass
import mimetypes
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
    from aibot import WSClient, WSClientOptions

    _WECOM_AVAILABLE = True
except ModuleNotFoundError:  # Parsers and policy helpers remain independently testable.
    WSClient = WSClientOptions = Any  # type: ignore[assignment,misc]
    _WECOM_AVAILABLE = False


def _csv_env(name: str) -> set[str]:
    return {item.strip() for item in os.getenv(name, "").split(",") if item.strip()}


def wecom_identity(user_id: str, display_name: str = "") -> PlatformIdentity:
    """Enterprise member userid is the official stable identity in one tenant."""

    stable_id = user_id.strip()
    return PlatformIdentity("wecom", stable_id, display_name.strip() or stable_id)


def wecom_message_allows(
    *, chat_type: str, mentioned_bot: bool, group_mode: str
) -> bool:
    mode = group_mode.strip().casefold()
    if mode not in {"mentions", "all", "disabled"}:
        raise ValueError("WECOM_GROUP_MODE must be mentions, all, or disabled")
    if chat_type.strip().casefold() in {"single", "private", "direct", "dm"}:
        return True
    if mode == "all":
        return True
    if mode == "disabled":
        return False
    return bool(mentioned_bot)


@dataclass(frozen=True, slots=True)
class WeComInbound:
    message_id: str
    user_id: str
    display_name: str
    chat_id: str
    chat_type: str
    text: str
    mentioned_bot: bool
    image_refs: tuple[tuple[str, str], ...]


def parse_wecom_frame(frame: dict[str, Any]) -> WeComInbound | None:
    """Normalize the official intelligent-robot WebSocket message body."""

    body = frame.get("body")
    if not isinstance(body, dict):
        return None
    sender = body.get("from") if isinstance(body.get("from"), dict) else {}
    user_id = str(sender.get("userid") or sender.get("user_id") or "").strip()
    if not user_id:
        return None
    chat_type = str(body.get("chattype") or "single").strip().casefold()
    chat_id = str(body.get("chatid") or user_id).strip()
    message_type = str(body.get("msgtype") or "").casefold()
    text_parts: list[str] = []
    image_refs: list[tuple[str, str]] = []

    def consume(item: dict[str, Any]) -> None:
        kind = str(item.get("msgtype") or "").casefold()
        if kind == "text" and isinstance(item.get("text"), dict):
            value = str(item["text"].get("content") or "").strip()
            if value:
                text_parts.append(value)
        elif kind == "image" and isinstance(item.get("image"), dict):
            image = item["image"]
            url = str(image.get("url") or "").strip()
            if url:
                image_refs.append((url, str(image.get("aeskey") or "")))

    if message_type == "mixed" and isinstance(body.get("mixed"), dict):
        rows = body["mixed"].get("msg_item") or body["mixed"].get("msgitem") or []
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict):
                consume(row)
    else:
        consume(body)

    explicit_mention = body.get("mentioned_bot")
    if explicit_mention is None:
        explicit_mention = body.get("is_at")
    # In group chat the intelligent-robot platform only routes messages aimed at
    # the bot; older payload revisions do not include an explicit mention flag.
    mentioned_bot = True if explicit_mention is None else bool(explicit_mention)
    return WeComInbound(
        message_id=str(body.get("msgid") or ""),
        user_id=user_id,
        display_name=str(sender.get("name") or sender.get("alias") or user_id),
        chat_id=chat_id,
        chat_type=chat_type,
        text="\n".join(text_parts),
        mentioned_bot=mentioned_bot,
        image_refs=tuple(image_refs),
    )


class WeComAdapter:
    """Official WeCom intelligent-robot long-connection adapter."""

    def __init__(self, client: Any, runtime: AdapterRuntime):
        self.client = client
        self.runtime = runtime
        self.dispatcher = DebouncedDispatcher(runtime)
        self.group_mode = os.getenv("WECOM_GROUP_MODE", "mentions").casefold()
        wecom_message_allows(
            chat_type="single", mentioned_bot=False, group_mode=self.group_mode
        )
        self.allowed_users = whitelist_values("wecom", "WECOM_ALLOWED_USERS")
        self.allowed_chats = whitelist_values("wecom", "WECOM_ALLOWED_CHATS")
        self._tasks: set[asyncio.Task[None]] = set()
        for event in ("message.text", "message.image", "message.mixed"):
            client.on(event, self._schedule_frame)

    def _accept(self, message: WeComInbound) -> bool:
        if self.allowed_users and message.user_id not in self.allowed_users:
            return False
        if self.allowed_chats and message.chat_id not in self.allowed_chats:
            return False
        if not wecom_message_allows(
            chat_type=message.chat_type,
            mentioned_bot=message.mentioned_bot,
            group_mode=self.group_mode,
        ):
            return False
        return True

    def _schedule_frame(self, frame: dict[str, Any]) -> None:
        """Return immediately to the SDK while media work continues asynchronously."""

        message = parse_wecom_frame(frame)
        if message is None or not self._accept(message):
            return
        task = asyncio.create_task(self._dispatch(message))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _download_images(
        self, refs: tuple[tuple[str, str], ...]
    ) -> list[tuple[bytes, str]]:
        maximum = int(os.getenv("WECOM_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
        images: list[tuple[bytes, str]] = []
        for url, aes_key in refs:
            try:
                payload, filename = await self.client.download_file(url, aes_key or None)
            except Exception:
                logger.exception("企业微信图片下载或解密失败")
                continue
            payload = bytes(payload)
            if not payload or len(payload) > maximum:
                continue
            mime_type = mimetypes.guess_type(str(filename or ""))[0] or "image/jpeg"
            if mime_type.startswith("image/"):
                images.append((payload, mime_type))
        return images

    async def _send_text(self, target_id: str, text: str) -> None:
        limit = int(os.getenv("WECOM_MESSAGE_LIMIT", "4000"))
        for chunk in split_message(text, limit):
            await self.client.send_message(
                target_id, {"msgtype": "markdown", "markdown": {"content": chunk}}
            )

    async def _dispatch(self, message: WeComInbound) -> None:
        images = await self._download_images(message.image_refs)
        if not message.text and not images:
            return
        identity = wecom_identity(message.user_id, message.display_name)
        target_id = message.chat_id or message.user_id
        self.dispatcher.submit(
            conversation_key=f"{target_id}:{identity.key}",
            identity=identity,
            text=message.text,
            images=images,
            send_text=lambda value: self._send_text(target_id, value),
            event_id=message.message_id or None,
            event_scope=str(getattr(self.client, "bot_id", "") or "wecom-bot"),
        )

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        await self.dispatcher.close()


async def start_wecom_adapter(runtime: AdapterRuntime | None = None) -> None:
    if not _WECOM_AVAILABLE:
        raise RuntimeError(
            "请先安装企业微信官方依赖：pip install wecom-aibot-python-sdk"
        )
    load_dotenv(PROJECT_ROOT / ".env")
    bot_id = os.getenv("WECOM_BOT_ID", "").strip()
    secret = os.getenv("WECOM_SECRET", "").strip()
    if not bot_id or not secret:
        raise RuntimeError("WECOM_BOT_ID and WECOM_SECRET are required")
    require_access_policy("wecom", "WECOM_ALLOWED_USERS", "WECOM_ALLOWED_CHATS")
    kwargs: dict[str, Any] = {"bot_id": bot_id, "secret": secret}
    ws_url = os.getenv("WECOM_WS_URL", "").strip()
    if ws_url:
        kwargs["ws_url"] = ws_url
    client = WSClient(WSClientOptions(**kwargs))
    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter = WeComAdapter(client, active_runtime)
    try:
        await client.connect()
        await asyncio.Event().wait()
    finally:
        await adapter.close()
        client.disconnect()
        if owns_runtime:
            await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_wecom_adapter())
