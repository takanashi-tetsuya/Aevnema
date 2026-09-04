from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import time
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
    import aiohttp
    import dingtalk_stream

    _DINGTALK_AVAILABLE = True
    _ChatbotHandler = dingtalk_stream.ChatbotHandler
except ModuleNotFoundError:  # Policy helpers remain importable without the SDK.
    aiohttp = dingtalk_stream = Any  # type: ignore[assignment]
    _ChatbotHandler = object
    _DINGTALK_AVAILABLE = False


def _csv_env(name: str) -> set[str]:
    return {item.strip() for item in os.getenv(name, "").split(",") if item.strip()}


def dingtalk_identity(
    sender_corp_id: str,
    sender_staff_id: str,
    sender_id: str,
    display_name: str = "",
) -> PlatformIdentity:
    """Namespace staffId by corp; fall back to DingTalk's opaque senderId."""

    corp = sender_corp_id.strip()
    staff = sender_staff_id.strip()
    stable_id = f"{corp}:{staff}" if corp and staff else sender_id.strip() or staff
    return PlatformIdentity("dingtalk", stable_id, display_name.strip() or stable_id)


def dingtalk_message_allows(
    *, conversation_type: str, is_in_at_list: bool, group_mode: str
) -> bool:
    mode = group_mode.strip().casefold()
    if mode not in {"mentions", "all", "disabled"}:
        raise ValueError("DINGTALK_GROUP_MODE must be mentions, all, or disabled")
    if str(conversation_type) == "1":
        return True
    if mode == "all":
        return True
    if mode == "disabled":
        return False
    return bool(is_in_at_list)


@dataclass(frozen=True, slots=True)
class DingTalkInbound:
    message_id: str
    user_key: str
    display_name: str
    conversation_id: str
    conversation_type: str
    text: str
    is_in_at_list: bool
    session_webhook: str
    session_webhook_expired_time: int
    sender_staff_id: str


def parse_dingtalk_message(message: Any) -> DingTalkInbound | None:
    text_content = getattr(getattr(message, "text", None), "content", "")
    sender_id = str(getattr(message, "sender_id", "") or "")
    staff_id = str(getattr(message, "sender_staff_id", "") or "")
    corp_id = str(getattr(message, "sender_corp_id", "") or "")
    identity = dingtalk_identity(
        corp_id,
        staff_id,
        sender_id,
        str(getattr(message, "sender_nick", "") or ""),
    )
    conversation_id = str(getattr(message, "conversation_id", "") or "")
    webhook = str(getattr(message, "session_webhook", "") or "")
    if not identity.platform_user_id or not conversation_id or not webhook:
        return None
    return DingTalkInbound(
        message_id=str(getattr(message, "message_id", "") or ""),
        user_key=identity.platform_user_id,
        display_name=identity.display_name,
        conversation_id=conversation_id,
        conversation_type=str(getattr(message, "conversation_type", "") or ""),
        text=str(text_content or "").strip(),
        is_in_at_list=bool(getattr(message, "is_in_at_list", False)),
        session_webhook=webhook,
        session_webhook_expired_time=int(
            getattr(message, "session_webhook_expired_time", 0) or 0
        ),
        sender_staff_id=staff_id,
    )


class DingTalkAdapter(_ChatbotHandler):
    """DingTalk Stream callback handler with non-blocking acknowledgement."""

    def __init__(
        self,
        runtime: AdapterRuntime,
        *,
        http_session: Any | None = None,
    ):
        super().__init__()
        self.runtime = runtime
        self.dispatcher = DebouncedDispatcher(runtime)
        self.group_mode = os.getenv("DINGTALK_GROUP_MODE", "mentions").casefold()
        dingtalk_message_allows(
            conversation_type="1", is_in_at_list=False, group_mode=self.group_mode
        )
        self.allowed_users = whitelist_values("dingtalk", "DINGTALK_ALLOWED_USERS")
        self.allowed_conversations = whitelist_values(
            "dingtalk", "DINGTALK_ALLOWED_CONVERSATIONS"
        )
        self._session = http_session
        self._owns_session = http_session is None

    def _accept(self, message: DingTalkInbound) -> bool:
        if self.allowed_users and message.user_key not in self.allowed_users:
            return False
        if (
            self.allowed_conversations
            and message.conversation_id not in self.allowed_conversations
        ):
            return False
        if not dingtalk_message_allows(
            conversation_type=message.conversation_type,
            is_in_at_list=message.is_in_at_list,
            group_mode=self.group_mode,
        ):
            return False
        return True

    async def _session_for_send(self) -> Any:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _send_text(self, message: DingTalkInbound, text: str) -> None:
        if (
            message.session_webhook_expired_time
            and int(time.time() * 1000) >= message.session_webhook_expired_time
        ):
            raise RuntimeError("DingTalk sessionWebhook expired before the reply")
        session = await self._session_for_send()
        limit = int(os.getenv("DINGTALK_MESSAGE_LIMIT", "4000"))
        for chunk in split_message(text, limit):
            payload: dict[str, Any] = {
                "msgtype": "text",
                "text": {"content": chunk},
            }
            if message.sender_staff_id:
                payload["at"] = {"atUserIds": [message.sender_staff_id]}
            async with session.post(message.session_webhook, json=payload) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(
                        f"DingTalk send failed ({response.status}): {body[:500]}"
                    )
                if body:
                    try:
                        result = await response.json(content_type=None)
                    except Exception:
                        result = {}
                    if isinstance(result, dict) and int(result.get("errcode", 0) or 0):
                        raise RuntimeError(f"DingTalk send failed: {result}")

    async def process(self, callback: Any) -> tuple[int, str]:
        """Queue work and return the Stream ACK before the LLM is invoked."""

        try:
            raw = callback.data
            if isinstance(raw, str):
                import json

                raw = json.loads(raw)
            incoming = dingtalk_stream.ChatbotMessage.from_dict(raw)
            message = parse_dingtalk_message(incoming)
            if message is not None and message.text and self._accept(message):
                identity = PlatformIdentity(
                    "dingtalk", message.user_key, message.display_name
                )
                self.dispatcher.submit(
                    conversation_key=f"{message.conversation_id}:{identity.key}",
                    identity=identity,
                    text=message.text,
                    send_text=lambda value: self._send_text(message, value),
                    event_id=message.message_id or None,
                    event_scope=message.conversation_id,
                )
        except Exception:
            # Malformed callbacks are acknowledged to prevent an endless retry storm.
            logger.exception("解析钉钉 Stream 消息失败")
        return dingtalk_stream.AckMessage.STATUS_OK, "OK"

    async def close(self) -> None:
        await self.dispatcher.close()
        if self._owns_session and self._session is not None:
            await self._session.close()


async def _close_dingtalk_transport(client: Any, task: asyncio.Task[Any]) -> None:
    websocket = getattr(client, "websocket", None)
    if websocket is not None:
        try:
            await websocket.close()
        except Exception:
            logger.exception("关闭钉钉 WebSocket 失败")
    # dingtalk-stream 0.24.3 has no stop() and consumes one CancelledError.
    task.cancel()
    await asyncio.sleep(0)
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def start_dingtalk_adapter(runtime: AdapterRuntime | None = None) -> None:
    if not _DINGTALK_AVAILABLE:
        raise RuntimeError("请先安装官方依赖：pip install dingtalk-stream")
    load_dotenv(PROJECT_ROOT / ".env")
    client_id = os.getenv("DINGTALK_CLIENT_ID", "").strip()
    client_secret = os.getenv("DINGTALK_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError("DINGTALK_CLIENT_ID and DINGTALK_CLIENT_SECRET are required")
    require_access_policy(
        "dingtalk", "DINGTALK_ALLOWED_USERS", "DINGTALK_ALLOWED_CONVERSATIONS"
    )
    client = dingtalk_stream.DingTalkStreamClient(
        dingtalk_stream.Credential(client_id, client_secret)
    )
    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter = DingTalkAdapter(active_runtime)
    client.register_callback_handler(dingtalk_stream.ChatbotMessage.TOPIC, adapter)
    stream_task = asyncio.create_task(client.start())
    try:
        # Shield lets this wrapper observe cancellation even though SDK start()
        # catches CancelledError internally.
        await asyncio.shield(stream_task)
    finally:
        await _close_dingtalk_transport(client, stream_task)
        await adapter.close()
        if owns_runtime:
            await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_dingtalk_adapter())
