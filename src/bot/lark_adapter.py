from __future__ import annotations

import asyncio
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
    from lark_channel import FeishuChannel, PolicyConfig

    _LARK_AVAILABLE = True
except ModuleNotFoundError:  # Pure policy helpers stay importable without the SDK.
    FeishuChannel = PolicyConfig = Any  # type: ignore[assignment,misc]
    _LARK_AVAILABLE = False


def _csv_env(name: str) -> set[str]:
    return {item.strip() for item in os.getenv(name, "").split(",") if item.strip()}


def lark_identity(open_id: str, display_name: str = "") -> PlatformIdentity:
    """Use the app-scoped open_id, never a mutable Feishu display name."""

    stable_id = open_id.strip()
    return PlatformIdentity("lark", stable_id, display_name.strip() or stable_id)


def lark_message_allows(
    *, chat_type: str, mentioned_bot: bool, group_mode: str
) -> bool:
    """Apply a conservative, structured-mention group policy."""

    mode = group_mode.strip().casefold()
    if mode not in {"mentions", "all", "disabled"}:
        raise ValueError("LARK_GROUP_MODE must be mentions, all, or disabled")
    if chat_type.strip().casefold() == "p2p":
        return True
    if mode == "all":
        return True
    if mode == "disabled":
        return False
    return bool(mentioned_bot)


class LarkAdapter:
    """Feishu/Lark adapter built on the official lark-channel SDK."""

    def __init__(self, channel: Any, runtime: AdapterRuntime):
        self.channel = channel
        self.runtime = runtime
        self.dispatcher = DebouncedDispatcher(runtime)
        self.group_mode = os.getenv("LARK_GROUP_MODE", "mentions").casefold()
        lark_message_allows(
            chat_type="p2p", mentioned_bot=False, group_mode=self.group_mode
        )
        self.allowed_users = whitelist_values("lark", "LARK_ALLOWED_USERS")
        self.allowed_chats = whitelist_values("lark", "LARK_ALLOWED_CHATS")
        self._unsubscribe = channel.on("message", self._on_message)

    def _accept(self, message: Any) -> bool:
        sender = getattr(message, "sender", None)
        sender_id = str(getattr(sender, "open_id", "") or "").strip()
        conversation = getattr(message, "conversation", None)
        chat_id = str(getattr(conversation, "chat_id", "") or "").strip()
        chat_type = str(getattr(conversation, "chat_type", "unknown") or "unknown")
        if not sender_id or not chat_id or bool(getattr(sender, "is_bot", False)):
            return False
        if self.allowed_users and sender_id not in self.allowed_users:
            return False
        if self.allowed_chats and chat_id not in self.allowed_chats:
            return False
        if not lark_message_allows(
            chat_type=chat_type,
            mentioned_bot=bool(getattr(message, "mentioned_bot", False)),
            group_mode=self.group_mode,
        ):
            return False
        return True

    async def _send_text(self, chat_id: str, reply_to: str, text: str) -> None:
        limit = int(os.getenv("LARK_MESSAGE_LIMIT", "4000"))
        for chunk in split_message(text, limit):
            opts = {"reply_target_gone": "fresh"}
            if reply_to:
                opts["reply_to"] = reply_to
            result = await self.channel.send(chat_id, {"text": chunk}, opts)
            if getattr(result, "success", True) is False:
                raise RuntimeError(f"Lark send failed: {getattr(result, 'error', result)}")

    async def _on_message(self, message: Any) -> None:
        if not self._accept(message):
            return
        text = str(
            getattr(message, "safe_content_text", "")
            or getattr(message, "body_text", "")
            or getattr(message, "content_text", "")
            or ""
        ).strip()
        if not text:
            return
        sender = message.sender
        conversation = message.conversation
        identity = lark_identity(
            str(sender.open_id), str(getattr(sender, "display_name", "") or "")
        )
        chat_id = str(conversation.chat_id)
        reply_to = str(getattr(message, "id", "") or "")
        self.dispatcher.submit(
            conversation_key=f"{chat_id}:{identity.key}",
            identity=identity,
            text=text,
            send_text=lambda value: self._send_text(chat_id, reply_to, value),
            event_id=reply_to or None,
            event_scope=chat_id,
        )

    async def close(self) -> None:
        await self.dispatcher.close()
        if callable(self._unsubscribe):
            self._unsubscribe()


async def start_lark_adapter(runtime: AdapterRuntime | None = None) -> None:
    """Start the adapter as a standalone long-lived WebSocket process."""

    if not _LARK_AVAILABLE:
        raise RuntimeError("请先安装官方依赖：pip install lark-channel-sdk")
    load_dotenv(PROJECT_ROOT / ".env")
    app_id = os.getenv("LARK_APP_ID", "").strip()
    app_secret = os.getenv("LARK_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        raise RuntimeError("LARK_APP_ID and LARK_APP_SECRET are required")
    require_access_policy("lark", "LARK_ALLOWED_USERS", "LARK_ALLOWED_CHATS")

    group_mode = os.getenv("LARK_GROUP_MODE", "mentions").casefold()
    lark_message_allows(chat_type="p2p", mentioned_bot=False, group_mode=group_mode)
    users = sorted(whitelist_values("lark", "LARK_ALLOWED_USERS"))
    chats = sorted(whitelist_values("lark", "LARK_ALLOWED_CHATS"))
    policy = PolicyConfig(
        dm_policy="allowlist" if users else "open",
        group_policy=(
            "disabled" if group_mode == "disabled" else "allowlist" if chats else "open"
        ),
        require_mention=group_mode == "mentions",
        allow_from=users or None,
        group_allowlist=chats or None,
    )
    kwargs: dict[str, Any] = {
        "app_id": app_id,
        "app_secret": app_secret,
        "policy": policy,
    }
    domain = os.getenv("LARK_DOMAIN", "").strip()
    if domain:
        kwargs["domain"] = domain
    channel = FeishuChannel(**kwargs)
    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter = LarkAdapter(channel, active_runtime)
    try:
        await channel.connect()
        await asyncio.Event().wait()
    finally:
        await adapter.close()
        await channel.disconnect()
        if owns_runtime:
            await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_lark_adapter())
