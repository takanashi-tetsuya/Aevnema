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
    import httpx
    from qqbot_agent_sdk import EventParser, QQApiClient, QQWebSocket, WSCallbacks

    _QQ_AVAILABLE = True
except ModuleNotFoundError:  # Identity/policy helpers remain importable in tests.
    httpx = Any  # type: ignore[assignment]
    EventParser = QQApiClient = QQWebSocket = WSCallbacks = Any  # type: ignore[assignment,misc]
    _QQ_AVAILABLE = False


def _csv_env(name: str) -> set[str]:
    return {item.strip() for item in os.getenv(name, "").split(",") if item.strip()}


def qq_identity(user_id: str, chat_scope: str, display_name: str = "") -> PlatformIdentity:
    """QQ openids are scene-scoped, so retain the scene in the memory key."""

    opaque_id = user_id.strip()
    scope = chat_scope.strip().casefold()
    stable_id = f"{scope}:{opaque_id}"
    return PlatformIdentity("qq", stable_id, display_name.strip() or opaque_id)


def qq_message_allows(
    *, chat_scope: str, event_type: str, group_mode: str
) -> bool:
    mode = group_mode.strip().casefold()
    if mode not in {"mentions", "all", "disabled"}:
        raise ValueError("QQ_GROUP_MODE must be mentions, all, or disabled")
    scope = chat_scope.strip().casefold()
    if scope in {"c2c", "dm"}:
        return True
    if mode == "all":
        return True
    if mode == "disabled":
        return False
    # QQ group delivery is explicitly GROUP_AT_MESSAGE_CREATE. Guilds have
    # both ordinary and @ event types, so only the latter is accepted here.
    return "_AT_MESSAGE_CREATE" in event_type.upper()


class QQAdapter:
    """QQ official-bot event parser and REST sender."""

    def __init__(self, api: Any, runtime: AdapterRuntime, parser: Any | None = None):
        self.api = api
        self.runtime = runtime
        self.parser = parser or EventParser()
        self.dispatcher = DebouncedDispatcher(runtime)
        self.group_mode = os.getenv("QQ_GROUP_MODE", "mentions").casefold()
        qq_message_allows(
            chat_scope="c2c", event_type="C2C_MESSAGE_CREATE", group_mode=self.group_mode
        )
        self.allowed_users = whitelist_values("qq", "QQ_ALLOWED_USERS")
        self.allowed_chats = whitelist_values("qq", "QQ_ALLOWED_CHATS")
        self.allowed_scopes = _csv_env("QQ_ALLOWED_SCOPES") or {
            "c2c",
            "group",
            "guild",
        }

    def _accept(self, event: Any) -> bool:
        scope = str(event.chat_scope).casefold()
        user_id = str(event.user_id or "")
        chat_id = str(event.chat_id or "")
        if not user_id or not chat_id or scope not in self.allowed_scopes:
            return False
        if self.allowed_users and user_id not in self.allowed_users:
            return False
        if self.allowed_chats and chat_id not in self.allowed_chats:
            return False
        if not qq_message_allows(
            chat_scope=scope,
            event_type=str(event.event_type),
            group_mode=self.group_mode,
        ):
            return False
        return True

    async def _send_text(self, event: Any, text: str) -> None:
        limit = int(os.getenv("QQ_MESSAGE_LIMIT", "4000"))
        for chunk in split_message(text, limit):
            await self.api.send_text(
                str(event.chat_scope),
                str(event.chat_id),
                chunk,
                reply_to=str(event.message_id or "") or None,
                markdown=False,
                max_length=limit,
            )

    async def _set_typing(self, event: Any, active: bool) -> None:
        if active and str(event.chat_scope).casefold() == "c2c":
            await self.api.send_typing(str(event.chat_id), str(event.message_id))

    async def on_message_event(self, event_type: str, raw: dict[str, Any]) -> None:
        """Parse and enqueue only; returning quickly releases the WS callback."""

        event = self.parser.parse(event_type, raw)
        if event is None or not str(event.content or "").strip() or not self._accept(event):
            return
        identity = qq_identity(
            str(event.user_id),
            str(event.chat_scope),
            str(getattr(event, "user_name", "") or ""),
        )
        self.dispatcher.submit(
            conversation_key=f"{event.chat_scope}:{event.chat_id}:{identity.key}",
            identity=identity,
            text=str(event.content).strip(),
            send_text=lambda value: self._send_text(event, value),
            set_typing=lambda active: self._set_typing(event, active),
            event_id=str(event.message_id or "") or None,
            event_scope=str(getattr(self.api, "app_id", "") or event.chat_scope),
        )

    async def close(self) -> None:
        await self.dispatcher.close()


class _QQGatewayState:
    """Small state holder satisfying the official WSCallbacks contract."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.last_seq: int | None = None
        self.heartbeat_interval = 0.0

    def get_session(self) -> tuple[str | None, int | None]:
        return self.session_id, self.last_seq

    def set_session(self, session_id: str | None, last_seq: int | None) -> None:
        self.session_id, self.last_seq = session_id, last_seq

    def set_heartbeat_interval(self, seconds: float) -> None:
        self.heartbeat_interval = seconds

    @staticmethod
    def connected() -> None:
        logger.info("QQ Bot WebSocket 已连接")

    @staticmethod
    def disconnected() -> None:
        logger.warning("QQ Bot WebSocket 已断开，SDK 将尝试恢复会话")

    @staticmethod
    def fatal_error(code: str, message: str) -> None:
        logger.error("QQ Bot WebSocket 致命错误 %s: %s", code, message)

    @staticmethod
    def fail_pending(reason: str) -> None:
        logger.warning("QQ Bot 待处理网关请求已失败：%s", reason)


def build_qq_callbacks(adapter: QQAdapter, api: Any) -> Any:
    state = _QQGatewayState()
    callbacks = WSCallbacks(
        on_message_event=adapter.on_message_event,
        on_connected=state.connected,
        on_disconnected=state.disconnected,
        on_fatal_error=state.fatal_error,
        get_token=api.ensure_token_sync,
        get_session=state.get_session,
        set_session=state.set_session,
        set_heartbeat_interval=state.set_heartbeat_interval,
        clear_token=api.clear_token,
        fail_pending=state.fail_pending,
        get_gateway_url=api.get_gateway_url_sync,
    )
    # Keep state alive and introspectable without extending the SDK dataclass.
    setattr(callbacks, "_adapter_state", state)
    return callbacks


async def start_qq_adapter(runtime: AdapterRuntime | None = None) -> None:
    if not _QQ_AVAILABLE:
        raise RuntimeError("请先安装官方依赖：pip install qqbot-agent-sdk")
    load_dotenv(PROJECT_ROOT / ".env")
    app_id = os.getenv("QQ_APP_ID", "").strip()
    client_secret = os.getenv("QQ_CLIENT_SECRET", "").strip()
    if not app_id or not client_secret:
        raise RuntimeError("QQ_APP_ID and QQ_CLIENT_SECRET are required")
    require_access_policy("qq", "QQ_ALLOWED_USERS", "QQ_ALLOWED_CHATS")

    http_client = httpx.AsyncClient()
    api = QQApiClient(app_id=app_id, client_secret=client_secret)
    api.setup(http_client)
    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter = QQAdapter(api, active_runtime)
    ws = QQWebSocket(callbacks=build_qq_callbacks(adapter, api))
    try:
        await api.ensure_token()
        gateway_url = await api.get_gateway_url()
        ws.start(gateway_url, asyncio.get_running_loop())
        await asyncio.Event().wait()
    finally:
        await ws.stop()
        await adapter.close()
        await http_client.aclose()
        if owns_runtime:
            await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_qq_adapter())
