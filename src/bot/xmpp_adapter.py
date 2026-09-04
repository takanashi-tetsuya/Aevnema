from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

from dotenv import load_dotenv

from src.bot.adapter_support import (
    AdapterRuntime,
    DebouncedDispatcher,
    PROJECT_ROOT,
    require_access_policy,
    split_message,
    whitelist_enabled,
)
from src.memory import PlatformIdentity
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

try:
    from slixmpp import ClientXMPP as _ClientXMPP
    from slixmpp import JID as _JID
except ModuleNotFoundError:  # Keep identity and policy helpers testable without extras.
    _ClientXMPP = object  # type: ignore[assignment,misc]
    _JID = None  # type: ignore[assignment,misc]


def xmpp_identity(jid: Any, display_name: str = "") -> PlatformIdentity:
    """Use the account's bare JID; a resource is a device, not a user."""

    bare = str(getattr(jid, "bare", jid)).strip().casefold()
    if "/" in bare:
        bare = bare.split("/", 1)[0]
    return PlatformIdentity("xmpp", bare, display_name or bare)


def _allowed_jids() -> set[str]:
    values: set[str] = set()
    for raw_value in os.getenv("XMPP_ALLOWED_JIDS", "").split(","):
        value = raw_value.strip()
        if not value:
            continue
        if _JID is None:
            bare = value.split("/", 1)[0].strip().casefold()
            local, separator, domain = bare.partition("@")
            if (
                separator != "@"
                or not local
                or not domain
                or "@" in domain
                or any(character.isspace() for character in bare)
            ):
                raise ValueError(
                    f"invalid JID in XMPP_ALLOWED_JIDS: {value!r}"
                )
            values.add(bare)
            continue
        try:
            jid = _JID(value)
        except ValueError as exc:
            raise ValueError(f"invalid JID in XMPP_ALLOWED_JIDS: {value!r}") from exc
        bare = str(jid.bare).strip().casefold()
        if not bare:
            raise ValueError("XMPP_ALLOWED_JIDS contains an empty JID")
        values.add(bare)
    return values


def _positive_int_env(name: str, default: str, *, maximum: int | None = None) -> int:
    raw_value = os.getenv(name, default).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1 or (maximum is not None and value > maximum):
        expected = f"between 1 and {maximum}" if maximum is not None else "positive"
        raise ValueError(f"{name} must be {expected}")
    return value


class XMPPAdapter(_ClientXMPP):  # type: ignore[misc]
    """Slixmpp transport for one-to-one chat and normal message stanzas."""

    def __init__(self, jid: str, password: str, runtime: AdapterRuntime):
        if _ClientXMPP is object:
            raise RuntimeError(
                "XMPP adapter requires slixmpp; install slixmpp>=1.17,<2"
            )
        super().__init__(jid, password)
        self.runtime = runtime
        self.dispatcher = DebouncedDispatcher(runtime)
        self.session_ready = asyncio.Event()
        self.authentication_failed = asyncio.Event()
        self.connection_closed = asyncio.Event()
        self.allowed_jids = _allowed_jids() if whitelist_enabled("xmpp") else set()
        self.message_limit = _positive_int_env("XMPP_MESSAGE_LIMIT", "8000")
        self._adapter_closed = False
        self.register_plugin("xep_0030")  # service discovery
        self.register_plugin("xep_0199")  # XMPP ping/keepalive
        self.add_event_handler("session_start", self._on_session_start)
        self.add_event_handler("failed_auth", self._on_failed_auth)
        self.add_event_handler("disconnected", self._on_disconnected)
        self.add_event_handler("message", self._on_message)

    async def _on_session_start(self, _event: Any) -> None:
        self.send_presence()
        try:
            await self.get_roster()
        except Exception as exc:
            # A roster is not needed to reply to an incoming direct stanza.
            logger.warning("XMPP roster request failed: %s", exc)
        self.session_ready.set()
        logger.info("XMPP Adapter 已登录：%s", self.boundjid.bare)

    def _on_failed_auth(self, _event: Any) -> None:
        logger.error("XMPP authentication failed")
        self.authentication_failed.set()

    def _on_disconnected(self, reason: Any) -> None:
        if not self._adapter_closed:
            logger.warning("XMPP connection closed: %s", reason or "unknown reason")
        self.connection_closed.set()

    async def _on_message(self, message: Any) -> None:
        message_type = str(message["type"])
        if message_type not in {"chat", "normal"}:
            return
        body = str(message["body"] or "").strip()
        if not body:
            return

        sender = message["from"]
        if not str(sender).strip():
            logger.warning("忽略缺少发送者 JID 的 XMPP 消息")
            return
        identity = xmpp_identity(sender)
        if identity.platform_user_id == str(self.boundjid.bare).casefold():
            return
        if self.allowed_jids and identity.platform_user_id not in self.allowed_jids:
            logger.warning("忽略不在 XMPP_ALLOWED_JIDS 中的消息：%s", identity.key)
            return

        reply_to = str(sender)

        async def send_text(text: str) -> None:
            for chunk in split_message(text, self.message_limit):
                self.send_message(mto=reply_to, mbody=chunk, mtype="chat")

        self.dispatcher.submit(
            # Keep transport conversations resource-specific so a burst that
            # moves between two clients cannot capture the wrong reply target.
            # DebouncedDispatcher still serializes work by the bare identity.
            conversation_key=f"{identity.key}/{reply_to}",
            identity=identity,
            text=body,
            send_text=send_text,
            event_id=str(message.get("id", "") or "") or None,
            event_scope=str(
                getattr(getattr(self, "boundjid", None), "bare", "")
                or getattr(self, "boundjid", "")
                or "xmpp-account"
            ),
        )

    async def close_adapter(self) -> None:
        if self._adapter_closed:
            return
        self._adapter_closed = True
        try:
            await self.dispatcher.close()
        finally:
            # slixmpp's disconnect() does not cancel an in-flight connection
            # attempt when no transport has been established yet.
            self.cancel_connection_attempt()
            result = self.disconnect()
            if inspect.isawaitable(result):
                await result


async def start_xmpp_adapter(runtime: AdapterRuntime | None = None) -> None:
    """Start the XMPP adapter and keep it alive until cancellation."""

    load_dotenv(PROJECT_ROOT / ".env")
    if _ClientXMPP is object:
        raise RuntimeError("XMPP adapter requires slixmpp>=1.17,<2")
    jid = os.getenv("XMPP_JID", "").strip()
    password = os.getenv("XMPP_PASSWORD", "")
    if not jid or not password:
        raise ValueError("XMPP_JID and XMPP_PASSWORD must be configured")
    require_access_policy("xmpp", "XMPP_ALLOWED_JIDS")
    try:
        configured_jid = _JID(jid)  # type: ignore[misc]
    except ValueError as exc:
        raise ValueError("XMPP_JID is not a valid JID") from exc
    if not str(configured_jid.bare):
        raise ValueError("XMPP_JID is not a valid JID")

    host = os.getenv("XMPP_HOST", "").strip()
    port_text = os.getenv("XMPP_PORT", "").strip()
    if bool(host) != bool(port_text):
        raise ValueError("XMPP_HOST and XMPP_PORT must be configured together")
    port = _positive_int_env("XMPP_PORT", "5222", maximum=65535) if host else None
    # Validate adapter-specific settings before starting shared background
    # workers. The constructor validates plugin configuration too.
    _positive_int_env("XMPP_MESSAGE_LIMIT", "8000")
    if whitelist_enabled("xmpp"):
        _allowed_jids()

    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter: XMPPAdapter | None = None
    logger.info("XMPP Adapter 正在连接……")
    try:
        adapter = XMPPAdapter(jid, password, active_runtime)
        connect_result = (
            adapter.connect(host=host, port=port)
            if host
            else adapter.connect()
        )
        if inspect.isawaitable(connect_result):
            await connect_result

        ready_wait = asyncio.create_task(adapter.session_ready.wait())
        failed_wait = asyncio.create_task(adapter.authentication_failed.wait())
        closed_wait = asyncio.create_task(adapter.connection_closed.wait())
        done, pending = await asyncio.wait(
            {ready_wait, failed_wait, closed_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if adapter.authentication_failed.is_set():
            raise RuntimeError("XMPP authentication failed")
        if closed_wait in done:
            raise ConnectionError("XMPP connection closed before session start")

        await adapter.connection_closed.wait()
        raise ConnectionError("XMPP connection closed")
    finally:
        logger.info("正在停止 XMPP Adapter 并提交待处理对话批次……")
        try:
            if adapter is not None:
                await adapter.close_adapter()
        finally:
            if owns_runtime:
                await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_xmpp_adapter())
