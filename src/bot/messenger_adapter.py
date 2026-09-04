from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hmac
import os
from typing import Any, Awaitable, Callable, Mapping

from dotenv import load_dotenv

from src.bot.adapter_support import (
    AdapterRuntime,
    DebouncedDispatcher,
    PROJECT_ROOT,
    require_access_policy,
    split_message,
    whitelist_values,
)
from src.bot.webhook_support import (
    AiohttpJsonPoster,
    JsonPoster,
    WebhookResponse,
    event_fingerprint,
    header_value,
    integer_env,
    parse_json_object,
    required_env,
    start_aiohttp_server,
    verify_meta_signature,
)
from src.bot.whatsapp_adapter import _graph_version
from src.memory import PlatformIdentity
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

ServerStarter = Callable[..., Awaitable[Any]]


@dataclass(slots=True, frozen=True)
class MessengerInboundMessage:
    page_id: str
    user_id: str
    display_name: str
    text: str
    event_id: str


def messenger_identity(
    page_id: str, user_id: str, display_name: str = ""
) -> PlatformIdentity:
    """A Messenger PSID is stable only in the scope of its Facebook Page."""

    stable_page = str(page_id).strip()
    stable_user = str(user_id).strip()
    if not stable_page or not stable_user:
        raise ValueError("Messenger page_id and PSID are required")
    return PlatformIdentity(
        "messenger",
        f"{stable_page}:{stable_user}",
        display_name or stable_user,
    )


def messenger_chat_allows(*, is_group: bool, mode: str) -> bool:
    normalized = mode.strip().casefold()
    if normalized != "direct":
        raise ValueError("MESSENGER_CHAT_MODE must be direct")
    return not is_group


class MessengerAdapter:
    """Facebook Page Messenger HTTPS webhook and Send API adapter."""

    def __init__(
        self,
        runtime: AdapterRuntime,
        *,
        app_secret: str,
        verify_token: str,
        page_access_token: str,
        graph_api_version: str,
        page_id: str,
        poster: JsonPoster,
        allowed_users: set[str] | None = None,
        chat_mode: str = "direct",
        message_limit: int = 2_000,
        max_body_bytes: int = 1_048_576,
    ):
        if not app_secret or not verify_token or not page_access_token:
            raise ValueError("Messenger webhook and Page credentials are required")
        self.runtime = runtime
        self.app_secret = app_secret
        self.verify_token = verify_token
        self.page_access_token = page_access_token
        self.graph_api_version = _graph_version(graph_api_version)
        self.page_id = page_id.strip()
        if not self.page_id:
            raise ValueError("Messenger page_id is required")
        messenger_chat_allows(is_group=False, mode=chat_mode)
        self.chat_mode = chat_mode.strip().casefold()
        self.poster = poster
        self.allowed_users = set(allowed_users or ())
        self.message_limit = message_limit
        self.max_body_bytes = max_body_bytes
        self.dispatcher = DebouncedDispatcher(runtime)
        self._closed = False

    async def handle_webhook(
        self,
        method: str,
        headers: Mapping[str, str],
        query: Mapping[str, str],
        raw_body: bytes,
    ) -> WebhookResponse:
        normalized_method = method.upper()
        if normalized_method == "GET":
            return self._verify_subscription(query)
        if normalized_method != "POST":
            return WebhookResponse.text("method not allowed", 405)

        signature = header_value(headers, "X-Hub-Signature-256")
        if not verify_meta_signature(raw_body, signature, self.app_secret):
            return WebhookResponse.text("unauthorized", 401)
        try:
            payload = parse_json_object(
                raw_body, max_bytes=self.max_body_bytes
            )
        except ValueError:
            return WebhookResponse.text("invalid payload", 400)

        for message in self._extract_messages(payload):
            self._dispatch(message)
        return WebhookResponse.text("EVENT_RECEIVED")

    def _verify_subscription(self, query: Mapping[str, str]) -> WebhookResponse:
        mode = str(query.get("hub.mode", ""))
        token = str(query.get("hub.verify_token", ""))
        challenge = str(query.get("hub.challenge", ""))
        if (
            mode == "subscribe"
            and challenge
            and hmac.compare_digest(token, self.verify_token)
        ):
            return WebhookResponse.text(challenge)
        return WebhookResponse.text("forbidden", 403)

    def _extract_messages(
        self, payload: Mapping[str, Any]
    ) -> list[MessengerInboundMessage]:
        if payload.get("object") != "page":
            return []
        extracted: list[MessengerInboundMessage] = []
        entries = payload.get("entry", [])
        if not isinstance(entries, list):
            return extracted
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            page_id = str(entry.get("id", "")).strip()
            if page_id != self.page_id:
                continue
            events = entry.get("messaging", [])
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict):
                    continue
                sender = event.get("sender", {})
                recipient = event.get("recipient", {})
                user_id = (
                    str(sender.get("id", "")).strip()
                    if isinstance(sender, dict)
                    else ""
                )
                recipient_page = (
                    str(recipient.get("id", "")).strip()
                    if isinstance(recipient, dict)
                    else ""
                )
                if (
                    not user_id
                    or recipient_page != self.page_id
                    or (self.allowed_users and user_id not in self.allowed_users)
                ):
                    continue
                is_group = bool(event.get("participants") or event.get("group_id"))
                if not messenger_chat_allows(
                    is_group=is_group, mode=self.chat_mode
                ):
                    continue

                message = event.get("message", {})
                postback = event.get("postback", {})
                if isinstance(message, dict) and message.get("is_echo"):
                    continue
                text = ""
                event_id = ""
                if isinstance(message, dict):
                    text = str(message.get("text", "")).strip()
                    event_id = str(message.get("mid", "")).strip()
                if not text and isinstance(postback, dict):
                    text = str(postback.get("payload", "")).strip()
                    event_id = str(postback.get("mid", "")).strip()
                if not text:
                    continue
                if not event_id:
                    event_id = event_fingerprint(
                        f"messenger:{page_id}", event
                    )
                extracted.append(
                    MessengerInboundMessage(
                        page_id=page_id,
                        user_id=user_id,
                        display_name=user_id,
                        text=text,
                        event_id=event_id,
                    )
                )
        return extracted

    def _dispatch(self, message: MessengerInboundMessage) -> None:
        identity = messenger_identity(
            message.page_id, message.user_id, message.display_name
        )

        async def send_text(text: str) -> None:
            await self.send_text(message.page_id, message.user_id, text)

        async def set_typing(enabled: bool) -> None:
            await self.set_typing(message.page_id, message.user_id, enabled)

        self.dispatcher.submit(
            conversation_key=f"messenger:{message.page_id}:{message.user_id}",
            identity=identity,
            text=message.text,
            send_text=send_text,
            set_typing=set_typing,
            event_id=message.event_id,
            event_scope=message.page_id,
        )

    def _send_url(self, page_id: str) -> str:
        if page_id != self.page_id:
            raise ValueError("refusing to send through an unconfigured Page")
        return (
            f"https://graph.facebook.com/{self.graph_api_version}/"
            f"{page_id}/messages"
        )

    def _send_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.page_access_token}",
            "Content-Type": "application/json",
        }

    async def send_text(self, page_id: str, recipient_id: str, text: str) -> None:
        for chunk in split_message(text, self.message_limit):
            await self.poster.post_json(
                self._send_url(page_id),
                headers=self._send_headers(),
                payload={
                    "recipient": {"id": recipient_id},
                    "messaging_type": "RESPONSE",
                    "message": {"text": chunk},
                },
            )

    async def set_typing(
        self, page_id: str, recipient_id: str, enabled: bool
    ) -> None:
        await self.poster.post_json(
            self._send_url(page_id),
            headers=self._send_headers(),
            payload={
                "recipient": {"id": recipient_id},
                "sender_action": "typing_on" if enabled else "typing_off",
            },
        )

    async def wait_background(self) -> None:
        await asyncio.sleep(0)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self.dispatcher.close()
        finally:
            await self.poster.close()


async def start_messenger_adapter(
    runtime: AdapterRuntime | None = None,
    *,
    shutdown_event: asyncio.Event | None = None,
    server_starter: ServerStarter | None = None,
    poster: JsonPoster | None = None,
) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    app_secret = required_env("MESSENGER_APP_SECRET", "META_APP_SECRET")
    verify_token = required_env("MESSENGER_VERIFY_TOKEN")
    page_access_token = required_env("MESSENGER_PAGE_ACCESS_TOKEN")
    graph_version = _graph_version(
        required_env("MESSENGER_GRAPH_API_VERSION", "META_GRAPH_API_VERSION")
    )
    page_id = required_env("MESSENGER_PAGE_ID")
    require_access_policy("messenger", "MESSENGER_ALLOWED_USERS")
    host = os.getenv("MESSENGER_WEBHOOK_HOST", "0.0.0.0").strip()
    port = integer_env("MESSENGER_WEBHOOK_PORT", "8082", maximum=65_535)
    path = os.getenv("MESSENGER_WEBHOOK_PATH", "/webhooks/messenger").strip()
    message_limit = integer_env("MESSENGER_MESSAGE_LIMIT", "2000")
    max_body_bytes = integer_env("MESSENGER_WEBHOOK_MAX_BYTES", "1048576")

    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    active_poster: JsonPoster | None = poster
    adapter: MessengerAdapter | None = None
    server: Any = None
    try:
        active_poster = active_poster or AiohttpJsonPoster()
        adapter = MessengerAdapter(
            active_runtime,
            app_secret=app_secret,
            verify_token=verify_token,
            page_access_token=page_access_token,
            graph_api_version=graph_version,
            page_id=page_id,
            poster=active_poster,
            allowed_users=whitelist_values("messenger", "MESSENGER_ALLOWED_USERS"),
            chat_mode=os.getenv("MESSENGER_CHAT_MODE", "direct"),
            message_limit=message_limit,
            max_body_bytes=max_body_bytes,
        )
        starter = server_starter or start_aiohttp_server
        server = await starter(
            handler=adapter.handle_webhook,
            host=host,
            port=port,
            path=path,
            max_body_bytes=max_body_bytes,
        )
        logger.info("Messenger Adapter 已在 %s:%s%s 启动", host, port, path)
        await (shutdown_event or asyncio.Event()).wait()
    finally:
        try:
            if server is not None:
                await server.close()
        finally:
            try:
                if adapter is not None:
                    await adapter.close()
                elif active_poster is not None:
                    await active_poster.close()
            finally:
                if owns_runtime:
                    await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_messenger_adapter())
