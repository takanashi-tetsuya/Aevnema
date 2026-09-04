from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hmac
import os
import re
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
from src.memory import PlatformIdentity
from src.utils.logger import setup_logger


logger = setup_logger(__name__)

ServerStarter = Callable[..., Awaitable[Any]]


@dataclass(slots=True, frozen=True)
class WhatsAppInboundMessage:
    phone_number_id: str
    user_id: str
    display_name: str
    text: str
    event_id: str


def whatsapp_identity(
    phone_number_id: str, user_id: str, display_name: str = ""
) -> PlatformIdentity:
    """Scope a WhatsApp address to the receiving business phone number."""

    phone_scope = str(phone_number_id).strip()
    stable_user = str(user_id).strip()
    if not phone_scope or not stable_user:
        raise ValueError("WhatsApp phone_number_id and user_id are required")
    return PlatformIdentity(
        "whatsapp",
        f"{phone_scope}:{stable_user}",
        display_name or stable_user,
    )


def whatsapp_chat_allows(*, is_group: bool, mode: str) -> bool:
    """Cloud API chatbot support is deliberately limited to direct chats."""

    normalized = mode.strip().casefold()
    if normalized != "direct":
        raise ValueError("WHATSAPP_CHAT_MODE must be direct")
    return not is_group


def _graph_version(value: str) -> str:
    normalized = value.strip()
    if not re.fullmatch(r"v[1-9][0-9]*\.[0-9]+", normalized):
        raise ValueError("Meta Graph API version must look like v25.0")
    return normalized


class WhatsAppAdapter:
    """Meta WhatsApp Cloud API HTTPS webhook and Send API adapter."""

    def __init__(
        self,
        runtime: AdapterRuntime,
        *,
        app_secret: str,
        verify_token: str,
        access_token: str,
        graph_api_version: str,
        phone_number_id: str,
        poster: JsonPoster,
        allowed_users: set[str] | None = None,
        allowed_phone_number_ids: set[str] | None = None,
        chat_mode: str = "direct",
        message_limit: int = 4_000,
        max_body_bytes: int = 1_048_576,
    ):
        if not app_secret or not verify_token or not access_token:
            raise ValueError("WhatsApp webhook and access credentials are required")
        self.runtime = runtime
        self.app_secret = app_secret
        self.verify_token = verify_token
        self.access_token = access_token
        self.graph_api_version = _graph_version(graph_api_version)
        self.phone_number_id = phone_number_id.strip()
        if not self.phone_number_id:
            raise ValueError("WhatsApp phone_number_id is required")
        whatsapp_chat_allows(is_group=False, mode=chat_mode)
        self.chat_mode = chat_mode.strip().casefold()
        self.poster = poster
        self.allowed_users = set(allowed_users or ())
        configured_scopes = set(allowed_phone_number_ids or ())
        configured_scopes.add(self.phone_number_id)
        self.allowed_phone_number_ids = configured_scopes
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
        # Meta only needs a prompt successful acknowledgement. LLM work is
        # intentionally detached from the webhook request.
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
    ) -> list[WhatsAppInboundMessage]:
        if payload.get("object") != "whatsapp_business_account":
            return []
        extracted: list[WhatsAppInboundMessage] = []
        entries = payload.get("entry", [])
        if not isinstance(entries, list):
            return extracted
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            changes = entry.get("changes", [])
            if not isinstance(changes, list):
                continue
            for change in changes:
                if not isinstance(change, dict) or change.get("field") != "messages":
                    continue
                value = change.get("value", {})
                if not isinstance(value, dict):
                    continue
                metadata = value.get("metadata", {})
                phone_number_id = (
                    str(metadata.get("phone_number_id", "")).strip()
                    if isinstance(metadata, dict)
                    else ""
                )
                if phone_number_id not in self.allowed_phone_number_ids:
                    continue
                names: dict[str, str] = {}
                contacts = value.get("contacts", [])
                if isinstance(contacts, list):
                    for contact in contacts:
                        if not isinstance(contact, dict):
                            continue
                        contact_id = str(contact.get("wa_id", "")).strip()
                        profile = contact.get("profile", {})
                        display_name = (
                            str(profile.get("name", "")).strip()
                            if isinstance(profile, dict)
                            else ""
                        )
                        if contact_id:
                            names[contact_id] = display_name
                messages = value.get("messages", [])
                if not isinstance(messages, list):
                    continue
                for raw_message in messages:
                    if not isinstance(raw_message, dict):
                        continue
                    user_id = str(raw_message.get("from", "")).strip()
                    if not user_id or (
                        self.allowed_users and user_id not in self.allowed_users
                    ):
                        continue
                    is_group = bool(
                        raw_message.get("group_id") or value.get("group_id")
                    )
                    if not whatsapp_chat_allows(
                        is_group=is_group, mode=self.chat_mode
                    ):
                        continue
                    if raw_message.get("type") != "text":
                        continue
                    text_value = raw_message.get("text", {})
                    text = (
                        str(text_value.get("body", "")).strip()
                        if isinstance(text_value, dict)
                        else ""
                    )
                    if not text:
                        continue
                    event_id = str(raw_message.get("id", "")).strip()
                    if not event_id:
                        event_id = event_fingerprint(
                            f"whatsapp:{phone_number_id}", raw_message
                        )
                    extracted.append(
                        WhatsAppInboundMessage(
                            phone_number_id=phone_number_id,
                            user_id=user_id,
                            display_name=names.get(user_id, ""),
                            text=text,
                            event_id=event_id,
                        )
                    )
        return extracted

    def _dispatch(self, message: WhatsAppInboundMessage) -> None:
        identity = whatsapp_identity(
            message.phone_number_id, message.user_id, message.display_name
        )

        async def send_text(text: str) -> None:
            await self.send_text(
                message.phone_number_id, message.user_id, text
            )

        self.dispatcher.submit(
            conversation_key=f"whatsapp:{message.phone_number_id}:{message.user_id}",
            identity=identity,
            text=message.text,
            send_text=send_text,
            event_id=message.event_id,
            event_scope=message.phone_number_id,
        )

    async def send_text(
        self, phone_number_id: str, recipient_id: str, text: str
    ) -> None:
        if phone_number_id not in self.allowed_phone_number_ids:
            raise ValueError("refusing to send through an unconfigured phone number")
        url = (
            f"https://graph.facebook.com/{self.graph_api_version}/"
            f"{phone_number_id}/messages"
        )
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        for chunk in split_message(text, self.message_limit):
            await self.poster.post_json(
                url,
                headers=headers,
                payload={
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": recipient_id,
                    "type": "text",
                    "text": {"body": chunk},
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


async def start_whatsapp_adapter(
    runtime: AdapterRuntime | None = None,
    *,
    shutdown_event: asyncio.Event | None = None,
    server_starter: ServerStarter | None = None,
    poster: JsonPoster | None = None,
) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    app_secret = required_env("WHATSAPP_APP_SECRET", "META_APP_SECRET")
    verify_token = required_env("WHATSAPP_VERIFY_TOKEN")
    access_token = required_env("WHATSAPP_ACCESS_TOKEN")
    graph_version = _graph_version(
        required_env("WHATSAPP_GRAPH_API_VERSION", "META_GRAPH_API_VERSION")
    )
    phone_number_id = required_env("WHATSAPP_PHONE_NUMBER_ID")
    require_access_policy(
        "whatsapp", "WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOWED_PHONE_NUMBER_IDS"
    )
    host = os.getenv("WHATSAPP_WEBHOOK_HOST", "0.0.0.0").strip()
    port = integer_env("WHATSAPP_WEBHOOK_PORT", "8081", maximum=65_535)
    path = os.getenv("WHATSAPP_WEBHOOK_PATH", "/webhooks/whatsapp").strip()
    message_limit = integer_env("WHATSAPP_MESSAGE_LIMIT", "4000")
    max_body_bytes = integer_env("WHATSAPP_WEBHOOK_MAX_BYTES", "1048576")

    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    active_poster: JsonPoster | None = poster
    adapter: WhatsAppAdapter | None = None
    server: Any = None
    try:
        active_poster = active_poster or AiohttpJsonPoster()
        adapter = WhatsAppAdapter(
            active_runtime,
            app_secret=app_secret,
            verify_token=verify_token,
            access_token=access_token,
            graph_api_version=graph_version,
            phone_number_id=phone_number_id,
            poster=active_poster,
            allowed_users=whitelist_values("whatsapp", "WHATSAPP_ALLOWED_USERS"),
            allowed_phone_number_ids=whitelist_values(
                "whatsapp", "WHATSAPP_ALLOWED_PHONE_NUMBER_IDS"
            ),
            chat_mode=os.getenv("WHATSAPP_CHAT_MODE", "direct"),
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
        logger.info("WhatsApp Adapter 已在 %s:%s%s 启动", host, port, path)
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
    asyncio.run(start_whatsapp_adapter())
