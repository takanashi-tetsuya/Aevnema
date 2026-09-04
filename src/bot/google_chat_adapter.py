from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hmac
import inspect
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Protocol

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
    WebhookResponse,
    bearer_token,
    event_fingerprint,
    integer_env,
    parse_json_object,
    required_env,
    start_aiohttp_server,
)
from src.memory import PlatformIdentity
from src.utils.logger import setup_logger


logger = setup_logger(__name__)
CHAT_ISSUER = "chat@system.gserviceaccount.com"
CHAT_BOT_SCOPE = "https://www.googleapis.com/auth/chat.bot"

ServerStarter = Callable[..., Awaitable[Any]]
OidcVerifier = Callable[[str], Awaitable[bool]]


class ChatMessageSender(Protocol):
    async def send_message(
        self, *, space_name: str, thread_name: str, text: str
    ) -> Any: ...

    async def close(self) -> None: ...


@dataclass(slots=True, frozen=True)
class GoogleChatInboundMessage:
    app_scope: str
    domain_id: str
    user_name: str
    display_name: str
    space_name: str
    thread_name: str
    text: str
    event_id: str


def google_chat_identity(
    app_scope: str,
    domain_id: str,
    user_name: str,
    display_name: str = "",
) -> PlatformIdentity:
    """Scope Chat's stable users/{id} by app and Workspace tenant."""

    stable_app = str(app_scope).strip()
    stable_user = str(user_name).strip()
    stable_domain = str(domain_id).strip() or "consumer"
    if not stable_app or not stable_user:
        raise ValueError("Google Chat app scope and user resource are required")
    return PlatformIdentity(
        "google_chat",
        f"{stable_app}:{stable_domain}:{stable_user}",
        display_name or stable_user,
    )


def google_chat_space_allows(*, is_direct: bool, mode: str) -> bool:
    normalized = mode.strip().casefold()
    if normalized not in {"direct", "mentions"}:
        raise ValueError("GOOGLE_CHAT_SPACE_MODE must be direct or mentions")
    # Chat interaction endpoints deliver group MESSAGE events only when the
    # app was invoked (normally @mentioned or called by a command).
    return is_direct or normalized == "mentions"


class GoogleChatOidcVerifier:
    """Verify the endpoint-URL OIDC mode documented for Google Chat apps."""

    def __init__(self, audience: str):
        self.audience = audience.strip()
        if not self.audience:
            raise ValueError("Google Chat OIDC audience is required")
        try:
            from google.auth.transport import requests as google_requests
            from google.oauth2 import id_token
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "google-auth is required to verify Google Chat OIDC requests"
            ) from exc
        self._request_type = google_requests.Request
        self._id_token = id_token

    async def __call__(self, token: str) -> bool:
        def verify() -> bool:
            try:
                claims = self._id_token.verify_oauth2_token(
                    token, self._request_type(), self.audience
                )
            except Exception:
                return False
            email = str(claims.get("email", ""))
            return bool(claims.get("email_verified")) and hmac.compare_digest(
                email, CHAT_ISSUER
            )

        return await asyncio.to_thread(verify)


class GoogleChatApiSender:
    def __init__(self, client: Any):
        self.client = client
        self._closed = False

    async def send_message(
        self, *, space_name: str, thread_name: str, text: str
    ) -> Any:
        message: dict[str, Any] = {"text": text}
        request: dict[str, Any] = {"parent": space_name, "message": message}
        if thread_name:
            message["thread"] = {"name": thread_name}
            request["message_reply_option"] = "REPLY_MESSAGE_OR_FAIL"
        return await self.client.create_message(request=request)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # google-apps-chat 0.10.x exposes close() on the transport rather than
        # ChatServiceAsyncClient itself.  Keep the direct-client branch for
        # newer clients and injected transports.
        close = getattr(self.client, "close", None)
        if close is None:
            close = getattr(getattr(self.client, "transport", None), "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def create_google_chat_sender(credentials_file: str) -> GoogleChatApiSender:
    path = Path(credentials_file).expanduser()
    if not path.is_file():
        raise ValueError(
            "Google Chat service-account file is missing: "
            "set GOOGLE_CHAT_SERVICE_ACCOUNT_FILE"
        )
    try:
        from google.apps import chat_v1
        from google.oauth2 import service_account
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "google-apps-chat and google-auth are required for Google Chat"
        ) from exc
    try:
        credentials = service_account.Credentials.from_service_account_file(
            str(path), scopes=[CHAT_BOT_SCOPE]
        )
    except Exception as exc:
        raise ValueError("Google Chat service-account credentials are invalid") from exc
    return GoogleChatApiSender(
        chat_v1.ChatServiceAsyncClient(credentials=credentials)
    )


class GoogleChatAdapter:
    """Google Chat interaction endpoint with detached Chat API responses."""

    def __init__(
        self,
        runtime: AdapterRuntime,
        *,
        oidc_verifier: OidcVerifier,
        sender: ChatMessageSender,
        app_scope: str,
        allowed_users: set[str] | None = None,
        allowed_spaces: set[str] | None = None,
        allowed_domains: set[str] | None = None,
        space_mode: str = "direct",
        message_limit: int = 8_000,
        max_body_bytes: int = 1_048_576,
    ):
        self.runtime = runtime
        self.oidc_verifier = oidc_verifier
        self.sender = sender
        self.app_scope = app_scope.strip()
        if not self.app_scope:
            raise ValueError("Google Chat app scope is required")
        google_chat_space_allows(is_direct=True, mode=space_mode)
        self.space_mode = space_mode.strip().casefold()
        self.allowed_users = set(allowed_users or ())
        self.allowed_spaces = set(allowed_spaces or ())
        self.allowed_domains = set(allowed_domains or ())
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
        del query
        if method.upper() != "POST":
            return WebhookResponse.text("method not allowed", 405)

        token = bearer_token(headers)
        if not token or not await self.oidc_verifier(token):
            return WebhookResponse.text("unauthorized", 401)
        try:
            payload = parse_json_object(
                raw_body, max_bytes=self.max_body_bytes
            )
        except ValueError:
            return WebhookResponse.text("invalid payload", 400)

        message = self._extract_message(payload)
        if message is not None:
            self._dispatch(message)
        # Never hold Google's 30-second synchronous response open for the LLM.
        return WebhookResponse.json({})

    def _extract_message(
        self, event: Mapping[str, Any]
    ) -> GoogleChatInboundMessage | None:
        event_type = event.get("type", event.get("eventType", ""))
        if not isinstance(event_type, str) or event_type != "MESSAGE":
            return None
        message = event.get("message", {})
        if not isinstance(message, dict):
            return None
        user = event.get("user", message.get("sender", {}))
        space = event.get("space", message.get("space", {}))
        if not isinstance(user, dict) or not isinstance(space, dict):
            return None
        if str(user.get("type", "HUMAN")) != "HUMAN":
            return None
        user_name = str(user.get("name", "")).strip()
        domain_id = str(user.get("domainId", "")).strip()
        space_name = str(space.get("name", "")).strip()
        if (
            not user_name
            or not space_name
            or (self.allowed_users and user_name not in self.allowed_users)
            or (self.allowed_spaces and space_name not in self.allowed_spaces)
            or (self.allowed_domains and domain_id not in self.allowed_domains)
        ):
            return None
        space_type = str(space.get("type", "")).upper()
        is_direct = space_type in {"DM", "DIRECT_MESSAGE"}
        if not google_chat_space_allows(
            is_direct=is_direct, mode=self.space_mode
        ):
            return None
        text = str(message.get("argumentText") or message.get("text") or "").strip()
        if not text:
            return None
        thread = message.get("thread", {})
        thread_name = (
            str(thread.get("name", "")).strip()
            if isinstance(thread, dict)
            else ""
        )
        event_id = str(message.get("name", "")).strip()
        if not event_id:
            event_id = event_fingerprint(
                f"google-chat:{self.app_scope}:{space_name}", event
            )
        return GoogleChatInboundMessage(
            app_scope=self.app_scope,
            domain_id=domain_id,
            user_name=user_name,
            display_name=str(user.get("displayName", "")).strip(),
            space_name=space_name,
            thread_name=thread_name,
            text=text,
            event_id=event_id,
        )

    def _dispatch(self, message: GoogleChatInboundMessage) -> None:
        identity = google_chat_identity(
            message.app_scope,
            message.domain_id,
            message.user_name,
            message.display_name,
        )

        async def send_text(text: str) -> None:
            await self.send_text(
                message.space_name, message.thread_name, text
            )

        thread_scope = message.thread_name or "root"
        self.dispatcher.submit(
            conversation_key=(
                f"google_chat:{message.app_scope}:{message.space_name}:"
                f"{thread_scope}:{message.user_name}"
            ),
            identity=identity,
            text=message.text,
            send_text=send_text,
            event_id=message.event_id,
            event_scope=message.app_scope,
        )

    async def send_text(
        self, space_name: str, thread_name: str, text: str
    ) -> None:
        for chunk in split_message(text, self.message_limit):
            await self.sender.send_message(
                space_name=space_name,
                thread_name=thread_name,
                text=chunk,
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
            await self.sender.close()


async def start_google_chat_adapter(
    runtime: AdapterRuntime | None = None,
    *,
    shutdown_event: asyncio.Event | None = None,
    server_starter: ServerStarter | None = None,
    oidc_verifier: OidcVerifier | None = None,
    sender: ChatMessageSender | None = None,
) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    audience = required_env("GOOGLE_CHAT_OIDC_AUDIENCE")
    require_access_policy(
        "google_chat",
        "GOOGLE_CHAT_ALLOWED_USERS",
        "GOOGLE_CHAT_ALLOWED_SPACES",
        "GOOGLE_CHAT_ALLOWED_DOMAINS",
    )
    app_scope = os.getenv("GOOGLE_CHAT_APP_SCOPE", audience).strip()
    host = os.getenv("GOOGLE_CHAT_WEBHOOK_HOST", "0.0.0.0").strip()
    port = integer_env("GOOGLE_CHAT_WEBHOOK_PORT", "8083", maximum=65_535)
    path = os.getenv("GOOGLE_CHAT_WEBHOOK_PATH", "/webhooks/google-chat").strip()
    message_limit = integer_env("GOOGLE_CHAT_MESSAGE_LIMIT", "8000")
    max_body_bytes = integer_env("GOOGLE_CHAT_WEBHOOK_MAX_BYTES", "1048576")

    active_verifier = oidc_verifier or GoogleChatOidcVerifier(audience)
    active_sender = sender
    if active_sender is None:
        credentials_file = required_env(
            "GOOGLE_CHAT_SERVICE_ACCOUNT_FILE", "GOOGLE_APPLICATION_CREDENTIALS"
        )
        active_sender = create_google_chat_sender(credentials_file)

    owns_runtime = runtime is None
    active_runtime: AdapterRuntime | None = runtime
    adapter: GoogleChatAdapter | None = None
    server: Any = None
    try:
        active_runtime = active_runtime or await AdapterRuntime.create()
        adapter = GoogleChatAdapter(
            active_runtime,
            oidc_verifier=active_verifier,
            sender=active_sender,
            app_scope=app_scope,
            allowed_users=whitelist_values(
                "google_chat", "GOOGLE_CHAT_ALLOWED_USERS"
            ),
            allowed_spaces=whitelist_values(
                "google_chat", "GOOGLE_CHAT_ALLOWED_SPACES"
            ),
            allowed_domains=whitelist_values(
                "google_chat", "GOOGLE_CHAT_ALLOWED_DOMAINS"
            ),
            space_mode=os.getenv("GOOGLE_CHAT_SPACE_MODE", "direct"),
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
        logger.info("Google Chat Adapter 已在 %s:%s%s 启动", host, port, path)
        await (shutdown_event or asyncio.Event()).wait()
    finally:
        try:
            if server is not None:
                await server.close()
        finally:
            try:
                if adapter is not None:
                    await adapter.close()
                elif active_sender is not None:
                    await active_sender.close()
            finally:
                if owns_runtime and active_runtime is not None:
                    await active_runtime.close()


if __name__ == "__main__":
    asyncio.run(start_google_chat_adapter())
