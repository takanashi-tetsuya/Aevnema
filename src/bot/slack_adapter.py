from __future__ import annotations

import os
import re
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
    from slack_bolt.async_app import AsyncApp as _AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import (
        AsyncSocketModeHandler as _AsyncSocketModeHandler,
    )

    _SLACK_AVAILABLE = True
except ModuleNotFoundError:  # Keep identity/event policy testable without extras.
    _AsyncApp = None  # type: ignore[assignment]
    _AsyncSocketModeHandler = None  # type: ignore[assignment]
    _SLACK_AVAILABLE = False


def _csv_env(name: str) -> set[str]:
    return {
        value.strip()
        for value in os.getenv(name, "").split(",")
        if value.strip()
    }


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


def slack_identity(
    team_id: str, user_id: str, display_name: str = ""
) -> PlatformIdentity:
    """Use Slack's documented workspace ID + workspace user ID stable key."""

    normalized_team = str(team_id or "").strip()
    normalized_user = str(user_id or "").strip()
    if not normalized_team:
        raise ValueError("Slack event has no team_id")
    if not normalized_user:
        raise ValueError("Slack event has no user ID")
    return PlatformIdentity(
        "slack",
        f"{normalized_team}:{normalized_user}",
        display_name or normalized_user,
    )


def slack_event_allows(*, event_type: str, channel_type: str) -> bool:
    """Only direct messages and explicit app mentions enter the chatbot."""

    return event_type == "app_mention" or (
        event_type == "message" and channel_type == "im"
    )


def strip_slack_bot_mention(text: str, bot_user_id: str) -> str:
    if not bot_user_id:
        return text.strip()
    return re.sub(
        rf"<@{re.escape(str(bot_user_id))}>",
        "",
        text,
        flags=re.IGNORECASE,
    ).lstrip(" :,-\t")


def slack_conversation_key(
    identity: PlatformIdentity,
    channel_id: str,
    thread_ts: str = "",
) -> str:
    target = f"{channel_id}:{thread_ts}" if thread_ts else channel_id
    return f"{identity.key}/{target}"


def _team_id_from_body(body: dict[str, Any]) -> str:
    direct = str(body.get("team_id") or "").strip()
    if direct:
        return direct
    team = body.get("team") or {}
    if isinstance(team, dict) and team.get("id"):
        return str(team["id"]).strip()
    for authorization in body.get("authorizations") or []:
        if isinstance(authorization, dict) and authorization.get("team_id"):
            return str(authorization["team_id"]).strip()
    # Org-wide Enterprise Grid installations can omit team_id.  Enterprise
    # IDs are also stable Slack IDs (with an E prefix), so they are a safe
    # final tenant scope rather than falling back to an unscoped user ID.
    enterprise_id = str(body.get("enterprise_id") or "").strip()
    if enterprise_id:
        return enterprise_id
    enterprise = body.get("enterprise") or {}
    if isinstance(enterprise, dict) and enterprise.get("id"):
        return str(enterprise["id"]).strip()
    for authorization in body.get("authorizations") or []:
        if isinstance(authorization, dict) and authorization.get("enterprise_id"):
            return str(authorization["enterprise_id"]).strip()
    return ""


class SlackAdapter:
    """Slack Bolt Socket Mode transport for DMs and ``app_mention`` events."""

    def __init__(self, app: Any, runtime: AdapterRuntime, *, bot_user_id: str):
        normalized_bot_id = str(bot_user_id or "").strip()
        if not normalized_bot_id:
            raise ValueError("Slack auth.test did not return a bot user ID")
        self.app = app
        self.runtime = runtime
        self.bot_user_id = normalized_bot_id
        self.dispatcher = DebouncedDispatcher(runtime)
        self.allowed_users = whitelist_values("slack", "SLACK_ALLOWED_USERS")
        self.allowed_channels = whitelist_values("slack", "SLACK_ALLOWED_CHANNELS")
        self.allowed_teams = whitelist_values("slack", "SLACK_ALLOWED_TEAMS")
        self.message_limit = _positive_int_env(
            "SLACK_MESSAGE_LIMIT", "3900", maximum=40000
        )
        self.reply_in_thread = (
            os.getenv("SLACK_REPLY_IN_THREAD", "true").strip().casefold() == "true"
        )
        self._adapter_closed = False

        # These are the official Bolt decorator APIs, invoked directly so the
        # transport can remain an injectable/testable object.
        app.event("app_mention")(self._on_app_mention)
        app.event("message")(self._on_message)

    async def _on_app_mention(
        self, event: dict[str, Any], body: dict[str, Any], client: Any
    ) -> None:
        await self.handle_event(
            event=event,
            body=body,
            client=client,
            event_type="app_mention",
        )

    async def _on_message(
        self, event: dict[str, Any], body: dict[str, Any], client: Any
    ) -> None:
        await self.handle_event(
            event=event,
            body=body,
            client=client,
            event_type="message",
        )

    async def handle_event(
        self,
        *,
        event: dict[str, Any],
        body: dict[str, Any],
        client: Any,
        event_type: str,
    ) -> None:
        channel_type = str(event.get("channel_type") or "")
        if not slack_event_allows(
            event_type=event_type, channel_type=channel_type
        ):
            return
        if event.get("subtype") is not None or event.get("bot_id") or event.get("hidden"):
            return

        user_id = str(event.get("user") or "").strip()
        if not user_id or user_id == self.bot_user_id:
            return
        channel_id = str(event.get("channel") or "").strip()
        if not channel_id:
            logger.warning("忽略缺少 channel ID 的 Slack 事件")
            return
        team_id = _team_id_from_body(body) or str(
            event.get("team") or event.get("team_id") or ""
        ).strip()
        if not team_id:
            logger.warning("忽略缺少 team_id 的 Slack 事件")
            return

        identity = slack_identity(team_id, user_id)
        if self.allowed_users and not {
            user_id,
            identity.platform_user_id,
        }.intersection(self.allowed_users):
            logger.warning("忽略不在 SLACK_ALLOWED_USERS 中的事件：%s", identity.key)
            return
        if self.allowed_channels and channel_id not in self.allowed_channels:
            return
        if self.allowed_teams and team_id not in self.allowed_teams:
            return
        text = str(event.get("text") or "").strip()
        if event_type == "app_mention":
            text = strip_slack_bot_mention(text, self.bot_user_id)
        if not text:
            return

        thread_ts = str(event.get("thread_ts") or "").strip()
        if event_type == "app_mention" and self.reply_in_thread and not thread_ts:
            thread_ts = str(event.get("ts") or "").strip()

        async def send_text(reply: str) -> None:
            for chunk in split_message(reply, self.message_limit):
                kwargs: dict[str, Any] = {
                    "channel": channel_id,
                    "text": chunk,
                }
                if thread_ts:
                    kwargs["thread_ts"] = thread_ts
                await client.chat_postMessage(**kwargs)

        self.dispatcher.submit(
            conversation_key=slack_conversation_key(
                identity, channel_id, thread_ts
            ),
            identity=identity,
            text=text,
            send_text=send_text,
            event_id=str(body.get("event_id") or event.get("event_ts") or "") or None,
            event_scope=team_id or "slack-app",
        )

    async def close_adapter(self) -> None:
        if self._adapter_closed:
            return
        self._adapter_closed = True
        await self.dispatcher.close()


async def start_slack_adapter(runtime: AdapterRuntime | None = None) -> None:
    """Start Slack Bolt using its official asyncio Socket Mode handler."""

    load_dotenv(PROJECT_ROOT / ".env")
    if not _SLACK_AVAILABLE:
        raise RuntimeError(
            "Slack adapter requires slack-bolt with aiohttp; "
            "install slack-bolt>=1.30,<2 and aiohttp"
        )
    bot_token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    app_token = os.getenv("SLACK_APP_TOKEN", "").strip()
    if not bot_token or not app_token:
        raise ValueError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be configured")
    require_access_policy(
        "slack", "SLACK_ALLOWED_USERS", "SLACK_ALLOWED_CHANNELS", "SLACK_ALLOWED_TEAMS"
    )

    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter: SlackAdapter | None = None
    handler: Any | None = None
    try:
        app = _AsyncApp(token=bot_token)
        bot_user_id = os.getenv("SLACK_BOT_USER_ID", "").strip()
        if not bot_user_id:
            auth = await app.client.auth_test()
            bot_user_id = str(auth.get("user_id") or "").strip()
        adapter = SlackAdapter(app, active_runtime, bot_user_id=bot_user_id)
        handler = _AsyncSocketModeHandler(app, app_token)
        logger.info("Slack Adapter 正在通过 Socket Mode 登录：%s", bot_user_id)
        await handler.start_async()
    finally:
        # Stop Socket Mode intake before cancelling queued adapter work.
        try:
            if handler is not None:
                await handler.close_async()
        finally:
            try:
                if adapter is not None:
                    await adapter.close_adapter()
            finally:
                if owns_runtime:
                    await active_runtime.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(start_slack_adapter())
