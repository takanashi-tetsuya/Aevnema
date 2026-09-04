from __future__ import annotations

import inspect
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
    import discord as _discord

    _DiscordClient = _discord.Client
    _DISCORD_AVAILABLE = True
except ModuleNotFoundError:  # Keep policy helpers importable without the extra.
    _discord = None  # type: ignore[assignment]
    _DiscordClient = object  # type: ignore[assignment,misc]
    _DISCORD_AVAILABLE = False


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


def discord_identity(user: Any) -> PlatformIdentity:
    """Build an identity from Discord's globally unique snowflake user ID."""

    user_id = str(getattr(user, "id", "") or "").strip()
    display_name = str(
        getattr(user, "display_name", "")
        or getattr(user, "global_name", "")
        or getattr(user, "name", "")
        or user_id
    ).strip()
    return PlatformIdentity("discord", user_id, display_name)


def discord_message_allows(
    *, is_direct: bool, mentioned: bool, replied_to_bot: bool, mode: str
) -> bool:
    """Apply the configured server/guild response policy.

    Direct messages are always eligible.  ``mentions`` is the conservative
    default for guilds and works without broad access to unrelated messages.
    """

    normalized = mode.strip().casefold()
    if normalized not in {"direct", "mentions", "all"}:
        raise ValueError("DISCORD_GUILD_MODE must be direct, mentions, or all")
    if is_direct:
        return True
    if normalized == "direct":
        return False
    if normalized == "mentions":
        return mentioned or replied_to_bot
    return True


def strip_discord_bot_mention(text: str, bot_user_id: str) -> str:
    """Remove Discord's structured ``<@id>``/``<@!id>`` bot mention."""

    if not bot_user_id:
        return text.strip()
    pattern = rf"<@!?{re.escape(str(bot_user_id))}>"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).lstrip(" :,-\t")


def discord_conversation_key(
    identity: PlatformIdentity, channel_id: str | int
) -> str:
    """Keep transport bursts scoped to both the user and Discord channel."""

    return f"{identity.key}/{channel_id}"


class DiscordAdapter(_DiscordClient):  # type: ignore[misc]
    """Discord Gateway transport for DMs and conservatively-triggered guild chat."""

    def __init__(self, runtime: AdapterRuntime, *, intents: Any | None = None):
        if not _DISCORD_AVAILABLE:
            raise RuntimeError(
                "Discord adapter requires discord.py; install discord.py>=2.7,<3"
            )

        self.guild_mode = os.getenv("DISCORD_GUILD_MODE", "mentions").strip().casefold()
        # Validate at startup even if the first event happens to be a DM.
        discord_message_allows(
            is_direct=True,
            mentioned=False,
            replied_to_bot=False,
            mode=self.guild_mode,
        )
        if intents is None:
            intents = _discord.Intents.default()
            intents.messages = True
            # DMs and messages that mention the app are content-intent exceptions.
            # Listening to every guild message is the only MVP mode that needs it.
            intents.message_content = self.guild_mode == "all"
        super().__init__(intents=intents)

        self.runtime = runtime
        self.dispatcher = DebouncedDispatcher(runtime)
        self.allowed_users = whitelist_values("discord", "DISCORD_ALLOWED_USERS")
        self.allowed_channels = whitelist_values("discord", "DISCORD_ALLOWED_CHANNELS")
        self.allowed_guilds = whitelist_values("discord", "DISCORD_ALLOWED_GUILDS")
        self.message_limit = _positive_int_env(
            "DISCORD_MESSAGE_LIMIT", "1900", maximum=2000
        )
        self._adapter_closed = False

    async def on_ready(self) -> None:
        logger.info("Discord Adapter 已登录：%s", self.user)

    def _is_reply_to_bot(self, message: Any, bot_user_id: str) -> bool:
        reference = getattr(message, "reference", None)
        resolved = getattr(reference, "resolved", None)
        author = getattr(resolved, "author", None)
        return bool(
            bot_user_id
            and str(getattr(author, "id", "") or "") == bot_user_id
        )

    async def on_message(self, message: Any) -> None:
        author = getattr(message, "author", None)
        bot_user = getattr(self, "user", None)
        bot_user_id = str(getattr(bot_user, "id", "") or "")
        author_id = str(getattr(author, "id", "") or "")
        if (
            not author_id
            or author_id == bot_user_id
            or bool(getattr(author, "bot", False))
        ):
            return

        channel = getattr(message, "channel", None)
        channel_id = str(getattr(channel, "id", "") or "")
        if not channel_id:
            logger.warning("忽略缺少 channel ID 的 Discord 消息")
            return
        guild = getattr(message, "guild", None)
        guild_id = str(getattr(guild, "id", "") or "")
        is_direct = guild is None

        mentions = getattr(message, "mentions", None) or []
        mentioned = any(
            str(getattr(user, "id", "") or "") == bot_user_id
            for user in mentions
        )
        replied_to_bot = self._is_reply_to_bot(message, bot_user_id)
        if not discord_message_allows(
            is_direct=is_direct,
            mentioned=mentioned,
            replied_to_bot=replied_to_bot,
            mode=self.guild_mode,
        ):
            return

        identity = discord_identity(author)
        if self.allowed_users and author_id not in self.allowed_users:
            logger.warning("忽略不在 DISCORD_ALLOWED_USERS 中的消息：%s", identity.key)
            return
        if self.allowed_channels and channel_id not in self.allowed_channels:
            return
        if guild is not None and self.allowed_guilds and guild_id not in self.allowed_guilds:
            return

        body = str(getattr(message, "content", "") or "").strip()
        if mentioned:
            body = strip_discord_bot_mention(body, bot_user_id)
        if not body:
            return

        async def send_text(text: str) -> None:
            for chunk in split_message(text, self.message_limit):
                kwargs: dict[str, Any] = {}
                if _DISCORD_AVAILABLE:
                    # Model output and quoted user content must not create surprise pings.
                    kwargs["allowed_mentions"] = _discord.AllowedMentions.none()
                await channel.send(chunk, **kwargs)

        # discord.py 2.7 exposes ``channel.typing()`` as an async context
        # manager.  Keeping that context open matches AdapterRuntime's
        # set_typing(True/False) lifecycle and refreshes the indicator for long
        # generations; ``trigger_typing`` is not a discord.py API.
        typing_context: Any | None = None

        async def set_typing(active: bool) -> None:
            nonlocal typing_context
            if active:
                if typing_context is not None:
                    return
                typing_factory = getattr(channel, "typing", None)
                if typing_factory is None:
                    return
                candidate = typing_factory()
                enter = getattr(candidate, "__aenter__", None)
                if enter is not None:
                    result = enter()
                    if inspect.isawaitable(result):
                        await result
                    typing_context = candidate
                elif inspect.isawaitable(candidate):
                    # Compatibility fallback for Messageable-like test doubles
                    # and older transports: this sends Discord's 10s indicator.
                    await candidate
                return

            current = typing_context
            typing_context = None
            if current is None:
                return
            exit_context = getattr(current, "__aexit__", None)
            if exit_context is not None:
                result = exit_context(None, None, None)
                if inspect.isawaitable(result):
                    await result

        self.dispatcher.submit(
            conversation_key=discord_conversation_key(identity, channel_id),
            identity=identity,
            text=body,
            send_text=send_text,
            set_typing=set_typing,
            event_id=str(getattr(message, "id", "") or "") or None,
            event_scope=str(getattr(self.user, "id", "") or "discord-bot"),
        )

    async def close_adapter(self) -> None:
        if self._adapter_closed:
            return
        self._adapter_closed = True
        try:
            await self.dispatcher.close()
        finally:
            result = self.close()
            if inspect.isawaitable(result):
                await result


async def start_discord_adapter(runtime: AdapterRuntime | None = None) -> None:
    """Start a Discord Gateway client until it is cancelled or disconnected."""

    load_dotenv(PROJECT_ROOT / ".env")
    if not _DISCORD_AVAILABLE:
        raise RuntimeError(
            "Discord adapter requires discord.py; install discord.py>=2.7,<3"
        )
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        raise ValueError("DISCORD_BOT_TOKEN must be configured")
    require_access_policy(
        "discord",
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOWED_CHANNELS",
        "DISCORD_ALLOWED_GUILDS",
    )

    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter: DiscordAdapter | None = None
    try:
        adapter = DiscordAdapter(active_runtime)
        await adapter.start(token, reconnect=True)
    finally:
        try:
            if adapter is not None:
                await adapter.close_adapter()
        finally:
            if owns_runtime:
                await active_runtime.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(start_discord_adapter())
