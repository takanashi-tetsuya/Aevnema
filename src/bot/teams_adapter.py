from __future__ import annotations

from contextlib import suppress
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
    from microsoft_teams.apps import App as _TeamsApp
    from microsoft_teams.api.auth.cloud_environment import (
        from_name as _teams_cloud_from_name,
    )

    _TEAMS_AVAILABLE = True
except ModuleNotFoundError:  # Keep Activity policy helpers testable without SDK.
    _TeamsApp = None  # type: ignore[assignment]
    _teams_cloud_from_name = None  # type: ignore[assignment]
    _TEAMS_AVAILABLE = False


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


def _field(value: Any, *names: str, default: Any = None) -> Any:
    """Read one Activity field from SDK models or JSON-shaped test doubles."""

    if value is None:
        return default
    for name in names:
        if isinstance(value, dict) and name in value:
            candidate = value[name]
            if candidate is not None:
                return candidate
            continue
        try:
            candidate = getattr(value, name)
        except (AttributeError, TypeError):
            continue
        if candidate is not None:
            return candidate
    return default


def _sender(activity: Any) -> Any:
    return _field(activity, "from_", "from", default={})


def _tenant_id(activity: Any) -> str:
    channel_data = _field(activity, "channel_data", "channelData", default={})
    tenant = _field(channel_data, "tenant", default={})
    tenant_id = _field(tenant, "id", "tenant_id", "tenantId", default="")
    if tenant_id:
        return str(tenant_id).strip()
    conversation = _field(activity, "conversation", default={})
    return str(
        _field(conversation, "tenant_id", "tenantId", default="") or ""
    ).strip()


def teams_identity(activity: Any) -> PlatformIdentity:
    """Prefer tenant-scoped Entra object ID; fall back to Bot-scoped user ID."""

    sender = _sender(activity)
    aad_object_id = str(
        _field(sender, "aad_object_id", "aadObjectId", default="") or ""
    ).strip()
    bot_scoped_id = str(_field(sender, "id", default="") or "").strip()
    stable_id = aad_object_id or bot_scoped_id
    if not stable_id:
        raise ValueError("Teams Activity has no sender ID")
    tenant_id = _tenant_id(activity)
    platform_user_id = f"{tenant_id}:{stable_id}" if tenant_id else stable_id
    display_name = str(
        _field(sender, "name", "display_name", "displayName", default="")
        or stable_id
    ).strip()
    return PlatformIdentity("teams", platform_user_id, display_name)


def teams_conversation_type(activity: Any) -> str:
    conversation = _field(activity, "conversation", default={})
    raw = str(
        _field(
            conversation,
            "conversation_type",
            "conversationType",
            default="",
        )
        or ""
    ).strip()
    if raw:
        return raw.casefold().replace("_", "").replace("-", "")
    channel_data = _field(activity, "channel_data", "channelData", default={})
    team = _field(channel_data, "team", default={})
    if _field(team, "id", default=""):
        return "channel"
    return "unknown"


def teams_message_allows(
    *, conversation_type: str, mentioned: bool, mode: str
) -> bool:
    """Apply a second, local guard around Teams' default @mention delivery."""

    normalized_mode = mode.strip().casefold()
    if normalized_mode not in {"direct", "mentions", "all"}:
        raise ValueError("TEAMS_GROUP_MODE must be direct, mentions, or all")
    normalized_type = (
        conversation_type.strip().casefold().replace("_", "").replace("-", "")
    )
    if normalized_type in {"personal", "direct", "oneonone", "1:1"}:
        return True
    if normalized_mode == "direct":
        return False
    if normalized_mode == "mentions":
        return mentioned
    return True


def teams_bot_is_mentioned(activity: Any) -> bool:
    recipient = _field(activity, "recipient", default={})
    bot_id = str(_field(recipient, "id", default="") or "")
    if not bot_id:
        return False
    for entity in _field(activity, "entities", default=[]) or []:
        if str(_field(entity, "type", default="") or "").casefold() != "mention":
            continue
        mentioned = _field(entity, "mentioned", default={})
        if str(_field(mentioned, "id", default="") or "") == bot_id:
            return True
    return False


def strip_teams_bot_mention(text: str, activity: Any) -> str:
    """Remove only mention entities that resolve to this Activity's recipient bot."""

    result = str(text or "")
    recipient = _field(activity, "recipient", default={})
    bot_id = str(_field(recipient, "id", default="") or "")
    for entity in _field(activity, "entities", default=[]) or []:
        if str(_field(entity, "type", default="") or "").casefold() != "mention":
            continue
        mentioned = _field(entity, "mentioned", default={})
        if not bot_id or str(_field(mentioned, "id", default="") or "") != bot_id:
            continue
        entity_text = str(_field(entity, "text", default="") or "")
        if entity_text:
            result = re.sub(re.escape(entity_text), "", result, flags=re.IGNORECASE)
            continue
        # Some clients omit entity.text.  Remove an exact canonical tag using
        # the mentioned account name, never a generic <at> tag that could
        # belong to a different participant in the same message.
        mention_name = str(
            _field(mentioned, "name", default="")
            or _field(recipient, "name", default="")
            or ""
        ).strip()
        if mention_name:
            result = re.sub(
                rf"<at>\s*{re.escape(mention_name)}\s*</at>",
                "",
                result,
                flags=re.IGNORECASE,
            )
    return result.lstrip(" :,-\t")


def teams_conversation_key(
    identity: PlatformIdentity,
    conversation_id: str,
    thread_id: str = "",
) -> str:
    target = f"{conversation_id}:{thread_id}" if thread_id else conversation_id
    return f"{identity.key}/{target}"


class TeamsAdapter:
    """Thin boundary around Microsoft's official ``microsoft-teams-apps`` SDK."""

    def __init__(self, app: Any, runtime: AdapterRuntime):
        self.app = app
        self.runtime = runtime
        self.dispatcher = DebouncedDispatcher(runtime)
        self.group_mode = os.getenv("TEAMS_GROUP_MODE", "mentions").strip().casefold()
        teams_message_allows(
            conversation_type="personal", mentioned=False, mode=self.group_mode
        )
        self.allowed_users = whitelist_values("teams", "TEAMS_ALLOWED_USERS")
        self.allowed_tenants = whitelist_values("teams", "TEAMS_ALLOWED_TENANTS")
        self.allowed_conversations = whitelist_values(
            "teams", "TEAMS_ALLOWED_CONVERSATIONS"
        )
        self.message_limit = _positive_int_env("TEAMS_MESSAGE_LIMIT", "12000")
        self._adapter_closed = False

        # Official SDK surface: @app.on_message and ActivityContext.send().
        app.on_message(self._on_message)

    async def _on_message(self, ctx: Any) -> None:
        activity = getattr(ctx, "activity", None)
        if activity is None:
            logger.warning("忽略缺少 Activity 的 Teams 消息")
            return

        sender = _sender(activity)
        sender_id = str(_field(sender, "id", default="") or "").strip()
        sender_type = str(
            _field(
                sender,
                "type",
                "role",
                "user_role",
                "userRole",
                default="",
            )
            or ""
        ).casefold()
        recipient = _field(activity, "recipient", default={})
        recipient_id = str(_field(recipient, "id", default="") or "").strip()
        if not sender_id or sender_id == recipient_id or sender_type == "bot":
            return

        conversation = _field(activity, "conversation", default={})
        conversation_id = str(_field(conversation, "id", default="") or "").strip()
        if not conversation_id:
            logger.warning("忽略缺少 conversation ID 的 Teams Activity")
            return
        conversation_type = teams_conversation_type(activity)
        mentioned = teams_bot_is_mentioned(activity)
        if not teams_message_allows(
            conversation_type=conversation_type,
            mentioned=mentioned,
            mode=self.group_mode,
        ):
            return

        try:
            identity = teams_identity(activity)
        except ValueError as exc:
            logger.warning("忽略无法识别用户的 Teams Activity：%s", exc)
            return
        aad_object_id = str(
            _field(sender, "aad_object_id", "aadObjectId", default="") or ""
        ).strip()
        if self.allowed_users and not {
            sender_id,
            aad_object_id,
            identity.platform_user_id,
        }.intersection(self.allowed_users):
            logger.warning("忽略不在 TEAMS_ALLOWED_USERS 中的消息：%s", identity.key)
            return
        tenant_id = _tenant_id(activity)
        if self.allowed_tenants and tenant_id not in self.allowed_tenants:
            return
        if (
            self.allowed_conversations
            and conversation_id not in self.allowed_conversations
        ):
            return

        text = str(_field(activity, "text", default="") or "").strip()
        if mentioned:
            text = strip_teams_bot_mention(text, activity)
        if not text:
            return

        # Teams channel replies carry reply_to_id.  Group chats and personal
        # chats are flat, so their debounce key remains the conversation ID.
        thread_id = ""
        if conversation_type in {"channel", "team", "teams"}:
            thread_id = str(
                _field(activity, "reply_to_id", "replyToId", "id", default="")
                or ""
            ).strip()

        async def send_text(reply: str) -> None:
            for chunk in split_message(reply, self.message_limit):
                await ctx.send(chunk)

        self.dispatcher.submit(
            conversation_key=teams_conversation_key(
                identity, conversation_id, thread_id
            ),
            identity=identity,
            text=text,
            send_text=send_text,
            event_id=str(_field(activity, "id", default="") or "") or None,
            event_scope=tenant_id or "teams-app",
        )

    async def close_adapter(self) -> None:
        if self._adapter_closed:
            return
        self._adapter_closed = True
        try:
            result = self.app.stop()
            if hasattr(result, "__await__"):
                await result
        finally:
            await self.dispatcher.close()


def _teams_app_options() -> dict[str, Any]:
    """Map adapter-prefixed settings onto documented ``AppOptions`` fields."""

    client_id = (
        os.getenv("TEAMS_CLIENT_ID", "").strip()
        or os.getenv("CLIENT_ID", "").strip()
    )
    tenant_id = (
        os.getenv("TEAMS_TENANT_ID", "").strip()
        or os.getenv("TENANT_ID", "").strip()
    )
    client_secret = (
        os.getenv("TEAMS_CLIENT_SECRET", "").strip()
        or os.getenv("CLIENT_SECRET", "").strip()
    )
    managed_identity_client_id = (
        os.getenv("TEAMS_MANAGED_IDENTITY_CLIENT_ID", "").strip()
        or os.getenv("MANAGED_IDENTITY_CLIENT_ID", "").strip()
    )
    if not client_id or not tenant_id:
        raise ValueError(
            "TEAMS_CLIENT_ID and TEAMS_TENANT_ID must be configured "
            "(CLIENT_ID/TENANT_ID aliases are also accepted)"
        )

    options: dict[str, Any] = {
        "client_id": client_id,
        "tenant_id": tenant_id,
    }
    if client_secret:
        options["client_secret"] = client_secret
    if managed_identity_client_id:
        options["managed_identity_client_id"] = managed_identity_client_id
    cloud = os.getenv("TEAMS_CLOUD", "").strip()
    if cloud:
        # AppOptions expects a CloudEnvironment instance, not the environment
        # name string.  The SDK's resolver recognizes Public/USGov/USGovDoD/China.
        options["cloud"] = _teams_cloud_from_name(cloud)
    endpoint = os.getenv("TEAMS_MESSAGING_ENDPOINT", "").strip()
    if endpoint:
        options["messaging_endpoint"] = endpoint
    service_url = os.getenv("TEAMS_SERVICE_URL", "").strip()
    if service_url:
        options["service_url"] = service_url
    return options


async def start_teams_adapter(runtime: AdapterRuntime | None = None) -> None:
    """Start the official Teams SDK's authenticated ``/api/messages`` server."""

    load_dotenv(PROJECT_ROOT / ".env")
    if not _TEAMS_AVAILABLE:
        raise RuntimeError(
            "Teams adapter requires Microsoft's Teams SDK; "
            "install microsoft-teams-apps>=2.0.16,<3"
        )
    require_access_policy(
        "teams",
        "TEAMS_ALLOWED_USERS",
        "TEAMS_ALLOWED_TENANTS",
        "TEAMS_ALLOWED_CONVERSATIONS",
    )

    options = _teams_app_options()
    port = _positive_int_env(
        "TEAMS_PORT", os.getenv("PORT", "3978"), maximum=65535
    )
    owns_runtime = runtime is None
    active_runtime = runtime or await AdapterRuntime.create()
    adapter: TeamsAdapter | None = None
    app: Any | None = None
    try:
        app = _TeamsApp(**options)
        adapter = TeamsAdapter(app, active_runtime)
        logger.info("Teams Adapter 正在监听 /api/messages（端口 %s）", port)
        await app.start(port=port)
    finally:
        if adapter is not None:
            with suppress(Exception):
                await adapter.close_adapter()
        elif app is not None:
            with suppress(Exception):
                await app.stop()
        if owns_runtime:
            await active_runtime.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(start_teams_adapter())
