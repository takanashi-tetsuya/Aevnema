from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import os
from typing import Awaitable, Callable

from src.bot.adapter_support import AdapterRuntime, require_access_policy


AdapterStarter = Callable[[AdapterRuntime | None], Awaitable[None]]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True, slots=True)
class AdapterSpec:
    name: str
    enable_env: str
    module: str
    starter: str
    dependency_modules: tuple[str, ...]
    required_env_groups: tuple[tuple[str, ...], ...]
    allowlist_envs: tuple[str, ...]

    def load_starter(self) -> AdapterStarter:
        module = importlib.import_module(self.module)
        return getattr(module, self.starter)


ADAPTER_SPECS: tuple[AdapterSpec, ...] = (
    AdapterSpec("telegram", "ENABLE_TELEGRAM", "src.bot.telegram_adapter", "start_telegram_adapter", ("telegram",), (("TELEGRAM_BOT_TOKEN",),), ("TELEGRAM_ALLOWED_USERS", "TELEGRAM_ALLOWED_CHATS")),
    AdapterSpec("discord", "ENABLE_DISCORD", "src.bot.discord_adapter", "start_discord_adapter", ("discord",), (("DISCORD_BOT_TOKEN",),), ("DISCORD_ALLOWED_USERS", "DISCORD_ALLOWED_CHANNELS", "DISCORD_ALLOWED_GUILDS")),
    AdapterSpec("slack", "ENABLE_SLACK", "src.bot.slack_adapter", "start_slack_adapter", ("slack_bolt", "aiohttp"), (("SLACK_BOT_TOKEN",), ("SLACK_APP_TOKEN",)), ("SLACK_ALLOWED_USERS", "SLACK_ALLOWED_CHANNELS", "SLACK_ALLOWED_TEAMS")),
    AdapterSpec("teams", "ENABLE_TEAMS", "src.bot.teams_adapter", "start_teams_adapter", ("microsoft_teams",), (("TEAMS_CLIENT_ID", "CLIENT_ID"), ("TEAMS_TENANT_ID", "TENANT_ID")), ("TEAMS_ALLOWED_USERS", "TEAMS_ALLOWED_TENANTS", "TEAMS_ALLOWED_CONVERSATIONS")),
    AdapterSpec("lark", "ENABLE_LARK", "src.bot.lark_adapter", "start_lark_adapter", ("lark_channel",), (("LARK_APP_ID",), ("LARK_APP_SECRET",)), ("LARK_ALLOWED_USERS", "LARK_ALLOWED_CHATS")),
    AdapterSpec("wecom", "ENABLE_WECOM", "src.bot.wecom_adapter", "start_wecom_adapter", ("aibot",), (("WECOM_BOT_ID",), ("WECOM_SECRET",)), ("WECOM_ALLOWED_USERS", "WECOM_ALLOWED_CHATS")),
    AdapterSpec("dingtalk", "ENABLE_DINGTALK", "src.bot.dingtalk_adapter", "start_dingtalk_adapter", ("dingtalk_stream", "aiohttp"), (("DINGTALK_CLIENT_ID",), ("DINGTALK_CLIENT_SECRET",)), ("DINGTALK_ALLOWED_USERS", "DINGTALK_ALLOWED_CONVERSATIONS")),
    AdapterSpec("qq", "ENABLE_QQ", "src.bot.qq_adapter", "start_qq_adapter", ("qqbot_agent_sdk", "httpx"), (("QQ_APP_ID",), ("QQ_CLIENT_SECRET",)), ("QQ_ALLOWED_USERS", "QQ_ALLOWED_CHATS")),
    AdapterSpec("line", "ENABLE_LINE", "src.bot.line_adapter", "start_line_adapter", ("linebot", "aiohttp"), (("LINE_CHANNEL_SECRET",), ("LINE_CHANNEL_ACCESS_TOKEN",)), ("LINE_ALLOWED_USERS", "LINE_ALLOWED_CHATS")),
    AdapterSpec("whatsapp", "ENABLE_WHATSAPP", "src.bot.whatsapp_adapter", "start_whatsapp_adapter", ("aiohttp",), (("WHATSAPP_APP_SECRET", "META_APP_SECRET"), ("WHATSAPP_VERIFY_TOKEN",), ("WHATSAPP_ACCESS_TOKEN",), ("WHATSAPP_GRAPH_API_VERSION", "META_GRAPH_API_VERSION"), ("WHATSAPP_PHONE_NUMBER_ID",)), ("WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOWED_PHONE_NUMBER_IDS")),
    AdapterSpec("messenger", "ENABLE_MESSENGER", "src.bot.messenger_adapter", "start_messenger_adapter", ("aiohttp",), (("MESSENGER_APP_SECRET", "META_APP_SECRET"), ("MESSENGER_VERIFY_TOKEN",), ("MESSENGER_PAGE_ACCESS_TOKEN",), ("MESSENGER_GRAPH_API_VERSION", "META_GRAPH_API_VERSION"), ("MESSENGER_PAGE_ID",)), ("MESSENGER_ALLOWED_USERS",)),
    AdapterSpec("google_chat", "ENABLE_GOOGLE_CHAT", "src.bot.google_chat_adapter", "start_google_chat_adapter", ("aiohttp", "google.auth", "google.apps.chat_v1"), (("GOOGLE_CHAT_OIDC_AUDIENCE",), ("GOOGLE_CHAT_SERVICE_ACCOUNT_FILE", "GOOGLE_APPLICATION_CREDENTIALS")), ("GOOGLE_CHAT_ALLOWED_USERS", "GOOGLE_CHAT_ALLOWED_SPACES", "GOOGLE_CHAT_ALLOWED_DOMAINS")),
    AdapterSpec("xmpp", "ENABLE_XMPP", "src.bot.xmpp_adapter", "start_xmpp_adapter", ("slixmpp",), (("XMPP_JID",), ("XMPP_PASSWORD",)), ("XMPP_ALLOWED_JIDS",)),
    AdapterSpec("matrix", "ENABLE_MATRIX", "src.bot.matrix_adapter", "start_matrix_adapter", ("nio",), (("MATRIX_HOMESERVER",), ("MATRIX_USER_ID",), ("MATRIX_ACCESS_TOKEN", "MATRIX_PASSWORD")), ("MATRIX_ALLOWED_USERS", "MATRIX_ALLOWED_ROOMS")),
)


def enabled_adapters() -> list[AdapterSpec]:
    return [spec for spec in ADAPTER_SPECS if env_flag(spec.enable_env)]


def validate_enabled_adapters(specs: list[AdapterSpec]) -> None:
    """Validate the complete launch set before the memory indexes are loaded."""

    errors: list[str] = []
    if not specs:
        errors.append(
            "no adapter is enabled; set at least one ENABLE_<PLATFORM>=true"
        )
    for spec in specs:
        missing_modules: list[str] = []
        for name in spec.dependency_modules:
            try:
                available = importlib.util.find_spec(name) is not None
            except (ImportError, ModuleNotFoundError, AttributeError):
                available = False
            if not available:
                missing_modules.append(name)
        if missing_modules:
            errors.append(
                f"{spec.name}: missing Python modules {', '.join(missing_modules)}"
            )
        for alternatives in spec.required_env_groups:
            if not any(os.getenv(name, "").strip() for name in alternatives):
                errors.append(
                    f"{spec.name}: configure one of {', '.join(alternatives)}"
                )
        try:
            require_access_policy(spec.name, *spec.allowlist_envs)
        except ValueError as exc:
            errors.append(
                str(exc)
            )

    listener_defaults = {
        "teams": ("0.0.0.0", "TEAMS_PORT", "3978"),
        "line": ("127.0.0.1", "LINE_WEBHOOK_PORT", "8081"),
        "whatsapp": ("0.0.0.0", "WHATSAPP_WEBHOOK_PORT", "8081"),
        "messenger": ("0.0.0.0", "MESSENGER_WEBHOOK_PORT", "8082"),
        "google_chat": ("0.0.0.0", "GOOGLE_CHAT_WEBHOOK_PORT", "8083"),
    }
    host_envs = {
        "line": "LINE_WEBHOOK_HOST",
        "whatsapp": "WHATSAPP_WEBHOOK_HOST",
        "messenger": "MESSENGER_WEBHOOK_HOST",
        "google_chat": "GOOGLE_CHAT_WEBHOOK_HOST",
    }
    listeners: list[tuple[str, str, int]] = []
    for spec in specs:
        defaults = listener_defaults.get(spec.name)
        if defaults is None:
            continue
        default_host, port_env, default_port = defaults
        host = os.getenv(host_envs.get(spec.name, ""), default_host).strip()
        try:
            port = int(os.getenv(port_env, default_port))
            if not 1 <= port <= 65_535:
                raise ValueError
        except ValueError:
            errors.append(f"{spec.name}: {port_env} must be a TCP port")
            continue
        listeners.append((spec.name, host or default_host, port))
    wildcard_hosts = {"0.0.0.0", "::", "[::]", ""}
    for index, (left_name, left_host, left_port) in enumerate(listeners):
        for right_name, right_host, right_port in listeners[index + 1 :]:
            hosts_overlap = (
                left_host == right_host
                or left_host in wildcard_hosts
                or right_host in wildcard_hosts
            )
            if left_port == right_port and hosts_overlap:
                errors.append(
                    f"listener collision: {left_name} and {right_name} both bind "
                    f"TCP {left_port} on overlapping hosts"
                )
    if errors:
        raise ValueError("adapter preflight failed:\n- " + "\n- ".join(errors))
