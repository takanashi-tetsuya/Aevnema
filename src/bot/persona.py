from __future__ import annotations

import os
from pathlib import Path
import tomllib
from typing import Any

from config.prompt_config import (
    PERSONA_FALLBACK_SYSTEM,
    render_persona_system_prompt,
)
from src.llm.user_settings import UserModelSettingsStore, describe_user_settings
from src.memory import PlatformIdentity
from src.utils.logger import setup_logger


logger = setup_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_persona_cache: dict[str, Any] = {
    "mtime": 0.0,
    "persona": {},
    "rules": {},
}


def _persona_path() -> Path:
    configured = os.getenv("PERSONA_CONFIG_PATH", "config/persona.toml")
    path = Path(configured)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_persona() -> dict[str, Any]:
    with _persona_path().open("rb") as handle:
        return tomllib.load(handle)


def get_dynamic_system_prompt(memory_context: str) -> str:
    """Render the shared persona for every transport adapter."""

    try:
        path = _persona_path()
        current_mtime = path.stat().st_mtime
        if current_mtime > float(_persona_cache["mtime"]):
            persona_data = load_persona()
            _persona_cache["persona"] = persona_data.get("persona", {})
            _persona_cache["rules"] = persona_data.get("rules", {})
            _persona_cache["mtime"] = current_mtime
        persona = _persona_cache["persona"]
        rules = _persona_cache["rules"]
        return render_persona_system_prompt(
            name=persona.get("name", "Bot"),
            description=persona.get("description", ""),
            style=rules.get("style", ""),
            memory_context=memory_context,
        )
    except Exception as exc:
        logger.error("读取 persona 失败: %s", exc)
        return PERSONA_FALLBACK_SYSTEM


def get_private_memory_emergency_reply() -> str:
    try:
        rules = load_persona().get("rules", {})
        configured = str(rules.get("private_memory_emergency_reply", "")).strip()
        if configured:
            return configured
    except Exception as exc:
        logger.error("读取私人记忆缺失回复失败: %s", exc)
    return "抱歉，我目前没有找到关于这个的可靠记忆，所以不想随便猜。可以再告诉我一次吗？"


def _parse_optional_number(value: str, parser: Any) -> Any:
    return None if value.casefold() in {"default", "inherit", "默认", "继承"} else parser(value)


def apply_model_setting(
    store: UserModelSettingsStore,
    identity: PlatformIdentity,
    args: list[str],
) -> str:
    """Platform-neutral implementation of the per-user ``/model`` command."""

    usage = (
        "可用设置：\n"
        "/model thinking inherit|auto|on|off\n"
        "/model thinking_budget <整数|default>\n"
        "/model temperature <0..2|default>\n"
        "/model top_p <0..1|default>\n"
        "/model max_tokens <整数|default>\n"
        "/model seed <整数|default>\n"
        "/model reset\n"
        "这些设置只影响当前平台 ID 的角色对话，不改变提取、审计和重写模型。"
    )
    if not args:
        return describe_user_settings(store.get(identity)) + "\n\n" + usage
    key = args[0].casefold()
    if key == "reset":
        settings = store.reset(identity)
        return "当前账号的模型覆盖已重置。\n" + describe_user_settings(settings)
    if len(args) != 2:
        return usage
    raw = args[1].strip()
    aliases = {
        "开启": "on",
        "打开": "on",
        "关闭": "off",
        "自动": "auto",
        "继承": "inherit",
    }
    try:
        if key == "thinking":
            settings = store.update(identity, thinking=aliases.get(raw.casefold(), raw.casefold()))
        elif key == "thinking_budget":
            settings = store.update(identity, thinking_budget=_parse_optional_number(raw, int))
        elif key == "temperature":
            settings = store.update(identity, temperature=_parse_optional_number(raw, float))
        elif key == "top_p":
            settings = store.update(identity, top_p=_parse_optional_number(raw, float))
        elif key == "max_tokens":
            settings = store.update(identity, max_tokens=_parse_optional_number(raw, int))
        elif key == "seed":
            settings = store.update(identity, seed=_parse_optional_number(raw, int))
        else:
            return usage
    except (TypeError, ValueError) as exc:
        return f"设置无效：{exc}\n\n{usage}"
    return "设置已保存，只对当前平台账号生效。\n" + describe_user_settings(settings)
