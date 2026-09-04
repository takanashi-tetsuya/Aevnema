"""Stable-ID access policy shared by all transport adapters."""

from __future__ import annotations

import os


_TRUE_VALUES = {"true", "1", "yes", "on"}
_FALSE_VALUES = {"false", "0", "no", "off", ""}


def _boolean_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def whitelist_enabled(platform: str) -> bool:
    return _boolean_env(f"{platform.upper()}_WHITELIST_ENABLED", False)


def whitelist_values(platform: str, env_name: str) -> set[str]:
    if not whitelist_enabled(platform):
        return set()
    return {
        part.strip()
        for part in os.getenv(env_name, "").split(",")
        if part.strip()
    }


def require_access_policy(platform: str, *allowlist_envs: str) -> None:
    if not whitelist_enabled(platform):
        return
    if any(os.getenv(name, "").strip() for name in allowlist_envs):
        return
    raise ValueError(
        f"{platform} whitelist is enabled but empty: configure one of "
        f"{', '.join(allowlist_envs)} or disable "
        f"{platform.upper()}_WHITELIST_ENABLED"
    )
