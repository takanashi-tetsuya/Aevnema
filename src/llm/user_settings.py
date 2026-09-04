from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
from typing import Any

from src.llm.engine import GenerationOptions
from src.memory.identity import PlatformIdentity


_THINKING_MODES = {"inherit", "auto", "on", "off"}


@dataclass(slots=True)
class UserModelSettings:
    """Per-platform-user overrides for the roleplay chat model only."""

    thinking: str = "inherit"
    thinking_budget: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.thinking = str(self.thinking).casefold()
        if self.thinking not in _THINKING_MODES:
            raise ValueError(
                "thinking must be one of: inherit, auto, on, off"
            )
        # Reuse the engine's validation for all numeric ranges.
        GenerationOptions.model_validate(self.request_options())

    def request_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.thinking == "on":
            options["enable_thinking"] = True
        elif self.thinking == "off":
            options["enable_thinking"] = False
        elif self.thinking == "auto":
            # Explicit None removes a model/task default and lets the provider decide.
            options["enable_thinking"] = None
        for name in (
            "thinking_budget",
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
        ):
            value = getattr(self, name)
            if value is not None:
                options[name] = value
        return options

    def is_default(self) -> bool:
        return self == UserModelSettings()


class UserModelSettingsStore:
    """Small atomic JSON store keyed by platform + immutable native user ID."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._lock = threading.RLock()
        self._users: dict[str, UserModelSettings] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            users = payload.get("users", {}) if isinstance(payload, dict) else {}
            for key, value in users.items():
                try:
                    self._users[str(key)] = UserModelSettings(**dict(value))
                except (TypeError, ValueError):
                    # One damaged user entry must not disable the bot for everyone.
                    continue

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "users": {
                key: asdict(value)
                for key, value in sorted(self._users.items())
            },
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def get(self, identity: PlatformIdentity) -> UserModelSettings:
        with self._lock:
            value = self._users.get(identity.key, UserModelSettings())
            return UserModelSettings(**asdict(value))

    def request_options(self, identity: PlatformIdentity) -> dict[str, Any]:
        return self.get(identity).request_options()

    def update(
        self, identity: PlatformIdentity, **changes: Any
    ) -> UserModelSettings:
        with self._lock:
            current = asdict(self._users.get(identity.key, UserModelSettings()))
            current.update(changes)
            updated = UserModelSettings(**current)
            if updated.is_default():
                self._users.pop(identity.key, None)
            else:
                self._users[identity.key] = updated
            self._save()
            return UserModelSettings(**asdict(updated))

    def reset(self, identity: PlatformIdentity) -> UserModelSettings:
        with self._lock:
            self._users.pop(identity.key, None)
            self._save()
        return UserModelSettings()


def describe_user_settings(settings: UserModelSettings) -> str:
    def shown(value: Any) -> str:
        return "继承配置" if value is None else str(value)

    return "\n".join(
        (
            f"thinking：{settings.thinking}",
            f"thinking_budget：{shown(settings.thinking_budget)}",
            f"temperature：{shown(settings.temperature)}",
            f"top_p：{shown(settings.top_p)}",
            f"max_tokens：{shown(settings.max_tokens)}",
            f"seed：{shown(settings.seed)}",
        )
    )
