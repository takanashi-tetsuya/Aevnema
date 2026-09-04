from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re


def _normalize_platform(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "_", value.strip().casefold())
    if not normalized:
        raise ValueError("platform is required")
    return normalized[:40]


@dataclass(slots=True, frozen=True)
class PlatformIdentity:
    """Stable cross-adapter identity; display_name is never a primary key."""

    platform: str
    platform_user_id: str
    display_name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "platform", _normalize_platform(self.platform))
        stable_id = str(self.platform_user_id).strip()
        if not stable_id:
            raise ValueError("platform_user_id is required")
        object.__setattr__(self, "platform_user_id", stable_id)
        object.__setattr__(self, "display_name", str(self.display_name or "").strip())

    @property
    def key(self) -> str:
        return f"{self.platform}:{self.platform_user_id}"

    @property
    def storage_key(self) -> str:
        readable = re.sub(r"[^0-9A-Za-z_-]+", "_", self.platform_user_id)
        readable = readable.strip("_")[:48] or "user"
        digest = sha256(self.key.encode("utf-8")).hexdigest()[:12]
        return f"{readable}-{digest}"

    def write_metadata(self, directory: str | Path) -> Path:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "identity.json"
        payload = asdict(self)
        if target.exists():
            current = json.loads(target.read_text(encoding="utf-8"))
            if (
                str(current.get("platform")) != self.platform
                or str(current.get("platform_user_id")) != self.platform_user_id
            ):
                raise ValueError(f"identity collision at {target_dir}")
            if current == payload:
                return target
        temporary = target.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
        return target

    @classmethod
    def from_metadata(cls, directory: str | Path) -> "PlatformIdentity":
        path = Path(directory) / "identity.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            platform=str(payload["platform"]),
            platform_user_id=str(payload["platform_user_id"]),
            display_name=str(payload.get("display_name", "")),
        )

