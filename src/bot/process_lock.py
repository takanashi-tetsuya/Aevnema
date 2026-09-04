from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path


class BotProcessLock:
    """Cross-platform presence lock used to block offline maintenance."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            details = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"bot process lock already exists: {self.path}\n{details}"
            ) from exc
        try:
            os.write(descriptor, payload)
        finally:
            os.close(descriptor)
        self._owned = True

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def __enter__(self) -> "BotProcessLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.release()
