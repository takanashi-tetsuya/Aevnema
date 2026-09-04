from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import threading
import time


@dataclass(frozen=True, slots=True)
class EventClaim:
    platform: str
    account_scope: str
    event_id: str


class PersistentEventDeduplicator:
    """Crash-persistent event reservation shared by every adapter.

    A claimed event is protected by a short lease. A completed event is kept
    for the configured TTL. Failed/cancelled work releases its claim so the
    platform may retry it. SQLite is deliberately synchronous here: each
    operation is a tiny indexed transaction and never spans network work.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ttl_seconds: float = 7 * 24 * 3600,
        lease_seconds: float = 15 * 60,
    ):
        if ttl_seconds <= 0 or lease_seconds <= 0:
            raise ValueError("dedup TTL and lease must be positive")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = float(ttl_seconds)
        self.lease_seconds = float(lease_seconds)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS adapter_event (
                platform TEXT NOT NULL,
                account_scope TEXT NOT NULL,
                event_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('claimed', 'completed')),
                lease_until REAL NOT NULL,
                expires_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(platform, account_scope, event_id)
            )
            """
        )
        self._closed = False
        self._operations = 0

    @staticmethod
    def _normalize(value: str, label: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"{label} must not be empty")
        return normalized

    def claim(
        self, platform: str, account_scope: str, event_id: str
    ) -> EventClaim | None:
        claim = EventClaim(
            self._normalize(platform, "platform"),
            self._normalize(account_scope, "account_scope"),
            self._normalize(event_id, "event_id"),
        )
        now = time.time()
        lease_until = now + self.lease_seconds
        expires_at = now + self.ttl_seconds
        with self._lock:
            if self._closed:
                raise RuntimeError("event deduplicator is closed")
            self._operations += 1
            if self._operations % 256 == 0:
                self._connection.execute(
                    "DELETE FROM adapter_event WHERE expires_at <= ?", (now,)
                )
            cursor = self._connection.execute(
                """
                INSERT INTO adapter_event(
                    platform, account_scope, event_id, status,
                    lease_until, expires_at, updated_at
                ) VALUES (?, ?, ?, 'claimed', ?, ?, ?)
                ON CONFLICT(platform, account_scope, event_id) DO UPDATE SET
                    status='claimed', lease_until=excluded.lease_until,
                    expires_at=excluded.expires_at, updated_at=excluded.updated_at
                WHERE adapter_event.expires_at <= excluded.updated_at
                   OR (adapter_event.status='claimed'
                       AND adapter_event.lease_until <= excluded.updated_at)
                """,
                (
                    claim.platform,
                    claim.account_scope,
                    claim.event_id,
                    lease_until,
                    expires_at,
                    now,
                ),
            )
            return claim if cursor.rowcount == 1 else None

    def complete(self, claim: EventClaim) -> None:
        now = time.time()
        with self._lock:
            if self._closed:
                return
            self._connection.execute(
                """
                UPDATE adapter_event
                SET status='completed', lease_until=0, expires_at=?, updated_at=?
                WHERE platform=? AND account_scope=? AND event_id=?
                """,
                (
                    now + self.ttl_seconds,
                    now,
                    claim.platform,
                    claim.account_scope,
                    claim.event_id,
                ),
            )

    def release(self, claim: EventClaim) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.execute(
                """
                DELETE FROM adapter_event
                WHERE platform=? AND account_scope=? AND event_id=?
                  AND status='claimed'
                """,
                (claim.platform, claim.account_scope, claim.event_id),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()
