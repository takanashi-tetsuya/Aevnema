from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
import importlib
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import uuid

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class MemoryMaintenanceConfig:
    project_root: Path
    engine_root: Path
    knowledge_database: Path
    public_database: Path
    user_database_dir: Path
    conversation_inbox_dir: Path
    growth_queue_dir: Path
    bot_lock_path: Path

    @classmethod
    def from_project(cls, project_root: str | Path) -> "MemoryMaintenanceConfig":
        root = Path(project_root).resolve()
        load_dotenv(root / ".env", override=False)
        engine_value = os.getenv("MEMORY_ENGINE_ROOT", "").strip()
        if not engine_value:
            raise ValueError("MEMORY_ENGINE_ROOT is required")
        return cls(
            project_root=root,
            engine_root=_resolve_path(engine_value, root),
            knowledge_database=_resolve_path(
                os.getenv(
                    "KNOWLEDGE_MEMORY_DB_PATH",
                    "data/knowledge/blue_archive.db",
                ),
                root,
            ),
            public_database=_resolve_path(
                os.getenv("PUBLIC_MEMORY_DB_PATH", "data/public/memory.db"),
                root,
            ),
            user_database_dir=_resolve_path(
                os.getenv("USER_MEMORY_DB_DIR", "data/users"), root
            ),
            conversation_inbox_dir=_resolve_path(
                os.getenv("CONVERSATION_INBOX_DIR", "data/conversation_inbox"),
                root,
            ),
            growth_queue_dir=_resolve_path(
                os.getenv("MEMORY_GROWTH_QUEUE_DIR", "data/growth_queue"), root
            ),
            bot_lock_path=_resolve_path(
                os.getenv("BOT_PROCESS_LOCK_PATH", "data/bot.lock"), root
            ),
        )


def _portable_path_text(value: str) -> str:
    raw = str(value).strip()
    windows_path = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
    if os.name != "nt" and windows_path:
        drive = windows_path.group(1).casefold()
        remainder = windows_path.group(2).replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{remainder}"
    wsl_path = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", raw)
    if os.name == "nt" and wsl_path:
        drive = wsl_path.group(1).upper()
        remainder = (wsl_path.group(2) or "").replace("/", "\\")
        return f"{drive}:\\{remainder}"
    return raw


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(_portable_path_text(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _assert_safe_target(
    path: Path,
    project_root: Path,
    *,
    directory: bool,
    allow_external: bool,
) -> None:
    resolved = path.resolve()
    anchor = Path(resolved.anchor).resolve()
    forbidden = {anchor, project_root.resolve(), Path.home().resolve()}
    if resolved in forbidden:
        raise ValueError(f"refusing to clear unsafe target: {resolved}")
    data_root = (project_root / "data").resolve()
    if not allow_external and not _is_relative_to(resolved, data_root):
        kind = "directory" if directory else "database"
        raise ValueError(
            f"refusing to clear external {kind} without --allow-external: {resolved}"
        )
    if not directory and resolved.suffix.casefold() not in {
        ".db",
        ".sqlite",
        ".sqlite3",
    }:
        raise ValueError(f"configured database has an unsafe extension: {resolved}")


def _database_class(engine_root: Path):
    package_root = engine_root / "src"
    package_dir = package_root / "memory_demo"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"associative-memory package not found under {package_dir}"
        )
    package_text = str(package_root)
    if package_text not in sys.path:
        sys.path.insert(0, package_text)
    module = importlib.import_module("memory_demo.database")
    return module.Database


def _sidecars(database_path: Path) -> tuple[Path, Path]:
    return (
        database_path.with_name(database_path.name + "-wal"),
        database_path.with_name(database_path.name + "-shm"),
    )


def verify_empty_database(database_path: Path) -> dict[str, int]:
    business_tables = (
        "source",
        "paragraph",
        "episode",
        "concept",
        "concept_alias",
        "association",
        "extraction_run",
        "extraction_task",
    )
    with closing(sqlite3.connect(database_path)) as connection:
        schema_row = connection.execute(
            "SELECT schema_version FROM schema_meta LIMIT 1"
        ).fetchone()
        if schema_row is None:
            raise RuntimeError(f"empty schema_meta in reset database: {database_path}")
        counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in business_tables
        }
    nonempty = {table: count for table, count in counts.items() if count}
    if nonempty:
        raise RuntimeError(f"memory reset verification failed: {nonempty}")
    return counts


def reset_database(
    database_path: Path,
    engine_root: Path,
    *,
    project_root: Path | None = None,
    allow_external: bool = False,
) -> None:
    """Atomically replace one configured memory database with an empty schema."""

    if project_root is not None:
        _assert_safe_target(
            database_path,
            project_root,
            directory=False,
            allow_external=allow_external,
        )
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_name(
        f".{database_path.name}.reset-{uuid.uuid4().hex}.tmp"
    )
    Database = _database_class(engine_root)
    try:
        Database(temporary).initialize()
        for sidecar in _sidecars(database_path):
            sidecar.unlink(missing_ok=True)
        os.replace(temporary, database_path)
    finally:
        temporary.unlink(missing_ok=True)
        for sidecar in _sidecars(temporary):
            sidecar.unlink(missing_ok=True)
    verify_empty_database(database_path)


def clear_directory_contents(
    path: Path,
    project_root: Path,
    *,
    allow_external: bool = False,
) -> None:
    """Remove children without deleting the configured directory itself."""

    _assert_safe_target(
        path,
        project_root,
        directory=True,
        allow_external=allow_external,
    )
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def knowledge_backup_paths(database_path: Path) -> list[Path]:
    patterns = (
        f"{database_path.stem}.pre-*.db",
        f"{database_path.stem}.backup-*.db",
        f"{database_path.name}.*.bak",
    )
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in database_path.parent.glob(pattern):
            found[str(path.resolve())] = path.resolve()
    return sorted(found.values(), key=str)


def ensure_bot_stopped(
    config: MemoryMaintenanceConfig,
    *,
    ignore_running_lock: bool = False,
) -> None:
    if config.bot_lock_path.exists() and not ignore_running_lock:
        details = config.bot_lock_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            "bot process lock exists; stop the bot before clearing memory: "
            f"{config.bot_lock_path}\n{details}"
        )


def clear_dynamic_memory(
    config: MemoryMaintenanceConfig,
    *,
    allow_external: bool = False,
) -> None:
    reset_database(
        config.public_database,
        config.engine_root,
        project_root=config.project_root,
        allow_external=allow_external,
    )
    clear_directory_contents(
        config.user_database_dir,
        config.project_root,
        allow_external=allow_external,
    )
    clear_directory_contents(
        config.conversation_inbox_dir,
        config.project_root,
        allow_external=allow_external,
    )
    clear_directory_contents(
        config.growth_queue_dir,
        config.project_root,
        allow_external=allow_external,
    )


def clear_knowledge_memory(
    config: MemoryMaintenanceConfig,
    *,
    preserve_backups: bool = False,
    allow_external: bool = False,
) -> list[Path]:
    backups = knowledge_backup_paths(config.knowledge_database)
    reset_database(
        config.knowledge_database,
        config.engine_root,
        project_root=config.project_root,
        allow_external=allow_external,
    )
    if not preserve_backups:
        for backup in backups:
            backup.unlink(missing_ok=True)
            for sidecar in _sidecars(backup):
                sidecar.unlink(missing_ok=True)
    return backups


def dynamic_memory_targets(config: MemoryMaintenanceConfig) -> tuple[Path, ...]:
    return (
        config.public_database,
        config.user_database_dir,
        config.conversation_inbox_dir,
        config.growth_queue_dir,
    )


def knowledge_memory_targets(config: MemoryMaintenanceConfig) -> tuple[Path, ...]:
    return (config.knowledge_database, *knowledge_backup_paths(config.knowledge_database))
