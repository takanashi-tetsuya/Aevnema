from __future__ import annotations

import os
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from dotenv import load_dotenv

from src.bot.process_lock import BotProcessLock
from src.memory.maintenance import (
    MemoryMaintenanceConfig,
    clear_directory_contents,
    clear_dynamic_memory,
    clear_knowledge_memory,
    ensure_bot_stopped,
    knowledge_backup_paths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)
ENGINE_ROOT = MemoryMaintenanceConfig.from_project(PROJECT_ROOT).engine_root


def _row_count(path: Path, table: str) -> int:
    with closing(sqlite3.connect(path)) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


class MemoryMaintenanceTests(unittest.TestCase):
    def _config(self, root: Path) -> MemoryMaintenanceConfig:
        values = {
            "MEMORY_ENGINE_ROOT": str(ENGINE_ROOT),
            "KNOWLEDGE_MEMORY_DB_PATH": "data/knowledge/knowledge.db",
            "PUBLIC_MEMORY_DB_PATH": "data/public/memory.db",
            "USER_MEMORY_DB_DIR": "data/users",
            "CONVERSATION_INBOX_DIR": "data/conversation_inbox",
            "MEMORY_GROWTH_QUEUE_DIR": "data/growth_queue",
        }
        with patch.dict(os.environ, values, clear=False):
            return MemoryMaintenanceConfig.from_project(root)

    @staticmethod
    def _seed_database(path: Path) -> None:
        from src.memory.maintenance import reset_database

        reset_database(path, ENGINE_ROOT)
        with closing(sqlite3.connect(path)) as connection:
            connection.create_function(
                "memory_bigram_tokens", 1, lambda value: str(value or "")
            )
            connection.execute("INSERT INTO source(raw_text) VALUES('memory')")
            connection.commit()

    def test_dynamic_reset_clears_public_users_inbox_and_growth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            self._seed_database(config.public_database)
            user_db = config.user_database_dir / "telegram" / "1" / "memory.db"
            self._seed_database(user_db)
            inbox = config.conversation_inbox_dir / "telegram" / "1" / "turn.json"
            inbox.parent.mkdir(parents=True)
            inbox.write_text("memory", encoding="utf-8")
            queued = config.growth_queue_dir / "job.json"
            queued.parent.mkdir(parents=True)
            queued.write_text("memory", encoding="utf-8")

            clear_dynamic_memory(config)

            self.assertEqual(_row_count(config.public_database, "source"), 0)
            self.assertEqual(list(config.user_database_dir.iterdir()), [])
            self.assertEqual(list(config.conversation_inbox_dir.iterdir()), [])
            self.assertEqual(list(config.growth_queue_dir.iterdir()), [])

    def test_knowledge_reset_removes_configured_backups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            self._seed_database(config.knowledge_database)
            backup = config.knowledge_database.with_name("knowledge.pre-v8.db")
            backup.write_bytes(config.knowledge_database.read_bytes())
            self.assertEqual(knowledge_backup_paths(config.knowledge_database), [backup])

            clear_knowledge_memory(config)

            self.assertEqual(_row_count(config.knowledge_database, "source"), 0)
            self.assertFalse(backup.exists())

    def test_knowledge_reset_can_preserve_backups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            self._seed_database(config.knowledge_database)
            backup = config.knowledge_database.with_name("knowledge.pre-v8.db")
            backup.write_bytes(config.knowledge_database.read_bytes())

            clear_knowledge_memory(config, preserve_backups=True)

            self.assertEqual(_row_count(config.knowledge_database, "source"), 0)
            self.assertTrue(backup.exists())

    def test_reset_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            self._seed_database(config.knowledge_database)

            clear_knowledge_memory(config)
            clear_knowledge_memory(config)

            self.assertEqual(_row_count(config.knowledge_database, "source"), 0)

    def test_external_directory_requires_explicit_permission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external-memory"
            external.mkdir()
            (external / "memory.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(ValueError):
                clear_directory_contents(external, root)

            self.assertTrue((external / "memory.txt").exists())

    def test_running_lock_blocks_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root)
            config.bot_lock_path.parent.mkdir(parents=True, exist_ok=True)
            config.bot_lock_path.write_text('{"pid": 42}', encoding="utf-8")

            with self.assertRaises(RuntimeError):
                ensure_bot_stopped(config)

            ensure_bot_stopped(config, ignore_running_lock=True)

    def test_bot_process_lock_is_exclusive_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "data" / "bot.lock"
            with BotProcessLock(lock_path):
                self.assertTrue(lock_path.exists())
                with self.assertRaises(RuntimeError):
                    BotProcessLock(lock_path).acquire()
            self.assertFalse(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
