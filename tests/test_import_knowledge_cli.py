from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from dotenv import load_dotenv

from src.cli import import_data as import_knowledge


class ImportKnowledgeCliTests(unittest.TestCase):
    def test_parser_accepts_document_map_episode_profile(self):
        args = import_knowledge.build_parser().parse_args(
            ["documents", "--episode-profile", "document_map_assisted"]
        )
        self.assertEqual(args.episode_profile, "document_map_assisted")

    def test_parser_accepts_audited_single_pass_profile(self):
        args = import_knowledge.build_parser().parse_args(
            ["documents", "--episode-profile", "single_pass_audited"]
        )
        self.assertEqual(args.episode_profile, "single_pass_audited")

    def test_parser_exposes_staged_database_and_import_controls(self):
        args = import_knowledge.build_parser().parse_args(
            [
                "documents",
                "--database",
                "data/knowledge/candidate.db",
                "--create-database",
                "--reasoning-max-tokens",
                "8000",
                "--reasoning-model",
                "deepseek-ai/DeepSeek-V3.2",
                "--fallback-model",
                "zai-org/GLM-4.5V",
                "--episode-factual-audit",
                "adaptive",
                "--episode-factual-audit-model",
                "zai-org/GLM-4.5V",
                "--episode-audit-always",
                "--episode-profile",
                "single_pass_evidence",
                "--progress-file",
                "data/knowledge/import-progress.json",
            ]
        )

        self.assertEqual(args.database, "data/knowledge/candidate.db")
        self.assertTrue(args.create_database)
        self.assertEqual(args.reasoning_max_tokens, 8000)
        self.assertEqual(args.reasoning_model, "deepseek-ai/DeepSeek-V3.2")
        self.assertEqual(args.fallback_model, "zai-org/GLM-4.5V")
        self.assertEqual(args.episode_factual_audit, "adaptive")
        self.assertEqual(args.episode_factual_audit_model, "zai-org/GLM-4.5V")
        self.assertTrue(args.episode_audit_always)
        self.assertEqual(args.episode_profile, "single_pass_evidence")
        self.assertEqual(args.progress_file, "data/knowledge/import-progress.json")

    def test_resumable_directory_import_skips_completed_unchanged_files(self):
        class FakeMemory:
            def __init__(self):
                self.calls = []
                self.knowledge = Mock()
                self.knowledge.refresh_indexes = AsyncMock()

            async def import_knowledge_file(
                self, path, source_root, *, refresh_indexes=True
            ):
                self.calls.append((path.name, refresh_indexes))
                return {
                    "status": "completed",
                    "run_id": len(self.calls),
                    "sources": 1,
                    "episodes": 2,
                    "paragraphs": 0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "a.txt").write_text("alpha", encoding="utf-8")
            (corpus / "b.md").write_text("beta", encoding="utf-8")
            progress = root / "progress.json"
            database = root / "candidate.db"
            memory = FakeMemory()

            first = asyncio.run(
                import_knowledge._run_resumable_knowledge_import(
                    memory, corpus, corpus, database, progress
                )
            )
            second = asyncio.run(
                import_knowledge._run_resumable_knowledge_import(
                    memory, corpus, corpus, database, progress
                )
            )

            self.assertEqual(first["status"], "completed")
            self.assertEqual(first["completed_this_run"], 2)
            self.assertEqual(second["skipped_completed"], 2)
            self.assertEqual(memory.calls, [("a.txt", False), ("b.md", False)])
            self.assertEqual(memory.knowledge.refresh_indexes.await_count, 1)

    def test_resumable_directory_import_reimports_changed_file(self):
        class FakeMemory:
            def __init__(self):
                self.knowledge = Mock()
                self.knowledge.refresh_indexes = AsyncMock()

            async def import_knowledge_file(self, *_args, **_kwargs):
                return {
                    "status": "completed",
                    "run_id": 1,
                    "sources": 1,
                    "episodes": 1,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus"
            corpus.mkdir()
            source = corpus / "a.txt"
            source.write_text("first", encoding="utf-8")
            progress = root / "progress.json"
            database = root / "candidate.db"
            memory = FakeMemory()
            asyncio.run(
                import_knowledge._run_resumable_knowledge_import(
                    memory, corpus, corpus, database, progress
                )
            )
            source.write_text("changed", encoding="utf-8")

            result = asyncio.run(
                import_knowledge._run_resumable_knowledge_import(
                    memory, corpus, corpus, database, progress
                )
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["changed_files"], 1)

    def test_source_key_rollback_does_not_touch_other_file_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "candidate.db"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    PRAGMA foreign_keys = ON;
                    CREATE TABLE source(id INTEGER PRIMARY KEY, raw_text TEXT);
                    CREATE TABLE paragraph(
                        id INTEGER PRIMARY KEY,
                        source_id INTEGER,
                        source_key TEXT
                    );
                    CREATE TABLE extraction_task(
                        id INTEGER PRIMARY KEY,
                        source_id INTEGER,
                        source_key TEXT
                    );
                    CREATE TABLE episode(
                        id INTEGER PRIMARY KEY,
                        source_id INTEGER,
                        source_key TEXT
                    );
                    CREATE TABLE association(
                        id INTEGER PRIMARY KEY,
                        from_type TEXT,
                        from_id INTEGER,
                        to_type TEXT,
                        to_id INTEGER
                    );
                    CREATE TABLE concept(
                        id INTEGER PRIMARY KEY,
                        canonical_concept_id INTEGER,
                        FOREIGN KEY(canonical_concept_id) REFERENCES concept(id)
                    );
                    INSERT INTO source VALUES(1, 'target'), (2, 'other');
                    INSERT INTO paragraph VALUES
                        (1, 1, 'a.txt'), (2, 2, 'b.txt');
                    INSERT INTO extraction_task VALUES
                        (1, 1, 'a.txt'), (2, 2, 'b.txt');
                    INSERT INTO episode VALUES
                        (10, 1, 'a.txt'), (20, 2, 'b.txt');
                    INSERT INTO association VALUES
                        (1, 'episode', 10, 'concept', 100),
                        (2, 'episode', 20, 'concept', 200),
                        (3, 'concept', 100, 'concept', 300);
                    INSERT INTO concept VALUES
                        (100, NULL), (200, NULL), (300, NULL);
                    """
                )
            # ``Connection`` context managers commit/rollback but do not
            # close the object.  Windows cannot remove the temporary database
            # while this test-local handle is still alive.
            connection.close()

            counts = import_knowledge._rollback_source_key_artifacts(database, "a.txt")

            self.assertEqual(counts["episodes"], 1)
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    connection.execute("SELECT id FROM episode ORDER BY id").fetchall(),
                    [(20,)],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT id FROM association ORDER BY id"
                    ).fetchall(),
                    [(2,), (3,)],
                )
                self.assertEqual(
                    connection.execute("SELECT id FROM source ORDER BY id").fetchall(),
                    [(2,)],
                )
                self.assertEqual(
                    connection.execute("SELECT id FROM concept ORDER BY id").fetchall(),
                    [(100,), (200,), (300,)],
                )
            connection.close()

    def test_source_key_rollback_prunes_only_orphaned_candidate_concepts(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "candidate.db"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE source(id INTEGER PRIMARY KEY, raw_text TEXT);
                    CREATE TABLE paragraph(id INTEGER PRIMARY KEY, source_id INTEGER);
                    CREATE TABLE extraction_task(
                        id INTEGER PRIMARY KEY, source_id INTEGER, source_key TEXT
                    );
                    CREATE TABLE episode(
                        id INTEGER PRIMARY KEY, source_id INTEGER, source_key TEXT
                    );
                    CREATE TABLE concept(
                        id INTEGER PRIMARY KEY,
                        canonical_concept_id INTEGER,
                        FOREIGN KEY(canonical_concept_id) REFERENCES concept(id)
                    );
                    CREATE TABLE concept_alias(
                        id INTEGER PRIMARY KEY,
                        concept_id INTEGER NOT NULL,
                        alias TEXT NOT NULL,
                        FOREIGN KEY(concept_id) REFERENCES concept(id)
                    );
                    CREATE TABLE association(
                        id INTEGER PRIMARY KEY,
                        from_type TEXT, from_id INTEGER,
                        to_type TEXT, to_id INTEGER
                    );
                    INSERT INTO source VALUES(1, 'target'), (2, 'other');
                    INSERT INTO extraction_task VALUES
                        (1, 1, 'a.txt'), (2, 2, 'b.txt');
                    INSERT INTO episode VALUES
                        (10, 1, 'a.txt'), (20, 2, 'b.txt');
                    INSERT INTO concept VALUES
                        (100, NULL), (200, NULL), (300, 100);
                    INSERT INTO concept_alias VALUES
                        (1, 100, 'orphan-parent'),
                        (2, 200, 'shared'),
                        (3, 300, 'merged-child');
                    INSERT INTO association VALUES
                        (1, 'episode', 10, 'concept', 100),
                        (2, 'episode', 10, 'concept', 200),
                        (3, 'episode', 20, 'concept', 200);
                    """
                )
            connection.close()

            counts = import_knowledge._rollback_source_key_artifacts(database, "a.txt")

            # Concept 100 lost its only graph edge but remains the canonical
            # target of merged Concept 300. Concept 200 is shared by another
            # Episode, so neither may be pruned.
            self.assertEqual(counts["concepts"], 0)
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    connection.execute("SELECT id FROM concept ORDER BY id").fetchall(),
                    [(100,), (200,), (300,)],
                )
            connection.close()

    def test_source_key_rollback_deletes_orphan_concept_dependents_first(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "candidate-with-alias.db"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    PRAGMA foreign_keys = ON;
                    CREATE TABLE source(id INTEGER PRIMARY KEY, raw_text TEXT);
                    CREATE TABLE paragraph(id INTEGER PRIMARY KEY, source_id INTEGER);
                    CREATE TABLE extraction_task(
                        id INTEGER PRIMARY KEY, source_id INTEGER, source_key TEXT
                    );
                    CREATE TABLE episode(
                        id INTEGER PRIMARY KEY, source_id INTEGER, source_key TEXT
                    );
                    CREATE TABLE concept(
                        id INTEGER PRIMARY KEY,
                        canonical_concept_id INTEGER,
                        FOREIGN KEY(canonical_concept_id) REFERENCES concept(id)
                    );
                    CREATE TABLE concept_alias(
                        id INTEGER PRIMARY KEY,
                        concept_id INTEGER NOT NULL,
                        alias TEXT NOT NULL,
                        FOREIGN KEY(concept_id) REFERENCES concept(id)
                    );
                    CREATE TABLE association(
                        id INTEGER PRIMARY KEY,
                        from_type TEXT, from_id INTEGER,
                        to_type TEXT, to_id INTEGER
                    );
                    INSERT INTO source VALUES(1, 'target'), (2, 'other');
                    INSERT INTO extraction_task VALUES
                        (1, 1, 'a.txt'), (2, 2, 'b.txt');
                    INSERT INTO episode VALUES
                        (10, 1, 'a.txt'), (20, 2, 'b.txt');
                    INSERT INTO concept VALUES(100, NULL), (200, NULL);
                    INSERT INTO concept_alias VALUES
                        (1, 100, 'orphan'), (2, 200, 'shared');
                    INSERT INTO association VALUES
                        (1, 'episode', 10, 'concept', 100),
                        (2, 'episode', 20, 'concept', 200);
                    """
                )
            connection.close()

            counts = import_knowledge._rollback_source_key_artifacts(database, "a.txt")

            self.assertEqual(counts["concepts"], 1)
            self.assertEqual(counts["concept_dependents"], 1)
            with sqlite3.connect(database) as connection:
                self.assertEqual(
                    connection.execute("SELECT id FROM concept ORDER BY id").fetchall(),
                    [(200,)],
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT concept_id FROM concept_alias ORDER BY id"
                    ).fetchall(),
                    [(200,)],
                )
            connection.close()

    def test_main_returns_nonzero_for_partial_import_by_default(self):
        args = argparse.Namespace(allow_partial=False)
        parser = Mock()
        parser.parse_args.return_value = args
        with (
            patch.object(import_knowledge, "build_parser", return_value=parser),
            patch.object(
                import_knowledge,
                "run",
                new=AsyncMock(return_value={"status": "partial"}),
            ),
            patch("builtins.print"),
        ):
            exit_code = import_knowledge.main()

        self.assertEqual(exit_code, 2)

    def test_explicit_database_initialization_creates_engine_schema(self):
        load_dotenv(import_knowledge.PROJECT_ROOT / ".env")
        engine_root = import_knowledge.MemorySystemConfig.from_env(
            import_knowledge.PROJECT_ROOT
        ).engine_root
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "candidate.db"

            import_knowledge._initialize_engine_database(engine_root, database_path)

            connection = sqlite3.connect(database_path)
            try:
                version = connection.execute(
                    "SELECT schema_version FROM schema_meta"
                ).fetchone()[0]
            finally:
                connection.close()
        self.assertGreaterEqual(int(version), 1)


if __name__ == "__main__":
    unittest.main()
