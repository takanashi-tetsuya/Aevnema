from __future__ import annotations

import argparse
import asyncio
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

from dotenv import load_dotenv

from src.memory import MemorySystem, MemorySystemConfig, PlatformIdentity


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import files through the associative memory pipeline"
    )
    parser.add_argument("path")
    parser.add_argument(
        "--domain", choices=("knowledge", "public", "private"), default="knowledge"
    )
    parser.add_argument("--source-root")
    parser.add_argument("--platform")
    parser.add_argument("--user-id")
    parser.add_argument("--display-name", default="")
    parser.add_argument(
        "--database",
        help="override the configured knowledge/public database for this process",
    )
    parser.add_argument(
        "--create-database",
        action="store_true",
        help="create the selected knowledge database when it does not exist",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        help="output-token ceiling used only by this import process",
    )
    parser.add_argument(
        "--reasoning-model",
        help="primary reasoning model used only by this knowledge import",
    )
    parser.add_argument(
        "--fallback-model",
        help="reasoning fallback model used only by this knowledge import",
    )
    parser.add_argument(
        "--prepare-workers",
        type=int,
        help="parallel model-preparation workers used only by this import process",
    )
    parser.add_argument(
        "--relation-batch-size",
        type=int,
        help="maximum relationship groups per model request for this import",
    )
    parser.add_argument(
        "--relation-workers",
        type=int,
        help="parallel relationship-judgement workers used only by this import",
    )
    parser.add_argument(
        "--episode-audit-always",
        action="store_true",
        help="run the combined Episode boundary audit for every Source segment",
    )
    parser.add_argument(
        "--episode-profile",
        choices=(
            "legacy",
            "single_pass_evidence",
            "single_pass_audited",
            "document_map_assisted",
            "document_map_contextual",
            "adaptive_anchor_map",
            "source_scoped_plain",
        ),
        help="select the reversible Episode extraction pipeline for this import",
    )
    parser.add_argument(
        "--episode-factual-audit",
        choices=("off", "adaptive", "always"),
        help="focused speaker/identity fact audit policy for this import",
    )
    parser.add_argument(
        "--episode-factual-audit-model",
        help="independent model for the focused factual audit; defaults to reasoning fallback",
    )
    parser.add_argument(
        "--episode-factual-audit-batch-size",
        type=int,
        help="maximum Episode reviews in one factual-audit request",
    )
    parser.add_argument(
        "--defer-inference-relations",
        action="store_true",
        help="import direct evidence nodes/edges but skip model-built inference relations",
    )
    parser.add_argument(
        "--progress-file",
        help=(
            "enable file-level resumable directory import and store its JSON checkpoint here"
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="return exit code 0 even when the import reports partial failures",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_progress(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "created_at": _utc_now(), "files": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("version") != 1
        or not isinstance(value.get("files"), dict)
    ):
        raise ValueError(f"invalid import progress file: {path}")
    return value


def _save_progress(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _rollback_source_key_artifacts(
    database_path: Path,
    source_key: str,
) -> dict[str, int]:
    """Remove durable rows owned by one failed resumable-import file.

    The cleanup is deliberately source-key scoped.  A Concept is removed only
    when it was connected to this file's Episodes and has no remaining graph
    edge after those Episodes are removed, so shared Concepts stay intact.
    """

    counts = {
        "associations": 0,
        "concept_dependents": 0,
        "concepts": 0,
        "episodes": 0,
        "paragraphs": 0,
        "sources": 0,
    }
    if not database_path.is_file():
        return counts
    # ``sqlite3.Connection``'s context manager only commits/rolls back; it
    # does not close the handle.  Pair it with ``closing`` so Windows can
    # remove a temporary candidate database immediately after rollback.
    with closing(sqlite3.connect(database_path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys = ON")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        required = {
            "source",
            "paragraph",
            "extraction_task",
            "episode",
            "association",
        }
        if not required.issubset(tables):
            return counts
        source_ids = [
            int(row[0])
            for row in connection.execute(
                """
                SELECT DISTINCT source_id
                FROM extraction_task
                WHERE source_key = ? AND source_id IS NOT NULL
                """,
                (source_key,),
            )
        ]
        episode_ids = [
            int(row[0])
            for row in connection.execute(
                "SELECT id FROM episode WHERE source_key = ?",
                (source_key,),
            )
        ]
        concept_ids: list[int] = []
        if episode_ids:
            placeholders = ",".join("?" for _ in episode_ids)
            if "concept" in tables:
                concept_ids = [
                    int(row[0])
                    for row in connection.execute(
                        f"""
                        SELECT DISTINCT CASE
                            WHEN from_type = 'concept' THEN from_id
                            ELSE to_id
                        END
                        FROM association
                        WHERE (from_type = 'episode' AND from_id IN ({placeholders})
                               AND to_type = 'concept')
                           OR (to_type = 'episode' AND to_id IN ({placeholders})
                               AND from_type = 'concept')
                        """,
                        [*episode_ids, *episode_ids],
                    )
                ]
            cursor = connection.execute(
                f"""
                DELETE FROM association
                WHERE (from_type = 'episode' AND from_id IN ({placeholders}))
                   OR (to_type = 'episode' AND to_id IN ({placeholders}))
                """,
                [*episode_ids, *episode_ids],
            )
            counts["associations"] = max(0, int(cursor.rowcount))
        cursor = connection.execute(
            "DELETE FROM episode WHERE source_key = ?", (source_key,)
        )
        counts["episodes"] = max(0, int(cursor.rowcount))
        if concept_ids:
            placeholders = ",".join("?" for _ in concept_ids)
            orphan_concept_ids = [
                int(row[0])
                for row in connection.execute(
                    f"""
                SELECT id
                FROM concept
                WHERE id IN ({placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM association
                      WHERE (from_type = 'concept' AND from_id = concept.id)
                         OR (to_type = 'concept' AND to_id = concept.id)
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM concept AS merged_concept
                      WHERE merged_concept.canonical_concept_id = concept.id
                  )
                """,
                    concept_ids,
                )
            ]
            if orphan_concept_ids:
                orphan_placeholders = ",".join("?" for _ in orphan_concept_ids)

                # Delete rows from every direct Concept child table before the
                # parent rows.  The dependency list comes from SQLite itself so
                # schema additions do not silently make resumable rollback fail.
                # Self-references are handled by the orphan query above.
                for table_name in sorted(tables - {"concept"}):
                    foreign_keys = connection.execute(
                        f'PRAGMA foreign_key_list("{table_name.replace(chr(34), chr(34) * 2)}")'
                    ).fetchall()
                    for foreign_key in foreign_keys:
                        referenced_table = str(foreign_key[2])
                        child_column = str(foreign_key[3])
                        parent_column = str(foreign_key[4] or "id")
                        if referenced_table != "concept" or parent_column != "id":
                            continue
                        quoted_table = table_name.replace('"', '""')
                        quoted_column = child_column.replace('"', '""')
                        cursor = connection.execute(
                            f'DELETE FROM "{quoted_table}" '
                            f'WHERE "{quoted_column}" IN ({orphan_placeholders})',
                            orphan_concept_ids,
                        )
                        counts["concept_dependents"] += max(0, int(cursor.rowcount))

                cursor = connection.execute(
                    f"DELETE FROM concept WHERE id IN ({orphan_placeholders})",
                    orphan_concept_ids,
                )
                counts["concepts"] = max(0, int(cursor.rowcount))
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            cursor = connection.execute(
                f"DELETE FROM paragraph WHERE source_id IN ({placeholders})",
                source_ids,
            )
            counts["paragraphs"] = max(0, int(cursor.rowcount))
        connection.execute(
            "UPDATE extraction_task SET source_id = NULL WHERE source_key = ?",
            (source_key,),
        )
        if source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            cursor = connection.execute(
                f"DELETE FROM source WHERE id IN ({placeholders})",
                source_ids,
            )
            counts["sources"] = max(0, int(cursor.rowcount))
    return counts


async def _run_resumable_knowledge_import(
    memory: MemorySystem,
    input_path: Path,
    source_root: Path,
    database_path: Path,
    progress_path: Path,
) -> dict:
    if not input_path.is_dir():
        raise ValueError("--progress-file requires a directory input")
    supported_suffixes = {".txt", ".md", ".json"}
    files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.casefold() in supported_suffixes
    )
    if not files:
        raise ValueError("directory contains no supported .txt, .md, or .json files")

    progress = _load_progress(progress_path)
    identity = {
        "input_root": str(input_path.resolve()),
        "source_root": str(source_root.resolve()),
        "database": str(database_path.resolve()),
    }
    existing_identity = progress.get("identity")
    if existing_identity is not None and existing_identity != identity:
        raise ValueError(
            "progress file belongs to a different input root, source root, or database"
        )
    progress["identity"] = identity

    for record in progress["files"].values():
        if isinstance(record, dict) and record.get("status") == "running":
            record["status"] = "interrupted"
            record["finished_at"] = _utc_now()
            record["error"] = "previous process ended while this file was running"
    _save_progress(progress_path, progress)

    counts = {
        "files": len(files),
        "completed_this_run": 0,
        "partial_this_run": 0,
        "failed_this_run": 0,
        "skipped_completed": 0,
        "skipped_unresolved": 0,
        "retried_unresolved": 0,
        "changed_files": 0,
        "rolled_back_artifacts": 0,
        "sources": 0,
        "episodes": 0,
        "paragraphs": 0,
    }
    for file_path in files:
        source_key = file_path.resolve().relative_to(source_root.resolve()).as_posix()
        fingerprint = _fingerprint_file(file_path)
        previous = progress["files"].get(source_key)
        if isinstance(previous, dict):
            previous_fingerprint = str(previous.get("sha256", ""))
            if previous_fingerprint and previous_fingerprint != fingerprint:
                rollback = _rollback_source_key_artifacts(database_path, source_key)
                counts["changed_files"] += 1
                counts["rolled_back_artifacts"] += sum(rollback.values())
                progress["files"].pop(source_key, None)
                await memory.knowledge.refresh_indexes()
                _save_progress(progress_path, progress)
            elif previous.get("status") == "completed":
                counts["skipped_completed"] += 1
                continue
            elif previous.get("status") in {
                "partial",
                "failed",
                "interrupted",
                "changed",
            }:
                rollback = _rollback_source_key_artifacts(database_path, source_key)
                counts["retried_unresolved"] += 1
                counts["rolled_back_artifacts"] += sum(rollback.values())
                progress["files"].pop(source_key, None)
                await memory.knowledge.refresh_indexes()
                _save_progress(progress_path, progress)

        record = {
            "status": "running",
            "sha256": fingerprint,
            "size": file_path.stat().st_size,
            "started_at": _utc_now(),
        }
        progress["files"][source_key] = record
        _save_progress(progress_path, progress)
        try:
            result = await memory.import_knowledge_file(
                file_path,
                source_root,
                refresh_indexes=False,
            )
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "finished_at": _utc_now(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            counts["failed_this_run"] += 1
        else:
            status = str(result.get("status", "partial"))
            if status not in {"completed", "partial", "failed"}:
                status = "partial"
            record.update(
                {
                    "status": status,
                    "finished_at": _utc_now(),
                    "run_id": result.get("run_id"),
                    "summary": result,
                }
            )
            counts[f"{status}_this_run"] += 1
            for field in ("sources", "episodes", "paragraphs"):
                counts[field] += int(result.get(field, 0) or 0)
        finally:
            _save_progress(progress_path, progress)

    if any(
        record.get("status") != "completed"
        for record in progress["files"].values()
        if isinstance(record, dict)
    ):
        status = "partial"
    else:
        status = "completed"
    progress["last_finished_at"] = _utc_now()
    progress["last_counts"] = counts
    _save_progress(progress_path, progress)
    if counts["completed_this_run"] or counts["partial_this_run"]:
        await memory.knowledge.refresh_indexes()
    return {
        "status": status,
        "mode": "resumable_directory",
        "progress_file": str(progress_path),
        **counts,
    }


def _initialize_engine_database(engine_root: Path, database_path: Path) -> None:
    package_root = str((engine_root / "src").resolve())
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from memory_demo.database import Database

    Database(database_path).initialize()


async def run(args: argparse.Namespace) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    path = Path(args.path).resolve()
    source_root = (
        Path(args.source_root).resolve()
        if args.source_root
        else path
        if path.is_dir()
        else path.parent
    )
    config = MemorySystemConfig.from_env(PROJECT_ROOT)
    if args.database:
        if args.domain == "private":
            raise ValueError("--database is not supported for private imports")
        database_path = Path(args.database).expanduser()
        if not database_path.is_absolute():
            database_path = PROJECT_ROOT / database_path
        database_path = database_path.resolve()
        if args.domain == "knowledge":
            config.knowledge_database_path = database_path
        else:
            config.public_database_path = database_path
    if args.reasoning_max_tokens is not None and args.reasoning_max_tokens < 1:
        raise ValueError("--reasoning-max-tokens must be greater than zero")
    if args.domain != "knowledge" and (
        args.reasoning_model is not None or args.fallback_model is not None
    ):
        raise ValueError(
            "--reasoning-model and --fallback-model currently apply only to knowledge imports"
        )
    if args.prepare_workers is not None:
        if args.prepare_workers < 1:
            raise ValueError("--prepare-workers must be greater than zero")
        os.environ["MEMORY_IMPORT_PREPARE_WORKERS"] = str(args.prepare_workers)
    if args.relation_batch_size is not None:
        if args.relation_batch_size < 1:
            raise ValueError("--relation-batch-size must be greater than zero")
        os.environ["MEMORY_IMPORT_RELATION_BATCH_SIZE"] = str(args.relation_batch_size)
    if args.relation_workers is not None:
        if args.relation_workers < 1:
            raise ValueError("--relation-workers must be greater than zero")
        os.environ["MEMORY_IMPORT_RELATION_WORKERS"] = str(args.relation_workers)
    if args.episode_audit_always:
        os.environ["MEMORY_IMPORT_EPISODE_AUDIT_ALWAYS"] = "true"
    if args.episode_profile is not None:
        os.environ["MEMORY_IMPORT_EPISODE_PROFILE"] = args.episode_profile
    if args.episode_factual_audit is not None:
        os.environ["MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODE"] = (
            args.episode_factual_audit
        )
    if args.episode_factual_audit_model is not None:
        factual_model = args.episode_factual_audit_model.strip()
        if not factual_model:
            raise ValueError("--episode-factual-audit-model cannot be empty")
        os.environ["MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_MODEL"] = factual_model
    if args.episode_factual_audit_batch_size is not None:
        if args.episode_factual_audit_batch_size < 1:
            raise ValueError(
                "--episode-factual-audit-batch-size must be greater than zero"
            )
        os.environ["MEMORY_IMPORT_EPISODE_FACTUAL_AUDIT_BATCH_SIZE"] = str(
            args.episode_factual_audit_batch_size
        )
    if args.defer_inference_relations:
        os.environ["MEMORY_IMPORT_BUILD_INFERENCE_RELATIONS"] = "false"
    if args.domain == "knowledge":
        if args.reasoning_model is not None:
            model_name = args.reasoning_model.strip()
            if not model_name:
                raise ValueError("--reasoning-model cannot be empty")
            config.knowledge_growth_reasoning_model = model_name
        if args.fallback_model is not None:
            fallback_name = args.fallback_model.strip()
            if not fallback_name:
                raise ValueError("--fallback-model cannot be empty")
            config.knowledge_growth_fallback_model = fallback_name
        import_token_limit = args.reasoning_max_tokens
        if import_token_limit is None:
            import_token_limit = int(
                os.getenv("KNOWLEDGE_IMPORT_REASONING_MAX_TOKENS", "8000")
            )
        config.knowledge_growth_reasoning_max_tokens = max(1, import_token_limit)
        if args.create_database and not config.knowledge_database_path.is_file():
            _initialize_engine_database(
                config.engine_root, config.knowledge_database_path
            )
    elif args.create_database and args.domain == "public":
        if not config.public_database_path.is_file():
            _initialize_engine_database(config.engine_root, config.public_database_path)
    elif args.create_database:
        raise ValueError("--create-database is not supported for private imports")
    memory = MemorySystem(config)
    await memory.initialize()
    if args.domain == "knowledge":
        if args.progress_file:
            progress_path = Path(args.progress_file).expanduser()
            if not progress_path.is_absolute():
                progress_path = PROJECT_ROOT / progress_path
            return await _run_resumable_knowledge_import(
                memory,
                path,
                source_root,
                config.knowledge_database_path,
                progress_path.resolve(),
            )
        return await memory.import_knowledge_file(path, source_root)
    if args.domain == "public":
        return await memory.import_public_file(path, source_root)
    if not args.platform or not args.user_id:
        raise ValueError("private import requires --platform and --user-id")
    identity = PlatformIdentity(args.platform, args.user_id, args.display_name)
    return await memory.import_private_file(identity, path, source_root)


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.allow_partial:
        return 0
    return 0 if result.get("status", "completed") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
