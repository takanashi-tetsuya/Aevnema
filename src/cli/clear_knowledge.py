from __future__ import annotations

import argparse
from pathlib import Path

from src.memory.maintenance import (
    MemoryMaintenanceConfig,
    clear_knowledge_memory,
    ensure_bot_stopped,
    knowledge_memory_targets,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace the configured knowledge database with an empty schema."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="execute the reset; without this flag only the target plan is shown",
    )
    parser.add_argument(
        "--preserve-backups",
        action="store_true",
        help="keep pre-migration and administrative knowledge database backups",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="allow a configured knowledge database outside the project data directory",
    )
    parser.add_argument(
        "--ignore-running-lock",
        action="store_true",
        help="ignore a stale bot lock only after independently confirming no bot is running",
    )
    args = parser.parse_args()
    config = MemoryMaintenanceConfig.from_project(PROJECT_ROOT)

    print("Knowledge memory reset targets:")
    targets = knowledge_memory_targets(config)
    for index, target in enumerate(targets):
        if index and args.preserve_backups:
            print(f"- {target} (preserved)")
        else:
            print(f"- {target}")
    print("The bot must be stopped before executing this command.")
    print("Diagnostic logs and experiment databases under logs/ are preserved.")
    if not args.yes:
        print("Dry run only. Add --yes to execute.")
        return 0

    ensure_bot_stopped(config, ignore_running_lock=args.ignore_running_lock)
    clear_knowledge_memory(
        config,
        preserve_backups=args.preserve_backups,
        allow_external=args.allow_external,
    )
    print("Knowledge memory cleared. Restart the bot before serving messages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
