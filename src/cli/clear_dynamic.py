from __future__ import annotations

import argparse
from pathlib import Path

from src.memory.maintenance import (
    MemoryMaintenanceConfig,
    clear_dynamic_memory,
    dynamic_memory_targets,
    ensure_bot_stopped,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Clear public memory, every private user memory, pending conversation "
            "imports, and queued association growth jobs."
        )
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="execute the reset; without this flag only the target plan is shown",
    )
    parser.add_argument(
        "--allow-external",
        action="store_true",
        help="allow configured memory targets outside the project data directory",
    )
    parser.add_argument(
        "--ignore-running-lock",
        action="store_true",
        help="ignore a stale bot lock only after independently confirming no bot is running",
    )
    args = parser.parse_args()
    config = MemoryMaintenanceConfig.from_project(PROJECT_ROOT)

    print("Dynamic memory reset targets:")
    for target in dynamic_memory_targets(config):
        print(f"- {target}")
    print("The bot must be stopped before executing this command.")
    print("Model settings and diagnostic logs are not memory and are preserved.")
    if not args.yes:
        print("Dry run only. Add --yes to execute.")
        return 0

    ensure_bot_stopped(config, ignore_running_lock=args.ignore_running_lock)
    clear_dynamic_memory(config, allow_external=args.allow_external)
    print("Dynamic memory cleared. Restart the bot before serving messages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
