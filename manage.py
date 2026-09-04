"""Single administrative entry point for the chatbot."""

from __future__ import annotations

import importlib
import sys


COMMANDS = {
    "import": "src.cli.import_data",
    "query": "src.cli.query",
    "stats": "src.cli.stats",
    "clear-dynamic": "src.cli.clear_dynamic",
    "clear-knowledge": "src.cli.clear_knowledge",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python manage.py <command> [options]")
        print("Commands: " + ", ".join(COMMANDS))
        print("Run 'python manage.py <command> --help' for command options.")
        return 0
    command = sys.argv.pop(1)
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Commands: " + ", ".join(COMMANDS), file=sys.stderr)
        return 2
    module = importlib.import_module(module_name)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
