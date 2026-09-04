from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only text search over Episode and Association rows"
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    database = args.database.resolve()
    uri = f"file:{database.as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        for pattern in args.patterns:
            print(f"\n## {pattern}")
            rows = connection.execute(
                "SELECT id, source_key, segment_index, text "
                "FROM episode WHERE text LIKE ? "
                "ORDER BY id LIMIT ?",
                (f"%{pattern}%", max(1, args.limit)),
            ).fetchall()
            for row in rows:
                text = " ".join(str(row["text"]).split())
                print(
                    f"E{int(row['id'])} {row['source_key']} "
                    f"s{int(row['segment_index'])}: {text}"
                )
            edge_rows = connection.execute(
                "SELECT id, from_type, from_id, to_type, to_id, relation_key, "
                "generation, audit_status, relation_text FROM association "
                "WHERE relation_text LIKE ? ORDER BY id LIMIT ?",
                (f"%{pattern}%", max(1, args.limit)),
            ).fetchall()
            for row in edge_rows:
                text = " ".join(str(row["relation_text"]).split())
                print(
                    f"A{int(row['id'])} {row['from_type']}:{int(row['from_id'])}"
                    f"->{row['to_type']}:{int(row['to_id'])} "
                    f"{row['relation_key']} g{int(row['generation'])} "
                    f"{row['audit_status']}: {text}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
