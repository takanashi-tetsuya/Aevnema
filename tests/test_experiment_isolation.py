from contextlib import closing
import sqlite3
import tempfile
from pathlib import Path
import unittest

from experiments.multi_candidate_retrieval_ab import (
    _apply_association_rows,
    _database_signature,
    _sqlite_backup,
)


class RetrievalExperimentIsolationTests(unittest.TestCase):
    def test_ab_databases_are_copied_without_mutating_formal_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal.db"
            baseline = root / "baseline.db"
            treatment = root / "treatment.db"
            with closing(sqlite3.connect(formal)) as connection:
                connection.execute(
                    "CREATE TABLE association ("
                    "id INTEGER PRIMARY KEY, relation_type TEXT, "
                    "use_count INTEGER, last_used TEXT)"
                )
                connection.execute(
                    "INSERT INTO association VALUES (1, 'semantic', 3, 'old')"
                )
                connection.commit()

            before = _database_signature(formal)
            _sqlite_backup(formal, baseline)
            _sqlite_backup(formal, treatment)
            _apply_association_rows(
                treatment,
                [
                    {
                        "id": 2,
                        "relation_type": "semantic",
                        "use_count": 0,
                        "last_used": None,
                    }
                ],
            )

            self.assertEqual(before, _database_signature(formal))
            with closing(sqlite3.connect(baseline)) as connection:
                self.assertEqual(
                    (1, 1),
                    connection.execute(
                        "SELECT COUNT(*), MAX(id) FROM association"
                    ).fetchone(),
                )
            with closing(sqlite3.connect(treatment)) as connection:
                self.assertEqual(
                    (2, 2),
                    connection.execute(
                        "SELECT COUNT(*), MAX(id) FROM association"
                    ).fetchone(),
                )


if __name__ == "__main__":
    unittest.main()
