import json
from pathlib import Path
import tempfile
import unittest

from experiments.memory_foreground_benchmark import _load_manifest, _score_groups, rescore
from experiments.memory_foreground_atomic_ab import (
    _nearest_rank_percentile,
    _paired_summary,
    _variant_summary,
)


class ForegroundBenchmarkTests(unittest.TestCase):
    def test_manifest_imports_engine_catalog_with_frozen_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.mkdir()
            (engine / "questions.json").write_text(
                json.dumps([{"id": "q1", "question": "why"}]),
                encoding="utf-8",
            )
            (engine / "evidence.json").write_text(
                json.dumps(
                    {
                        "questions": {
                            "q1": {
                                "required_episode_groups": [[7, 8], [9]],
                                "required_sources": ["main/one.json"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "test-import-v1",
                        "catalog_imports": [
                            {
                                "questions_path": "engine://questions.json",
                                "evidence_path": "engine://evidence.json",
                                "include_ids": ["q1"],
                                "group_labels": {"q1": ["first", "second"]},
                                "group_alternative_additions": {
                                    "q1": [[10], []]
                                },
                                "group_alternative_overrides": {
                                    "q1": [None, [11, 12]]
                                },
                                "group_required_source_additions": {
                                    "q1": [[], ["main/two.json"]]
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = _load_manifest(manifest_path, engine)

            self.assertEqual(["q1"], [item["id"] for item in manifest["questions"]])
            self.assertEqual(
                [7, 8, 10],
                manifest["questions"][0]["evidence_groups"][0]["alternatives"],
            )
            self.assertEqual(
                [11, 12],
                manifest["questions"][0]["evidence_groups"][1]["alternatives"],
            )
            self.assertTrue(
                manifest["questions"][0]["evidence_groups"][1][
                    "oracle_adjustment"
                ]["override_applied"]
            )
            self.assertEqual(
                ["main/one.json", "main/two.json"],
                manifest["questions"][0]["evidence_groups"][1][
                    "required_sources"
                ],
            )
            self.assertEqual(
                "second",
                manifest["questions"][0]["evidence_groups"][1]["description"],
            )

    def test_atomic_ab_summary_uses_nearest_rank_and_fact_slots(self):
        rows = [
            {
                "elapsed_seconds": value,
                "passed": index < 2,
                "evidence_groups": [
                    {"passed": True},
                    {"passed": index != 2},
                ],
                "timings": {
                    "total_seconds": value - 0.1,
                    "phases_seconds": {"evidence_rerank": value / 2},
                },
            }
            for index, value in enumerate([1.0, 2.0, 9.0])
        ]

        summary = _variant_summary(rows)

        self.assertEqual(9.0, _nearest_rank_percentile([1, 2, 9], 0.9))
        self.assertEqual(2, summary["passed_runs"])
        self.assertEqual(5, summary["matched_fact_slots"])
        self.assertEqual(6, summary["required_fact_slots"])
        self.assertEqual(2.0, summary["elapsed_seconds"]["median"])
        self.assertEqual(9.0, summary["elapsed_seconds"]["p90_nearest_rank"])

    def test_atomic_ab_pair_compares_matching_candidate_sequences(self):
        def row(variant, elapsed, rerank, position):
            return {
                "pair": 1,
                "position": position,
                "variant": variant,
                "elapsed_seconds": elapsed,
                "candidate_episode_ids": [4, 8, 15],
                "passed": True,
                "matched_fact_slots": 3,
                "timings": {
                    "phases_seconds": {"evidence_rerank": rerank}
                },
            }

        summary = _paired_summary(
            [row("atomic40", 10.0, 8.0, 1), row("atomic12", 4.0, 2.0, 2)]
        )

        self.assertEqual(1, summary["complete_pairs"])
        self.assertEqual(1, summary["candidate_sequence_equal_pairs"])
        self.assertEqual(6.0, summary["pairs"][0]["elapsed_seconds_saved_by_12"])
        self.assertEqual(0.6, summary["pairs"][0]["elapsed_relative_reduction"])

    def test_equivalent_episode_is_accepted_for_a_fact_slot(self):
        item = {
            "evidence_groups": [
                {
                    "id": "surface_exam_setup",
                    "description": "表面上帮助成绩不佳的学生",
                    "alternatives": [232, 253, 300],
                }
            ]
        }

        groups = _score_groups(item, {232, 999})

        self.assertTrue(groups[0]["passed"])
        self.assertEqual([232], groups[0]["matches"])

    def test_rescore_reuses_old_episode_ids_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            replay_path = root / "old-report.json"
            output_root = root / "out"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "questions": [
                            {
                                "id": "q1",
                                "question": "question",
                                "evidence_groups": [
                                    {
                                        "id": "fact",
                                        "description": "fact",
                                        "alternatives": [232, 253],
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            replay_path.write_text(
                json.dumps(
                    {
                        "version": "foreground-benchmark-report-v1",
                        "run_id": "old",
                        "repeat": 1,
                        "rows": [
                            {
                                "id": "q1",
                                "elapsed_seconds": 1.25,
                                "error": "",
                                "episode_ids": [232],
                                "evidence_groups": [],
                                "passed": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = rescore(manifest_path, replay_path, output_root)

            self.assertEqual(1, report["summary"]["passed_runs"])
            self.assertEqual(1.0, report["summary"]["fact_slot_recall"])
            self.assertEqual(str(replay_path.resolve()), report["rescored_from"])
            self.assertTrue(Path(report["report_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
