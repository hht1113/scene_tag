import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from filter_bootstrap_candidates import build_candidates, iter_input_files, load_records


class TestFilterBootstrapCandidates(unittest.TestCase):
    def test_iter_input_files_supports_paths_and_globs(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "a.json"
            p2 = Path(d) / "b.json"
            p1.write_text("[]", encoding="utf-8")
            p2.write_text("[]", encoding="utf-8")

            files = iter_input_files([str(p1)], [str(Path(d) / "*.json")])
            self.assertEqual(set(files), {str(p1), str(p2)})

    def test_build_candidates_filters_and_dedups(self):
        rows = [
            (
                "file1.json",
                {
                    "video_path": "v1.mp4",
                    "judge_verdict": "accepted",
                    "judge_reason": ["ok"],
                    "accepted_for_bootstrap": True,
                    "final_output": "x",
                    "annotator_raw_output": "draft",
                    "final_segments": [{"label": "L1", "start": 0.0, "end": 5.0}],
                },
            ),
            (
                "file2.json",
                {
                    "video_path": "v1.mp4",
                    "judge_verdict": "corrected",
                    "judge_reason": ["fix"],
                    "accepted_for_bootstrap": True,
                    "final_output": "y",
                    "annotator_raw_output": "draft2",
                    "final_segments": [{"label": "L1", "start": 0.0, "end": 5.0}],
                },
            ),
            (
                "file3.json",
                {
                    "video_path": "v2.mp4",
                    "judge_verdict": "rejected",
                    "accepted_for_bootstrap": False,
                    "final_segments": [{"label": "L2", "start": 1.0, "end": 2.0}],
                },
            ),
        ]

        candidates, stats = build_candidates(rows)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["label"], "L1")
        self.assertEqual(candidates[0]["source_record_count"], 2)
        self.assertEqual(set(candidates[0]["source_files"]), {"file1.json", "file2.json"})
        self.assertEqual(stats["eligible_records"], 2)
        self.assertEqual(stats["unique_candidates"], 1)

    def test_load_records_accepts_json_array(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "input.json"
            payload = [{"video_path": "a.mp4"}, {"video_path": "b.mp4"}]
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_records([str(path)])
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], str(path))


if __name__ == "__main__":
    unittest.main()
