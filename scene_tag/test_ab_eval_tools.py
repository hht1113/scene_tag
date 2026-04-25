import json
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from generate_ab_compare_report import build_report, load_method_stats
from prepare_ab_eval_subset import (
    filter_annotation_records,
    parse_vehicle_id,
    stratified_sample_by_vehicle,
)


class TestPrepareAbEvalSubset(unittest.TestCase):
    def test_parse_vehicle_id(self):
        path = "/a/b/raw_clips/X6S5001/2025-01-01/clip.mp4"
        self.assertEqual(parse_vehicle_id(path), "X6S5001")

    def test_stratified_sample_by_vehicle(self):
        paths = [
            f"/root/raw_clips/V1/c{i}.mp4" for i in range(10)
        ] + [
            f"/root/raw_clips/V2/c{i}.mp4" for i in range(20)
        ]
        sampled = stratified_sample_by_vehicle(paths, sample_size=6, seed=1)
        self.assertEqual(len(sampled), 6)
        self.assertTrue(any("/V1/" in p for p in sampled))
        self.assertTrue(any("/V2/" in p for p in sampled))

    def test_filter_annotation_records(self):
        records = [
            {"video_path": "a.mp4", "segments": []},
            {"video_path": "b.mp4", "segments": []},
        ]
        filtered = filter_annotation_records(records, {"b.mp4"})
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["video_path"], "b.mp4")


class TestGenerateAbCompareReport(unittest.TestCase):
    def test_load_method_stats_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            dpath = Path(d)
            ann = [
                {
                    "video_path": "v1.mp4",
                    "segments": [{"label": "L1", "start": 0.0, "end": 5.0}],
                },
                {
                    "video_path": "v2.mp4",
                    "segments": [{"label": "L2", "start": 0.0, "end": 5.0}],
                },
            ]
            review = {
                "v1.mp4": {"segments": {"0": "correct"}},
                "v2.mp4": {"segments": {"0": "wrong"}},
            }
            anno_path = dpath / "mining_demo.json"
            review_path = dpath / "mining_demo_review.json"
            anno_path.write_text(json.dumps(ann), encoding="utf-8")
            review_path.write_text(json.dumps(review), encoding="utf-8")

            stats = load_method_stats(d)
            self.assertEqual(stats["overall"]["reviewed"], 2)
            self.assertEqual(stats["overall"]["correct"], 1)
            self.assertEqual(stats["label_distribution"]["L1"]["precision"], 100.0)
            self.assertEqual(stats["label_distribution"]["L2"]["precision"], 0.0)

            report = build_report(stats, stats, "Demo Report")
            self.assertIn("Demo Report", report)
            self.assertIn("`L1`", report)
            self.assertIn("Single Agent", report)


if __name__ == "__main__":
    unittest.main()
