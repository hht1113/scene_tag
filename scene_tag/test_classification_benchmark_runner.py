import json
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from classification_benchmark_runner import (
    TaskRecord,
    build_eval_dataset,
    compute_binary_metrics,
    load_task_records,
    parse_boolean_result,
)


class TestClassificationBenchmarkRunner(unittest.TestCase):
    def test_parse_boolean_result(self):
        cases = [
            ('{"result": true, "reason": "ok"}', True),
            ('{"result": "false", "reason": "no"}', False),
            ('{"result": ["true"], "reason": ["yes"]}', True),
            ("RESULT: true\nREASON: ok", True),
            ("结论：符合", True),
            ("结论：不符合", False),
        ]
        for raw_text, expected in cases:
            actual, _ = parse_boolean_result(raw_text)
            self.assertEqual(actual, expected)

        actual, mode = parse_boolean_result("I cannot decide")
        self.assertIsNone(actual)
        self.assertEqual(mode, "parse_error")

    def test_build_eval_dataset_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            task_a_file = root / "task_a.json"
            task_b_file = root / "task_b.json"
            task_a_file.write_text(
                json.dumps(
                    {
                        "items": [
                            {"image_path": "/img/a1.webp"},
                            {"image_path": "/img/a2.webp"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            task_b_file.write_text(
                json.dumps(
                    {
                        "items": [
                            {"image_path": "/img/b1.webp"},
                            {"image_path": "/img/b2.webp"},
                            {"image_path": "/img/b3.webp"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            tasks = [
                TaskRecord(
                    task_id="task_a",
                    task_name="任务A",
                    task_slug="task_a",
                    task_file=task_a_file,
                    source_component="classification_positive_set",
                    positive_item_count=2,
                ),
                TaskRecord(
                    task_id="task_b",
                    task_name="任务B",
                    task_slug="task_b",
                    task_file=task_b_file,
                    source_component="classification_positive_set",
                    positive_item_count=3,
                ),
            ]

            dataset = build_eval_dataset(
                task_records=tasks,
                max_positives_per_task=2,
                negatives_per_task=2,
                seed=123,
            )
            self.assertEqual(len(dataset["tasks"]), 2)
            self.assertGreaterEqual(dataset["unique_image_count"], 4)

            for task in dataset["tasks"]:
                self.assertEqual(task["positive_count"], 2)
                self.assertEqual(task["negative_count"], 2)
                self.assertEqual(len(task["samples"]), 4)

            metrics = compute_binary_metrics(
                [
                    {"label": 1, "predicted_label": 1, "predicted_positive": True, "latency_seconds": 1.0, "parse_mode": "json_bool"},
                    {"label": 1, "predicted_label": 0, "predicted_positive": False, "latency_seconds": 2.0, "parse_mode": "json_bool"},
                    {"label": 0, "predicted_label": 0, "predicted_positive": False, "latency_seconds": 3.0, "parse_mode": "regex"},
                    {"label": 0, "predicted_label": None, "predicted_positive": None, "latency_seconds": 4.0, "parse_mode": "parse_error"},
                ]
            )
            self.assertEqual(metrics["tp"], 1)
            self.assertEqual(metrics["tn"], 1)
            self.assertEqual(metrics["fp"], 1)
            self.assertEqual(metrics["fn"], 1)
            self.assertEqual(metrics["accuracy"], 0.5)
            self.assertEqual(metrics["precision"], 0.5)
            self.assertEqual(metrics["recall"], 0.5)
            self.assertEqual(metrics["f1"], 0.5)
            self.assertEqual(metrics["parse_error_count"], 1)
            self.assertEqual(metrics["avg_latency_seconds"], 2.5)

    def test_load_task_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            task_index = root / "classification_compatible_tasks.json"
            task_index.write_text(
                json.dumps(
                    [
                        {
                            "task_id": "1",
                            "task_name": "交通灯组",
                            "task_slug": "traffic_lights",
                            "task_file": "/tmp/traffic_lights.json",
                            "source_component": "retrieval_as_classification_positive_set",
                            "positive_item_count": 9,
                        },
                        {
                            "task_id": "2",
                            "task_name": "锥桶",
                            "task_slug": "cones",
                            "task_file": "/tmp/cones.json",
                            "source_component": "retrieval_as_classification_positive_set",
                            "positive_item_count": 6,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            records = load_task_records(task_index, "交通灯", None)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].task_name, "交通灯组")


if __name__ == "__main__":
    unittest.main()
