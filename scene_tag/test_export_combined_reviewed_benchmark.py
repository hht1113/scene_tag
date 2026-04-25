import json
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from export_combined_reviewed_benchmark import (
    flatten_classification_tasks,
    flatten_retrieval_tasks,
    project_retrieval_to_classification_tasks,
    write_outputs,
)


class TestExportCombinedReviewedBenchmark(unittest.TestCase):
    def test_flatten_and_write_outputs(self):
        classification_manifest = {
            "task_count": 2,
            "unique_image_count": 3,
            "reviewed_tp_occurrence_count": 5,
            "tasks": [
                {
                    "task_id": "区域限速标志",
                    "task_name": "区域限速标志",
                    "task_slug": "finetune_area_speed_limit_sign",
                    "task_type": "finetune",
                    "item_count": 2,
                    "output_path": "/tmp/a.json",
                },
                {
                    "task_id": "img_road_surface_water",
                    "task_name": "雨天路面积水",
                    "task_slug": "img_road_surface_water",
                    "task_type": "generalize",
                    "item_count": 1,
                    "output_path": "/tmp/b.json",
                },
            ],
        }
        retrieval_json = {
            "model": "tongyi-plus",
            "top_k": 10,
            "source_judgment_file": "/tmp/judgments.json",
            "task_groups": {
                "交通标识/信号灯": [
                    {
                        "task_name": "红色左转箭头灯",
                        "task_type": "image_to_image",
                        "query_text": None,
                        "query_image": "/tmp/query.jpg",
                        "positive_count": 2,
                        "positive_samples": [
                            {"path": "/tmp/gallery1.jpg", "filename": "gallery1.jpg", "rank": 1, "score": 0.9},
                            {"path": "/tmp/gallery2.jpg", "filename": "gallery2.jpg", "rank": 2, "score": 0.8},
                        ],
                    }
                ],
                "道路施工/障碍物": [
                    {
                        "task_name": "锥桶",
                        "task_type": "text_to_image",
                        "query_text": "锥桶",
                        "query_image": None,
                        "positive_count": 1,
                        "positive_samples": [
                            {"path": "/tmp/gallery2.jpg", "filename": "gallery2.jpg", "rank": 3, "score": 0.7}
                        ],
                    }
                ],
            },
        }

        classification_tasks, classification_summary = flatten_classification_tasks(classification_manifest)
        retrieval_tasks, retrieval_summary = flatten_retrieval_tasks(retrieval_json)
        retrieval_cls_tasks, retrieval_cls_summary = project_retrieval_to_classification_tasks(
            retrieval_json
        )

        self.assertEqual(len(classification_tasks), 2)
        self.assertEqual(classification_summary["positive_item_count"], 3)
        self.assertEqual(len(retrieval_tasks), 2)
        self.assertEqual(retrieval_summary["task_count"], 2)
        self.assertEqual(retrieval_summary["positive_sample_count"], 3)
        self.assertEqual(retrieval_summary["unique_positive_image_count"], 2)
        self.assertEqual(len(retrieval_cls_tasks), 2)
        self.assertEqual(retrieval_cls_summary["task_count"], 2)
        self.assertEqual(retrieval_cls_summary["positive_item_count"], 3)
        self.assertEqual(retrieval_cls_summary["unique_image_count"], 2)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "reviewed_benchmark"
            manifest_path, summary_md_path, manifest = write_outputs(
                output_dir=output_dir,
                classification_tasks=classification_tasks,
                retrieval_tasks=retrieval_tasks,
                retrieval_classification_tasks=retrieval_cls_tasks,
                classification_summary=classification_summary,
                retrieval_summary=retrieval_summary,
                retrieval_classification_summary=retrieval_cls_summary,
                source_paths={
                    "classification_manifest": "/tmp/classification_manifest.json",
                    "retrieval_json": "/tmp/retrieval.json",
                    "retrieval_markdown": "/tmp/retrieval.md",
                },
            )

            self.assertTrue(manifest_path.exists())
            self.assertTrue(summary_md_path.exists())
            self.assertEqual(manifest["overall"]["classification_task_count"], 2)
            self.assertEqual(manifest["overall"]["retrieval_task_count"], 2)
            self.assertEqual(manifest["overall"]["retrieval_as_classification_task_count"], 2)

            classification_tasks_path = output_dir / "classification_tasks.json"
            retrieval_tasks_path = output_dir / "retrieval_tasks.json"
            retrieval_cls_tasks_path = output_dir / "retrieval_as_classification_tasks.json"
            classification_compatible_tasks_path = output_dir / "classification_compatible_tasks.json"
            self.assertTrue(classification_tasks_path.exists())
            self.assertTrue(retrieval_tasks_path.exists())
            self.assertTrue(retrieval_cls_tasks_path.exists())
            self.assertTrue(classification_compatible_tasks_path.exists())

            written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(written_manifest["components"]), 3)
            self.assertEqual(
                written_manifest["overall"]["classification_compatible_tasks_file"],
                str(classification_compatible_tasks_path),
            )


if __name__ == "__main__":
    unittest.main()
