import json
import sys
import tempfile
import unittest
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from export_reviewed_tp_datasets import (
    build_task_datasets,
    collect_reviewed_occurrences,
    write_task_datasets,
)


class TestExportReviewedTpDatasets(unittest.TestCase):
    def test_collect_and_export_reviewed_tp(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            finetune_dir = root / "finetune_review"
            generalize_dir = root / "generalize_review"
            output_dir = root / "reviewed_tp_datasets"
            finetune_dir.mkdir()
            generalize_dir.mkdir()

            finetune_records = [
                {
                    "序号": 1,
                    "图片路径": "/data/a.webp",
                    "模型判定": {
                        "区域限速标志": True,
                        "禁止掉头标志": True,
                        "掉头箭头": False,
                    },
                    "人工判定": "正确",
                    "备注": "多标签正确",
                },
                {
                    "序号": 2,
                    "图片路径": "/data/b.webp",
                    "模型判定": {
                        "区域限速标志": True,
                    },
                    "人工判定": "错误",
                    "备注": "",
                },
            ]
            (finetune_dir / "qwen235b_区域限速标志_sample2.json").write_text(
                json.dumps(finetune_records, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            generalize_records_model_a = [
                {
                    "序号": 1,
                    "图片路径": "/data/c.webp",
                    "模型结果": "road_surface_water",
                    "已知正样本": "是",
                    "人工判定": "正确",
                    "备注": "",
                }
            ]
            generalize_records_model_b = [
                {
                    "序号": 3,
                    "图片路径": "/data/c.webp",
                    "模型结果": "road_surface_water",
                    "已知正样本": "否",
                    "人工判定": "正确",
                    "备注": "重复命中",
                }
            ]
            (generalize_dir / "doubao_img_road_surface_water_sample1.json").write_text(
                json.dumps(generalize_records_model_a, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (generalize_dir / "qwen235b_img_road_surface_water_sample1.json").write_text(
                json.dumps(generalize_records_model_b, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            occurrences = collect_reviewed_occurrences([finetune_dir, generalize_dir])
            self.assertEqual(len(occurrences), 4)

            task_datasets = build_task_datasets(occurrences)
            self.assertEqual(len(task_datasets), 3)

            finetune_speed = task_datasets[("finetune", "区域限速标志")]
            self.assertEqual(finetune_speed["item_count"], 1)

            finetune_no_uturn = task_datasets[("finetune", "禁止掉头标志")]
            self.assertEqual(finetune_no_uturn["item_count"], 1)

            generalize_water = task_datasets[("generalize", "img_road_surface_water")]
            self.assertEqual(generalize_water["item_count"], 1)
            self.assertEqual(
                sorted(generalize_water["items"][0]["source_models"]),
                ["doubao", "qwen235b"],
            )

            manifest_path, manifest = write_task_datasets(
                task_datasets=task_datasets,
                output_dir=output_dir,
                occurrences=occurrences,
                review_dirs=[finetune_dir, generalize_dir],
            )

            self.assertTrue(manifest_path.exists())
            self.assertEqual(manifest["task_count"], 3)
            self.assertEqual(manifest["unique_image_count"], 2)

            exported_generalize = output_dir / "by_task" / "generalize" / "img_road_surface_water.json"
            self.assertTrue(exported_generalize.exists())


if __name__ == "__main__":
    unittest.main()
