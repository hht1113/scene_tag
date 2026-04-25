#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


GENERALIZE_TASKS = {
    "img_road_surface_water": {
        "task_name": "雨天路面积水",
        "positive_label": "road_surface_water",
        "slug": "img_road_surface_water",
    },
    "img_dual_countdown": {
        "task_name": "双灯+中间倒计时",
        "positive_label": "dual_countdown",
        "slug": "img_dual_countdown",
    },
    "img_portable_tld": {
        "task_name": "便携式临时信号灯",
        "positive_label": "portable_tld",
        "slug": "img_portable_tld",
    },
}

FINETUNE_LABEL_SLUGS = {
    "掉头箭头": "finetune_uturn_arrow",
    "左转掉头组合箭头": "finetune_left_uturn_arrow",
    "道路施工标志": "finetune_construction_sign",
    "注意儿童标志": "finetune_children_sign",
    "禁止掉头标志": "finetune_no_uturn_sign",
    "区域限速标志": "finetune_area_speed_limit_sign",
}


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description="导出 review 文件中的人工确认 TP 数据集")
    parser.add_argument(
        "--review-dir",
        action="append",
        dest="review_dirs",
        default=[],
        help="review 目录，可重复传入；默认使用 finetune_review 和 generalize_review",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare/reviewed_tp_datasets"),
        help="导出目录",
    )
    return parser.parse_args()


def default_review_dirs():
    base = Path("/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare")
    return [base / "finetune_review", base / "generalize_review"]


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def unique_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def is_reviewed_correct(item):
    return str(item.get("人工判定", "")).strip() == "正确"


def safe_slug(task_type, task_id, task_name):
    if task_type == "generalize":
        return GENERALIZE_TASKS.get(task_id, {}).get("slug", task_id)

    mapped = FINETUNE_LABEL_SLUGS.get(task_name)
    if mapped:
        return mapped

    digest = hashlib.md5(f"{task_type}:{task_id}:{task_name}".encode("utf-8")).hexdigest()[:8]
    return f"{task_type}_task_{digest}"


def parse_review_file_meta(review_path):
    filename = review_path.name
    parent = review_path.parent.name

    if parent == "generalize_review":
        match = re.match(r"^(?P<model>[^_]+)_(?P<task_id>img_.+)_sample\d+\.json$", filename)
        if not match:
            raise ValueError(f"无法解析泛化 review 文件名: {filename}")
        task_id = match.group("task_id")
        task_cfg = GENERALIZE_TASKS.get(task_id, {})
        return {
            "task_type": "generalize",
            "model": match.group("model"),
            "task_id": task_id,
            "task_name": task_cfg.get("task_name", task_id),
            "positive_label": task_cfg.get("positive_label", task_id),
            "slug": safe_slug("generalize", task_id, task_cfg.get("task_name", task_id)),
        }

    if parent == "finetune_review":
        match = re.match(r"^(?P<model>[^_]+)_(?P<label>.+)_sample\d+\.json$", filename)
        if not match:
            raise ValueError(f"无法解析精调 review 文件名: {filename}")
        label = match.group("label")
        return {
            "task_type": "finetune",
            "model": match.group("model"),
            "task_id": label,
            "task_name": label,
            "positive_label": label,
            "slug": safe_slug("finetune", label, label),
        }

    raise ValueError(f"未知 review 目录: {review_path.parent}")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_positive_labels(item, meta):
    if meta["task_type"] == "generalize":
        return [meta["positive_label"]]

    labels = item.get("模型判定", {})
    if isinstance(labels, dict):
        positives = [label for label, value in labels.items() if value]
        if positives:
            return positives

    return [meta["task_name"]]


def build_occurrence(meta, review_path, item, positive_label):
    task_name = positive_label if meta["task_type"] == "finetune" else meta["task_name"]
    task_id = positive_label if meta["task_type"] == "finetune" else meta["task_id"]

    return {
        "task_type": meta["task_type"],
        "task_id": task_id,
        "task_name": task_name,
        "task_slug": safe_slug(meta["task_type"], task_id, task_name),
        "positive_label": positive_label,
        "image_path": item["图片路径"],
        "source_model": meta["model"],
        "source_review_file": review_path.name,
        "source_review_path": str(review_path),
        "review_index": item.get("序号"),
        "known_positive": item.get("已知正样本"),
        "note": (item.get("备注") or "").strip(),
        "raw_positive_labels": extract_positive_labels(item, meta),
    }


def collect_reviewed_occurrences(review_dirs):
    occurrences = []

    for review_dir in review_dirs:
        if not review_dir.exists():
            continue

        for review_path in sorted(review_dir.glob("*.json")):
            meta = parse_review_file_meta(review_path)
            records = load_json(review_path)

            for item in records:
                if not is_reviewed_correct(item):
                    continue
                for positive_label in extract_positive_labels(item, meta):
                    occurrences.append(build_occurrence(meta, review_path, item, positive_label))

    return occurrences


def build_task_datasets(occurrences):
    grouped = {}

    for occ in occurrences:
        task_key = (occ["task_type"], occ["task_id"])
        if task_key not in grouped:
            grouped[task_key] = {
                "task_type": occ["task_type"],
                "task_id": occ["task_id"],
                "task_name": occ["task_name"],
                "task_slug": occ["task_slug"],
                "positive_label": occ["positive_label"],
                "items_by_image": {},
            }

        dataset = grouped[task_key]
        image_entry = dataset["items_by_image"].setdefault(
            occ["image_path"],
            {
                "image_path": occ["image_path"],
                "source_models": [],
                "source_review_files": [],
                "source_review_paths": [],
                "review_indices": [],
                "notes": [],
                "known_positive_flags": [],
                "evidence": [],
            },
        )

        image_entry["source_models"].append(occ["source_model"])
        image_entry["source_review_files"].append(occ["source_review_file"])
        image_entry["source_review_paths"].append(occ["source_review_path"])
        if occ["review_index"] is not None:
            image_entry["review_indices"].append(occ["review_index"])
        if occ["note"]:
            image_entry["notes"].append(occ["note"])
        if occ["known_positive"]:
            image_entry["known_positive_flags"].append(occ["known_positive"])

        image_entry["evidence"].append(
            {
                "source_model": occ["source_model"],
                "source_review_file": occ["source_review_file"],
                "source_review_path": occ["source_review_path"],
                "review_index": occ["review_index"],
                "known_positive": occ["known_positive"],
                "note": occ["note"],
                "raw_positive_labels": occ["raw_positive_labels"],
            }
        )

    finalized = {}
    for task_key, dataset in grouped.items():
        items = []
        for image_path, item in sorted(dataset["items_by_image"].items()):
            item["source_models"] = sorted(set(item["source_models"]))
            item["source_review_files"] = sorted(set(item["source_review_files"]))
            item["source_review_paths"] = sorted(set(item["source_review_paths"]))
            item["review_indices"] = sorted(set(item["review_indices"]))
            item["notes"] = unique_preserve_order(item["notes"])
            item["known_positive_flags"] = unique_preserve_order(item["known_positive_flags"])
            item["review_count"] = len(item["evidence"])
            items.append(item)

        finalized[task_key] = {
            "task_type": dataset["task_type"],
            "task_id": dataset["task_id"],
            "task_name": dataset["task_name"],
            "task_slug": dataset["task_slug"],
            "positive_label": dataset["positive_label"],
            "item_count": len(items),
            "items": items,
        }

    return finalized


def write_task_datasets(task_datasets, output_dir, occurrences, review_dirs):
    output_dir.mkdir(parents=True, exist_ok=True)
    by_task_dir = output_dir / "by_task"
    by_task_dir.mkdir(parents=True, exist_ok=True)

    manifest_tasks = []
    total_unique_images = set()

    for (_, _), dataset in sorted(task_datasets.items(), key=lambda x: (x[1]["task_type"], x[1]["task_name"])):
        task_dir = by_task_dir / dataset["task_type"]
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / f"{dataset['task_slug']}.json"

        payload = {
            "generated_at": utc_now_iso(),
            "task_type": dataset["task_type"],
            "task_id": dataset["task_id"],
            "task_name": dataset["task_name"],
            "task_slug": dataset["task_slug"],
            "positive_label": dataset["positive_label"],
            "item_count": dataset["item_count"],
            "items": dataset["items"],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        total_unique_images.update(item["image_path"] for item in dataset["items"])
        manifest_tasks.append(
            {
                "task_type": dataset["task_type"],
                "task_id": dataset["task_id"],
                "task_name": dataset["task_name"],
                "task_slug": dataset["task_slug"],
                "item_count": dataset["item_count"],
                "output_path": str(output_path),
            }
        )

    all_occurrences_path = output_dir / "all_reviewed_tp_instances.json"
    with open(all_occurrences_path, "w", encoding="utf-8") as f:
        json.dump(sorted(occurrences, key=lambda x: (x["task_type"], x["task_name"], x["image_path"])), f, ensure_ascii=False, indent=2)

    manifest = {
        "generated_at": utc_now_iso(),
        "review_dirs": [str(path) for path in review_dirs],
        "output_dir": str(output_dir),
        "all_reviewed_tp_instances_path": str(all_occurrences_path),
        "task_count": len(manifest_tasks),
        "unique_image_count": len(total_unique_images),
        "reviewed_tp_occurrence_count": len(occurrences),
        "tasks": manifest_tasks,
    }

    manifest_path = output_dir / "tp_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path, manifest


def main():
    args = parse_args()
    review_dirs = [Path(p) for p in args.review_dirs] if args.review_dirs else default_review_dirs()
    occurrences = collect_reviewed_occurrences(review_dirs)
    task_datasets = build_task_datasets(occurrences)
    manifest_path, manifest = write_task_datasets(task_datasets, args.output_dir, occurrences, review_dirs)

    generalize_tasks = sum(1 for task in manifest["tasks"] if task["task_type"] == "generalize")
    finetune_tasks = sum(1 for task in manifest["tasks"] if task["task_type"] == "finetune")

    print("=" * 60)
    print("人工确认 TP 数据集导出完成")
    print(f"review 目录: {', '.join(str(p) for p in review_dirs)}")
    print(f"任务数: {manifest['task_count']} (generalize={generalize_tasks}, finetune={finetune_tasks})")
    print(f"TP occurrence 数: {manifest['reviewed_tp_occurrence_count']}")
    print(f"去重图片数: {manifest['unique_image_count']}")
    print(f"manifest: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
