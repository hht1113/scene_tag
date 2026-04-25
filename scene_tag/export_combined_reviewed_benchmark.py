#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description="汇总人工核查后的分类/检索 benchmark")
    parser.add_argument(
        "--classification-manifest",
        type=Path,
        default=Path(
            "/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare/reviewed_tp_datasets/tp_manifest.json"
        ),
    )
    parser.add_argument(
        "--retrieval-json",
        type=Path,
        default=Path("/root/workspace/Gemini/tongyi_plus_reviewed_testset.json"),
    )
    parser.add_argument(
        "--retrieval-markdown",
        type=Path,
        default=Path("/root/workspace/Gemini/tongyi_plus_reviewed_testset.md"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare/reviewed_benchmark"
        ),
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def task_slug(prefix, *parts):
    raw = "::".join(str(p) for p in parts if p)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def flatten_classification_tasks(classification_manifest):
    tasks = []
    total_positive_items = 0

    for task in classification_manifest["tasks"]:
        total_positive_items += task["item_count"]
        tasks.append(
            {
                "benchmark_family": "classification_positive_set",
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "task_slug": task["task_slug"],
                "task_type": task["task_type"],
                "positive_item_count": task["item_count"],
                "task_file": task["output_path"],
            }
        )

    summary = {
        "benchmark_family": "classification_positive_set",
        "task_count": len(tasks),
        "unique_image_count": classification_manifest["unique_image_count"],
        "positive_item_count": total_positive_items,
        "reviewed_tp_occurrence_count": classification_manifest["reviewed_tp_occurrence_count"],
        "source_manifest": classification_manifest,
    }
    return tasks, summary


def flatten_retrieval_tasks(retrieval_json):
    flattened = []
    unique_positive_paths = set()
    total_positive_samples = 0

    for task_group, tasks in retrieval_json.get("task_groups", {}).items():
        for task in tasks:
            positive_samples = task.get("positive_samples", [])
            total_positive_samples += len(positive_samples)
            unique_positive_paths.update(sample["path"] for sample in positive_samples)

            flattened.append(
                {
                    "benchmark_family": "retrieval_reviewed_testset",
                    "task_group": task_group,
                    "task_name": task["task_name"],
                    "task_slug": task_slug(
                        "retrieval",
                        task_group,
                        task["task_name"],
                        task["task_type"],
                        task.get("query_text"),
                        task.get("query_image"),
                    ),
                    "task_type": task["task_type"],
                    "query_text": task.get("query_text"),
                    "query_image": task.get("query_image"),
                    "positive_count": task.get("positive_count", len(positive_samples)),
                    "positive_samples": positive_samples,
                }
            )

    group_summaries = []
    group_map = {}
    for task in flattened:
        group_map.setdefault(task["task_group"], {"task_count": 0, "positive_sample_count": 0})
        group_map[task["task_group"]]["task_count"] += 1
        group_map[task["task_group"]]["positive_sample_count"] += task["positive_count"]

    for group_name, group_info in sorted(group_map.items()):
        group_summaries.append(
            {
                "task_group": group_name,
                "task_count": group_info["task_count"],
                "positive_sample_count": group_info["positive_sample_count"],
            }
        )

    summary = {
        "benchmark_family": "retrieval_reviewed_testset",
        "model": retrieval_json.get("model"),
        "top_k": retrieval_json.get("top_k"),
        "task_count": len(flattened),
        "task_group_count": len(group_summaries),
        "positive_sample_count": total_positive_samples,
        "unique_positive_image_count": len(unique_positive_paths),
        "task_groups": group_summaries,
        "source_judgment_file": retrieval_json.get("source_judgment_file"),
    }
    return flattened, summary


def project_retrieval_to_classification_tasks(retrieval_json):
    projected = {}

    for task_group, tasks in retrieval_json.get("task_groups", {}).items():
        for task in tasks:
            task_name = task["task_name"]
            task_key = (task_group, task_name)
            task_slug_value = task_slug("retrieval_cls", task_group, task_name)
            task_entry = projected.setdefault(
                task_key,
                {
                    "benchmark_family": "retrieval_as_classification_positive_set",
                    "task_group": task_group,
                    "task_id": f"{task_group}::{task_name}",
                    "task_name": task_name,
                    "task_slug": task_slug_value,
                    "positive_label": task_name,
                    "source_query_tasks": [],
                    "items_by_path": {},
                },
            )

            task_entry["source_query_tasks"].append(
                {
                    "task_type": task["task_type"],
                    "query_text": task.get("query_text"),
                    "query_image": task.get("query_image"),
                    "positive_count": task.get("positive_count", len(task.get("positive_samples", []))),
                }
            )

            for sample in task.get("positive_samples", []):
                item = task_entry["items_by_path"].setdefault(
                    sample["path"],
                    {
                        "image_path": sample["path"],
                        "filenames": [],
                        "scores": [],
                        "ranks": [],
                        "source_query_types": [],
                        "source_queries": [],
                        "evidence": [],
                    },
                )
                item["filenames"].append(sample.get("filename"))
                item["scores"].append(sample.get("score"))
                item["ranks"].append(sample.get("rank"))
                item["source_query_types"].append(task["task_type"])
                item["source_queries"].append(
                    {
                        "task_type": task["task_type"],
                        "query_text": task.get("query_text"),
                        "query_image": task.get("query_image"),
                    }
                )
                item["evidence"].append(
                    {
                        "rank": sample.get("rank"),
                        "score": sample.get("score"),
                        "filename": sample.get("filename"),
                        "assigned_label": sample.get("assigned_label"),
                        "task_type": task["task_type"],
                        "query_text": task.get("query_text"),
                        "query_image": task.get("query_image"),
                    }
                )

    task_list = []
    unique_paths = set()

    for (_, _), task in sorted(projected.items(), key=lambda x: (x[1]["task_group"], x[1]["task_name"])):
        items = []
        for image_path, item in sorted(task["items_by_path"].items()):
            item["filenames"] = sorted({name for name in item["filenames"] if name})
            item["ranks"] = sorted({rank for rank in item["ranks"] if rank is not None})
            item["scores"] = sorted({score for score in item["scores"] if score is not None}, reverse=True)
            item["source_query_types"] = sorted(set(item["source_query_types"]))
            item["support_count"] = len(item["evidence"])
            items.append(item)
            unique_paths.add(image_path)

        task_list.append(
            {
                "benchmark_family": "retrieval_as_classification_positive_set",
                "task_group": task["task_group"],
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "task_slug": task["task_slug"],
                "positive_label": task["positive_label"],
                "positive_item_count": len(items),
                "source_query_task_count": len(task["source_query_tasks"]),
                "source_query_tasks": task["source_query_tasks"],
                "items": items,
            }
        )

    summary = {
        "benchmark_family": "retrieval_as_classification_positive_set",
        "task_count": len(task_list),
        "unique_image_count": len(unique_paths),
        "positive_item_count": sum(task["positive_item_count"] for task in task_list),
        "source_retrieval_task_count": sum(
            len(tasks) for tasks in retrieval_json.get("task_groups", {}).values()
        ),
    }
    return task_list, summary


def build_markdown_summary(
    classification_summary,
    retrieval_summary,
    retrieval_classification_summary,
    manifest_path,
):
    lines = [
        "# Reviewed Benchmark Summary",
        "",
        f"- 生成时间: `{utc_now_iso()}`",
        f"- manifest: `{manifest_path}`",
        "",
        "## 分类挖掘 Benchmark",
        "",
        f"- 任务数: `{classification_summary['task_count']}`",
        f"- 去重图片数: `{classification_summary['unique_image_count']}`",
        f"- 正样本条目数: `{classification_summary['positive_item_count']}`",
        f"- TP 证据数: `{classification_summary['reviewed_tp_occurrence_count']}`",
        "",
        "## 检索投影为分类 Benchmark",
        "",
        f"- 任务数: `{retrieval_classification_summary['task_count']}`",
        f"- 去重图片数: `{retrieval_classification_summary['unique_image_count']}`",
        f"- 正样本条目数: `{retrieval_classification_summary['positive_item_count']}`",
        f"- 来源检索 query 数: `{retrieval_classification_summary['source_retrieval_task_count']}`",
        "",
        "## 检索 Benchmark",
        "",
        f"- 模型: `{retrieval_summary['model']}`",
        f"- Top-K: `{retrieval_summary['top_k']}`",
        f"- 任务组数: `{retrieval_summary['task_group_count']}`",
        f"- Query 任务数: `{retrieval_summary['task_count']}`",
        f"- 正样本条目数: `{retrieval_summary['positive_sample_count']}`",
        f"- 去重正样本图片数: `{retrieval_summary['unique_positive_image_count']}`",
        "",
        "## 检索任务组分布",
        "",
        "| 任务组 | Query 数 | 正样本条目数 |",
        "|---|---:|---:|",
    ]

    for group in retrieval_summary["task_groups"]:
        lines.append(
            f"| {group['task_group']} | {group['task_count']} | {group['positive_sample_count']} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_outputs(
    output_dir,
    classification_tasks,
    retrieval_tasks,
    retrieval_classification_tasks,
    classification_summary,
    retrieval_summary,
    retrieval_classification_summary,
    source_paths,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    classification_tasks_path = output_dir / "classification_tasks.json"
    retrieval_tasks_path = output_dir / "retrieval_tasks.json"
    retrieval_classification_tasks_path = output_dir / "retrieval_as_classification_tasks.json"
    classification_compatible_tasks_path = output_dir / "classification_compatible_tasks.json"
    retrieval_cls_by_task_dir = output_dir / "by_task" / "retrieval_as_classification"
    retrieval_cls_by_task_dir.mkdir(parents=True, exist_ok=True)

    with open(classification_tasks_path, "w", encoding="utf-8") as f:
        json.dump(classification_tasks, f, ensure_ascii=False, indent=2)

    with open(retrieval_tasks_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_tasks, f, ensure_ascii=False, indent=2)

    retrieval_classification_task_refs = []
    for task in retrieval_classification_tasks:
        task_path = retrieval_cls_by_task_dir / f"{task['task_slug']}.json"
        task_payload = {
            "generated_at": utc_now_iso(),
            "benchmark_family": task["benchmark_family"],
            "task_group": task["task_group"],
            "task_id": task["task_id"],
            "task_name": task["task_name"],
            "task_slug": task["task_slug"],
            "positive_label": task["positive_label"],
            "positive_item_count": task["positive_item_count"],
            "source_query_task_count": task["source_query_task_count"],
            "source_query_tasks": task["source_query_tasks"],
            "items": task["items"],
        }
        with open(task_path, "w", encoding="utf-8") as f:
            json.dump(task_payload, f, ensure_ascii=False, indent=2)

        retrieval_classification_task_refs.append(
            {
                "benchmark_family": task["benchmark_family"],
                "task_group": task["task_group"],
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "task_slug": task["task_slug"],
                "positive_item_count": task["positive_item_count"],
                "task_file": str(task_path),
            }
        )

    with open(retrieval_classification_tasks_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_classification_task_refs, f, ensure_ascii=False, indent=2)

    classification_compatible_tasks = []
    for task in classification_tasks:
        merged_task = dict(task)
        merged_task["source_component"] = "classification_positive_set"
        classification_compatible_tasks.append(merged_task)

    for task in retrieval_classification_task_refs:
        merged_task = dict(task)
        merged_task["source_component"] = "retrieval_as_classification_positive_set"
        classification_compatible_tasks.append(merged_task)

    with open(classification_compatible_tasks_path, "w", encoding="utf-8") as f:
        json.dump(classification_compatible_tasks, f, ensure_ascii=False, indent=2)

    manifest = {
        "benchmark_name": "reviewed_scene_benchmark",
        "generated_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "sources": source_paths,
        "components": [
            {
                "benchmark_family": "classification_positive_set",
                "summary": classification_summary,
                "tasks_file": str(classification_tasks_path),
            },
            {
                "benchmark_family": "retrieval_reviewed_testset",
                "summary": retrieval_summary,
                "tasks_file": str(retrieval_tasks_path),
            },
            {
                "benchmark_family": "retrieval_as_classification_positive_set",
                "summary": retrieval_classification_summary,
                "tasks_file": str(retrieval_classification_tasks_path),
                "by_task_dir": str(retrieval_cls_by_task_dir),
            },
        ],
        "overall": {
            "component_count": 3,
            "classification_task_count": classification_summary["task_count"],
            "retrieval_task_count": retrieval_summary["task_count"],
            "retrieval_as_classification_task_count": retrieval_classification_summary["task_count"],
            "classification_unique_image_count": classification_summary["unique_image_count"],
            "retrieval_unique_positive_image_count": retrieval_summary["unique_positive_image_count"],
            "retrieval_as_classification_unique_image_count": retrieval_classification_summary["unique_image_count"],
            "classification_compatible_task_count": (
                classification_summary["task_count"] + retrieval_classification_summary["task_count"]
            ),
            "classification_compatible_tasks_file": str(classification_compatible_tasks_path),
        },
    }

    manifest_path = output_dir / "benchmark_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    summary_md_path = output_dir / "benchmark_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(
            build_markdown_summary(
                classification_summary,
                retrieval_summary,
                retrieval_classification_summary,
                manifest_path,
            )
        )

    return manifest_path, summary_md_path, manifest


def main():
    args = parse_args()

    classification_manifest = load_json(args.classification_manifest)
    retrieval_json = load_json(args.retrieval_json)

    classification_tasks, classification_summary = flatten_classification_tasks(classification_manifest)
    retrieval_tasks, retrieval_summary = flatten_retrieval_tasks(retrieval_json)
    retrieval_classification_tasks, retrieval_classification_summary = (
        project_retrieval_to_classification_tasks(retrieval_json)
    )

    source_paths = {
        "classification_manifest": str(args.classification_manifest),
        "retrieval_json": str(args.retrieval_json),
        "retrieval_markdown": str(args.retrieval_markdown) if args.retrieval_markdown.exists() else None,
    }

    manifest_path, summary_md_path, manifest = write_outputs(
        output_dir=args.output_dir,
        classification_tasks=classification_tasks,
        retrieval_tasks=retrieval_tasks,
        retrieval_classification_tasks=retrieval_classification_tasks,
        classification_summary=classification_summary,
        retrieval_summary=retrieval_summary,
        retrieval_classification_summary=retrieval_classification_summary,
        source_paths=source_paths,
    )

    print("=" * 60)
    print("统一 reviewed benchmark 导出完成")
    print(f"classification 任务数: {classification_summary['task_count']}")
    print(f"retrieval 任务数: {retrieval_summary['task_count']}")
    print(f"retrieval_as_classification 任务数: {retrieval_classification_summary['task_count']}")
    print(f"manifest: {manifest_path}")
    print(f"summary: {summary_md_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
