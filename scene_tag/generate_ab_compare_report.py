#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 single-agent / dual-agent 审核结果生成 A/B 对比报告。
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def infer_review_path(annotation_path: str) -> str:
    return os.path.splitext(annotation_path)[0] + "_review.json"


def load_method_stats(annotation_dir: str) -> Dict[str, object]:
    annotation_paths = sorted(
        str(p) for p in Path(annotation_dir).glob("*.json")
        if p.is_file() and not p.name.endswith("_review.json")
    )

    per_label = defaultdict(lambda: {"total": 0, "reviewed": 0, "correct": 0, "wrong": 0, "unsure": 0})
    file_rows = []
    total_segments = 0
    total_reviewed = 0
    total_correct = 0
    total_wrong = 0
    total_unsure = 0

    for anno_path in annotation_paths:
        review_path = infer_review_path(anno_path)
        if not os.path.exists(review_path):
            continue

        with open(anno_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        with open(review_path, "r", encoding="utf-8") as f:
            reviews = json.load(f)

        file_seg_total = 0
        file_reviewed = 0
        file_correct = 0

        for ann in annotations:
            vp = ann.get("video_path", "")
            segs = ann.get("segments", []) or []
            rv = reviews.get(vp, {}) if isinstance(reviews, dict) else {}
            seg_reviews = rv.get("segments", {}) if isinstance(rv, dict) else {}

            for idx, seg in enumerate(segs):
                label = seg.get("label", "")
                if not label or label == "not_applicable":
                    continue

                file_seg_total += 1
                total_segments += 1
                per_label[label]["total"] += 1

                verdict = seg_reviews.get(str(idx), "")
                if verdict:
                    file_reviewed += 1
                    total_reviewed += 1
                    per_label[label]["reviewed"] += 1

                    if verdict == "correct":
                        file_correct += 1
                        total_correct += 1
                        per_label[label]["correct"] += 1
                    elif verdict == "wrong":
                        total_wrong += 1
                        per_label[label]["wrong"] += 1
                    else:
                        total_unsure += 1
                        per_label[label]["unsure"] += 1

        file_rows.append(
            {
                "file": os.path.basename(anno_path),
                "segments": file_seg_total,
                "reviewed": file_reviewed,
                "correct": file_correct,
                "precision": round(file_correct / file_reviewed * 100, 1) if file_reviewed else None,
            }
        )

    label_distribution = {}
    for label, stats in per_label.items():
        precision = round(stats["correct"] / stats["reviewed"] * 100, 1) if stats["reviewed"] else None
        label_distribution[label] = {**stats, "precision": precision}

    return {
        "files": file_rows,
        "overall": {
            "segments": total_segments,
            "reviewed": total_reviewed,
            "correct": total_correct,
            "wrong": total_wrong,
            "unsure": total_unsure,
            "precision": round(total_correct / total_reviewed * 100, 1) if total_reviewed else None,
        },
        "label_distribution": dict(sorted(label_distribution.items())),
    }


def build_report(single_stats: Dict[str, object], dual_stats: Dict[str, object], title: str) -> str:
    single_overall = single_stats["overall"]
    dual_overall = dual_stats["overall"]

    all_labels = sorted(set(single_stats["label_distribution"].keys()) | set(dual_stats["label_distribution"].keys()))

    lines = [
        f"# {title}",
        "",
        "## 1. 整体结果",
        "",
        "| 方法 | 送审段数 | 已审核 | 正确 | Precision |",
        "|---|---:|---:|---:|---:|",
        f"| Single Agent | {single_overall['segments']} | {single_overall['reviewed']} | {single_overall['correct']} | {single_overall['precision'] if single_overall['precision'] is not None else '—'}% |",
        f"| Dual Agent | {dual_overall['segments']} | {dual_overall['reviewed']} | {dual_overall['correct']} | {dual_overall['precision'] if dual_overall['precision'] is not None else '—'}% |",
        "",
        "## 2. 逐标签对比",
        "",
        "| 标签 | Single 已审 | Single Precision | Dual 已审 | Dual Precision | 差值(pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for label in all_labels:
        s = single_stats["label_distribution"].get(label, {})
        d = dual_stats["label_distribution"].get(label, {})
        sp = s.get("precision")
        dp = d.get("precision")
        delta = None
        if sp is not None and dp is not None:
            delta = round(dp - sp, 1)
        lines.append(
            f"| `{label}` | {s.get('reviewed', 0)} | "
            f"{sp if sp is not None else '—'}% | "
            f"{d.get('reviewed', 0)} | "
            f"{dp if dp is not None else '—'}% | "
            f"{delta if delta is not None else '—'} |"
        )

    lines.extend(
        [
            "",
            "## 3. 分组文件对比",
            "",
            "### Single Agent",
            "",
            "| 文件 | 送审段数 | 已审核 | 正确 | Precision |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in single_stats["files"]:
        lines.append(
            f"| `{row['file']}` | {row['segments']} | {row['reviewed']} | {row['correct']} | "
            f"{row['precision'] if row['precision'] is not None else '—'}% |"
        )

    lines.extend(
        [
            "",
            "### Dual Agent",
            "",
            "| 文件 | 送审段数 | 已审核 | 正确 | Precision |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in dual_stats["files"]:
        lines.append(
            f"| `{row['file']}` | {row['segments']} | {row['reviewed']} | {row['correct']} | "
            f"{row['precision'] if row['precision'] is not None else '—'}% |"
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 single-agent vs dual-agent A/B 审核报告")
    parser.add_argument("--single_dir", required=True, help="single-agent 标注与审核目录")
    parser.add_argument("--dual_dir", required=True, help="dual-agent 标注与审核目录")
    parser.add_argument("--output_md", required=True, help="输出 markdown 报告")
    parser.add_argument("--output_json", required=True, help="输出 json 报告")
    parser.add_argument("--title", default="Single Agent vs Dual Agent A/B Report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    single_stats = load_method_stats(args.single_dir)
    dual_stats = load_method_stats(args.dual_dir)
    report_md = build_report(single_stats, dual_stats, args.title)

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "title": args.title,
                "single": single_stats,
                "dual": dual_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"报告已生成: {args.output_md}")
    print(f"数据已生成: {args.output_json}")


if __name__ == "__main__":
    main()
