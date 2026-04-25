#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从双 agent 结果中筛出可进入冷启动集的候选样本。

输入:
- 一个或多个 dual_agent_distillation.py 的输出 JSON 文件

输出:
- 扁平化后的 segment 级候选 JSON
- 可选 JSONL
- 统计 JSON
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def iter_input_files(paths: List[str], patterns: List[str]) -> List[str]:
    files: List[str] = []
    seen = set()

    for path in paths:
        if os.path.isfile(path) and path not in seen:
            files.append(path)
            seen.add(path)

    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            if os.path.isfile(path) and path not in seen:
                files.append(path)
                seen.add(path)

    return files


def load_records(input_files: Iterable[str]) -> List[Tuple[str, Dict[str, Any]]]:
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"输入文件不是 JSON 数组: {path}")
        for item in data:
            if isinstance(item, dict):
                rows.append((path, item))
    return rows


def segment_key(video_path: str, label: str, start: float, end: float) -> Tuple[str, str, float, float]:
    return (video_path, label, round(float(start), 1), round(float(end), 1))


def build_candidates(rows: List[Tuple[str, Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dedup: Dict[Tuple[str, str, float, float], Dict[str, Any]] = {}
    label_counter: Counter[str] = Counter()
    verdict_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()

    total_records = 0
    eligible_records = 0

    for source_file, item in rows:
        total_records += 1
        verdict = item.get("judge_verdict", "")
        verdict_counter[verdict] += 1

        if not item.get("accepted_for_bootstrap", False):
            continue

        final_segments = item.get("final_segments") or []
        if not final_segments:
            continue

        eligible_records += 1
        source_counter[Path(source_file).name] += 1

        for seg in final_segments:
            label = seg.get("label", "")
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            if not label:
                continue

            key = segment_key(item.get("video_path", ""), label, start, end)
            label_counter[label] += 1

            if key not in dedup:
                dedup[key] = {
                    "video_path": item.get("video_path", ""),
                    "label": label,
                    "start": round(float(start), 1),
                    "end": round(float(end), 1),
                    "judge_verdict": verdict,
                    "judge_reason": item.get("judge_reason", []),
                    "final_output": item.get("final_output", ""),
                    "annotator_raw_output": item.get("annotator_raw_output", ""),
                    "source_files": [source_file],
                    "source_record_count": 1,
                }
            else:
                entry = dedup[key]
                entry["source_record_count"] += 1
                if source_file not in entry["source_files"]:
                    entry["source_files"].append(source_file)

    candidates = sorted(
        dedup.values(),
        key=lambda x: (x["label"], x["video_path"], x["start"], x["end"]),
    )

    stats = {
        "total_input_records": total_records,
        "eligible_records": eligible_records,
        "unique_candidates": len(candidates),
        "label_distribution": dict(sorted(label_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "verdict_distribution": dict(sorted(verdict_counter.items())),
        "source_distribution": dict(sorted(source_counter.items())),
    }
    return candidates, stats


def write_outputs(
    candidates: List[Dict[str, Any]],
    stats: Dict[str, Any],
    output_json: str,
    output_stats: str,
    output_jsonl: str = "",
) -> None:
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_stats) or ".", exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)

    with open(output_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if output_jsonl:
        os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for item in candidates:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="筛选双 agent 冷启动候选样本")
    parser.add_argument(
        "--input_files",
        nargs="*",
        default=[],
        help="一个或多个 dual-agent 结果 JSON 文件",
    )
    parser.add_argument(
        "--input_globs",
        nargs="*",
        default=[],
        help='glob 模式，例如 "scene_tag/results_dual/*.json"',
    )
    parser.add_argument("--output_json", required=True, help="输出 JSON")
    parser.add_argument("--output_stats", required=True, help="输出统计 JSON")
    parser.add_argument("--output_jsonl", default="", help="可选输出 JSONL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = iter_input_files(args.input_files, args.input_globs)
    if not input_files:
        raise SystemExit("没有找到任何输入文件")

    rows = load_records(input_files)
    candidates, stats = build_candidates(rows)
    write_outputs(
        candidates=candidates,
        stats=stats,
        output_json=args.output_json,
        output_stats=args.output_stats,
        output_jsonl=args.output_jsonl,
    )

    print(f"输入文件数: {len(input_files)}")
    print(f"输入记录数: {stats['total_input_records']}")
    print(f"可用记录数: {stats['eligible_records']}")
    print(f"唯一候选数: {stats['unique_candidates']}")
    print(f"输出 JSON: {args.output_json}")
    print(f"输出统计: {args.output_stats}")
    if args.output_jsonl:
        print(f"输出 JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
