#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为单 agent vs 双 agent 对比准备公平子集。

策略:
- 从 batch2 的 10k 视频列表中按车辆分层抽样 1k（默认）
- 导出固定子集视频列表
- 从 batch2 已完成的单 agent 结果中提取同一批视频，形成可直接审核的对照文件
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_GROUPS = [
    "04_Intersection",
    "05_LaneCruising",
    "02_TrafficLight",
]


def parse_vehicle_id(video_path: str) -> str:
    parts = Path(video_path).parts
    if "raw_clips" in parts:
        idx = parts.index("raw_clips")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def load_video_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def stratified_sample_by_vehicle(video_paths: Sequence[str], sample_size: int, seed: int) -> List[str]:
    random.seed(seed)
    by_vehicle: Dict[str, List[str]] = defaultdict(list)
    for path in video_paths:
        by_vehicle[parse_vehicle_id(path)].append(path)

    total = len(video_paths)
    sampled: List[str] = []
    for vehicle_id in sorted(by_vehicle.keys()):
        items = by_vehicle[vehicle_id]
        ratio = sample_size / total
        n = max(1, round(len(items) * ratio))
        n = min(n, len(items))
        sampled.extend(random.sample(items, n))

    # 如因四舍五入不足 sample_size，则从剩余未抽样样本中补齐，保证最终条数精确一致。
    sampled_set = set(sampled)
    if len(sampled) < sample_size:
        remain = [path for path in video_paths if path not in sampled_set]
        random.shuffle(remain)
        sampled.extend(remain[: sample_size - len(sampled)])

    random.shuffle(sampled)
    if len(sampled) > sample_size:
        sampled = sampled[:sample_size]
    return sampled


def filter_annotation_records(records: List[dict], keep_set: Iterable[str]) -> List[dict]:
    keep = set(keep_set)
    return [item for item in records if item.get("video_path") in keep]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 A/B 对比公平子集")
    parser.add_argument(
        "--source_video_list",
        default="/root/workspace/LLaMA-Factory/scene_tag/mining_10k_video_list_batch2.txt",
    )
    parser.add_argument(
        "--source_results_dir",
        default="/root/workspace/LLaMA-Factory/scene_tag/results_batch2",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/workspace/LLaMA-Factory/scene_tag/ab_eval/batch2_samepool_1k",
    )
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260402)
    parser.add_argument("--groups", nargs="*", default=DEFAULT_GROUPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_videos = load_video_list(args.source_video_list)
    sampled_videos = stratified_sample_by_vehicle(source_videos, args.sample_size, args.seed)

    output_dir = Path(args.output_dir)
    single_dir = output_dir / "single_agent"
    single_dir.mkdir(parents=True, exist_ok=True)

    subset_list_path = output_dir / "subset_video_list.txt"
    subset_meta_path = output_dir / "subset_meta.json"

    with open(subset_list_path, "w", encoding="utf-8") as f:
        for path in sampled_videos:
            f.write(path + "\n")

    sampled_set = set(sampled_videos)

    exported = {}
    for group in args.groups:
        src_path = Path(args.source_results_dir) / f"mining_{group}.json"
        if not src_path.exists():
            raise FileNotFoundError(f"结果文件不存在: {src_path}")
        with open(src_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        filtered = filter_annotation_records(records, sampled_set)
        out_path = single_dir / f"mining_{group}_single_agent.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, ensure_ascii=False, indent=2)
        exported[group] = {
            "source": str(src_path),
            "output": str(out_path),
            "count": len(filtered),
        }

    meta = {
        "source_video_list": args.source_video_list,
        "source_results_dir": args.source_results_dir,
        "sample_size": len(sampled_videos),
        "requested_sample_size": args.sample_size,
        "seed": args.seed,
        "groups": args.groups,
        "single_agent_exports": exported,
        "same_pool_as": "batch2",
        "comparison_principle": "same_videos_same_prompts_compare_single_vs_dual_agent",
    }
    with open(subset_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"源视频数: {len(source_videos)}")
    print(f"抽样视频数: {len(sampled_videos)}")
    print(f"子集列表: {subset_list_path}")
    print(f"元信息: {subset_meta_path}")
    for group, info in exported.items():
        print(f"{group}: {info['count']} 条 -> {info['output']}")


if __name__ == "__main__":
    main()
