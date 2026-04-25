"""
从 154K 挖掘池中分层抽样 10,000 个 segment。

分割方式：
  - 总池: 154,548 segments (57辆车)
  - 抽样: 10,000 segments (6.5%)
  - 方法: 按车辆ID分层等比例随机抽样
  - 随机种子: 42 (可复现)
  - 保证每辆车都有代表性

输出:
  scene_tag/mining_10k_video_list.txt  (每行一个视频路径)
  scene_tag/mining_10k_stats.json       (抽样统计)
"""

import os
import json
import random
from collections import defaultdict
from pathlib import Path

POOL_DIR = "/mnt/pfs/houhaotian/junction_videos_segment/raw_clips/"
TARGET_COUNT = 10000
SEED = 42
OUTPUT_LIST = "/root/workspace/LLaMA-Factory/scene_tag/mining_10k_video_list.txt"
OUTPUT_STATS = "/root/workspace/LLaMA-Factory/scene_tag/mining_10k_stats.json"


def main():
    random.seed(SEED)

    vehicle_segments = defaultdict(list)

    print("扫描挖掘池...")
    for vehicle_id in sorted(os.listdir(POOL_DIR)):
        vehicle_dir = os.path.join(POOL_DIR, vehicle_id)
        if not os.path.isdir(vehicle_dir):
            continue
        for root, dirs, files in os.walk(vehicle_dir):
            for f in files:
                if f.endswith(".mp4"):
                    vehicle_segments[vehicle_id].append(os.path.join(root, f))

    total = sum(len(v) for v in vehicle_segments.values())
    print(f"总 segment 数: {total}")
    print(f"车辆数: {len(vehicle_segments)}")

    sampled = []
    stats_per_vehicle = {}

    for vehicle_id in sorted(vehicle_segments.keys()):
        segs = vehicle_segments[vehicle_id]
        ratio = TARGET_COUNT / total
        n_sample = max(1, round(len(segs) * ratio))
        n_sample = min(n_sample, len(segs))

        chosen = random.sample(segs, n_sample)
        sampled.extend(chosen)
        stats_per_vehicle[vehicle_id] = {
            "total": len(segs),
            "sampled": len(chosen),
            "ratio": round(len(chosen) / len(segs) * 100, 1),
        }
        print(f"  {vehicle_id}: {len(segs)} → 抽 {len(chosen)} ({len(chosen)/len(segs)*100:.1f}%)")

    random.shuffle(sampled)

    if len(sampled) > TARGET_COUNT:
        sampled = sampled[:TARGET_COUNT]

    with open(OUTPUT_LIST, "w") as f:
        for path in sampled:
            f.write(path + "\n")

    stats = {
        "seed": SEED,
        "pool_total": total,
        "sampled_total": len(sampled),
        "sample_ratio": round(len(sampled) / total * 100, 2),
        "vehicle_count": len(vehicle_segments),
        "method": "stratified_random_by_vehicle",
        "per_vehicle": stats_per_vehicle,
    }
    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n抽样完成: {len(sampled)} segments")
    print(f"  列表: {OUTPUT_LIST}")
    print(f"  统计: {OUTPUT_STATS}")


if __name__ == "__main__":
    main()
