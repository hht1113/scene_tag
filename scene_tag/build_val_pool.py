"""
构建验证集挖掘池：从扩展池中提取不在主池和已有Round-A结果中的新segment，
确保与训练集完全隔离。

用法:
    python build_val_pool.py --sample 3000 --seed 42
    python build_val_pool.py                          # 全量差集，不采样
"""
import json
import argparse
import random
from pathlib import Path


def load_video_paths_from_pool(pool_path):
    with open(pool_path) as f:
        pool = json.load(f)
    return {item["videos"][0]: item for item in pool}


def load_video_paths_from_jsonl(jsonl_path):
    paths = set()
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vp = obj.get("video_path", "")
                if vp:
                    paths.add(vp)
            except json.JSONDecodeError:
                continue
    return paths


def main():
    parser = argparse.ArgumentParser(description="构建验证集挖掘池")
    parser.add_argument("--expanded", default="/root/workspace/LLaMA-Factory/data/mining_pool_expanded.json")
    parser.add_argument("--main-pool", default="/root/workspace/LLaMA-Factory/data/mining_pool_all.json")
    parser.add_argument("--round-a-results", default="/mnt/pfs/chenruize/dataset/round_a_results_v2_final.jsonl")
    parser.add_argument("--output", default="/root/workspace/LLaMA-Factory/data/val_pool_new.json")
    parser.add_argument("--sample", type=int, default=0, help="采样数量，0表示不采样")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[1/4] 加载扩展池: {args.expanded}")
    expanded_map = load_video_paths_from_pool(args.expanded)
    print(f"  扩展池总量: {len(expanded_map)}")

    print(f"[2/4] 加载主池: {args.main_pool}")
    main_map = load_video_paths_from_pool(args.main_pool)
    main_paths = set(main_map.keys())
    print(f"  主池总量: {len(main_paths)}")

    print(f"[3/4] 加载已有Round-A结果: {args.round_a_results}")
    ra_paths = set()
    if Path(args.round_a_results).exists():
        ra_paths = load_video_paths_from_jsonl(args.round_a_results)
        print(f"  已有Round-A结果: {len(ra_paths)}")
    else:
        print(f"  文件不存在，跳过")

    exclude_paths = main_paths | ra_paths
    new_items = [item for path, item in expanded_map.items() if path not in exclude_paths]
    print(f"  差集（新segment）: {len(new_items)}")

    overlap_main = len([p for p in expanded_map if p in main_paths])
    overlap_ra = len([p for p in expanded_map if p in ra_paths])
    print(f"  其中与主池重叠: {overlap_main}, 与Round-A结果重叠: {overlap_ra}")

    if args.sample > 0 and args.sample < len(new_items):
        print(f"[4/4] 随机采样 {args.sample} 条 (seed={args.seed})")
        random.seed(args.seed)
        new_items = random.sample(new_items, args.sample)
    else:
        print(f"[4/4] 不采样，保留全部 {len(new_items)} 条")

    with open(args.output, "w") as f:
        json.dump(new_items, f, ensure_ascii=False, indent=2)
    print(f"已写入: {args.output} ({len(new_items)} 条)")


if __name__ == "__main__":
    main()
