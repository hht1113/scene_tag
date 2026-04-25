"""
重建平衡数据集：测试集每类 30 条，训练集去除后做动态采样。

流程:
  1. 合并所有数据源（合并后训练集 + 当前测试集），按 (video, time_range) 去重
  2. 每个类别随机抽取 30 条作为测试集
  3. 剩余数据作为训练集
  4. 对训练集执行动态采样
  5. 保存新的训练集和测试集
"""

import json
import random
import copy
import os
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# 动态采样倍率配置（与之前 merge_and_upsample_with_review.py 保持一致）
SAMPLING_MULTIPLIERS = {
    "LaneChange_NavForIntersection":                3.0,
    "StartStop_StartFromMainRoad":                  3.0,
    "DynamicInteraction_VRUInLaneCrossing":          2.5,
    "TrafficLight_StraightStopOrGo":                2.0,
    "TrafficLight_LeftTurnStopOrGo":                1.0,
    "LaneChange_AvoidSlowVRU":                      2.0,
    "LaneChange_AvoidStaticVehicle":                1.0,
    "DynamicInteraction_VehicleInLaneCrossing":      1.0,
    "DynamicInteraction_StandardVehicleCutIn":       1.0,
    "StartStop_ParkRoadside":                       2.0,
    "Intersection_StandardUTurn":                   1.0,
    "LaneCruising_Straight":                        0.3,
}

DEFAULT_MULTIPLIER = 1.0
TEST_PER_CLASS = 30


def get_dedup_key(item):
    vid = item["videos"][0] if isinstance(item.get("videos"), list) and item["videos"] else ""
    tr = tuple(item.get("time_range_in_slice", []))
    return (vid, tr)


def dynamic_sample(data, multipliers, default_multiplier):
    category_map = defaultdict(list)
    for item in data:
        category_map[item["label_en"]].append(item)

    sampled = []
    for label, items in category_map.items():
        current = len(items)
        mult = multipliers.get(label, default_multiplier)
        target = max(1, int(current * mult))

        if target <= current:
            chosen = random.sample(items, target)
            sampled.extend(chosen)
        else:
            sampled.extend(items)
            needed = target - current
            extras = random.choices(items, k=needed)
            sampled.extend([copy.deepcopy(x) for x in extras])

    random.shuffle(sampled)
    return sampled


def print_distribution(counts, title, total=None):
    if total is None:
        total = sum(counts.values())
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"{'标签':<50} {'数量':>8} {'占比':>8}")
    print("-" * 70)
    for label, count in counts.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"{label:<50} {count:>8} {pct:>7.1f}%")
    print("-" * 70)
    print(f"{'总计':<50} {total:>8}")


def main():
    data_dir = Path("/root/workspace/LLaMA-Factory/data")

    # ── Step 1: 读取所有数据源 ──
    merged_train_file = data_dir / "qwen3_sft_train_segment_add_tags.json"
    test_file = data_dir / "qwen3_sft_test_segment_upsample.json"

    print(f"[1/5] 读取数据源...")
    with open(merged_train_file) as f:
        merged_train = json.load(f)
    with open(test_file) as f:
        test_data = json.load(f)

    print(f"      合并后训练集: {len(merged_train)} 条")
    print(f"      当前测试集:   {len(test_data)} 条")

    # ── Step 2: 合并 + 去重 ──
    print(f"\n[2/5] 合并数据并去重...")
    all_data = merged_train + test_data
    seen_keys = set()
    unique_data = []
    dup_count = 0
    for item in all_data:
        key = get_dedup_key(item)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_data.append(item)
        else:
            dup_count += 1

    print(f"      合并前总数: {len(all_data)}")
    print(f"      去重后总数: {len(unique_data)} (去除 {dup_count} 条重复)")

    pool_counts = Counter(d["label_en"] for d in unique_data)
    print_distribution(pool_counts, "去重后全部数据池分布")

    # ── Step 3: 每类抽 30 条测试集 ──
    print(f"\n[3/5] 每类抽取 {TEST_PER_CLASS} 条作为测试集...")
    category_map = defaultdict(list)
    for item in unique_data:
        category_map[item["label_en"]].append(item)

    new_test = []
    new_train = []
    for label in sorted(category_map.keys()):
        items = category_map[label]
        random.shuffle(items)
        n_available = len(items)

        if n_available < TEST_PER_CLASS:
            print(f"      [WARN] {label} 仅有 {n_available} 条，不足 {TEST_PER_CLASS}，全部放入测试集")
            new_test.extend(items)
        else:
            new_test.extend(items[:TEST_PER_CLASS])
            new_train.extend(items[TEST_PER_CLASS:])
            print(f"      {label}: 测试={TEST_PER_CLASS}, 训练={n_available - TEST_PER_CLASS}")

    test_counts = Counter(d["label_en"] for d in new_test)
    train_counts_before = Counter(d["label_en"] for d in new_train)

    print_distribution(test_counts, f"新测试集分布 (每类 {TEST_PER_CLASS} 条)")
    print_distribution(train_counts_before, "新训练集分布 (动态采样前)")

    # ── Step 4: 动态采样 ──
    print(f"\n[4/5] 对训练集执行动态采样...")
    print(f"      采样前训练集: {len(new_train)} 条")
    sampled_train = dynamic_sample(new_train, SAMPLING_MULTIPLIERS, DEFAULT_MULTIPLIER)
    print(f"      采样后训练集: {len(sampled_train)} 条")

    sampled_counts = Counter(d["label_en"] for d in sampled_train)
    print_distribution(sampled_counts, "新训练集分布 (动态采样后)")

    # 打印采样前后对比
    print(f"\n{'=' * 90}")
    print(f"  动态采样前后对比")
    print(f"{'=' * 90}")
    all_labels = sorted(set(list(train_counts_before.keys()) + list(sampled_counts.keys())))
    print(f"{'标签':<50} {'采样前':>8} {'采样后':>8} {'变化':>8} {'倍率':>8}")
    print("-" * 90)
    for label in all_labels:
        b = train_counts_before.get(label, 0)
        a = sampled_counts.get(label, 0)
        diff = a - b
        ratio = a / b if b > 0 else float("inf")
        mult = SAMPLING_MULTIPLIERS.get(label, DEFAULT_MULTIPLIER)
        print(f"{label:<50} {b:>8} {a:>8} {diff:>+8} {ratio:>7.2f}x (配置: {mult}x)")
    print("-" * 90)
    tb = sum(train_counts_before.values())
    ta = sum(sampled_counts.values())
    print(f"{'总计':<50} {tb:>8} {ta:>8} {ta-tb:>+8} {ta/tb if tb>0 else 0:>7.2f}x")

    # ── Step 5: 保存 ──
    print(f"\n[5/5] 保存新数据集...")

    test_out = data_dir / "qwen3_sft_test_segment_balanced.json"
    train_out = data_dir / "qwen3_sft_train_segment_add_tags_balanced_dynamic.json"

    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(new_test, f, ensure_ascii=False, indent=2)
    print(f"      测试集: {test_out}")
    print(f"        样本数: {len(new_test)}, 大小: {os.path.getsize(test_out)/1024:.1f} KB")

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(sampled_train, f, ensure_ascii=False, indent=2)
    print(f"      训练集: {train_out}")
    print(f"        样本数: {len(sampled_train)}, 大小: {os.path.getsize(train_out)/1024/1024:.1f} MB")

    print(f"\n[DONE] 数据集构建完成！")
    print(f"  测试集: {len(new_test)} 条 ({len(test_counts)} 类, 每类 {TEST_PER_CLASS} 条)")
    print(f"  训练集: {len(sampled_train)} 条 (动态采样后)")


if __name__ == "__main__":
    main()
