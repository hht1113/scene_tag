"""
重建平衡数据集 v2：标注员数据优先补充测试集。

设计原则:
  - 不打乱原始 train/test 分割，避免不必要的重训练
  - 标注员数据（新一轮）优先填充测试集，使每类达到 30 条
  - 原始测试集中超过 30 条的类别，多余部分回流训练集
  - 标注员数据填完测试集后，剩余部分加入训练集
  - 训练集合并后执行动态采样

数据流:
  原始测试集 (script 4)  ─┐
                          ├─► 按类拆分 ──► 每类取 30 条 ──► 新测试集
  标注员数据 (新一轮)    ─┘                    │
                                              剩余
                                               │
  原始训练集 (script 4) ──────────────────── 合并 ──► 动态采样 ──► 新训练集
"""

import json
import random
import copy
import os
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# ─── 配置 ───
TEST_PER_CLASS = 30

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

ALLOWED_LABELS = set(SAMPLING_MULTIPLIERS.keys())

INSTRUCTION_TEMPLATES = [
    "<video>\nWhat is the vehicle's action shown in this 20-second video?",
    "<video>\nPlease explain the ego vehicle's action in this 20-second video.",
    "<video>\nWhat is the ego vehicle's operation in this 20-second video?",
    "<video>\nWhat did the ego vehicle do in this 20-second video?",
    "<video>\nWhat is the driving maneuver of the ego vehicle in this 20-second video?",
    "<video>\nDescribe the behavior of the ego vehicle in this 20-second video.",
    "<video>\nWhat is the driving behavior of the ego vehicle in this 20-second clip?",
    "<video>\nPlease analyze the ego vehicle's action in this 20-second video.",
    "<video>\nWhat is the ego vehicle's action in this 20-second video clip?",
    "<video>\nWhat action is the ego vehicle completing in this 20-second video?",
    "<video>\nIdentify the ego vehicle's action in this 20-second video clip.",
    "<video>\nPlease tell me the ego vehicle's action in this 20-second video.",
    "<video>\nWhat action is the ego vehicle executing in this 20-second clip?",
    "<video>\nWhat is the ego vehicle doing in this 20-second video?",
    "<video>\nWhat is the ego vehicle's behavior in this 20-second video?",
    "<video>\nWhat operation is the ego vehicle currently executing in this 20-second clip?",
    "<video>\nWhat is the operation of the ego vehicle in this 20-second clip?",
    "<video>\nWhat is the driving maneuver of the ego vehicle in this 20-second clip?",
    "<video>\nWhat is the behavior of the ego vehicle in this 20-second clip?",
    "<video>\nWhat is the ego vehicle's action in this 20-second video clip?",
]

OUTPUT_TEMPLATES = [
    "Based on the 20-second video, the ego vehicle's behavior from <start_time>{start}</start_time> to <end_time>{end}</end_time> seconds is <driving_maneuver>{label}</driving_maneuver>.",
    "From the 20-second video, the ego vehicle performs <driving_maneuver>{label}</driving_maneuver> between <start_time>{start}</start_time> and <end_time>{end}</end_time> seconds.",
    "In this 20-second video, from <start_time>{start}</start_time> to <end_time>{end}</end_time> seconds, the ego vehicle's action is <driving_maneuver>{label}</driving_maneuver>.",
    "The 20-second video shows the ego vehicle exhibits <driving_maneuver>{label}</driving_maneuver> behavior during <start_time>{start}</start_time> to <end_time>{end}</end_time> seconds.",
    "Based on the 20-second video content, the primary action of the ego vehicle is <driving_maneuver>{label}</driving_maneuver> from <start_time>{start}</start_time> to <end_time>{end}</end_time> seconds.",
    "From watching this 20-second video, between <start_time>{start}</start_time> and <end_time>{end}</end_time> seconds, the ego vehicle is <driving_maneuver>{label}</driving_maneuver>.",
    "The 20-second video depicts that during the interval <start_time>{start}</start_time> to <end_time>{end}</end_time> seconds, the ego vehicle's behavior is <driving_maneuver>{label}</driving_maneuver>.",
    "In this 20-second video, the ego vehicle executes <driving_maneuver>{label}</driving_maneuver> from <start_time>{start}</start_time> to <end_time>{end}</end_time> seconds.",
    "Based on the 20-second video footage, from <start_time>{start}</start_time> to <end_time>{end}</end_time> seconds, the ego vehicle engages in <driving_maneuver>{label}</driving_maneuver>.",
    "The 20-second video demonstrates that the ego vehicle's driving maneuver is <driving_maneuver>{label}</driving_maneuver> between <start_time>{start}</start_time> and <end_time>{end}</end_time> seconds.",
]


def get_dedup_key(item):
    vid = item["videos"][0] if isinstance(item.get("videos"), list) and item["videos"] else ""
    tr = tuple(item.get("time_range_in_slice", []))
    return (vid, tr)


def convert_annotation(ann, system_prompt):
    start = float(ann["start"])
    end = float(ann["end"])
    label = ann["label"]
    instruction = random.choice(INSTRUCTION_TEMPLATES)
    output_template = random.choice(OUTPUT_TEMPLATES)
    output = output_template.format(label=label, start=f"{start:.1f}", end=f"{end:.1f}")
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "videos": [ann["video_path"]],
        "system": system_prompt,
        "slice_key": ann["video_path"],
        "time_range_in_slice": [start, end],
        "label_en": label,
    }


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


def main():
    data_dir = Path("/root/workspace/LLaMA-Factory/data")

    # ── Step 1: 加载原始数据 ──
    # 正确配对：train_segment_upsample + test_segment_upsample (重叠=0)
    print("[1/6] 加载原始数据...")

    orig_train_file = data_dir / "qwen3_sft_train_segment_upsample.json"
    orig_test_file = data_dir / "qwen3_sft_test_segment_upsample.json"

    with open(orig_train_file) as f:
        orig_train = json.load(f)
    with open(orig_test_file) as f:
        orig_test = json.load(f)

    system_prompt = orig_train[0]["system"]

    print(f"      原始训练集: {len(orig_train)} 条  (train_segment_upsample.json)")
    print(f"      原始测试集: {len(orig_test)} 条  (test_segment_upsample.json)")

    orig_test_counts = Counter(d["label_en"] for d in orig_test)

    # ── Step 2: 加载标注员数据并转换 ──
    print("\n[2/6] 加载标注员数据...")

    ann1_file = "/mnt/pfs/houhaotian/review_annotator_1/training_export/_all.json"
    ann2_file = "/mnt/pfs/houhaotian/review_annotator_2/training_export/_all.json"

    with open(ann1_file) as f:
        ann1_raw = json.load(f)
    with open(ann2_file) as f:
        ann2_raw = json.load(f)

    ann1_filtered = [a for a in ann1_raw if a["label"] in ALLOWED_LABELS]
    ann2_filtered = [a for a in ann2_raw if a["label"] in ALLOWED_LABELS]
    all_ann = ann1_filtered + ann2_filtered

    print(f"      标注员一: {len(ann1_raw)} -> {len(ann1_filtered)} (过滤 {len(ann1_raw)-len(ann1_filtered)})")
    print(f"      标注员二: {len(ann2_raw)} -> {len(ann2_filtered)} (过滤 {len(ann2_raw)-len(ann2_filtered)})")
    print(f"      标注员合计 (过滤后): {len(all_ann)} 条")

    converted_ann = [convert_annotation(a, system_prompt) for a in all_ann]

    # 去重（标注员数据内部 + 与原始数据的重叠）
    orig_train_keys = set(get_dedup_key(d) for d in orig_train)
    orig_test_keys = set(get_dedup_key(d) for d in orig_test)

    seen_keys = orig_train_keys | orig_test_keys
    unique_ann = []
    dup_with_orig = 0
    for item in converted_ann:
        k = get_dedup_key(item)
        if k not in seen_keys:
            seen_keys.add(k)
            unique_ann.append(item)
        else:
            dup_with_orig += 1

    print(f"      与原始数据去重后: {len(unique_ann)} 条 (去除 {dup_with_orig} 条重叠)")

    ann_counts = Counter(d["label_en"] for d in unique_ann)

    # ── Step 3: 构建测试集（每类 30 条） ──
    print(f"\n[3/6] 构建平衡测试集 (每类 {TEST_PER_CLASS} 条)...")

    test_pool_by_label = defaultdict(list)
    for item in orig_test:
        test_pool_by_label[item["label_en"]].append(item)

    ann_by_label = defaultdict(list)
    for item in unique_ann:
        ann_by_label[item["label_en"]].append(item)

    new_test = []
    ann_to_train = []
    test_overflow_to_train = []

    print(f"\n      {'类别':<48} {'原测试':>6} {'需补':>6} {'标注补':>6} {'回流':>6} {'最终':>6}")
    print(f"      {'─' * 80}")

    for label in sorted(ALLOWED_LABELS):
        orig_items = test_pool_by_label.get(label, [])
        ann_items = ann_by_label.get(label, [])
        random.shuffle(ann_items)

        current_test = len(orig_items)
        needed = max(0, TEST_PER_CLASS - current_test)

        if current_test >= TEST_PER_CLASS:
            random.shuffle(orig_items)
            new_test.extend(orig_items[:TEST_PER_CLASS])
            test_overflow_to_train.extend(orig_items[TEST_PER_CLASS:])
            ann_to_train.extend(ann_items)
            overflow = current_test - TEST_PER_CLASS
            print(f"      {label:<48} {current_test:>6} {0:>6} {0:>6} {overflow:>6} {TEST_PER_CLASS:>6}")
        else:
            new_test.extend(orig_items)
            filled = min(needed, len(ann_items))
            new_test.extend(ann_items[:filled])
            ann_to_train.extend(ann_items[filled:])
            final = current_test + filled
            print(f"      {label:<48} {current_test:>6} {needed:>6} {filled:>6} {0:>6} {final:>6}")

    print(f"      {'─' * 80}")
    print(f"      {'总计':<48} {len(orig_test):>6} {'':>6} {'':>6} {len(test_overflow_to_train):>6} {len(new_test):>6}")

    test_counts = Counter(d["label_en"] for d in new_test)

    # ── Step 4: 构建训练集（合并） ──
    print(f"\n[4/6] 构建训练集...")
    new_train_raw = orig_train + test_overflow_to_train + ann_to_train

    # 去重
    seen = set()
    new_train_dedup = []
    for item in new_train_raw:
        k = get_dedup_key(item)
        if k not in seen:
            seen.add(k)
            new_train_dedup.append(item)

    print(f"      原始训练集:         {len(orig_train)} 条")
    print(f"      测试集回流:         {len(test_overflow_to_train)} 条")
    print(f"      标注员剩余:         {len(ann_to_train)} 条")
    print(f"      合并后 (去重前):    {len(new_train_raw)} 条")
    print(f"      合并后 (去重后):    {len(new_train_dedup)} 条")

    train_counts_before = Counter(d["label_en"] for d in new_train_dedup)

    # ── Step 5: 动态采样 ──
    print(f"\n[5/6] 执行动态采样...")
    sampled_train = dynamic_sample(new_train_dedup, SAMPLING_MULTIPLIERS, DEFAULT_MULTIPLIER)
    print(f"      采样前: {len(new_train_dedup)} 条")
    print(f"      采样后: {len(sampled_train)} 条")

    sampled_counts = Counter(d["label_en"] for d in sampled_train)

    # ── Step 6: 保存 ──
    print(f"\n[6/6] 保存数据集...")

    test_out = data_dir / "qwen3_sft_test_segment_balanced_v2.json"
    train_out = data_dir / "qwen3_sft_train_segment_balanced_dynamic_v2.json"

    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(new_test, f, ensure_ascii=False, indent=2)

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(sampled_train, f, ensure_ascii=False, indent=2)

    print(f"      测试集: {test_out}")
    print(f"        样本数: {len(new_test)}, 大小: {os.path.getsize(test_out)/1024:.1f} KB")
    print(f"      训练集: {train_out}")
    print(f"        样本数: {len(sampled_train)}, 大小: {os.path.getsize(train_out)/1024/1024:.1f} MB")

    # ── 数据泄露分析 ──
    print(f"\n{'=' * 90}")
    print(f"  数据泄露风险分析（vs 旧模型训练集）")
    print(f"{'=' * 90}")

    old_model_train_file = data_dir / "qwen3_sft_train_segment_add_tags_dynamic_sample.json"
    with open(old_model_train_file) as f:
        old_model_train = json.load(f)
    old_model_keys = set(get_dedup_key(d) for d in old_model_train)

    new_test_keys = set(get_dedup_key(d) for d in new_test)
    leaked = new_test_keys & old_model_keys
    safe = new_test_keys - old_model_keys

    print(f"  新测试集总数:                   {len(new_test_keys)}")
    print(f"  在旧训练集中 (泄露):            {len(leaked)} ({len(leaked)/len(new_test_keys)*100:.1f}%)")
    print(f"  不在旧训练集中 (安全):          {len(safe)} ({len(safe)/len(new_test_keys)*100:.1f}%)")

    print(f"\n  按类别泄露明细:")
    print(f"  {'类别':<50} {'测试':>6} {'泄露':>6} {'安全':>6} {'泄露率':>8}")
    print(f"  {'─' * 80}")
    test_by_label = defaultdict(list)
    for item in new_test:
        test_by_label[item["label_en"]].append(get_dedup_key(item))
    for label in sorted(test_by_label.keys()):
        keys = set(test_by_label[label])
        lk = keys & old_model_keys
        sk = keys - old_model_keys
        print(f"  {label:<50} {len(keys):>6} {len(lk):>6} {len(sk):>6} {len(lk)/len(keys)*100:>7.1f}%")

    # 新数据集内部一致性
    new_train_keys = set(get_dedup_key(d) for d in sampled_train)
    internal_overlap = new_test_keys & new_train_keys
    print(f"\n  新训练集 vs 新测试集 重叠: {len(internal_overlap)} 条")
    if len(internal_overlap) == 0:
        print(f"  [OK] 新训练集和新测试集之间无数据泄露")
    else:
        print(f"  [ERROR] 存在泄露!")

    # ── 完整分布统计 ──
    all_labels = sorted(ALLOWED_LABELS)

    print(f"\n{'=' * 100}")
    print(f"  完整数据分布")
    print(f"{'=' * 100}")
    print(f"  {'类别':<50} {'测试集':>8} {'训练(采样前)':>14} {'采样倍率':>10} {'训练(采样后)':>14} {'占比':>8}")
    print(f"  {'─' * 95}")
    for label in all_labels:
        tc = test_counts.get(label, 0)
        tb = train_counts_before.get(label, 0)
        ta = sampled_counts.get(label, 0)
        mult = SAMPLING_MULTIPLIERS.get(label, DEFAULT_MULTIPLIER)
        pct = ta / len(sampled_train) * 100 if sampled_train else 0
        print(f"  {label:<50} {tc:>8} {tb:>14} {mult:>9.1f}x {ta:>14} {pct:>7.1f}%")
    print(f"  {'─' * 95}")
    print(f"  {'总计':<50} {len(new_test):>8} {len(new_train_dedup):>14} {'':>10} {len(sampled_train):>14}")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
