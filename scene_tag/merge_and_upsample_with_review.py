"""
合并标注员数据 + 按类别动态采样平衡训练集。

流程:
  1. 读取原始训练集 qwen3_sft_train_segment.json (采样前)
  2. 读取标注人一/二数据，过滤保留允许的 12 个标签
  3. 合并到训练集 -> qwen3_sft_train_segment_add_tags.json
  4. 按类别动态采样（低 precision 类别加强）-> qwen3_sft_train_segment_add_tags_dynamic_sample.json
"""

import json
import random
import copy
import os
from collections import Counter, defaultdict
from pathlib import Path

random.seed(42)

# ============================================================================
#  动态采样倍率配置（基于合并后各类别的原始数量）
#
#  multiplier > 1  : 上采样（复制样本增多）
#  multiplier == 1 : 保持原样
#  multiplier < 1  : 下采样（随机抽取减少）
#
#  例: 某标签合并后有 100 条，multiplier=2.0 → 采样后 200 条
#      某标签合并后有 500 条，multiplier=0.3 → 采样后 150 条
# ============================================================================
SAMPLING_MULTIPLIERS = {
    # ── precision < 50%，大幅加强 ──
    "LaneChange_NavForIntersection":                3.0,   # 导航变道 (precision 28.6%)
    "StartStop_StartFromMainRoad":                  3.0,   # 主路发车 (precision 37.5%)
    "DynamicInteraction_VRUInLaneCrossing":          2.5,   # 车道内VRU横穿 (precision 42.3%)
    "TrafficLight_StraightStopOrGo":                2.0,   # 直行红绿灯起停 (precision 46.2%)

    # ── precision 适中，适当保持或小幅增强 ──
    "TrafficLight_LeftTurnStopOrGo":                1.0,   # 左转红绿灯起停 (precision 69.2%)
    "LaneChange_AvoidSlowVRU":                      2.0,   # 避让慢行VRU变道 (precision 73.7%)
    "LaneChange_AvoidStaticVehicle":                1.0,   # 避让静止车辆变道 (precision 72.8%)
    "DynamicInteraction_VehicleInLaneCrossing":      1.0,   # 车道内车辆横穿 (precision 87.2%)
    "DynamicInteraction_StandardVehicleCutIn":       1.0,   # 标准车辆加塞 (precision 82.6%)
    "StartStop_ParkRoadside":                       2.0,   # 路边停车 (precision 76.9%)
    "Intersection_StandardUTurn":                   1.0,   # 标准掉头 (precision 75.0%)
    "LaneCruising_Straight":                        0.3,   # 直行巡航（数量最多，下采样）
}

# 需要拆分到测试集的新标签及数量（已清空，不再拆分新标签）
TEST_SPLIT_COUNTS = {}

# 未在上表中列出的标签使用此默认倍率
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


def get_system_prompt(existing_data):
    """从现有数据中提取 system prompt"""
    return existing_data[0]["system"]


def update_system_prompt(old_prompt):
    """在 system prompt 中插入新增的标签定义（RightTurn / LeftTurn / LeadVehicleEmergencyBrake）"""
    new_prompt = old_prompt

    new_prompt = new_prompt.replace(
        "Intersection_StandardUTurn\nLaneCruising_Straight",
        "Intersection_StandardUTurn\nIntersection_RightTurn\nIntersection_LeftTurn\n"
        "DynamicInteraction_LeadVehicleEmergencyBrake\nLaneCruising_Straight",
    )

    new_prompt = new_prompt.replace(
        "11. Intersection_StandardUTurn: Making a U-turn at an intersection\n"
        "12. LaneCruising_Straight: Straight-line cruising without notable events\n"
        "13. else: Default for all other behaviors not covered by the predefined categories",

        "11. Intersection_StandardUTurn: Making a U-turn at an intersection\n"
        "12. Intersection_RightTurn: Making a right turn at an intersection\n"
        "13. Intersection_LeftTurn: Making a left turn at an intersection\n"
        "14. DynamicInteraction_LeadVehicleEmergencyBrake: The lead vehicle suddenly brakes, requiring the ego vehicle to react\n"
        "15. LaneCruising_Straight: Straight-line cruising without notable events\n"
        "16. else: Default for all other behaviors not covered by the predefined categories",
    )
    return new_prompt


def convert_annotation(ann, system_prompt, source_tag):
    """将标注员数据转换为训练集格式"""
    start = float(ann["start"])
    end = float(ann["end"])
    label = ann["label"]

    instruction = random.choice(INSTRUCTION_TEMPLATES)
    output_template = random.choice(OUTPUT_TEMPLATES)
    output = output_template.format(
        label=label,
        start=f"{start:.1f}",
        end=f"{end:.1f}",
    )

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
    """
    按类别动态采样:
    - 每个类别的目标数量 = 原始数量 × 对应倍率
    - 倍率 > 1: 上采样（重复样本）
    - 倍率 < 1: 下采样（随机抽取）
    - 倍率 == 1: 保持不变
    """
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


def print_comparison(before_counts, after_counts, title):
    """打印前后对比表"""
    all_labels = sorted(set(list(before_counts.keys()) + list(after_counts.keys())))
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")
    print("{:<50} {:>8} {:>8} {:>8} {:>8}".format("标签", "采样前", "采样后", "变化", "倍率"))
    print("-" * 90)
    for label in all_labels:
        b = before_counts.get(label, 0)
        a = after_counts.get(label, 0)
        diff = a - b
        ratio = a / b if b > 0 else float("inf")
        marker = " [NEW]" if b == 0 else ""
        print("{:<50} {:>8} {:>8} {:>+8} {:>7.2f}x{}".format(label, b, a, diff, ratio, marker))
    print("-" * 90)
    tb = sum(before_counts.values())
    ta = sum(after_counts.values())
    print("{:<50} {:>8} {:>8} {:>+8} {:>7.2f}x".format("总计", tb, ta, ta - tb, ta / tb if tb > 0 else 0))


def print_sampling_config():
    """打印当前采样配置"""
    print(f"\n{'=' * 90}")
    print(f"  当前动态采样倍率配置")
    print(f"{'=' * 90}")
    print("{:<50} {:>10} {:>12}".format("标签", "倍率", "说明"))
    print("-" * 75)
    for label, mult in sorted(SAMPLING_MULTIPLIERS.items()):
        if mult > 1:
            desc = f"上采样 ×{mult}"
        elif mult < 1:
            desc = f"下采样 ×{mult}"
        else:
            desc = "保持原样"
        print("{:<50} {:>10.1f} {:>12}".format(label, mult, desc))
    print("-" * 75)
    print(f"  未列出标签的默认倍率: {DEFAULT_MULTIPLIER}")


def main():
    data_dir = Path("/root/workspace/LLaMA-Factory/data")

    print_sampling_config()

    # ── Step 1: 读取原始训练集 (采样前) ──
    orig_file = data_dir / "qwen3_sft_train_segment.json"
    print(f"\n[1/5] 读取原始训练集: {orig_file}")
    with open(orig_file, "r") as f:
        orig_data = json.load(f)
    print(f"      样本数: {len(orig_data)}")
    orig_counts = Counter(d["label_en"] for d in orig_data)

    system_prompt = get_system_prompt(orig_data)
    print(f"      system prompt: 使用原始 12 标签 prompt（不新增标签）")

    for item in orig_data:
        item["system"] = system_prompt

    # ── Step 2: 读取标注员数据, 过滤, 转换 ──
    ann1_file = "/mnt/pfs/houhaotian/review_annotator_1/training_export/_all.json"
    ann2_file = "/mnt/pfs/houhaotian/review_annotator_2/training_export/_all.json"

    print(f"\n[2/5] 读取标注员数据并过滤...")
    with open(ann1_file, "r") as f:
        ann1_raw = json.load(f)
    with open(ann2_file, "r") as f:
        ann2_raw = json.load(f)

    ann1_filtered = [a for a in ann1_raw if a["label"] in ALLOWED_LABELS]
    ann2_filtered = [a for a in ann2_raw if a["label"] in ALLOWED_LABELS]
    print(f"      标注人一: {len(ann1_raw)} -> {len(ann1_filtered)} (过滤 {len(ann1_raw)-len(ann1_filtered)})")
    print(f"      标注人二: {len(ann2_raw)} -> {len(ann2_filtered)} (过滤 {len(ann2_raw)-len(ann2_filtered)})")

    all_ann_filtered = ann1_filtered + ann2_filtered
    converted_all = [convert_annotation(a, system_prompt, "annotator") for a in all_ann_filtered]

    # ── Step 3: 拆分测试集（新标签各取 N 条） ──
    print(f"\n[3/5] 拆分测试集样本...")
    test_file = data_dir / "qwen3_sft_test_segment_upsample.json"
    with open(test_file, "r") as f:
        test_data = json.load(f)
    print(f"      现有测试集样本数: {len(test_data)}")

    for item in test_data:
        item["system"] = system_prompt

    test_new = []
    train_from_ann = []
    label_pools = defaultdict(list)
    for item in converted_all:
        label_pools[item["label_en"]].append(item)

    for label, n_test in TEST_SPLIT_COUNTS.items():
        pool = label_pools.get(label, [])
        if len(pool) < n_test:
            print(f"      [WARN] {label} 仅有 {len(pool)} 条，不足 {n_test}，全部放入测试集")
            test_new.extend(pool)
        else:
            random.shuffle(pool)
            test_new.extend(pool[:n_test])
            train_from_ann.extend(pool[n_test:])
        print(f"      {label}: 测试集 {min(len(pool), n_test)} 条, 训练集 {max(0, len(pool)-n_test)} 条")

    for label, items in label_pools.items():
        if label not in TEST_SPLIT_COUNTS:
            train_from_ann.extend(items)

    test_data_updated = test_data + test_new
    test_out_file = data_dir / "qwen3_sft_test_segment_upsample.json"
    with open(test_out_file, "w", encoding="utf-8") as f:
        json.dump(test_data_updated, f, ensure_ascii=False, indent=2)
    print(f"      更新后测试集样本数: {len(test_data_updated)} (+{len(test_new)})")

    test_counts = Counter(d["label_en"] for d in test_data_updated)
    print(f"      测试集分布:")
    for label, count in test_counts.most_common():
        print(f"        {label}: {count}")

    # ── Step 4: 合并训练数据并保存 (采样前) ──
    merged = orig_data + train_from_ann
    merged_counts = Counter(d["label_en"] for d in merged)

    add_tags_file = data_dir / "qwen3_sft_train_segment_add_tags.json"
    print(f"\n[4/5] 保存合并数据 (采样前): {add_tags_file}")
    print(f"      合并总样本数: {len(merged)}")
    with open(add_tags_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print_comparison(orig_counts, merged_counts, "合并前后对比 (原始 vs 合并)")

    # ── Step 5: 动态采样 ──
    print(f"\n[5/5] 执行动态采样...")
    balanced = dynamic_sample(merged, SAMPLING_MULTIPLIERS, DEFAULT_MULTIPLIER)
    balanced_counts = Counter(d["label_en"] for d in balanced)

    upsample_file = data_dir / "qwen3_sft_train_segment_add_tags_dynamic_sample.json"
    print(f"      保存采样数据: {upsample_file}")
    print(f"      采样后总样本数: {len(balanced)}")
    with open(upsample_file, "w", encoding="utf-8") as f:
        json.dump(balanced, f, ensure_ascii=False, indent=2)

    # ── 最终统计 ──
    print_comparison(merged_counts, balanced_counts, "动态采样前后对比 (合并数据 vs 采样数据)")
    print_comparison(orig_counts, balanced_counts, "完整对比 (原始数据 vs 最终采样数据)")

    print(f"\n{'=' * 90}")
    print(f"  最终采样数据分布")
    print(f"{'=' * 90}")
    print("{:<50} {:>8} {:>8} {:>10}".format("标签", "数量", "占比", "采样倍率"))
    print("-" * 80)
    for label, count in balanced_counts.most_common():
        pct = count / len(balanced) * 100
        mult = SAMPLING_MULTIPLIERS.get(label, DEFAULT_MULTIPLIER)
        print("{:<50} {:>8} {:>7.1f}% {:>9.1f}x".format(label, count, pct, mult))
    print("-" * 80)
    print("{:<50} {:>8}".format("总计", len(balanced)))

    print(f"\n文件大小:")
    print(f"  {add_tags_file.name}: {os.path.getsize(add_tags_file)/1024/1024:.1f} MB")
    print(f"  {upsample_file.name}: {os.path.getsize(upsample_file)/1024/1024:.1f} MB")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
