"""
构造多标签微调数据集：每个视频输出 1-2 个标签（按时长排序）。

变化:
  - 旧格式: 一个视频一个标签 → 模型只输出一个 driving_maneuver
  - 新格式: 一个视频 1-2 个标签 → 模型按时长/可信度排序输出多个 segment
  - 评估时: 预测的标签匹配到任意一个真实标签即算正确

数据来源:
  - 原始训练集: train_segment_upsample.json (与 test_segment_upsample 配对, 0重叠)
  - 标注员数据: annotator 1 + annotator 2
  - 测试基底: test_segment_upsample.json
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
MAX_LABELS_PER_VIDEO = 2

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
    "<video>\nWhat are the ego vehicle's driving maneuvers in this 20-second video? List the most significant ones ranked by confidence.",
    "<video>\nIdentify the top driving maneuvers of the ego vehicle in this 20-second video, ranked by significance.",
    "<video>\nWhat are the primary driving behaviors of the ego vehicle in this 20-second video? Rank by confidence.",
    "<video>\nAnalyze this 20-second video and list the ego vehicle's main driving maneuvers in order of significance.",
    "<video>\nWhat driving maneuvers does the ego vehicle perform in this 20-second video? List the most important ones.",
    "<video>\nDescribe the ego vehicle's key driving behaviors in this 20-second video, ranked by confidence.",
    "<video>\nIn this 20-second video, what are the ego vehicle's driving maneuvers? Rank from most to least significant.",
    "<video>\nPlease identify and rank the ego vehicle's driving maneuvers in this 20-second video clip.",
    "<video>\nWhat are the main actions of the ego vehicle in this 20-second video? List by order of importance.",
    "<video>\nAnalyze the ego vehicle's behavior in this 20-second video and list the top maneuvers by confidence.",
]

OUTPUT_TEMPLATE_SINGLE = (
    "The primary driving maneuver is "
    "<driving_maneuver>{label1}</driving_maneuver> "
    "from <start_time>{start1}</start_time> to <end_time>{end1}</end_time> seconds."
)

OUTPUT_TEMPLATE_DOUBLE = (
    "The primary driving maneuver is "
    "<driving_maneuver>{label1}</driving_maneuver> "
    "from <start_time>{start1}</start_time> to <end_time>{end1}</end_time> seconds"
    " and the secondary maneuver is "
    "<driving_maneuver>{label2}</driving_maneuver> "
    "from <start_time>{start2}</start_time> to <end_time>{end2}</end_time> seconds."
)


def get_dedup_key(item):
    vid = item["videos"][0] if isinstance(item.get("videos"), list) and item["videos"] else ""
    tr = tuple(item.get("time_range_in_slice", []))
    label = item.get("label_en", "")
    return (vid, tr, label)


def group_by_video(items):
    """将同一视频的多个标注合并为一条，按时长排序取前 MAX_LABELS_PER_VIDEO 个"""
    video_map = defaultdict(list)
    for item in items:
        vid = item["videos"][0]
        video_map[vid].append(item)

    grouped = []
    for vid, entries in video_map.items():
        segments = []
        seen_labels = set()
        for e in entries:
            label = e["label_en"]
            start, end = e["time_range_in_slice"]
            duration = end - start
            if label not in seen_labels:
                segments.append({
                    "label": label,
                    "start": start,
                    "end": end,
                    "duration": duration,
                })
                seen_labels.add(label)

        segments.sort(key=lambda x: -x["duration"])
        segments = segments[:MAX_LABELS_PER_VIDEO]

        primary_label = segments[0]["label"]

        if len(segments) == 1:
            output = OUTPUT_TEMPLATE_SINGLE.format(
                label1=segments[0]["label"],
                start1=f"{segments[0]['start']:.1f}",
                end1=f"{segments[0]['end']:.1f}",
            )
        else:
            output = OUTPUT_TEMPLATE_DOUBLE.format(
                label1=segments[0]["label"],
                start1=f"{segments[0]['start']:.1f}",
                end1=f"{segments[0]['end']:.1f}",
                label2=segments[1]["label"],
                start2=f"{segments[1]['start']:.1f}",
                end2=f"{segments[1]['end']:.1f}",
            )

        all_labels = [s["label"] for s in segments]

        grouped.append({
            "instruction": random.choice(INSTRUCTION_TEMPLATES),
            "input": "",
            "output": output,
            "videos": [vid],
            "system": entries[0]["system"],
            "slice_key": vid,
            "time_range_in_slice": [segments[0]["start"], segments[0]["end"]],
            "label_en": primary_label,
            "all_labels": all_labels,
            "num_labels": len(segments),
        })

    return grouped


def convert_annotation(ann, system_prompt):
    start = float(ann["start"])
    end = float(ann["end"])
    label = ann["label"]
    return {
        "instruction": "",
        "input": "",
        "output": "",
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
    print("[1/7] 加载原始数据...")

    with open(data_dir / "qwen3_sft_train_segment_upsample.json") as f:
        orig_train = json.load(f)
    with open(data_dir / "qwen3_sft_test_segment_upsample.json") as f:
        orig_test = json.load(f)

    system_prompt = orig_train[0]["system"]
    print(f"      原始训练集: {len(orig_train)} 条")
    print(f"      原始测试集: {len(orig_test)} 条")

    # ── Step 2: 加载标注员数据 ──
    print("\n[2/7] 加载标注员数据...")

    with open("/mnt/pfs/houhaotian/review_annotator_1/training_export/_all.json") as f:
        ann1 = [a for a in json.load(f) if a["label"] in ALLOWED_LABELS]
    with open("/mnt/pfs/houhaotian/review_annotator_2/training_export/_all.json") as f:
        ann2 = [a for a in json.load(f) if a["label"] in ALLOWED_LABELS]

    all_ann = ann1 + ann2
    converted_ann = [convert_annotation(a, system_prompt) for a in all_ann]
    print(f"      标注员合计: {len(converted_ann)} 条")

    # 去重
    orig_keys = set(get_dedup_key(d) for d in orig_train) | set(get_dedup_key(d) for d in orig_test)
    unique_ann = []
    for item in converted_ann:
        k = get_dedup_key(item)
        if k not in orig_keys:
            orig_keys.add(k)
            unique_ann.append(item)
    print(f"      去重后: {len(unique_ann)} 条")

    # ── Step 3: 按视频分组合并多标签 ──
    print("\n[3/7] 按视频分组（合并同一视频的多标签）...")

    # 测试集分组
    test_grouped = group_by_video(orig_test)
    print(f"      测试集: {len(orig_test)} 条 → {len(test_grouped)} 个视频")

    multi_test = sum(1 for g in test_grouped if g["num_labels"] > 1)
    print(f"      其中多标签视频: {multi_test} 个")

    # 标注员数据分组
    ann_grouped = group_by_video(unique_ann)
    print(f"      标注员: {len(unique_ann)} 条 → {len(ann_grouped)} 个视频")

    multi_ann = sum(1 for g in ann_grouped if g["num_labels"] > 1)
    print(f"      其中多标签视频: {multi_ann} 个")

    # 原始训练集去重后取 unique 样本（上采样前），然后分组
    seen_train = set()
    orig_train_unique = []
    for item in orig_train:
        k = get_dedup_key(item)
        if k not in seen_train:
            seen_train.add(k)
            orig_train_unique.append(item)

    train_grouped = group_by_video(orig_train_unique)
    print(f"      原始训练: {len(orig_train)} 条 → {len(orig_train_unique)} unique → {len(train_grouped)} 个视频")

    multi_train = sum(1 for g in train_grouped if g["num_labels"] > 1)
    print(f"      其中多标签视频: {multi_train} 个")

    # ── Step 4: 构建平衡测试集 ──
    print(f"\n[4/7] 构建平衡测试集 (每类 {TEST_PER_CLASS} 条)...")

    test_by_label = defaultdict(list)
    for item in test_grouped:
        test_by_label[item["label_en"]].append(item)

    ann_by_label = defaultdict(list)
    for item in ann_grouped:
        ann_by_label[item["label_en"]].append(item)

    new_test = []
    ann_to_train = []
    test_overflow_to_train = []

    print(f"\n      {'类别':<48} {'原测试':>6} {'需补':>6} {'标注补':>6} {'回流':>6} {'最终':>6}")
    print(f"      {'─' * 80}")

    for label in sorted(ALLOWED_LABELS):
        t_items = test_by_label.get(label, [])
        a_items = ann_by_label.get(label, [])
        random.shuffle(a_items)

        current = len(t_items)
        needed = max(0, TEST_PER_CLASS - current)

        if current >= TEST_PER_CLASS:
            random.shuffle(t_items)
            new_test.extend(t_items[:TEST_PER_CLASS])
            test_overflow_to_train.extend(t_items[TEST_PER_CLASS:])
            ann_to_train.extend(a_items)
            overflow = current - TEST_PER_CLASS
            print(f"      {label:<48} {current:>6} {0:>6} {0:>6} {overflow:>6} {TEST_PER_CLASS:>6}")
        else:
            new_test.extend(t_items)
            filled = min(needed, len(a_items))
            new_test.extend(a_items[:filled])
            ann_to_train.extend(a_items[filled:])
            final = current + filled
            print(f"      {label:<48} {current:>6} {needed:>6} {filled:>6} {0:>6} {final:>6}")

    print(f"      {'─' * 80}")
    print(f"      {'总计':<48} {len(test_grouped):>6} {'':>6} {'':>6} {len(test_overflow_to_train):>6} {len(new_test):>6}")

    # ── Step 5: 构建训练集 ──
    print(f"\n[5/7] 构建训练集...")
    new_train_raw = train_grouped + test_overflow_to_train + ann_to_train

    seen = set()
    new_train_dedup = []
    for item in new_train_raw:
        k = (item["videos"][0], item["label_en"])
        if k not in seen:
            seen.add(k)
            new_train_dedup.append(item)

    print(f"      原始训练 (分组): {len(train_grouped)} 个视频")
    print(f"      测试回流:        {len(test_overflow_to_train)}")
    print(f"      标注员剩余:      {len(ann_to_train)}")
    print(f"      合并去重后:      {len(new_train_dedup)} 个视频")

    train_counts_before = Counter(d["label_en"] for d in new_train_dedup)

    # ── Step 6: 动态采样 ──
    print(f"\n[6/7] 执行动态采样...")
    sampled_train = dynamic_sample(new_train_dedup, SAMPLING_MULTIPLIERS, DEFAULT_MULTIPLIER)
    print(f"      采样前: {len(new_train_dedup)} 条")
    print(f"      采样后: {len(sampled_train)} 条")

    sampled_counts = Counter(d["label_en"] for d in sampled_train)

    # ── Step 7: 保存 ──
    print(f"\n[7/7] 保存数据集...")

    test_out = data_dir / "qwen3_sft_test_segment_multilabel_v1.json"
    train_out = data_dir / "qwen3_sft_train_segment_multilabel_v1.json"

    with open(test_out, "w", encoding="utf-8") as f:
        json.dump(new_test, f, ensure_ascii=False, indent=2)

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(sampled_train, f, ensure_ascii=False, indent=2)

    print(f"      测试集: {test_out}")
    print(f"        样本数: {len(new_test)}, 大小: {os.path.getsize(test_out)/1024:.1f} KB")
    print(f"      训练集: {train_out}")
    print(f"        样本数: {len(sampled_train)}, 大小: {os.path.getsize(train_out)/1024/1024:.1f} MB")

    # ── 统计 ──
    test_counts = Counter(d["label_en"] for d in new_test)
    test_multi = sum(1 for d in new_test if d.get("num_labels", 1) > 1)
    train_multi = sum(1 for d in sampled_train if d.get("num_labels", 1) > 1)

    print(f"\n{'=' * 100}")
    print(f"  数据集统计")
    print(f"{'=' * 100}")

    print(f"\n  测试集: {len(new_test)} 条 (多标签: {test_multi} 条, {test_multi/len(new_test)*100:.1f}%)")
    print(f"  训练集: {len(sampled_train)} 条 (多标签: {train_multi} 条, {train_multi/len(sampled_train)*100:.1f}%)")

    print(f"\n  {'类别':<48} {'测试':>6} {'训练(采样前)':>14} {'倍率':>8} {'训练(采样后)':>14} {'占比':>8}")
    print(f"  {'─' * 95}")
    for label in sorted(ALLOWED_LABELS):
        tc = test_counts.get(label, 0)
        tb = train_counts_before.get(label, 0)
        ta = sampled_counts.get(label, 0)
        mult = SAMPLING_MULTIPLIERS.get(label, DEFAULT_MULTIPLIER)
        pct = ta / len(sampled_train) * 100
        print(f"  {label:<48} {tc:>6} {tb:>14} {mult:>7.1f}x {ta:>14} {pct:>7.1f}%")
    print(f"  {'─' * 95}")
    print(f"  {'总计':<48} {len(new_test):>6} {len(new_train_dedup):>14} {'':>8} {len(sampled_train):>14}")

    # 输出格式示例
    print(f"\n{'=' * 100}")
    print(f"  输出格式示例")
    print(f"{'=' * 100}")
    for item in sampled_train:
        if item.get("num_labels", 1) == 1:
            print(f"\n  [单标签] {item['output']}")
            break
    for item in sampled_train:
        if item.get("num_labels", 1) == 2:
            print(f"\n  [双标签] {item['output']}")
            break

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
