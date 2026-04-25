"""
构造多标签微调数据集 v2：使用新版 system_prompt_v2.txt，按置信度/时长输出 1-2 个标签。

主要变化 (相比 v1):
  - 使用修复后的 system_prompt_v2.txt（解决了标签定义冲突、格式错误等问题）
  - 标注员数据自带 confidence 字段，排序时同时考虑 duration 和 confidence
  - 输出格式与新 prompt 保持一致：primary + optional secondary
  - 评估时：预测的标签匹配到任意一个真实标签即算正确

数据来源:
  - 原始训练集: qwen3_sft_train_segment_upsample.json
  - 原始测试集: qwen3_sft_test_segment_upsample.json
  - 标注员数据: annotator 1 + annotator 2 (含 confidence 字段)
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

REMOVED_LABELS = {
    "LaneCruising_Straight",
    "LaneChange_NavForIntersection",
}

SAMPLING_MULTIPLIERS = {
    # 目标 ~300/类, 倍率 = 300 / 采样前数量
    "TrafficLight_StraightStopOrGo":                0.52,  # 574 → ~298
    "TrafficLight_LeftTurnStopOrGo":                0.50,  # 600 → ~300
    "LaneChange_AvoidSlowVRU":                      3.5,   #  85 → ~297
    "LaneChange_AvoidStaticVehicle":                0.72,  # 419 → ~301
    "DynamicInteraction_VRUInLaneCrossing":          1.35,  # 223 → ~301
    "DynamicInteraction_VehicleInLaneCrossing":      0.72,  # 418 → ~300
    "DynamicInteraction_StandardVehicleCutIn":       0.97,  # 310 → ~300
    "StartStop_StartFromMainRoad":                  5.7,   #  53 → ~302
    "StartStop_ParkRoadside":                       3.1,   #  98 → ~303
    "Intersection_StandardUTurn":                   0.51,  # 592 → ~301
}

DEFAULT_MULTIPLIER = 1.0
ALLOWED_LABELS = set(SAMPLING_MULTIPLIERS.keys())

SYSTEM_PROMPT_FILE = Path(__file__).parent / "system_prompt_v2.txt"

INSTRUCTION_TEMPLATES = [
    "<video>\nWhat are the ego vehicle's driving maneuvers in this 20-second video?",
    "<video>\nIdentify the driving maneuvers of the ego vehicle in this 20-second video.",
    "<video>\nWhat driving maneuvers does the ego vehicle perform in this 20-second video?",
    "<video>\nAnalyze this 20-second video and identify the ego vehicle's driving maneuvers.",
    "<video>\nDescribe the ego vehicle's driving maneuvers in this 20-second video.",
    "<video>\nIn this 20-second video, what are the ego vehicle's driving maneuvers?",
    "<video>\nPlease identify the ego vehicle's driving maneuvers in this 20-second video clip.",
    "<video>\nWhat are the driving behaviors of the ego vehicle in this 20-second video?",
    "<video>\nWhat driving actions does the ego vehicle execute in this 20-second video?",
    "<video>\nAnalyze the ego vehicle's behavior in this 20-second video and identify the maneuvers.",
]

OUTPUT_TEMPLATE_SINGLE = (
    "<driving_maneuver>{label1}</driving_maneuver> "
    "from <start_time>{start1}</start_time> to <end_time>{end1}</end_time> seconds."
)

OUTPUT_TEMPLATE_DOUBLE = (
    "<driving_maneuver>{label1}</driving_maneuver> "
    "from <start_time>{start1}</start_time> to <end_time>{end1}</end_time> seconds"
    " and "
    "<driving_maneuver>{label2}</driving_maneuver> "
    "from <start_time>{start2}</start_time> to <end_time>{end2}</end_time> seconds."
)


def load_system_prompt():
    """从 system_prompt_v2.txt 加载 system prompt"""
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_dedup_key(item):
    vid = item["videos"][0] if isinstance(item.get("videos"), list) and item["videos"] else ""
    tr = tuple(item.get("time_range_in_slice", []))
    label = item.get("label_en", "")
    return (vid, tr, label)


def compute_score(duration, confidence=None):
    """综合 duration 和 confidence 计算排序分数。
    confidence 范围 0-100，归一化到 0-1 后与 duration 加权组合。
    """
    dur_weight = 0.6
    conf_weight = 0.4
    if confidence is not None and confidence > 0:
        return dur_weight * duration + conf_weight * (confidence / 100.0) * 20.0
    return duration


def group_by_video(items):
    """将同一视频的多个标注合并为一条，按综合分数排序取前 MAX_LABELS_PER_VIDEO 个"""
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
            if label in REMOVED_LABELS:
                continue
            raw_start, raw_end = e["time_range_in_slice"]
            start, end = clamp_time(raw_start, raw_end)
            duration = end - start
            confidence = e.get("confidence")
            if label not in seen_labels:
                segments.append({
                    "label": label,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "confidence": confidence,
                    "score": compute_score(duration, confidence),
                })
                seen_labels.add(label)

        if not segments:
            continue

        segments.sort(key=lambda x: -x["score"])
        segments = segments[:MAX_LABELS_PER_VIDEO]

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
            "label_en": segments[0]["label"],
            "all_labels": all_labels,
            "num_labels": len(segments),
        })

    return grouped


def clamp_time(start, end, min_t=0.0, max_t=20.0):
    """将时间范围截断到 [min_t, max_t]"""
    start = max(min_t, min(start, max_t))
    end = max(min_t, min(end, max_t))
    if end <= start:
        end = start + 1.0
        end = min(end, max_t)
    return round(start, 1), round(end, 1)


def convert_annotation(ann, system_prompt):
    """将标注员原始格式转换为统一格式，保留 confidence"""
    start, end = clamp_time(float(ann["start"]), float(ann["end"]))
    label = ann["label"]
    confidence = ann.get("confidence")
    return {
        "instruction": "",
        "input": "",
        "output": "",
        "videos": [ann["video_path"]],
        "system": system_prompt,
        "slice_key": ann["video_path"],
        "time_range_in_slice": [start, end],
        "label_en": label,
        "confidence": confidence,
    }


def dynamic_sample(data, multipliers, default_multiplier):
    """按类别动态采样"""
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


def update_system_prompts(data, new_system_prompt):
    """将所有样本的 system prompt 替换为新版本"""
    for item in data:
        item["system"] = new_system_prompt
    return data


def main():
    data_dir = Path("/root/workspace/LLaMA-Factory/data")

    # ── Step 0: 加载新版 system prompt ──
    print("[0/8] 加载新版 system_prompt_v2.txt...")
    new_system_prompt = load_system_prompt()
    print(f"      长度: {len(new_system_prompt)} 字符")

    # ── Step 1: 加载原始数据 ──
    print("\n[1/8] 加载原始数据...")

    with open(data_dir / "qwen3_sft_train_segment_upsample.json") as f:
        orig_train_raw = json.load(f)
    with open(data_dir / "qwen3_sft_test_segment_upsample.json") as f:
        orig_test_raw = json.load(f)

    orig_train = [d for d in orig_train_raw if d.get("label_en", "") not in REMOVED_LABELS]
    orig_test = [d for d in orig_test_raw if d.get("label_en", "") not in REMOVED_LABELS]

    print(f"      原始训练集: {len(orig_train_raw)} 条 -> 过滤后 {len(orig_train)} 条 (移除 {len(orig_train_raw)-len(orig_train)} 条)")
    print(f"      原始测试集: {len(orig_test_raw)} 条 -> 过滤后 {len(orig_test)} 条 (移除 {len(orig_test_raw)-len(orig_test)} 条)")

    # ── Step 2: 加载标注员数据 ──
    print("\n[2/8] 加载标注员数据...")

    ann1_path = "/mnt/pfs/houhaotian/review_annotator_1/training_export/_all.json"
    ann2_path = "/mnt/pfs/houhaotian/review_annotator_2/training_export/_all.json"

    ann1, ann2 = [], []
    if os.path.exists(ann1_path):
        with open(ann1_path) as f:
            ann1 = [a for a in json.load(f) if a["label"] in ALLOWED_LABELS]
        print(f"      标注员 1: {len(ann1)} 条")
    else:
        print(f"      标注员 1: 文件不存在, 跳过")

    if os.path.exists(ann2_path):
        with open(ann2_path) as f:
            ann2 = [a for a in json.load(f) if a["label"] in ALLOWED_LABELS]
        print(f"      标注员 2: {len(ann2)} 条")
    else:
        print(f"      标注员 2: 文件不存在, 跳过")

    all_ann = ann1 + ann2
    converted_ann = [convert_annotation(a, new_system_prompt) for a in all_ann]
    print(f"      标注员合计: {len(converted_ann)} 条")

    conf_values = [a.get("confidence") for a in all_ann if a.get("confidence") is not None]
    if conf_values:
        print(f"      置信度分布: min={min(conf_values)}, max={max(conf_values)}, avg={sum(conf_values)/len(conf_values):.1f}")

    # 去重
    orig_keys = set(get_dedup_key(d) for d in orig_train) | set(get_dedup_key(d) for d in orig_test)
    unique_ann = []
    for item in converted_ann:
        k = get_dedup_key(item)
        if k not in orig_keys:
            orig_keys.add(k)
            unique_ann.append(item)
    print(f"      去重后: {len(unique_ann)} 条")

    # ── Step 3: 统一替换 system prompt ──
    print("\n[3/8] 替换所有样本的 system prompt 为新版本...")
    update_system_prompts(orig_train, new_system_prompt)
    update_system_prompts(orig_test, new_system_prompt)
    print("      完成")

    # ── Step 4: 按视频分组合并多标签 ──
    print("\n[4/8] 按视频分组（合并同一视频的多标签）...")

    test_grouped = group_by_video(orig_test)
    print(f"      测试集: {len(orig_test)} 条 -> {len(test_grouped)} 个视频")
    multi_test = sum(1 for g in test_grouped if g["num_labels"] > 1)
    print(f"      其中多标签视频: {multi_test} 个")

    ann_grouped = group_by_video(unique_ann)
    print(f"      标注员: {len(unique_ann)} 条 -> {len(ann_grouped)} 个视频")
    multi_ann = sum(1 for g in ann_grouped if g["num_labels"] > 1)
    print(f"      其中多标签视频: {multi_ann} 个")

    seen_train = set()
    orig_train_unique = []
    for item in orig_train:
        k = get_dedup_key(item)
        if k not in seen_train:
            seen_train.add(k)
            orig_train_unique.append(item)

    train_grouped = group_by_video(orig_train_unique)
    print(f"      原始训练: {len(orig_train)} 条 -> {len(orig_train_unique)} unique -> {len(train_grouped)} 个视频")
    multi_train = sum(1 for g in train_grouped if g["num_labels"] > 1)
    print(f"      其中多标签视频: {multi_train} 个")

    # ── Step 5: 构建平衡测试集 ──
    print(f"\n[5/8] 构建平衡测试集 (每类 {TEST_PER_CLASS} 条)...")

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

    # ── Step 6: 构建训练集（排除测试集视频） ──
    print(f"\n[6/8] 构建训练集...")
    new_train_raw = train_grouped + test_overflow_to_train + ann_to_train

    test_video_set = set(d["videos"][0] for d in new_test)

    seen = set()
    new_train_dedup = []
    leaked = 0
    for item in new_train_raw:
        vid = item["videos"][0]
        if vid in test_video_set:
            leaked += 1
            continue
        k = (vid, item["label_en"])
        if k not in seen:
            seen.add(k)
            new_train_dedup.append(item)

    print(f"      原始训练 (分组): {len(train_grouped)} 个视频")
    print(f"      测试回流:        {len(test_overflow_to_train)}")
    print(f"      标注员剩余:      {len(ann_to_train)}")
    print(f"      排除测试集重叠:  {leaked} 条")
    print(f"      合并去重后:      {len(new_train_dedup)} 个视频")

    train_counts_before = Counter(d["label_en"] for d in new_train_dedup)

    # ── Step 7: 动态采样 ──
    print(f"\n[7/8] 执行动态采样...")
    sampled_train = dynamic_sample(new_train_dedup, SAMPLING_MULTIPLIERS, DEFAULT_MULTIPLIER)
    print(f"      采样前: {len(new_train_dedup)} 条")
    print(f"      采样后: {len(sampled_train)} 条")

    sampled_counts = Counter(d["label_en"] for d in sampled_train)

    # ── Step 8: 保存 ──
    print(f"\n[8/8] 保存数据集...")

    test_out = data_dir / "qwen3_sft_test_segment_multilabel_v2.json"
    train_out = data_dir / "qwen3_sft_train_segment_multilabel_v2.json"

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
    print(f"  {'─' * 100}")
    for label in sorted(ALLOWED_LABELS):
        tc = test_counts.get(label, 0)
        tb = train_counts_before.get(label, 0)
        ta = sampled_counts.get(label, 0)
        mult = SAMPLING_MULTIPLIERS.get(label, DEFAULT_MULTIPLIER)
        pct = ta / len(sampled_train) * 100 if sampled_train else 0
        print(f"  {label:<48} {tc:>6} {tb:>14} {mult:>7.1f}x {ta:>14} {pct:>7.1f}%")
    print(f"  {'─' * 100}")
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

    print(f"\n{'=' * 100}")
    print(f"  新版 system prompt (前 300 字符)")
    print(f"{'=' * 100}")
    print(f"  {new_system_prompt[:300]}...")

    print(f"\n[DONE] v2 多标签数据集构建完成")
    print(f"  训练集: {train_out}")
    print(f"  测试集: {test_out}")
    print(f"\n  下一步:")
    print(f"  1. 在 data/dataset_info.json 中注册新数据集")
    print(f"  2. 修改训练 yaml 中的 dataset 字段指向新数据集")


if __name__ == "__main__":
    main()
