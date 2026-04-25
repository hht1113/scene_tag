#!/usr/bin/env python3
"""
Bad Case 分析脚本

分析模型预测错误的样本，重点关注：
1. 预测标签是否可能是视频的合理标签（多标签问题）
2. 哪些类别对之间容易混淆
3. 基于语义关系判断哪些"错误"可能是标注不完整导致的
"""

import json
import re
import os
from collections import Counter, defaultdict

PRED_FILE = "/root/workspace/LLaMA-Factory/VQA/json/12tags_Qwen3-VL-30B_full_add_tags_dynamic.jsonl"
TEST_FILE = "/root/workspace/LLaMA-Factory/data/qwen3_sft_test_segment.json"

ALL_LABELS = [
    "TrafficLight_StraightStopOrGo",
    "TrafficLight_LeftTurnStopOrGo",
    "LaneChange_NavForIntersection",
    "LaneChange_AvoidSlowVRU",
    "LaneChange_AvoidStaticVehicle",
    "DynamicInteraction_VRUInLaneCrossing",
    "DynamicInteraction_VehicleInLaneCrossing",
    "DynamicInteraction_StandardVehicleCutIn",
    "StartStop_StartFromMainRoad",
    "StartStop_ParkRoadside",
    "Intersection_StandardUTurn",
    "LaneCruising_Straight",
    "else",
]

MULTI_LABEL_PAIRS = {
    ("LaneChange_AvoidStaticVehicle", "LaneChange_NavForIntersection"):
        "在路口附近换道避让静态车辆，可以同时属于两个标签",
    ("LaneChange_AvoidStaticVehicle", "LaneChange_AvoidSlowVRU"):
        "避让慢行VRU和静态车辆可能在同一场景中出现",
    ("LaneChange_NavForIntersection", "LaneChange_AvoidSlowVRU"):
        "在路口附近换道时可能同时在避让慢行VRU",
    ("DynamicInteraction_VehicleInLaneCrossing", "DynamicInteraction_StandardVehicleCutIn"):
        "车辆加塞本质上也是车辆横穿车道的一种形式",
    ("TrafficLight_StraightStopOrGo", "TrafficLight_LeftTurnStopOrGo"):
        "同一视频中可能同时涉及直行和左转的红绿灯场景",
    ("TrafficLight_StraightStopOrGo", "StartStop_StartFromMainRoad"):
        "在主路红绿灯处起步，可以同时属于两个标签",
    ("TrafficLight_LeftTurnStopOrGo", "Intersection_StandardUTurn"):
        "U型掉头经常在有交通灯的路口发生，需要等左转灯",
    ("DynamicInteraction_VehicleInLaneCrossing", "LaneChange_NavForIntersection"):
        "在路口换道时可能同时有车辆横穿",
    ("LaneChange_NavForIntersection", "LaneCruising_Straight"):
        "换道前后有直行巡航的阶段",
    ("DynamicInteraction_StandardVehicleCutIn", "LaneCruising_Straight"):
        "直行巡航中遇到车辆加塞",
    ("LaneChange_AvoidStaticVehicle", "DynamicInteraction_VRUInLaneCrossing"):
        "避让静态车辆时可能同时有VRU横穿",
    ("LaneChange_AvoidStaticVehicle", "DynamicInteraction_StandardVehicleCutIn"):
        "换道避让静态车辆时可能有其他车辆加塞",
    ("DynamicInteraction_VRUInLaneCrossing", "DynamicInteraction_StandardVehicleCutIn"):
        "VRU横穿和车辆加塞可能在不同时段同时出现",
    ("TrafficLight_StraightStopOrGo", "DynamicInteraction_VehicleInLaneCrossing"):
        "在红绿灯等待时可能有车辆横穿",
    ("StartStop_StartFromMainRoad", "StartStop_ParkRoadside"):
        "从路边起步也可能被标为主路起步",
    ("LaneChange_AvoidSlowVRU", "DynamicInteraction_VRUInLaneCrossing"):
        "VRU在车道中横穿时，ego可能同时在换道避让",
    ("DynamicInteraction_VehicleInLaneCrossing", "LaneChange_AvoidStaticVehicle"):
        "车辆横穿后变成静态障碍需要换道避让",
    ("LaneChange_AvoidStaticVehicle", "LaneCruising_Straight"):
        "换道前后的直行巡航阶段",
    ("DynamicInteraction_StandardVehicleCutIn", "DynamicInteraction_VRUInLaneCrossing"):
        "加塞场景中可能也有VRU出现",
    ("TrafficLight_LeftTurnStopOrGo", "LaneChange_NavForIntersection"):
        "左转前可能先换道，或者左转灯后需要换道",
}


def normalize_pair(a, b):
    return tuple(sorted([a, b]))


def parse_maneuvers(text):
    return re.findall(r'<driving_maneuver>(.*?)</driving_maneuver>', text)


def is_known_multi_label_pair(gt_label, pred_label):
    pair = normalize_pair(gt_label, pred_label)
    return pair in MULTI_LABEL_PAIRS


def get_multi_label_reason(gt_label, pred_label):
    pair = normalize_pair(gt_label, pred_label)
    return MULTI_LABEL_PAIRS.get(pair, "")


def main():
    with open(PRED_FILE, 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    with open(TEST_FILE, 'r') as f:
        test_data = json.load(f)

    print(f"预测样本数: {len(predictions)}")
    print(f"测试样本数: {len(test_data)}")

    n = min(len(predictions), len(test_data))

    bad_cases = []
    correct_cases = []
    confusion_counter = Counter()
    total_by_gt = Counter()

    for i in range(n):
        pred_obj = predictions[i]
        test_obj = test_data[i]

        gt_maneuvers = parse_maneuvers(pred_obj['label'])
        pred_maneuvers = parse_maneuvers(pred_obj['predict'])

        gt_label = gt_maneuvers[0] if gt_maneuvers else "Unrecognized"
        pred_label = pred_maneuvers[0] if pred_maneuvers else "Unrecognized"

        total_by_gt[gt_label] += 1

        video_path = test_obj.get('videos', [''])[0] if i < len(test_data) else ''
        slice_key = test_obj.get('slice_key', '')

        if gt_label != pred_label:
            is_multi = is_known_multi_label_pair(gt_label, pred_label)
            reason = get_multi_label_reason(gt_label, pred_label)
            bad_cases.append({
                'index': i,
                'gt_label': gt_label,
                'pred_label': pred_label,
                'video_path': video_path,
                'slice_key': slice_key,
                'is_potential_multi_label': is_multi,
                'multi_label_reason': reason,
                'gt_text': pred_obj['label'],
                'pred_text': pred_obj['predict'],
            })
            confusion_counter[normalize_pair(gt_label, pred_label)] += 1
        else:
            correct_cases.append({
                'index': i,
                'label': gt_label,
            })

    print(f"\n{'='*80}")
    print(f"总体统计")
    print(f"{'='*80}")
    print(f"总样本数:   {n}")
    print(f"正确预测:   {len(correct_cases)} ({len(correct_cases)/n*100:.1f}%)")
    print(f"错误预测:   {len(bad_cases)} ({len(bad_cases)/n*100:.1f}%)")

    multi_label_cases = [c for c in bad_cases if c['is_potential_multi_label']]
    non_multi_cases = [c for c in bad_cases if not c['is_potential_multi_label']]

    print(f"\n其中可能是多标签问题 (预测标签可能实际上也是正确的): {len(multi_label_cases)} ({len(multi_label_cases)/n*100:.1f}%)")
    print(f"真正的预测错误: {non_multi_cases.__len__()} ({len(non_multi_cases)/n*100:.1f}%)")

    if multi_label_cases:
        adjusted_accuracy = (len(correct_cases) + len(multi_label_cases)) / n * 100
        print(f"\n如果多标签case都算对，调整后准确率: {adjusted_accuracy:.1f}% (原始: {len(correct_cases)/n*100:.1f}%)")

    # 按混淆类别对分组的 bad case
    print(f"\n{'='*80}")
    print(f"混淆类别对分析 (按出现频率排序)")
    print(f"{'='*80}")

    for pair, count in confusion_counter.most_common():
        is_multi = pair in MULTI_LABEL_PAIRS
        reason = MULTI_LABEL_PAIRS.get(pair, "")
        marker = " ⚠️ 可能是多标签问题" if is_multi else ""
        print(f"\n  {pair[0]} ↔ {pair[1]}: {count}次{marker}")
        if reason:
            print(f"    原因: {reason}")

    # 按GT标签分组的详细分析
    print(f"\n{'='*80}")
    print(f"按GT标签分组的 Bad Case 详情")
    print(f"{'='*80}")

    gt_groups = defaultdict(list)
    for case in bad_cases:
        gt_groups[case['gt_label']].append(case)

    for gt_label in sorted(gt_groups.keys()):
        cases = gt_groups[gt_label]
        total = total_by_gt[gt_label]
        correct_count = total - len(cases)
        multi_count = sum(1 for c in cases if c['is_potential_multi_label'])

        print(f"\n{'─'*70}")
        print(f"GT标签: {gt_label} (总数={total}, 正确={correct_count}, 错误={len(cases)}, 疑似多标签={multi_count})")
        print(f"{'─'*70}")

        pred_counter = Counter()
        for c in cases:
            pred_counter[c['pred_label']] += 1

        for pred_label, cnt in pred_counter.most_common():
            is_multi = is_known_multi_label_pair(gt_label, pred_label)
            marker = " [疑似多标签]" if is_multi else ""
            print(f"  → 被预测为 {pred_label}: {cnt}次{marker}")

    # 可能是多标签问题的详细列表
    print(f"\n{'='*80}")
    print(f"疑似多标签问题的 Bad Case 详细列表 (共{len(multi_label_cases)}个)")
    print(f"{'='*80}")

    pair_groups = defaultdict(list)
    for case in multi_label_cases:
        pair = normalize_pair(case['gt_label'], case['pred_label'])
        pair_groups[pair].append(case)

    for pair, cases in sorted(pair_groups.items(), key=lambda x: -len(x[1])):
        reason = MULTI_LABEL_PAIRS.get(pair, "")
        print(f"\n{'─'*70}")
        print(f"混淆对: {pair[0]} ↔ {pair[1]} ({len(cases)}个)")
        print(f"原因: {reason}")
        print(f"{'─'*70}")

        for case in cases:
            video_name = os.path.basename(case['video_path']) if case['video_path'] else 'N/A'
            print(f"  [{case['index']:3d}] GT={case['gt_label']}")
            print(f"        Pred={case['pred_label']}")
            print(f"        视频: {video_name}")
            print(f"        slice_key: {case['slice_key']}")
            print()

    # 不太可能是多标签的 bad case
    print(f"\n{'='*80}")
    print(f"不太可能是多标签的 Bad Case (共{len(non_multi_cases)}个)")
    print(f"{'='*80}")

    for case in non_multi_cases:
        video_name = os.path.basename(case['video_path']) if case['video_path'] else 'N/A'
        print(f"  [{case['index']:3d}] GT={case['gt_label']:<45s} Pred={case['pred_label']}")
        print(f"        视频: {video_name}")

    # 汇总表
    print(f"\n{'='*80}")
    print(f"汇总: 各GT标签错误分析")
    print(f"{'='*80}")
    print(f"  {'GT标签':<45s} {'总数':>4s} {'正确':>4s} {'错误':>4s} {'疑似多标签':>8s} {'真错误':>6s} {'原始Acc':>8s} {'调整Acc':>8s}")
    print(f"  {'-'*100}")

    total_correct_all = 0
    total_multi_all = 0
    total_error_all = 0
    total_all = 0

    for gt_label in sorted(total_by_gt.keys()):
        total = total_by_gt[gt_label]
        errors = gt_groups.get(gt_label, [])
        n_errors = len(errors)
        n_correct = total - n_errors
        n_multi = sum(1 for c in errors if c['is_potential_multi_label'])
        n_real_error = n_errors - n_multi
        orig_acc = n_correct / total if total > 0 else 0
        adj_acc = (n_correct + n_multi) / total if total > 0 else 0

        total_correct_all += n_correct
        total_multi_all += n_multi
        total_error_all += n_real_error
        total_all += total

        print(f"  {gt_label:<45s} {total:>4d} {n_correct:>4d} {n_errors:>4d} {n_multi:>8d} {n_real_error:>6d} {orig_acc:>7.1%} {adj_acc:>7.1%}")

    print(f"  {'-'*100}")
    orig_total_acc = total_correct_all / total_all if total_all > 0 else 0
    adj_total_acc = (total_correct_all + total_multi_all) / total_all if total_all > 0 else 0
    print(f"  {'合计':<45s} {total_all:>4d} {total_correct_all:>4d} {len(bad_cases):>4d} {total_multi_all:>8d} {total_error_all:>6d} {orig_total_acc:>7.1%} {adj_total_acc:>7.1%}")

    # 输出 JSON 报告
    report = {
        'summary': {
            'total': n,
            'correct': len(correct_cases),
            'errors': len(bad_cases),
            'potential_multi_label': len(multi_label_cases),
            'real_errors': len(non_multi_cases),
            'original_accuracy': len(correct_cases) / n,
            'adjusted_accuracy': (len(correct_cases) + len(multi_label_cases)) / n,
        },
        'confusion_pairs': {
            f"{p[0]} ↔ {p[1]}": {
                'count': c,
                'is_potential_multi_label': p in MULTI_LABEL_PAIRS,
                'reason': MULTI_LABEL_PAIRS.get(p, ''),
            }
            for p, c in confusion_counter.most_common()
        },
        'bad_cases': [
            {
                'index': case['index'],
                'gt_label': case['gt_label'],
                'pred_label': case['pred_label'],
                'video_path': case['video_path'],
                'slice_key': case['slice_key'],
                'is_potential_multi_label': case['is_potential_multi_label'],
                'multi_label_reason': case['multi_label_reason'],
            }
            for case in bad_cases
        ],
    }

    report_path = '/root/workspace/LLaMA-Factory/scene_tag/bad_case_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
