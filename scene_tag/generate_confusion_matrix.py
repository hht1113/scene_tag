#!/usr/bin/env python3
"""
生成 dynamic 模型的混淆矩阵图
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 9

PRED_FILE = "/root/workspace/LLaMA-Factory/VQA/json/12tags_Qwen3-VL-30B_full_add_tags_dynamic.jsonl"

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
    "Unrecognized",
]

SHORT_LABELS = [
    "TL_Straight\nStopOrGo",
    "TL_LeftTurn\nStopOrGo",
    "LC_NavFor\nIntersection",
    "LC_Avoid\nSlowVRU",
    "LC_Avoid\nStaticVehicle",
    "DI_VRUIn\nLaneCrossing",
    "DI_VehicleIn\nLaneCrossing",
    "DI_Standard\nVehicleCutIn",
    "SS_StartFrom\nMainRoad",
    "SS_Park\nRoadside",
    "Intersection\nStandardUTurn",
    "LaneCruising\nStraight",
    "else",
    "Unrecognized",
]


def parse_maneuvers(text):
    return re.findall(r'<driving_maneuver>(.*?)</driving_maneuver>', text)


def main():
    with open(PRED_FILE, 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    n_labels = len(ALL_LABELS)
    label_to_idx = {label: i for i, label in enumerate(ALL_LABELS)}
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for obj in predictions:
        gt_list = parse_maneuvers(obj['label'])
        pred_list = parse_maneuvers(obj['predict'])
        gt = gt_list[0] if gt_list else "Unrecognized"
        pred = pred_list[0] if pred_list else "Unrecognized"
        gi = label_to_idx.get(gt, label_to_idx["Unrecognized"])
        pi = label_to_idx.get(pred, label_to_idx["Unrecognized"])
        cm[gi][pi] += 1

    used_rows = [i for i in range(n_labels) if cm[i].sum() > 0]
    used_cols = [i for i in range(n_labels) if cm[:, i].sum() > 0]
    used = sorted(set(used_rows) | set(used_cols))

    cm_used = cm[np.ix_(used, used)]
    labels_used = [SHORT_LABELS[i] for i in used]
    full_labels_used = [ALL_LABELS[i] for i in used]

    fig, ax = plt.subplots(figsize=(14, 11))
    cmap = plt.cm.Blues
    im = ax.imshow(cm_used, interpolation='nearest', cmap=cmap)
    plt.colorbar(im, ax=ax, label='Count')

    for i in range(len(used)):
        for j in range(len(used)):
            val = cm_used[i, j]
            if val > 0:
                color = 'white' if val > cm_used.max() * 0.5 else 'black'
                fontweight = 'bold' if i == j else 'normal'
                ax.text(j, i, str(val), ha='center', va='center',
                        color=color, fontsize=9, fontweight=fontweight)

    ax.set_xticks(range(len(used)))
    ax.set_yticks(range(len(used)))
    ax.set_xticklabels(labels_used, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels_used, fontsize=8)
    ax.set_xlabel('Predicted (incl. Unrecognized)', fontsize=11)
    ax.set_ylabel('Ground Truth', fontsize=11)
    ax.set_title('Confusion Matrix - 30B Dynamic Model\n(12tags_Qwen3-VL-30B_full_add_tags_dynamic)',
                 fontsize=13, fontweight='bold')

    diag_total = sum(cm_used[i, i] for i in range(len(used)))
    total = cm_used.sum()
    acc = diag_total / total if total > 0 else 0
    ax.text(0.02, 0.98, f'Accuracy: {acc:.1%} ({diag_total}/{total})',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    out_path = '/root/workspace/LLaMA-Factory/scene_tag/confusion_matrix_dynamic.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {out_path}")

    print(f"\n各标签统计:")
    for i, idx in enumerate(used):
        row_sum = cm_used[i].sum()
        diag = cm_used[i, i]
        recall = diag / row_sum if row_sum > 0 else 0
        col_sum = cm_used[:, i].sum()
        precision = diag / col_sum if col_sum > 0 else 0
        print(f"  {full_labels_used[i]:<45s} GT={row_sum:>3d}  Pred={col_sum:>3d}  "
              f"TP={diag:>3d}  Precision={precision:.1%}  Recall={recall:.1%}")


if __name__ == "__main__":
    main()
