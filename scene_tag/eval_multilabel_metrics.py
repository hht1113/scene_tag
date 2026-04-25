#!/usr/bin/env python3
"""
Evaluate multi-label classification results for driving maneuver prediction.
Computes mAP and per-label Precision / Recall / F1.
"""

import json
import re
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)

INPUT_FILE = "/root/workspace/LLaMA-Factory/VQA/json/test_multilabel_v3_30B_checkpoint282.jsonl"
TAG_PATTERN = re.compile(r"<driving_maneuver>(.*?)</driving_maneuver>")


def extract_labels(text: str) -> set[str]:
    return set(TAG_PATTERN.findall(text))


def main():
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred_labels = extract_labels(obj.get("predict", ""))
            true_labels = extract_labels(obj.get("label", ""))
            records.append((pred_labels, true_labels))

    all_labels = sorted(
        set().union(*(p | t for p, t in records))
    )
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    n_samples = len(records)
    n_labels = len(all_labels)

    y_true = np.zeros((n_samples, n_labels), dtype=int)
    y_pred = np.zeros((n_samples, n_labels), dtype=int)

    for i, (pred_set, true_set) in enumerate(records):
        for lbl in true_set:
            y_true[i, label_to_idx[lbl]] = 1
        for lbl in pred_set:
            if lbl in label_to_idx:
                y_pred[i, label_to_idx[lbl]] = 1

    # ---- per-label AP ----
    per_label_ap = {}
    for j, lbl in enumerate(all_labels):
        if y_true[:, j].sum() == 0:
            per_label_ap[lbl] = float("nan")
        else:
            per_label_ap[lbl] = average_precision_score(y_true[:, j], y_pred[:, j])

    valid_aps = [v for v in per_label_ap.values() if not np.isnan(v)]
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0

    # ---- per-label P / R / F1 ----
    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    label_metrics = []
    for j, lbl in enumerate(all_labels):
        label_metrics.append(
            {
                "label": lbl,
                "precision": precision_arr[j],
                "recall": recall_arr[j],
                "f1": f1_arr[j],
                "ap": per_label_ap[lbl],
                "support_true": int(y_true[:, j].sum()),
                "support_pred": int(y_pred[:, j].sum()),
            }
        )

    label_metrics.sort(key=lambda x: x["precision"], reverse=True)

    # ---- micro / macro averages ----
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # ---- sample-level exact-match accuracy ----
    exact_match = sum(
        1 for i in range(n_samples) if np.array_equal(y_true[i], y_pred[i])
    ) / n_samples

    # ---- print results ----
    sep = "=" * 120
    print(sep)
    print(f"  评估文件: {INPUT_FILE}")
    print(f"  样本数: {n_samples}    标签种类数: {n_labels}")
    print(sep)

    print(f"\n{'Overall Metrics':=^120}")
    print(f"  mAP (mean Average Precision) : {mean_ap:.4f}")
    print(f"  Micro  Precision / Recall / F1 : {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
    print(f"  Macro  Precision / Recall / F1 : {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    print(f"  Exact Match Accuracy           : {exact_match:.4f}")

    print(f"\n{'Per-Label Metrics (sorted by Precision ↓)':=^120}")
    header = f"{'Label':<50} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AP':>8} {'#True':>7} {'#Pred':>7} {'Status':>10}"
    print(header)
    print("-" * 120)

    high_prec_labels = []
    low_prec_labels = []

    for m in label_metrics:
        status = "✓ ≥70%" if m["precision"] >= 0.70 else "✗ <70%"
        if m["precision"] >= 0.70:
            high_prec_labels.append(m["label"])
        else:
            low_prec_labels.append(m["label"])

        ap_str = f"{m['ap']:.4f}" if not np.isnan(m["ap"]) else "  N/A "
        print(
            f"  {m['label']:<48} {m['precision']:>8.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {ap_str:>8} {m['support_true']:>7d} {m['support_pred']:>7d} {status:>10}"
        )

    print("-" * 120)

    print(f"\n{'Summary':=^120}")
    print(f"  Precision ≥ 70% 的标签 ({len(high_prec_labels)}):")
    for lbl in high_prec_labels:
        print(f"    ✓ {lbl}")
    print(f"\n  Precision < 70% 的标签 ({len(low_prec_labels)}):")
    for lbl in low_prec_labels:
        print(f"    ✗ {lbl}")
    print(sep)

    # ---- confusion details: per-label FP / FN breakdown ----
    print(f"\n{'Per-Label Error Analysis':=^120}")
    for m in label_metrics:
        j = label_to_idx[m["label"]]
        tp = int(((y_true[:, j] == 1) & (y_pred[:, j] == 1)).sum())
        fp = int(((y_true[:, j] == 0) & (y_pred[:, j] == 1)).sum())
        fn = int(((y_true[:, j] == 1) & (y_pred[:, j] == 0)).sum())
        tn = int(((y_true[:, j] == 0) & (y_pred[:, j] == 0)).sum())
        print(f"  {m['label']:<48}  TP={tp:>4}  FP={fp:>4}  FN={fn:>4}  TN={tn:>4}")
    print(sep)


if __name__ == "__main__":
    main()
