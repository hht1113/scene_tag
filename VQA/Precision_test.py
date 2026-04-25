import json
import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def parse_ground_truth(gt):
    parts = gt.split('\n')
    maneuvers = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        behavior_match = re.search(r'<driving_maneuver>(.*?)</driving_maneuver>', part)
        if behavior_match:
            behavior = behavior_match.group(1).strip()
            maneuvers.append(behavior)
    return maneuvers


def parse_prediction(pred):
    lines = pred.split('\n')
    maneuvers = []
    for line in lines:
        behavior_match = re.search(r'<driving_maneuver>(.*?)</driving_maneuver>', line)
        if behavior_match:
            behavior = behavior_match.group(1).strip()
            maneuvers.append(behavior)
    return maneuvers


def compute_ap(tp_list, num_gt):
    """
    计算单个类别的 AP (Average Precision)。
    tp_list: 按置信度降序排列的检测列表，1=TP, 0=FP
    num_gt: 该类别的 GT 总数
    返回: AP = Precision-Recall 曲线下的面积 (VOC 全点插值法)
    """
    if num_gt == 0:
        return 0.0
    if len(tp_list) == 0:
        return 0.0
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - t for t in tp_list])
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap


def compute_classification_map(jsonl_file):
    """
    纯分类 mAP：只判断动作类别是否匹配，不考虑时间段。

    对每个样本：
      - 提取 GT 类别列表和 Pred 类别列表
      - 按类别名匹配（同一样本内，每个 GT 实例只匹配一次）
      - 匹配成功 = TP，多余预测 = FP，未匹配的 GT = FN

    对每个类别：
      - 汇总所有样本的 TP/FP，TP 排在 FP 前面（最优排序）
      - 计算 Precision-Recall 曲线，AP = 曲线下面积

    mAP = 所有类别 AP 的平均值
    """
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    class_tp_fp = {}
    class_gt_count = Counter()

    for obj in data:
        gt_classes = parse_ground_truth(obj['label'])
        pred_classes = parse_prediction(obj['predict'])

        gt_counter = Counter(gt_classes)
        pred_counter = Counter(pred_classes)

        for cls in gt_counter:
            class_gt_count[cls] += gt_counter[cls]

        all_cls = set(gt_counter.keys()) | set(pred_counter.keys())
        for cls in all_cls:
            if cls not in class_tp_fp:
                class_tp_fp[cls] = []
            n_gt = gt_counter.get(cls, 0)
            n_pred = pred_counter.get(cls, 0)
            tp = min(n_gt, n_pred)
            fp = max(0, n_pred - n_gt)
            class_tp_fp[cls].extend([1] * tp)
            class_tp_fp[cls].extend([0] * fp)

    all_classes = sorted(set(class_gt_count.keys()) | set(class_tp_fp.keys()))

    per_class = {}
    for cls in all_classes:
        preds = class_tp_fp.get(cls, [])
        num_gt = class_gt_count.get(cls, 0)

        preds_sorted = sorted(preds, reverse=True)
        ap = compute_ap(preds_sorted, num_gt)

        tp_total = sum(preds)
        fp_total = len(preds) - tp_total
        fn_total = num_gt - tp_total
        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cls] = {
            'num_gt': num_gt,
            'num_pred': len(preds),
            'TP': tp_total,
            'FP': fp_total,
            'FN': fn_total,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AP': ap,
        }

    valid_aps = [per_class[cls]['AP'] for cls in all_classes if per_class[cls]['num_gt'] > 0]
    map_val = np.mean(valid_aps) if valid_aps else 0.0

    return {
        'per_class': per_class,
        'mAP': map_val,
        'all_classes': all_classes,
    }


def main():
    jsonl_dir = r'/root/workspace/LLaMA-Factory/VQA/json'
    jsonl_files = glob.glob(os.path.join(jsonl_dir, '*.jsonl')) + glob.glob(os.path.join(jsonl_dir, '*.json'))
    jsonl_files = sorted(set(jsonl_files))

    if not jsonl_files:
        print(f"No jsonl files found in {jsonl_dir}")
        return

    print("=" * 110)
    print("分类 mAP（只评估动作类别是否正确，不考虑时间段）")
    print("=" * 110)

    all_results = {}
    for jsonl_file in jsonl_files:
        file_name = os.path.splitext(os.path.basename(jsonl_file))[0]
        result = compute_classification_map(jsonl_file)
        all_results[file_name] = result

        print(f"\n{'─' * 100}")
        print(f"模型: {file_name}")
        print(f"{'─' * 100}")
        print(f"  {'类别':<45s} {'GT':>5s} {'Pred':>5s} {'TP':>5s} {'FP':>5s} {'FN':>5s} "
              f"{'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'AP':>7s}")
        print(f"  {'-' * 95}")

        total_tp = 0
        total_fp = 0
        total_fn = 0
        for cls in result['all_classes']:
            d = result['per_class'][cls]
            total_tp += d['TP']
            total_fp += d['FP']
            total_fn += d['FN']
            print(f"  {cls:<45s} {d['num_gt']:>5d} {d['num_pred']:>5d} {d['TP']:>5d} "
                  f"{d['FP']:>5d} {d['FN']:>5d} {d['Precision']:>7.4f} {d['Recall']:>7.4f} "
                  f"{d['F1']:>7.4f} {d['AP']:>7.4f}")

        micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

        print(f"  {'-' * 95}")
        print(f"  {'Micro Avg':<45s} {'':>5s} {'':>5s} {total_tp:>5d} {total_fp:>5d} {total_fn:>5d} "
              f"{micro_prec:>7.4f} {micro_rec:>7.4f} {micro_f1:>7.4f} {'':>7s}")
        print(f"  {'mAP (Macro Avg of AP)':<45s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} {'':>5s} "
              f"{'':>7s} {'':>7s} {'':>7s} {result['mAP']:>7.4f}")

    # ========== 汇总表 ==========
    print(f"\n\n{'=' * 110}")
    print("mAP 汇总对比")
    print(f"{'=' * 110}")
    file_names = sorted(all_results.keys())
    header = f"  {'模型':<58s} {'mAP':>8s} {'Micro-P':>8s} {'Micro-R':>8s} {'Micro-F1':>9s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fn in file_names:
        r = all_results[fn]
        total_tp = sum(r['per_class'][c]['TP'] for c in r['all_classes'])
        total_fp = sum(r['per_class'][c]['FP'] for c in r['all_classes'])
        total_fn = sum(r['per_class'][c]['FN'] for c in r['all_classes'])
        mp = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        mr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        mf = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
        print(f"  {fn:<58s} {r['mAP']:>8.4f} {mp:>8.4f} {mr:>8.4f} {mf:>9.4f}")

    # ========== 绘图 ==========
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    n_files = len(file_names)
    colors = [color_list[i % len(color_list)] for i in range(n_files)]

    all_classes = sorted(set(
        cls for fn in file_names for cls in all_results[fn]['all_classes']
    ))
    n_classes = len(all_classes)

    # 图1: 各模型各类别 Precision 柱状图
    bar_width = 0.18
    group_gap = 0.5
    group_width = n_files * bar_width + group_gap
    x_cls = np.arange(n_classes) * group_width

    fig, ax = plt.subplots(figsize=(max(20, n_classes * group_width * 1.5), 10))
    for i, (fn, color) in enumerate(zip(file_names, colors)):
        precs = [all_results[fn]['per_class'].get(cls, {}).get('Precision', 0.0) for cls in all_classes]
        offset = (i - (n_files - 1) / 2) * bar_width
        positions = x_cls + offset
        bars = ax.bar(positions, precs, width=bar_width, color=color,
                      label=fn, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, precs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_xticks(x_cls)
    ax.set_xticklabels(all_classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Per-Class Precision (Classification Only)', fontsize=14)
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, title='Model')
    plt.tight_layout()
    plt.savefig('/root/workspace/LLaMA-Factory/VQA/plot/per_class_Precision_chart.png', dpi=150)
    print("\nPer-class Precision chart saved to /root/workspace/LLaMA-Factory/VQA/plot/per_class_Precision_chart.png")

    # 图2: 各模型 mAP 对比柱状图
    fig2, ax2 = plt.subplots(figsize=(max(10, n_files * 2), 7))
    map_vals = [all_results[fn]['mAP'] for fn in file_names]
    short_names = [fn.replace('12tags_', '').replace('_segment', '\nsegment')
                   .replace('_upsample', '\nupsample').replace('_upstream', '\nupstream')
                   for fn in file_names]
    bars = ax2.bar(range(n_files), map_vals, color=colors[:n_files],
                   edgecolor='black', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, map_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(n_files))
    ax2.set_xticklabels(short_names, fontsize=9, ha='center')
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel('mAP', fontsize=12)
    ax2.set_title('mAP Comparison (Classification Only)', fontsize=14)
    ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('/root/workspace/LLaMA-Factory/VQA/plot/mAP_chart.png', dpi=150)
    print("mAP comparison chart saved to /root/workspace/LLaMA-Factory/VQA/plot/mAP_chart.png")


if __name__ == "__main__":
    main()
