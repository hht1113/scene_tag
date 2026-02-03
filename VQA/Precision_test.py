import json
import re
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def parse_ground_truth(gt):
    parts = gt.split('\n')
    maneuvers = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        behavior_match = re.search(r'<driving_maneuver>(.*?)</driving_maneuver>', part)
        start_match = re.search(r'<start_time>(\d+(?:\.\d+)?)</start_time>', part)
        end_match = re.search(r'<end_time>(\d+(?:\.\d+)?)</end_time>', part)
        if behavior_match and start_match and end_match:
            behavior = behavior_match.group(1).strip()
            start = float(start_match.group(1))
            end = float(end_match.group(1))
            maneuvers.append((behavior, start, end))
    return maneuvers

def parse_prediction(pred):
    lines = pred.split('\n')
    maneuvers = []
    for line in lines:
        behavior_match = re.search(r'<driving_maneuver>(.*?)</driving_maneuver>', line)
        start_match = re.search(r'<start_time>(\d+(?:\.\d+)?)</start_time>', line)
        end_match = re.search(r'<end_time>(\d+(?:\.\d+)?)</end_time>', line)
        if behavior_match and start_match and end_match:
            behavior = behavior_match.group(1).strip()
            start = float(start_match.group(1))
            end = float(end_match.group(1))
            maneuvers.append((behavior, start, end))
    return maneuvers

def iou(start1, end1, start2, end2):
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    if inter_end <= inter_start:
        return 0.0
    inter = inter_end - inter_start
    union = (end1 - start1) + (end2 - start2) - inter
    return inter / union if union > 0 else 0.0

def main():
    # 读取指定路径下的所有jsonl文件
    jsonl_dir = r'/root/workspace/LLaMA-Factory/VQA/json'
    jsonl_files = glob.glob(os.path.join(jsonl_dir, '*.jsonl'))
    jsonl_files = sorted(jsonl_files)  # 按照文件名排序
    
    if not jsonl_files:
        print(f"No jsonl files found in {jsonl_dir}")
        return
    
    # 存储每个文件的统计数据
    all_file_stats = {}  # {filename: {action: {'correct': x, 'total': y, 'ious': []}}}
    all_actions = set()  # 收集所有动作类型
    
    for jsonl_file in jsonl_files:
        file_name = os.path.basename(jsonl_file).replace('.jsonl', '')
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]
        
        action_stats = {}

        # 先统计整个文件中每个行为在预测中的总出现次数（分母 = TP + FP）
        pred_count_by_behavior = {}
        for obj in data:
            pred_maneuvers = parse_prediction(obj['predict'])
            for pred_behav, pred_start, pred_end in pred_maneuvers:
                pred_count_by_behavior.setdefault(pred_behav, 0)
                pred_count_by_behavior[pred_behav] += 1

        # 遍历每个对象，计算 TP（匹配成功的预测数）
        for obj in data:
            gt_maneuvers = parse_ground_truth(obj['label'])
            pred_maneuvers = parse_prediction(obj['predict'])

            for pred_behav, pred_start, pred_end in pred_maneuvers:
                if pred_behav not in action_stats:
                    action_stats[pred_behav] = {
                        'correct': 0,
                        'total': pred_count_by_behavior.get(pred_behav, 0),
                        'ious': []
                    }

                # 在该对象的 GT 中找同名行为，计算最大 IoU
                same_gt = [(gt_start, gt_end) for gt_behav, gt_start, gt_end in gt_maneuvers if gt_behav == pred_behav]
                if not same_gt:
                    continue  # 无同名 GT，跳过（FP）
                max_iou = 0
                for gt_start, gt_end in same_gt:
                    current_iou = iou(pred_start, pred_end, gt_start, gt_end)
                    if current_iou > max_iou:
                        max_iou = current_iou
                # 如果最大 IoU >= 0，则算 TP
                if max_iou >= 0:
                    action_stats[pred_behav]['correct'] += 1
                    action_stats[pred_behav]['ious'].append(max_iou)
        
        all_file_stats[file_name] = action_stats
        all_actions.update(action_stats.keys())
    
    # 对动作排序以保持一致性
    actions = sorted(list(all_actions))
    file_names = [os.path.basename(f).replace('.jsonl', '') for f in jsonl_files]
    n_files = len(file_names)
    n_actions = len(actions)
    print(f"n_files: {n_files}, n_actions: {n_actions}, actions: {actions}")
    
    # 设置颜色 - 使用更鲜明的颜色
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [color_list[i % len(color_list)] for i in range(n_files)]
    
    # 设置条形图参数 - 关键：让条形并排显示
    bar_width = 0.25  # 单个条形的宽度，增加以适应标签宽度
    group_gap = 0.5   # 组与组之间的间隙，增加间距
    
    # 计算每组的中心位置，考虑组间距
    group_width = n_files * bar_width + group_gap
    x = np.arange(n_actions) * group_width
    
    # 计算图表宽度，适应条形宽度变化
    fig_width = max(20, n_actions * group_width * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 12))
    
    # 绘制分组条形图 - 每个文件一组条形，并排放置
    for i, (file_name, color) in enumerate(zip(file_names, colors)):
        stats = all_file_stats[file_name]
        accuracies = []
        labels = []
        
        for action in actions:
            if action in stats and stats[action]['total'] > 0:
                acc = stats[action]['correct'] / stats[action]['total']
                correct = stats[action]['correct']
                total = stats[action]['total']
            else:
                acc = 0
                correct = 0
                total = 0
            accuracies.append(acc)
            labels.append(f"{acc:.2f}\n({correct}/{total})")
        
        # 计算偏移量，使条形并排显示在组中心两侧
        offset = (i - (n_files - 1) / 2) * bar_width
        positions = x + offset
        
        bars = ax.bar(positions, accuracies, width=bar_width, color=color, 
                     label=file_name, edgecolor='black', linewidth=0.5)
        
        # 在每个条形上方添加标签
        for bar, label in zip(bars, labels):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, label, 
                   ha='center', va='bottom', rotation=0, fontsize=10)
    
    # 设置x轴刻度位置和标签 - 放在每组的中心
    ax.set_xticks(x)
    ax.set_xticklabels(actions, rotation=45, ha='right')
    ax.set_xlim(-group_gap, (n_actions-1)*group_width + group_gap)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Behavior', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision by Behavior (Grouped Comparison)', fontsize=14)
    
    # 添加横向红色虚线表示 >=0.8
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5)
    ax.text(ax.get_xlim()[1], 0.82, '>=0.8', ha='right', va='bottom', color='red', fontsize=10)
    
    # 添加网格线便于阅读
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10, title='JSONL Files')
    
    plt.tight_layout()
    plt.savefig('/root/workspace/LLaMA-Factory/VQA/plot/precision_chart.png')
    print("Bar chart saved as bar_chart.png")

if __name__ == "__main__":
    main()
