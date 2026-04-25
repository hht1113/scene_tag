"""
将 vllm_infer 的 JSONL 推理结果转换为 13_review.py 的审核格式。

支持边推理边转换：读取已有的 JSONL 行，转换后即可审核。

用法:
    python convert_infer_to_review.py \
        --input infer_results/mining_pool_5000_30B_65k_2ep.jsonl \
        --output infer_results/mining_pool_5000_for_review.json

    # 然后启动审核工具
    python 13_review.py --port 9000
    # 在浏览器中粘贴: /root/workspace/LLaMA-Factory/infer_results/mining_pool_5000_for_review.json
"""

import json
import re
import argparse
from collections import Counter
from pathlib import Path


def extract_segments(text):
    """从模型输出中提取所有 (label, start, end) 段"""
    segments = []
    pattern = r'<driving_maneuver>(.*?)</driving_maneuver>.*?<start_time>([\d.]+)</start_time>.*?<end_time>([\d.]+)</end_time>'
    for m in re.finditer(pattern, text):
        label = m.group(1)
        start = float(m.group(2))
        end = float(m.group(3))
        segments.append({"label": label, "start": start, "end": end})
    return segments


def extract_video_path_from_prompt(prompt):
    """从 prompt 中提取视频路径（在 mining 数据集中视频路径在 dataset JSON 里，不在 prompt 中）"""
    return None


def convert(input_path, output_path, dataset_path=None):
    # 读取 JSONL（支持不完整文件）
    results = []
    with open(input_path, 'r') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [WARN] 第 {line_no} 行 JSON 解析失败，跳过")

    print(f"读取 {len(results)} 条推理结果")

    # 如果提供了 dataset_path，从中获取视频路径
    video_paths = {}
    if dataset_path and Path(dataset_path).exists():
        with open(dataset_path) as f:
            dataset = json.load(f)
        for i, item in enumerate(dataset):
            video_paths[i] = item['videos'][0]
        print(f"从数据集加载 {len(video_paths)} 个视频路径")

    # 转换
    annotations = []
    label_counter = Counter()

    for i, r in enumerate(results):
        predict = r.get('predict', '')
        segments = extract_segments(predict)

        video_path = video_paths.get(i, f"unknown_video_{i}")

        for seg in segments:
            label_counter[seg['label']] += 1

        annotations.append({
            "video_path": video_path,
            "segments": segments,
            "raw_predict": predict,
        })

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

    print(f"\n转换完成: {output_path}")
    print(f"  视频数: {len(annotations)}")
    print(f"  总段数: {sum(label_counter.values())}")
    print(f"\n  标签分布:")
    for label, count in label_counter.most_common():
        print(f"    {label}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='vllm_infer 输出的 JSONL 文件')
    parser.add_argument('--output', required=True, help='输出的审核格式 JSON')
    parser.add_argument('--dataset', default='/root/workspace/LLaMA-Factory/data/mining_pool_5000.json',
                        help='原始数据集 JSON（用于获取视频路径）')
    args = parser.parse_args()
    convert(args.input, args.output, args.dataset)


if __name__ == '__main__':
    main()
