#!/usr/bin/env python3
"""
生成视频验证表格
用于人工检查切片视频和模型预测标签是否匹配
"""

import json
import pandas as pd
from pathlib import Path

def generate_validation_table(predict_jsonl_path, dataset_json_path, output_csv_path="video_validation.csv"):
    """生成视频验证表格"""
    
    # 读取数据
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)
    
    predictions = []
    with open(predict_jsonl_path, 'r') as f:
        for line in f:
            predictions.append(json.loads(line.strip()))
    
    # 生成验证数据
    validation_data = []
    
    for i, (data, pred) in enumerate(zip(dataset, predictions)):
        # 获取视频路径
        video_path = data['videos'][0] if 'videos' in data and data['videos'] else ''
        
        # 获取预测结果
        predict_text = pred.get('predict', '')
        
        # 提取标签
        import re
        labels = re.findall(r'<driving_maneuver>(.*?)</driving_maneuver>', predict_text)
        
        if labels:
            for label in labels:
                validation_data.append({
                    '视频路径': video_path,
                    '预测标签': label,
                    '完整预测': predict_text[:100] + '...' if len(predict_text) > 100 else predict_text,
                    '序号': i + 1
                })
        else:
            validation_data.append({
                '视频路径': video_path,
                '预测标签': '无标签',
                '完整预测': predict_text[:100] + '...' if len(predict_text) > 100 else predict_text,
                '序号': i + 1
            })
    
    # 保存为CSV
    df = pd.DataFrame(validation_data)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 验证表格已生成: {output_csv_path}")
    print(f"  数据集样本数: {len(dataset)}")
    print(f"  预测样本数: {len(predictions)}")
    print(f"  验证条目数: {len(df)}")
    
    # 显示标签统计
    if not df.empty:
        print(f"\n标签统计:")
        label_counts = df['预测标签'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
    
    return output_csv_path

if __name__ == "__main__":
    # 设置你的文件路径
    predict_file = "/root/workspace/LLaMA-Factory/infer_results/12tags_Qwen3-VL-4B_segment_upstream_1epoch_digged.jsonl"  # 替换为你的预测文件路径
    dataset_file = "/mnt/pfs/houhaotian/segemnt_inference_dataset.json"  # 数据集文件路径
    output_file = "/root/workspace/LLaMA-Factory/dig_result/video_validation.csv"  # 输出文件路径（可自定义）
    
    # 检查文件是否存在
    for path, desc in [(predict_file, "预测结果文件"), (dataset_file, "数据集文件")]:
        if not Path(path).exists():
            print(f"❌ 错误: {desc}不存在: {path}")
            exit(1)
    
    # 生成验证表格
    csv_path = generate_validation_table(predict_file, dataset_file, output_file)
    
    print(f"\n📋 使用说明:")
    print(f"1. 表格已保存到: {csv_path}")
    print(f"2. 用Excel或文本编辑器打开查看")
    print(f"3. 根据'视频路径'找到视频文件，播放查看")
    print(f"4. 对照'预测标签'判断模型预测是否正确")
    print(f"5. 可以按'预测标签'排序，批量检查同类视频")