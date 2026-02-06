#!/usr/bin/env python3
"""
将验证CSV中的视频按类别分类到不同目录
用于分配给不同人员检查
"""

import pandas as pd
import os
import shutil
from pathlib import Path
import argparse

def categorize_videos_by_label(csv_path, output_dir, categories):
    """
    将视频按标签分类到不同目录
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出根目录
        categories: 类别列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 确保CSV有必要的列
    required_columns = ['视频路径', '预测标签']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误: CSV文件缺少必要的列: {col}")
            return
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输出目录是否与CSV文件在同一目录，避免覆盖
    csv_dir = Path(csv_path).parent
    if output_dir.samefile(csv_dir):
        # 在输出目录下创建一个分类子目录
        classified_dir = output_dir / "classified_videos"
        classified_dir.mkdir(exist_ok=True)
        output_dir = classified_dir
        print(f"注意: 避免与CSV文件冲突，分类结果将保存到: {output_dir}")
    
    # 为每个类别创建目录
    category_dirs = {}
    for category in categories:
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)
        category_dirs[category] = category_dir
    
    # 统计信息
    stats = {category: 0 for category in categories}
    
    # 处理每个视频
    for idx, row in df.iterrows():
        video_path = Path(row['视频路径'])
        label = str(row['预测标签']).strip()
        
        # 检查视频文件是否存在
        if not video_path.exists():
            print(f"警告: 视频文件不存在: {video_path}")
            continue
        
        # 确定类别（如果不是预定义类别，则归为else）
        if label in category_dirs:
            target_category = label
        else:
            target_category = 'else'
        
        # 如果else目录不存在，创建它
        if target_category not in category_dirs:
            else_dir = output_dir / 'else'
            else_dir.mkdir(exist_ok=True)
            category_dirs[target_category] = else_dir
        
        # 目标路径
        target_dir = category_dirs[target_category]
        # 使用原始文件名，避免重名冲突
        target_path = target_dir / video_path.name
        
        # 如果目标文件已存在，添加序号
        counter = 1
        original_target = target_path
        while target_path.exists():
            name = original_target.stem
            suffix = original_target.suffix
            target_path = target_dir / f"{name}_{counter}{suffix}"
            counter += 1
        
        try:
            # 复制视频文件
            shutil.copy2(video_path, target_path)
            stats[target_category] += 1
            
            # 每处理100个文件打印进度
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(df)} 个文件")
                
        except Exception as e:
            print(f"错误: 复制文件失败 {video_path} -> {target_path}: {str(e)}")
    
    # 为每个类别创建说明文件
    for category, category_dir in category_dirs.items():
        # 创建说明文件
        readme_path = category_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"类别: {category}\n")
            f.write(f"视频数量: {stats[category]}\n")
            f.write(f"类别定义: {get_category_definition(category)}\n")
            f.write("\n检查说明:\n")
            f.write("1. 播放本目录下的所有视频文件\n")
            f.write("2. 检查预测标签是否正确\n")
            f.write("3. 如发现标签错误，请在文件名前加上 WRONG_ 前缀\n")
            f.write("4. 正确标签的视频无需修改\n")
    
    # 创建汇总统计文件
    summary_path = output_dir / "summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("视频分类统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"CSV文件: {csv_path}\n")
        f.write(f"总行数: {len(df)}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"处理时间: {pd.Timestamp.now()}\n")
        f.write("\n各类别视频数量:\n")
        
        for category, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {category}: {count} 个\n")
        
        f.write(f"\n总计: {sum(stats.values())} 个视频文件\n")
    
    return stats, output_dir

def get_category_definition(category):
    """获取类别定义"""
    definitions = {
        "TrafficLight_StraightStopOrGo": "Ego vehicle stops or starts at a traffic light for straight-line movement",
        "TrafficLight_LeftTurnStopOrGo": "Ego vehicle stops or starts at a traffic light for left-turn movement",
        "LaneChange_NavForIntersection": "Lane change for navigation purposes approaching an intersection",
        "LaneChange_AvoidSlowVRU": "Lane change to avoid slow-moving vulnerable road users (pedestrians, cyclists)",
        "LaneChange_AvoidStaticVehicle": "Lane change to avoid stationary vehicles",
        "DynamicInteraction_VRUInLaneCrossing": "Interaction with vulnerable road users crossing the ego's lane",
        "DynamicInteraction_VehicleInLaneCrossing": "Interaction with other vehicles crossing the ego's lane",
        "DynamicInteraction_StandardVehicleCutIn": "Another vehicle cuts in front of the ego vehicle",
        "StartStop_StartFromMainRoad": "Starting from a stopped position on a main road",
        "StartStop_ParkRoadside": "Parking or stopping at roadside",
        "Intersection_StandardUTurn": "Making a U-turn at an intersection",
        "LaneCruising_Straight": "Straight-line cruising without notable events",
        "else": "其他未定义的类别"
    }
    return definitions.get(category, "未定义类别")

def main():
    # 定义12个类别
    categories = [
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
        "else"  # 其他类别
    ]
    
    parser = argparse.ArgumentParser(description='将视频按类别分类到不同目录')
    parser.add_argument('--csv', type=str, required=False,
                       default='/root/workspace/LLaMA-Factory/dig_result/video_validation.csv',
                       help='输入CSV文件路径，默认: video_validation.csv')
    parser.add_argument('--output', type=str, required=False,
                       default='/root/workspace/LLaMA-Factory/dig_result',
                       help='输出根目录，默认: /root/workspace/LLaMA-Factory/dig_result')
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    print(f"开始处理CSV文件: {csv_path}")
    print(f"输出根目录: {args.output}")
    print(f"分类数量: {len(categories)} 个")
    
    # 执行分类
    stats, final_output_dir = categorize_videos_by_label(csv_path, args.output, categories)
    
    # 显示结果
    print(f"\n✅ 视频分类完成!")
    print(f"最终输出目录: {final_output_dir}")
    print("\n各类别视频数量:")
    for category, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} 个")
    print(f"\n总计: {sum(stats.values())} 个视频文件")
    
    # 显示使用说明
    print(f"\n📋 使用说明:")
    print(f"1. 查看汇总统计: cat {final_output_dir}/summary.txt")
    print(f"2. 每个类别目录下都有README.txt说明文件")
    print(f"3. 可以按类别分配给不同人员检查")
    print(f"4. 检查人员进入对应类别目录，播放视频验证标签")
    print(f"5. 错误的视频请重命名为 WRONG_原文件名")

if __name__ == "__main__":
    main()