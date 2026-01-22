#!/usr/bin/env python3
"""
构造用于挖掘的JSON数据集文件
视频路径为已下载的视频本地路径
便于后续加载微调模型进行推理
"""

import os
import json
import logging
import glob
from pathlib import Path
from typing import List, Dict, Any
import argparse

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_video_files(video_root: str) -> List[str]:
    """
    在指定目录下查找所有video.mp4文件
    
    Args:
        video_root: 视频根目录
        
    Returns:
        视频文件路径列表
    """
    # 使用glob递归查找所有video.mp4文件
    pattern = os.path.join(video_root, "**", "video.mp4")
    video_files = glob.glob(pattern, recursive=True)
    
    # 过滤出有效的文件路径
    valid_videos = []
    for video_path in video_files:
        if os.path.isfile(video_path):
            valid_videos.append(video_path)
    
    logger.info(f"在 {video_root} 下找到 {len(valid_videos)} 个视频文件")
    return sorted(valid_videos)

def extract_video_metadata(video_path: str) -> Dict[str, str]:
    """
    从视频路径中提取元数据信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        包含元数据的字典
    """
    metadata = {}
    
    # 尝试从路径中解析车号、日期、clip等信息
    parts = Path(video_path).parts
    
    for i, part in enumerate(parts):
        if part == "raw_clips" and i + 1 < len(parts):
            # 车号
            metadata["car_id"] = parts[i + 1]
        elif "clips" in part and i + 1 < len(parts):
            # clip ID
            metadata["clip_id"] = parts[i + 1]
        elif len(part) == 19 and part[4] == '-' and part[7] == '-':  # 类似 2025-12-23_16-09-31
            metadata["date_time"] = part
    
    return metadata

def create_dataset_entry(video_path: str) -> Dict[str, Any]:
    """
    为单个视频创建数据集条目
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        数据集条目字典
    """
    # 提取元数据
    metadata = extract_video_metadata(video_path)
    
    # 基础指令
    instruction = "<video>\nWhat is the ego vehicle's behavior in this video clip?"
    
    # 系统提示 - 与微调数据集保持一致
    system = """You are an expert in autonomous driving scene annotation. 
Based on a 60-second video, you need to identify the ego vehicle's actions.

You MUST choose labels ONLY from this specific list:
1. TrafficLight_StraightStopOrGo
2. TrafficLight_LeftTurnStopOrGo
3. LaneChange_NavForIntersection

Please use the format: <driving_maneuver>action_label</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.
If there are multiple actions, list them in chronological order separated by " and ".
IMPORTANT: Only use the exact labels from the list above. Do NOT create new labels."""
    
    # 构建数据集条目
    entry = {
        "instruction": instruction,
        "input": "",
        "output": "",  # 留空，用于模型推理
        "videos": [video_path],  # 使用本地路径
        "system": system
    }
    
    # 添加元数据作为额外信息
    if metadata:
        entry["metadata"] = metadata
    
    return entry

def create_dataset(video_files: List[str], output_path: str) -> None:
    """
    创建完整的数据集文件
    
    Args:
        video_files: 视频文件路径列表
        output_path: 输出JSON文件路径
    """
    dataset = []
    
    logger.info(f"开始构建数据集，共 {len(video_files)} 个视频")
    
    for i, video_path in enumerate(video_files, 1):
        try:
            # 验证视频文件存在
            if not os.path.exists(video_path):
                logger.warning(f"视频文件不存在: {video_path}")
                continue
            
            # 验证文件大小
            file_size = os.path.getsize(video_path)
            if file_size < 1024:  # 小于1KB认为是无效文件
                logger.warning(f"视频文件过小 ({file_size} bytes): {video_path}")
                continue
            
            # 创建数据集条目
            entry = create_dataset_entry(video_path)
            dataset.append(entry)
            
            if i % 10 == 0 or i == len(video_files):
                logger.info(f"已处理 {i}/{len(video_files)} 个视频")
                
        except Exception as e:
            logger.error(f"处理视频失败 {video_path}: {str(e)}")
            continue
    
    # 保存数据集
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 数据集创建成功!")
        logger.info(f"  文件路径: {output_path}")
        logger.info(f"  总条目数: {len(dataset)}")
        logger.info(f"  视频总数: {sum(len(entry['videos']) for entry in dataset)}")
        
    except Exception as e:
        logger.error(f"保存数据集失败: {str(e)}")
        raise

def create_dataset_jsonl(video_files: List[str], output_path: str) -> None:
    """
    创建JSONL格式的数据集文件（每行一个JSON对象）
    
    Args:
        video_files: 视频文件路径列表
        output_path: 输出JSONL文件路径
    """
    logger.info(f"开始构建JSONL数据集，共 {len(video_files)} 个视频")
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            valid_count = 0
            for i, video_path in enumerate(video_files, 1):
                try:
                    # 验证视频文件存在
                    if not os.path.exists(video_path):
                        logger.warning(f"视频文件不存在: {video_path}")
                        continue
                    
                    # 验证文件大小
                    file_size = os.path.getsize(video_path)
                    if file_size < 1024:  # 小于1KB认为是无效文件
                        logger.warning(f"视频文件过小 ({file_size} bytes): {video_path}")
                        continue
                    
                    # 创建数据集条目
                    entry = create_dataset_entry(video_path)
                    
                    # 写入JSONL行
                    json_line = json.dumps(entry, ensure_ascii=False)
                    f.write(json_line + '\n')
                    valid_count += 1
                    
                    if i % 10 == 0 or i == len(video_files):
                        logger.info(f"已处理 {i}/{len(video_files)} 个视频")
                        
                except Exception as e:
                    logger.error(f"处理视频失败 {video_path}: {str(e)}")
                    continue
        
        logger.info(f"✅ JSONL数据集创建成功!")
        logger.info(f"  文件路径: {output_path}")
        logger.info(f"  总条目数: {valid_count}")
        
    except Exception as e:
        logger.error(f"保存数据集失败: {str(e)}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建用于视频推理的JSON数据集')
    parser.add_argument('--video-dir', type=str, required=True,
                       default='/root/workspace/digged_videos',
                       help='视频文件目录，默认: /root/workspace/digged_videos')
    parser.add_argument('--output', type=str, required=True,
                       default='/root/workspace/inference_dataset.json',
                       help='输出JSON文件路径，默认: /root/workspace/inference_dataset.json')
    parser.add_argument('--format', type=str, choices=['json', 'jsonl'], default='json',
                       help='输出格式: json或jsonl，默认: json')
    parser.add_argument('--max-videos', type=int, default=None,
                       help='最大视频数量（用于测试），默认: 无限制')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("开始创建推理数据集")
    logger.info(f"视频目录: {args.video_dir}")
    logger.info(f"输出文件: {args.output}")
    logger.info(f"输出格式: {args.format}")
    if args.max_videos:
        logger.info(f"最大视频数: {args.max_videos}")
    logger.info("=" * 60)
    
    # 检查视频目录是否存在
    if not os.path.exists(args.video_dir):
        logger.error(f"视频目录不存在: {args.video_dir}")
        return
    
    # 查找所有视频文件
    video_files = find_video_files(args.video_dir)
    
    if not video_files:
        logger.error("未找到任何视频文件")
        return
    
    # 限制视频数量（用于测试）
    if args.max_videos and args.max_videos < len(video_files):
        logger.info(f"限制为前 {args.max_videos} 个视频")
        video_files = video_files[:args.max_videos]
    
    # 创建数据集
    if args.format == 'json':
        create_dataset(video_files, args.output)
    else:  # jsonl
        create_dataset_jsonl(video_files, args.output)
    
    # 显示数据集样本示例
    logger.info("\n" + "=" * 60)
    logger.info("数据集样本示例:")
    
    # 读取并显示第一个样本
    try:
        if args.format == 'json':
            with open(args.output, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            if dataset:
                sample = dataset[0]
                logger.info(json.dumps(sample, ensure_ascii=False, indent=2)[:500] + "...")
        else:  # jsonl
            with open(args.output, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            if first_line:
                sample = json.loads(first_line)
                logger.info(json.dumps(sample, ensure_ascii=False, indent=2)[:500] + "...")
    except Exception as e:
        logger.warning(f"无法读取输出文件以显示示例: {str(e)}")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

'''
# 基本用法：为已下载的视频创建数据集
python create_inference_dataset.py --video-dir /root/workspace/digged_videos --output /root/workspace/inference_dataset.json

# 创建JSONL格式的数据集
python create_inference_dataset.py --video-dir /root/workspace/digged_videos --output /root/workspace/inference_dataset.jsonl --format jsonl

# 限制处理视频数量（用于测试）
python create_inference_dataset.py --video-dir /root/workspace/digged_videos --output /root/workspace/test_dataset.json --max-videos 5
'''