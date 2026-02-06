#!/usr/bin/env python3
"""
构造用于视频挖掘的JSON数据集文件
使用切片抽帧后的视频路径
"""

import json
import logging
import glob
from pathlib import Path
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 类别定义
DRIVING_MANEUVER_CATEGORIES = {
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
    "LaneCruising_Straight": "Straight-line cruising without notable events"
}

# 获取类别列表
CATEGORY_LABELS = list(DRIVING_MANEUVER_CATEGORIES.keys())
CATEGORY_LIST_STR = "\n".join(CATEGORY_LABELS)

# 生成类别定义的文本
CATEGORY_DEFINITIONS = "\n".join(
    [f"{i+1}. {label}: {definition}" 
     for i, (label, definition) in enumerate(DRIVING_MANEUVER_CATEGORIES.items())]
)

# 主系统提示
SYSTEM_PROMPT = f"""You are an expert in autonomous driving scene annotation.
Based on the input video and the question about the ego vehicle's behavior, analyze the 20-second video to identify the ego vehicle's actions with strict precision, focusing on predefined driving maneuver categories.

DRIVING MANEUVER CATEGORIES:
You MUST use ONLY these predefined labels for the ego vehicle's actions:

{CATEGORY_LIST_STR}
else (ONLY when NO label above matches, meaning the ego vehicle's action does not fit any of the predefined categories)

LABELING RULES:
1. Assign a label ONLY if the action clearly matches the definition of one of the predefined categories
2. NEVER force-match ambiguous scenes to predefined labels
3. Use "else" when:
   • The ego vehicle's action does not match any predefined category
   • The scene is ambiguous or uncertain (confidence < 90%)
   • No clearly identifiable maneuver occurs
4. For "else" segments: Cover ONLY time periods with NO identifiable predefined maneuver
5. Time segments MUST be contiguous
6. Minimum segment duration: 1.0 second. Ignore shorter or transient actions
7. Base times on video timeline (0 to 20 seconds)

OUTPUT FORMAT:
<driving_maneuver>action_label</driving_maneuver> from <start_time>XX</start_time> to <end_time>YY</end_time> seconds
• Use one of the predefined category labels or "else" for each time segment
• Time precision: 0 decimal places (e.g., 5, 23)
• NO additional text or explanations—only output the formatted segments

CATEGORY DEFINITIONS:
{CATEGORY_DEFINITIONS}
13. else: Default for all other behaviors not covered by the predefined categories

IMPORTANT GUIDELINES:
1. Analyze the entire 20-second video thoroughly
2. Match actions to the most specific appropriate category
3. If multiple categories could apply, choose the one that best describes the primary action
4. Ensure time segments accurately reflect when each maneuver occurs
5. Maintain chronological order in output
"""

def find_segmented_videos(video_root: str) -> list:
    """查找所有切片后的视频文件"""
    pattern = str(Path(video_root) / "**" / "*_segment_*.mp4")
    video_files = glob.glob(pattern, recursive=True)
    video_files.sort()
    logger.info(f"找到 {len(video_files)} 个切片视频文件")
    return video_files

def extract_video_info(video_path: str) -> dict:
    """从视频路径中提取信息"""
    path = Path(video_path)
    
    # 提取元数据
    info = {
        "video_path": video_path,
        "filename": path.name,
        "segment_id": path.stem.split("_segment_")[-1] if "_segment_" in path.stem else "unknown"
    }
    
    # 提取车号
    parts = path.parts
    try:
        raw_clips_idx = parts.index("raw_clips")
        if raw_clips_idx + 1 < len(parts):
            info["car_id"] = parts[raw_clips_idx + 1]
    except ValueError:
        pass
    
    return info

def create_dataset_entry(video_path: str) -> dict:
    """为单个视频创建数据集条目"""
    video_info = extract_video_info(video_path)
    
    entry = {
        "instruction": "<video>\nWhat is the ego vehicle's behavior in this 20-second video clip?",
        "input": "",
        "output": "",
        "videos": [video_path],
        "system": SYSTEM_PROMPT,
        "metadata": video_info
    }
    
    return entry

def create_dataset(video_files: list, output_path: str) -> None:
    """创建数据集文件"""
    dataset = []
    
    for i, video_path in enumerate(video_files):
        try:
            if not Path(video_path).exists():
                logger.warning(f"视频文件不存在: {video_path}")
                continue
            
            entry = create_dataset_entry(video_path)
            dataset.append(entry)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(video_files)} 个视频")
                
        except Exception as e:
            logger.error(f"处理视频失败 {video_path}: {str(e)}")
            continue
    
    # 保存数据集
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ 数据集创建成功! 文件: {output_path}, 条目数: {len(dataset)}")

def main():
    parser = argparse.ArgumentParser(description='创建用于视频挖掘的JSON数据集')
    parser.add_argument('--video-dir', type=str, required=False,
                       default='/mnt/pfs/houhaotian/junction_videos_segment',
                       help='视频切片目录')
    parser.add_argument('--output', type=str, required=False,
                       default='/mnt/pfs/houhaotian/junction_segemnt_inference_dataset.json',
                       help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    logger.info(f"开始创建推理数据集，视频目录: {args.video_dir}")
    
    # 查找所有切片视频
    video_files = find_segmented_videos(args.video_dir)
    
    if not video_files:
        logger.error("未找到任何切片视频文件")
        return
    
    # 创建数据集
    create_dataset(video_files, args.output)

if __name__ == "__main__":
    main()