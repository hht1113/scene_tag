import os
import json
import random
import time
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from tqdm import tqdm
import traceback
from collections import defaultdict
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/video_vqa_dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 问题模板列表 - 关于自车动作
ENGLISH_QUESTION_TEMPLATES = [
    "What is the ego vehicle's action in the video?",
    "What is the ego vehicle doing in this video clip?",
    "What is the behavior of the ego vehicle?",
    "Please tell me the ego vehicle's action.",
    "What operation is the ego vehicle currently executing?",
    "What is the driving maneuver of the ego vehicle in this video?",
    "Identify the ego vehicle's action in the video.",
    "Describe the behavior of the ego vehicle.",
    "What is the operation of the ego vehicle?",
    "What is the vehicle's action shown in the video?",
    "What action is the ego vehicle executing?",
    "What is the ego vehicle's behavior in this video clip?",
    "Please explain the ego vehicle's action.",
    "What is the driving maneuver of the ego vehicle?",
    "What is the ego vehicle's operation in the video?",
    "What action is the ego vehicle completing in this video?",
    "What is the driving behavior of the ego vehicle?",
    "Please analyze the ego vehicle's action.",
    "What is the ego vehicle's action in the video?",
    "What did the ego vehicle do in the video?"
]

# 单动作答案模板列表
SINGLE_ANSWER_TEMPLATES = [
    "The ego vehicle's behavior from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds is <driving_maneuver>action</driving_maneuver>.",
    "The ego vehicle performs <driving_maneuver>action</driving_maneuver> between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds.",
    "From <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle's action is <driving_maneuver>action</driving_maneuver>.",
    "The ego vehicle exhibits <driving_maneuver>action</driving_maneuver> behavior during <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "The primary action of the ego vehicle is <driving_maneuver>action</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "Between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds, the ego vehicle is <driving_maneuver>action</driving_maneuver>.",
    "During the interval <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle's behavior is <driving_maneuver>action</driving_maneuver>.",
    "The ego vehicle executes <driving_maneuver>action</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "From <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle engages in <driving_maneuver>action</driving_maneuver>.",
    "The ego vehicle's driving maneuver is <driving_maneuver>action</driving_maneuver> between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds."
]

class VideoVQADatasetBuilder:
    """视频VQA数据集构建器（视频粒度，合并多个动作）"""
    
    def __init__(self, annotations_file: str, output_dir: str, train_ratio: float = 0.8, 
                 merge_interval: int = 1):
        """
        初始化数据集构建器
        
        Args:
            annotations_file: 标注文件路径（使用转换后的标注文件）
            output_dir: 输出目录
            train_ratio: 训练集比例
            merge_interval: 合并间隔（秒），相邻动作间隔小于等于此值会被合并
        """
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.merge_interval = merge_interval
        
    def load_all_annotations(self) -> List[Dict]:
        """加载所有标注数据，并进行去重"""
        all_annotations = []
        
        try:
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"从 {self.annotations_file} 加载数据，数据类型: {type(data)}")
            
            # 根据文件格式处理
            if isinstance(data, list):
                all_annotations = data
                logger.info(f"直接加载列表，共 {len(all_annotations)} 个标注")
            elif isinstance(data, dict) and "data" in data:
                all_annotations = data["data"]
                logger.info(f"从data字段加载，共 {len(all_annotations)} 个标注")
            else:
                logger.error(f"标注文件格式不支持: {self.annotations_file}")
                return []
            
            logger.info(f"初始加载了 {len(all_annotations)} 个标注")
            
            # 去重：基于id去重
            seen_ids = set()
            unique_annotations = []
            duplicate_count = 0
            
            for ann in all_annotations:
                ann_id = ann.get('id', '')
                if not ann_id:
                    logger.warning(f"发现没有id的标注: {ann}")
                    unique_annotations.append(ann)  # 没有id的保留
                elif ann_id in seen_ids:
                    duplicate_count += 1
                    logger.debug(f"发现重复标注，id: {ann_id}")
                else:
                    seen_ids.add(ann_id)
                    unique_annotations.append(ann)
            
            if duplicate_count > 0:
                logger.warning(f"发现 {duplicate_count} 个重复标注，已去重")
            
            all_annotations = unique_annotations
            logger.info(f"去重后保留 {len(all_annotations)} 个唯一标注")
            
            # 只保留视频存在的标注
            filtered_annotations = []
            for ann in all_annotations:
                video_exists = ann.get("video_exists", False)
                video_path = ann.get("video_path", "")
                
                if video_exists and video_path and os.path.exists(video_path):
                    filtered_annotations.append(ann)
                else:
                    logger.debug(f"跳过视频不存在的标注: {ann.get('id', 'unknown')}")
            
            logger.info(f"过滤后保留 {len(filtered_annotations)} 个视频存在的标注")
            
            return filtered_annotations
            
        except Exception as e:
            logger.error(f"加载标注文件失败 {self.annotations_file}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def group_by_video(self, annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """按视频路径分组标注，并对每个视频内的标注去重"""
        video_groups = defaultdict(list)
        
        for ann in annotations:
            video_path = ann.get('video_path', '')
            if video_path:
                video_groups[video_path].append(ann)
        
        logger.info(f"按视频分组完成: {len(video_groups)} 个视频")
        
        # 对每个视频内的标注进行去重
        clean_video_groups = {}
        for video_path, anns in video_groups.items():
            # 基于id去重
            seen_ids = set()
            unique_anns = []
            
            for ann in anns:
                ann_id = ann.get('id', '')
                if ann_id in seen_ids:
                    logger.warning(f"视频 {os.path.basename(video_path)} 中有重复标注: {ann_id}")
                else:
                    seen_ids.add(ann_id)
                    unique_anns.append(ann)
            
            if len(anns) != len(unique_anns):
                logger.info(f"视频 {os.path.basename(video_path)}: {len(anns)} -> {len(unique_anns)} 个标注")
            
            clean_video_groups[video_path] = unique_anns
        
        # 统计每个视频的标注数量
        for video_path, anns in list(clean_video_groups.items())[:5]:  # 显示前5个
            logger.info(f"视频: {os.path.basename(video_path)}, 标注数: {len(anns)}")
            if len(anns) > 1:
                # 检查是否有完全相同的标注
                for i in range(len(anns)):
                    for j in range(i+1, len(anns)):
                        if (anns[i].get('label_en') == anns[j].get('label_en') and
                            anns[i].get('time_range') == anns[j].get('time_range')):
                            logger.warning(f"视频 {os.path.basename(video_path)} 中有完全相同标注: {anns[i].get('id')}")
        
        return clean_video_groups
    
    def remove_duplicate_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """移除重复的标注（基于标签和时间范围）"""
        if not annotations:
            return []
        
        seen = set()
        unique_annotations = []
        
        for ann in annotations:
            label_en = ann.get('label_en', '')
            time_range = tuple(ann.get('time_range', []))
            ann_id = ann.get('id', '')
            
            # 创建唯一标识
            key = (label_en, time_range, ann_id)
            
            if key in seen:
                logger.debug(f"移除重复标注: {ann_id} - {label_en} - {time_range}")
            else:
                seen.add(key)
                unique_annotations.append(ann)
        
        if len(annotations) != len(unique_annotations):
            logger.info(f"去重: {len(annotations)} -> {len(unique_annotations)}")
        return unique_annotations
    
    def merge_overlapping_actions(self, annotations: List[Dict]) -> List[Dict]:
        """
        合并重叠或相邻的相同动作
        
        合并条件：
        1. 相同标签的动作
        2. 时间范围重叠或相邻（间隔小于等于merge_interval秒）
        3. 合并后的时间范围取最早开始时间和最晚结束时间
        """
        if not annotations:
            return []
        
        # 按标签分组
        label_groups = defaultdict(list)
        for ann in annotations:
            label = ann.get('label_en', '')
            if label:
                label_groups[label].append(ann)
        
        merged_annotations = []
        
        for label, label_anns in label_groups.items():
            if len(label_anns) == 1:
                # 只有一个动作，直接添加
                merged_annotations.append(label_anns[0])
                continue
            
            # 按开始时间排序
            sorted_anns = sorted(label_anns, key=lambda x: x.get('time_range', [0])[0])
            
            # 合并重叠或相邻的时间区间
            merged_ranges = []
            current_range = None
            current_anns = []
            
            for ann in sorted_anns:
                time_range = ann.get('time_range', [])
                if len(time_range) < 2:
                    continue
                
                start_time = time_range[0]
                end_time = time_range[1]
                
                if current_range is None:
                    # 第一个区间
                    current_range = [start_time, end_time]
                    current_anns = [ann]
                else:
                    # 检查是否重叠或相邻
                    if start_time <= current_range[1] + self.merge_interval:
                        # 重叠或相邻，合并
                        current_range[1] = max(current_range[1], end_time)
                        current_anns.append(ann)
                    else:
                        # 不重叠，保存当前区间，开始新的区间
                        if current_range:
                            # 创建合并后的标注
                            merged_ann = self._create_merged_annotation(current_anns, current_range)
                            merged_annotations.append(merged_ann)
                        
                        current_range = [start_time, end_time]
                        current_anns = [ann]
            
            # 处理最后一个区间
            if current_range and current_anns:
                merged_ann = self._create_merged_annotation(current_anns, current_range)
                merged_annotations.append(merged_ann)
        
        if len(annotations) != len(merged_annotations):
            logger.info(f"合并动作: {len(annotations)} -> {len(merged_annotations)}")
            for ann in merged_annotations:
                if 'merged_from' in ann:
                    logger.debug(f"合并动作: {ann['label_en']} {ann['time_range']} 来自 {len(ann['merged_from'])} 个标注")
        
        return merged_annotations
    
    def _create_merged_annotation(self, original_anns: List[Dict], merged_range: List[int]) -> Dict:
        """创建合并后的标注"""
        if not original_anns:
            return None
        
        # 使用第一个标注作为基础
        base_ann = original_anns[0].copy()
        
        # 更新时间范围
        base_ann['time_range'] = merged_range
        base_ann['duration'] = merged_range[1] - merged_range[0]
        
        # 记录合并信息
        base_ann['merged_from'] = [
            {
                'id': ann.get('id', ''),
                'time_range': ann.get('time_range', []),
                'duration': ann.get('duration', 0)
            }
            for ann in original_anns
        ]
        
        # 更新ID
        base_ann['id'] = f"merged_{len(original_anns)}_{hash(tuple(merged_range)) % 10000:04d}"
        
        return base_ann
    
    def generate_single_action_description(self, action: Dict) -> str:
        """生成单个动作的描述"""
        label_en = action.get('label_en', '')
        time_range = action.get('time_range', [])
        
        if not label_en or len(time_range) < 2:
            return ""
        
        # 获取时间范围
        start_time = int(time_range[0])
        end_time = int(time_range[1])
        
        # 随机选择一个单动作模板
        template = random.choice(SINGLE_ANSWER_TEMPLATES)
        
        # 替换模板中的标签
        description = template.replace(
            "<start_time>start_time_value</start_time>", 
            f"<start_time>{start_time}</start_time>"
        ).replace(
            "<end_time>end_time_value</end_time>", 
            f"<end_time>{end_time}</end_time>"
        ).replace(
            "<driving_maneuver>action</driving_maneuver>", 
            f"<driving_maneuver>{label_en}</driving_maneuver>"
        )
        
        return description
    
    def merge_actions_for_video(self, video_annotations: List[Dict]) -> Dict:
        """合并同一视频的多个动作为一个综合描述，先进行去重和合并"""
        if not video_annotations:
            return None
        
        # 先去重
        unique_annotations = self.remove_duplicate_annotations(video_annotations)
        if not unique_annotations:
            logger.warning(f"去重后没有标注")
            return None
        
        # 合并重叠或相邻的相同动作
        merged_annotations = self.merge_overlapping_actions(unique_annotations)
        if not merged_annotations:
            logger.warning(f"合并后没有标注")
            return None
        
        # 按开始时间排序
        sorted_annotations = sorted(merged_annotations, 
                                   key=lambda x: x.get('time_range', [0])[0])
        
        video_path = sorted_annotations[0].get('video_path', '')
        video_exists = sorted_annotations[0].get('video_exists', False)
        
        if not video_path or not video_exists:
            return None
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            logger.warning(f"视频文件不存在: {video_path}")
            return None
        
        # 获取视频时长（从标注中获取）
        durations = [ann.get('duration', 0) for ann in sorted_annotations]
        if durations:
            video_duration = max(durations)  # 使用最大的持续时间
        else:
            video_duration = 60  # 默认60秒
        
        # 生成问题
        question = random.choice(ENGLISH_QUESTION_TEMPLATES)
        
        # 生成每个动作的描述
        action_descriptions = []
        for ann in sorted_annotations:
            description = self.generate_single_action_description(ann)
            if description:
                action_descriptions.append(description)
        
        if not action_descriptions:
            logger.warning(f"无法为视频生成动作描述: {video_path}")
            return None
        
        # 连接所有动作描述
        if len(action_descriptions) == 1:
            answer = action_descriptions[0]
        else:
            # 随机选择连接方式
            connector = random.choice(["; ", " and "])
            answer = connector.join(action_descriptions)
        
        # 获取所有标签
        all_labels = []
        for ann in sorted_annotations:
            label_en = ann.get('label_en', '')
            if label_en:
                all_labels.append(label_en)
        
        # 获取所有标注的详细信息
        annotations_info = []
        for ann in sorted_annotations:
            time_range = ann.get('time_range', [])
            if len(time_range) >= 2:
                start_time = int(time_range[0])
                end_time = int(time_range[1])
            else:
                start_time = 0
                end_time = 0
            
            annotation_info = {
                "label_en": ann.get('label_en', ''),
                "label_zh": ann.get('label_zh', ''),
                "time_range_seconds": time_range,
                "time_range_frames": [start_time, end_time],  # 帧数等于秒数
                "duration_seconds": ann.get('duration', 0),
                "original_annotation_id": ann.get('id', '')
            }
            
            # 添加合并信息
            if 'merged_from' in ann:
                annotation_info['merged_from'] = ann['merged_from']
                annotation_info['merged_count'] = len(ann['merged_from'])
            
            annotations_info.append(annotation_info)
        
        # 计算主要标签（出现次数最多的标签）
        if all_labels:
            from collections import Counter
            label_counter = Counter(all_labels)
            primary_label = label_counter.most_common(1)[0][0]
        else:
            primary_label = ""
        
        # 获取视频文件大小
        try:
            file_size = os.path.getsize(video_path)
            file_size_mb = file_size / (1024 * 1024)
        except:
            file_size = 0
            file_size_mb = 0
        
        return {
            "id": f"video_{len(sorted_annotations)}_{hash(video_path) % 10000:04d}",
            "video_path": video_path,
            "video_filename": os.path.basename(video_path),
            "video_exists": True,
            "video_duration": video_duration,
            "video_size": file_size,
            "video_size_mb": file_size_mb,
            "question": question,
            "answer": answer,
            "primary_label": primary_label,
            "all_labels": list(set(all_labels)),
            "num_actions": len(sorted_annotations),
            "merged_actions_count": sum(1 for ann in sorted_annotations if 'merged_from' in ann),
            "total_original_actions": len(video_annotations),
            "annotations": annotations_info,
            "video_info": {
                "video_duration": video_duration,
                "total_frames": video_duration,  # 每秒1帧
                "has_multiple_actions": len(sorted_annotations) > 1
            }
        }
    
    def process_video_groups(self, video_groups: Dict[str, List[Dict]]) -> List[Dict]:
        """处理所有视频组，生成视频粒度的数据集"""
        video_samples = []
        skipped_videos = 0
        
        for video_path, annotations in tqdm(video_groups.items(), desc="处理视频"):
            # 检查标注数量
            if len(annotations) > 10:
                logger.warning(f"视频 {os.path.basename(video_path)} 有 {len(annotations)} 个标注，可能存在重复或需要合并")
            
            # 合并同一视频的所有动作
            video_sample = self.merge_actions_for_video(annotations)
            
            if video_sample:
                video_samples.append(video_sample)
            else:
                skipped_videos += 1
        
        logger.info(f"生成了 {len(video_samples)} 个视频样本，跳过了 {skipped_videos} 个视频")
        
        # 检查是否有重复的样本
        video_paths = set()
        duplicate_samples = 0
        for sample in video_samples:
            video_path = sample.get('video_path', '')
            if video_path in video_paths:
                duplicate_samples += 1
                logger.warning(f"发现重复视频样本: {video_path}")
            else:
                video_paths.add(video_path)
        
        if duplicate_samples > 0:
            logger.warning(f"发现 {duplicate_samples} 个重复视频样本")
        
        return video_samples
    
    def split_by_category(self, video_samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """按类别划分训练集和测试集（每个类别80%训练，20%测试）"""
        if not video_samples:
            return [], []
        
        # 按主要标签分组
        category_groups = defaultdict(list)
        for sample in video_samples:
            primary_label = sample.get('primary_label', 'unknown')
            category_groups[primary_label].append(sample)
        
        train_data = []
        test_data = []
        
        # 对每个类别进行划分
        for category, items in category_groups.items():
            if len(items) < 2:  # 如果类别样本太少，全部放入训练集
                train_data.extend(items)
                logger.warning(f"类别 '{category}' 只有 {len(items)} 个样本，全部放入训练集")
                continue
            
            # 打乱顺序
            random.shuffle(items)
            
            # 计算分割点
            split_idx = int(len(items) * self.train_ratio)
            
            if split_idx == 0:  # 确保训练集至少有一个样本
                split_idx = 1
            
            train_data.extend(items[:split_idx])
            test_data.extend(items[split_idx:])
            
            logger.info(f"类别 '{category}': {len(items)}个样本 -> 训练{len(items[:split_idx])}, 测试{len(items[split_idx:])}")
        
        # 再次打乱
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        logger.info(f"总体划分: 训练集{len(train_data)}个视频样本, 测试集{len(test_data)}个视频样本")
        return train_data, test_data
    
    def save_dataset(self, train_data: List[Dict], test_data: List[Dict]):
        """保存数据集"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"video_vqa_dataset_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        
        # 1. 保存训练集
        train_file = os.path.join(output_path, "train.json")
        train_dataset = {
            "version": "1.0.0",
            "description": "Video VQA Training Dataset (Video-level, multiple actions merged)",
            "created": datetime.now().isoformat(),
            "config": {
                "merge_interval": self.merge_interval,
                "train_ratio": self.train_ratio
            },
            "statistics": {
                "total_samples": len(train_data),
                "categories_count": len(set([item.get('primary_label', '') for item in train_data])),
                "total_actions": sum([item.get('num_actions', 0) for item in train_data]),
                "merged_actions": sum([item.get('merged_actions_count', 0) for item in train_data]),
                "avg_actions_per_video": sum([item.get('num_actions', 0) for item in train_data]) / len(train_data) if train_data else 0
            },
            "data": train_data
        }
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"保存训练集: {train_file} ({len(train_data)} 个样本)")
        
        # 2. 保存测试集
        test_file = os.path.join(output_path, "test.json")
        test_dataset = {
            "version": "1.0.0",
            "description": "Video VQA Test Dataset (Video-level, multiple actions merged)",
            "created": datetime.now().isoformat(),
            "config": {
                "merge_interval": self.merge_interval,
                "train_ratio": self.train_ratio
            },
            "statistics": {
                "total_samples": len(test_data),
                "categories_count": len(set([item.get('primary_label', '') for item in test_data])),
                "total_actions": sum([item.get('num_actions', 0) for item in test_data]),
                "merged_actions": sum([item.get('merged_actions_count', 0) for item in test_data]),
                "avg_actions_per_video": sum([item.get('num_actions', 0) for item in test_data]) / len(test_data) if test_data else 0
            },
            "data": test_data
        }
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"保存测试集: {test_file} ({len(test_data)} 个样本)")
        
        # 3. 保存完整数据集
        all_data = train_data + test_data
        all_file = os.path.join(output_path, "all_data.json")
        all_dataset = {
            "version": "1.0.0",
            "description": "Video VQA Complete Dataset (Video-level, multiple actions merged)",
            "created": datetime.now().isoformat(),
            "config": {
                "merge_interval": self.merge_interval,
                "train_ratio": self.train_ratio
            },
            "statistics": {
                "total_samples": len(all_data),
                "train_samples": len(train_data),
                "test_samples": len(test_data),
                "train_ratio": self.train_ratio,
                "categories_count": len(set([item.get('primary_label', '') for item in all_data])),
                "total_actions": sum([item.get('num_actions', 0) for item in all_data]),
                "merged_actions": sum([item.get('merged_actions_count', 0) for item in all_data]),
                "avg_actions_per_video": sum([item.get('num_actions', 0) for item in all_data]) / len(all_data) if all_data else 0
            },
            "data": all_data
        }
        
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(all_dataset, f, ensure_ascii=False, indent=2)
        logger.info(f"保存完整数据集: {all_file} ({len(all_data)} 个样本)")
        
        # 4. 保存统计信息
        stats = self.calculate_statistics(train_data, test_data)
        stats_file = os.path.join(output_path, "statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 5. 保存类别信息
        categories = self.extract_category_info(all_data)
        categories_file = os.path.join(output_path, "categories.json")
        with open(categories_file, 'w', encoding='utf-8') as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据集已保存到: {output_path}")
        return output_path, stats
    
    def calculate_statistics(self, train_data: List[Dict], test_data: List[Dict]) -> Dict:
        """计算数据集统计信息"""
        # 训练集统计
        train_videos = set()
        train_categories = defaultdict(int)
        train_actions_counts = []
        train_merged_counts = []
        train_original_actions_counts = []
        
        for item in train_data:
            video_path = item.get('video_path', '')
            if video_path:
                train_videos.add(video_path)
            
            primary_label = item.get('primary_label', 'unknown')
            train_categories[primary_label] += 1
            
            num_actions = item.get('num_actions', 0)
            train_actions_counts.append(num_actions)
            
            merged_actions = item.get('merged_actions_count', 0)
            train_merged_counts.append(merged_actions)
            
            total_original = item.get('total_original_actions', 0)
            train_original_actions_counts.append(total_original)
        
        # 测试集统计
        test_videos = set()
        test_categories = defaultdict(int)
        test_actions_counts = []
        test_merged_counts = []
        test_original_actions_counts = []
        
        for item in test_data:
            video_path = item.get('video_path', '')
            if video_path:
                test_videos.add(video_path)
            
            primary_label = item.get('primary_label', 'unknown')
            test_categories[primary_label] += 1
            
            num_actions = item.get('num_actions', 0)
            test_actions_counts.append(num_actions)
            
            merged_actions = item.get('merged_actions_count', 0)
            test_merged_counts.append(merged_actions)
            
            total_original = item.get('total_original_actions', 0)
            test_original_actions_counts.append(total_original)
        
        # 计算平均动作数量
        avg_train_actions = sum(train_actions_counts) / len(train_actions_counts) if train_actions_counts else 0
        avg_test_actions = sum(test_actions_counts) / len(test_actions_counts) if test_actions_counts else 0
        
        # 计算合并统计
        total_merged_train = sum(train_merged_counts)
        total_merged_test = sum(test_merged_counts)
        total_original_train = sum(train_original_actions_counts)
        total_original_test = sum(test_original_actions_counts)
        
        # 计算视频时长统计
        train_durations = [item.get('video_duration', 0) for item in train_data]
        test_durations = [item.get('video_duration', 0) for item in test_data]
        
        stats = {
            "dataset_info": {
                "total_videos": len(train_data) + len(test_data),
                "train_videos": len(train_data),
                "test_videos": len(test_data),
                "train_ratio": self.train_ratio,
                "merge_interval": self.merge_interval,
                "generation_time": datetime.now().isoformat()
            },
            "video_info": {
                "unique_videos_train": len(train_videos),
                "unique_videos_test": len(test_videos),
                "unique_videos_total": len(train_videos.union(test_videos)),
                "avg_video_duration_train": sum(train_durations) / len(train_durations) if train_durations else 0,
                "avg_video_duration_test": sum(test_durations) / len(test_durations) if test_durations else 0,
                "max_video_duration_train": max(train_durations) if train_durations else 0,
                "max_video_duration_test": max(test_durations) if test_durations else 0
            },
            "category_info": {
                "total_categories": len(set(list(train_categories.keys()) + list(test_categories.keys()))),
                "train_categories": dict(sorted(train_categories.items(), key=lambda x: x[1], reverse=True)),
                "test_categories": dict(sorted(test_categories.items(), key=lambda x: x[1], reverse=True))
            },
            "action_info": {
                "avg_actions_per_video_train": avg_train_actions,
                "avg_actions_per_video_test": avg_test_actions,
                "max_actions_train": max(train_actions_counts) if train_actions_counts else 0,
                "max_actions_test": max(test_actions_counts) if test_actions_counts else 0,
                "min_actions_train": min(train_actions_counts) if train_actions_counts else 0,
                "min_actions_test": min(test_actions_counts) if test_actions_counts else 0,
                "total_actions_train": sum(train_actions_counts),
                "total_actions_test": sum(test_actions_counts)
            },
            "merge_info": {
                "total_merged_actions_train": total_merged_train,
                "total_merged_actions_test": total_merged_test,
                "total_original_actions_train": total_original_train,
                "total_original_actions_test": total_original_test,
                "compression_rate_train": (total_original_train - sum(train_actions_counts)) / total_original_train if total_original_train > 0 else 0,
                "compression_rate_test": (total_original_test - sum(test_actions_counts)) / total_original_test if total_original_test > 0 else 0
            },
            "generation_info": {
                "question_templates": len(ENGLISH_QUESTION_TEMPLATES),
                "answer_templates": len(SINGLE_ANSWER_TEMPLATES)
            }
        }
        
        return stats
    
    def extract_category_info(self, all_data: List[Dict]) -> Dict:
        """提取类别信息"""
        categories = {}
        
        for item in all_data:
            primary_label = item.get('primary_label', '')
            all_labels = item.get('all_labels', [])
            
            if not primary_label:
                continue
                
            if primary_label not in categories:
                categories[primary_label] = {
                    "label": primary_label,
                    "count": 0,
                    "all_labels_in_category": set(),
                    "example_questions": set(),
                    "example_answers": set(),
                    "videos": []
                }
            
            categories[primary_label]["count"] += 1
            
            # 添加所有标签
            for label in all_labels:
                categories[primary_label]["all_labels_in_category"].add(label)
            
            # 添加示例问题和答案
            categories[primary_label]["example_questions"].add(item.get('question', ''))
            categories[primary_label]["example_answers"].add(item.get('answer', ''))
            
            # 添加视频信息
            video_info = {
                "id": item.get('id', ''),
                "video_filename": item.get('video_filename', ''),
                "num_actions": item.get('num_actions', 0),
                "merged_actions": item.get('merged_actions_count', 0)
            }
            categories[primary_label]["videos"].append(video_info)
        
        # 转换set为list
        for cat in categories.values():
            cat["all_labels_in_category"] = list(cat["all_labels_in_category"])
            cat["example_questions"] = list(cat["example_questions"])[:3]  # 只保留3个示例问题
            cat["example_answers"] = list(cat["example_answers"])[:3]  # 只保留3个示例答案
        
        return categories
    
    def generate_sample_output(self, train_data: List[Dict], test_data: List[Dict], output_path: str):
        """生成样本输出文件，用于查看格式"""
        samples_file = os.path.join(output_path, "samples.json")
        
        samples = {
            "train_samples": train_data[:2] if len(train_data) >= 2 else train_data,
            "test_samples": test_data[:2] if len(test_data) >= 2 else test_data
        }
        
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"样本文件已保存: {samples_file}")
        
        # 在控制台显示样本
        print("\n" + "=" * 60)
        print("📋 数据集样本示例")
        print("=" * 60)
        
        if train_data:
            print("\n训练集样本 (前2个):")
            for i, sample in enumerate(train_data[:2], 1):
                print(f"\n样本 {i}:")
                print(f"  ID: {sample.get('id', 'N/A')}")
                print(f"  视频: {sample.get('video_filename', 'N/A')}")
                print(f"  视频路径: {sample.get('video_path', 'N/A')[:80]}...")
                print(f"  视频时长: {sample.get('video_duration', 'N/A')}秒")
                print(f"  视频大小: {sample.get('video_size_mb', 0):.1f} MB")
                print(f"  问题: {sample.get('question', 'N/A')}")
                print(f"  答案: {sample.get('answer', 'N/A')}")
                print(f"  主要标签: {sample.get('primary_label', 'N/A')}")
                print(f"  所有标签: {sample.get('all_labels', [])}")
                print(f"  动作数量: {sample.get('num_actions', 0)}")
                print(f"  合并动作数: {sample.get('merged_actions_count', 0)}")
                print(f"  原始标注数: {sample.get('total_original_actions', 0)}")
                
                # 检查是否有重复动作
                annotations = sample.get('annotations', [])
                if annotations:
                    # 检查合并信息
                    for j, ann in enumerate(annotations[:3], 1):  # 显示前3个标注
                        label = ann.get('label_en', 'unknown')
                        time_range = ann.get('time_range_frames', [0, 0])
                        merged_count = ann.get('merged_count', 0)
                        if merged_count > 0:
                            print(f"    {j}. {label}: {time_range[0]}-{time_range[1]}秒 (合并了{merged_count}个标注)")
                        else:
                            print(f"    {j}. {label}: {time_range[0]}-{time_range[1]}秒")
                    if len(annotations) > 3:
                        print(f"    ... 还有 {len(annotations) - 3} 个标注")
        
        if test_data:
            print(f"\n测试集样本 (前2个):")
            for i, sample in enumerate(test_data[:2], 1):
                print(f"\n样本 {i}:")
                print(f"  ID: {sample.get('id', 'N/A')}")
                print(f"  视频: {sample.get('video_filename', 'N/A')}")
                print(f"  视频路径: {sample.get('video_path', 'N/A')[:80]}...")
                print(f"  视频时长: {sample.get('video_duration', 'N/A')}秒")
                print(f"  视频大小: {sample.get('video_size_mb', 0):.1f} MB")
                print(f"  问题: {sample.get('question', 'N/A')}")
                print(f"  答案: {sample.get('answer', 'N/A')}")
                print(f"  主要标签: {sample.get('primary_label', 'N/A')}")
                print(f"  所有标签: {sample.get('all_labels', [])}")
                print(f"  动作数量: {sample.get('num_actions', 0)}")
                print(f"  合并动作数: {sample.get('merged_actions_count', 0)}")
                print(f"  原始标注数: {sample.get('total_original_actions', 0)}")
        
        print("=" * 60)

def main():
    """主函数"""
    # 使用转换后的标注文件
    # 注意：这里使用您之前代码生成的 existing_videos_dataset.json
    # 如果您有不同的文件，请修改这个路径
    ANNOTATIONS_FILE = "/root/workspace/vqa_dataset_prepared/converted_annotations/existing_videos_dataset.json"
    OUTPUT_DIR = "/root/workspace/video_vqa_dataset"
    
    print("=" * 60)
    print("视频VQA数据集生成工具（视频粒度，合并动作）- 增强去重和合并版")
    print("=" * 60)
    print(f"📁 标注文件: {ANNOTATIONS_FILE}")
    print(f"📦 输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    print("📋 功能说明:")
    print("  - 从转换后的标注文件生成视频粒度的VQA数据集")
    print("  - 合并同一视频的多个动作为一个综合答案")
    print("  - 每个动作都有独立的开始和结束时间（秒）")
    print("  - 使用双边闭合标签<xxx>目标内容</xxx>")
    print("  - 适应视频，每秒1帧处理")
    print("  - 按类别80%训练集、20%测试集划分")
    print("  - 只使用视频存在的标注")
    print("  - 多动作用分号或and连接")
    print("  - 增强去重功能，避免重复动作")
    print("  - 合并重叠或相邻的相同动作")
    print("=" * 60)
    
    # 检查标注文件
    if not os.path.exists(ANNOTATIONS_FILE):
        logger.error(f"标注文件不存在: {ANNOTATIONS_FILE}")
        print(f"\n❌ 错误: 标注文件不存在: {ANNOTATIONS_FILE}")
        print("请先运行标签转换脚本生成标注文件")
        return
    
    # 检查文件是否为空
    if os.path.getsize(ANNOTATIONS_FILE) == 0:
        logger.error(f"标注文件为空: {ANNOTATIONS_FILE}")
        print(f"\n❌ 错误: 标注文件为空: {ANNOTATIONS_FILE}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化数据集构建器
    builder = VideoVQADatasetBuilder(
        annotations_file=ANNOTATIONS_FILE,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,
        merge_interval=1  # 相邻1秒内的相同动作会合并
    )
    
    # 加载所有标注
    all_annotations = builder.load_all_annotations()
    if not all_annotations:
        logger.error("没有找到标注数据")
        print("\n❌ 错误: 没有找到标注数据")
        print("请检查标注文件格式是否正确")
        return
    
    # 按视频分组
    video_groups = builder.group_by_video(all_annotations)
    if not video_groups:
        logger.error("没有找到有效的视频标注")
        print("\n❌ 错误: 没有找到有效的视频标注")
        print("请确保标注文件中包含有效的视频路径")
        return
    
    # 处理视频组，生成视频粒度的样本
    video_samples = builder.process_video_groups(video_groups)
    if not video_samples:
        logger.error("没有生成有效的视频样本")
        print("\n❌ 错误: 没有生成有效的视频样本")
        print("请检查标注数据是否包含有效的时间范围和标签")
        return
    
    # 按类别划分训练集和测试集
    train_data, test_data = builder.split_by_category(video_samples)
    
    if not train_data and not test_data:
        logger.error("无法划分训练集和测试集")
        print("\n❌ 错误: 无法划分训练集和测试集")
        return
    
    # 保存数据集
    output_path, stats = builder.save_dataset(train_data, test_data)
    
    # 生成样本输出
    builder.generate_sample_output(train_data, test_data, output_path)
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("🎉 数据集生成完成")
    print("=" * 60)
    print(f"📊 数据集统计:")
    print(f"  ✅ 总视频数: {stats['dataset_info']['total_videos']}")
    print(f"  📚 训练集: {stats['dataset_info']['train_videos']} 个视频")
    print(f"  📊 测试集: {stats['dataset_info']['test_videos']} 个视频")
    print(f"  🎯 训练比例: {stats['dataset_info']['train_ratio'] * 100}%")
    print(f"  🔄 合并间隔: {stats['dataset_info']['merge_interval']} 秒")
    
    print(f"\n📹 视频统计:")
    print(f"  🎬 唯一视频数: {stats['video_info']['unique_videos_total']}")
    print(f"  🎯 训练集视频: {stats['video_info']['unique_videos_train']}")
    print(f"  📊 测试集视频: {stats['video_info']['unique_videos_test']}")
    print(f"  ⏱️  平均视频时长: {stats['video_info']['avg_video_duration_train']:.1f}秒 (训练集)")
    
    print(f"\n🏷️  类别统计:")
    print(f"  📂 总类别数: {stats['category_info']['total_categories']}")
    print(f"  🎯 训练集前5个类别:")
    for i, (category, count) in enumerate(list(stats['category_info']['train_categories'].items())[:5], 1):
        print(f"     {i}. {category}: {count} 个视频")
    
    print(f"\n🎬 动作统计:")
    print(f"  📈 训练集平均动作数/视频: {stats['action_info']['avg_actions_per_video_train']:.2f}")
    print(f"  📈 测试集平均动作数/视频: {stats['action_info']['avg_actions_per_video_test']:.2f}")
    print(f"  📊 训练集最大动作数: {stats['action_info']['max_actions_train']}")
    print(f"  📊 测试集最大动作数: {stats['action_info']['max_actions_test']}")
    
    print(f"\n🔄 合并统计:")
    print(f"  📉 训练集合并动作数: {stats['merge_info']['total_merged_actions_train']}")
    print(f"  📉 测试集合并动作数: {stats['merge_info']['total_merged_actions_test']}")
    print(f"  📈 训练集原始标注数: {stats['merge_info']['total_original_actions_train']}")
    print(f"  📈 测试集原始标注数: {stats['merge_info']['total_original_actions_test']}")
    print(f"  📊 训练集压缩率: {stats['merge_info']['compression_rate_train']:.2%}")
    print(f"  📊 测试集压缩率: {stats['merge_info']['compression_rate_test']:.2%}")
    
    print(f"\n📁 输出目录: {output_path}")
    print("生成的文件:")
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  📄 {file} ({size_kb:.1f} KB)")
    
    print(f"\n🚀 下一步建议:")
    print("1. 检查生成的数据集格式是否正确")
    print("2. 查看samples.json文件了解数据格式")
    print("3. 在微调脚本中，使用视频路径和开始结束时间进行抽帧")
    print("4. 使用训练集训练视频VQA模型")
    print("5. 使用测试集评估模型性能")
    
    print("=" * 60)
    
    # 显示特殊标记使用说明
    print("\n🔤 特殊标记说明:")
    print("  <start_time>起始时间</start_time>: 动作起始时间（秒）")
    print("  <end_time>结束时间</end_time>: 动作结束时间（秒）")
    print("  <driving_maneuver>动作标签</driving_maneuver>: 驾驶动作标签")
    print("\n📝 示例问题-答案对:")
    if train_data:
        sample = train_data[0]
        print(f"\n  问题: {sample.get('question', '')}")
        print(f"  答案: {sample.get('answer', '')}")
        
        # 显示多动作示例
        if sample.get('num_actions', 0) > 1:
            print(f"\n  🔄 多动作示例解析:")
            annotations = sample.get('annotations', [])
            for i, ann in enumerate(annotations, 1):
                label = ann.get('label_en', 'unknown')
                time_range = ann.get('time_range_frames', [0, 0])
                merged_count = ann.get('merged_count', 0)
                if merged_count > 0:
                    print(f"    动作{i}: {label} ({time_range[0]}-{time_range[1]}秒, 合并了{merged_count}个标注)")
                else:
                    print(f"    动作{i}: {label} ({time_range[0]}-{time_range[1]}秒)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()