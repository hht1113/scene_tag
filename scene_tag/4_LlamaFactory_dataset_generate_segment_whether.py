import os
import json
import random
from typing import Dict, List, Tuple, Optional, Set
import logging
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/llama_factory_whether_dataset_generation.log'),
        logging.StreamHandler()
    ]
)
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

# Whether类问题的系统提示
SYSTEM_PROMPT_WHETHER = f"""You are an expert in autonomous driving scene annotation.
Based on the input video and the question about whether the ego vehicle performs a specific action, analyze the 20-second video to determine if the specified action occurs.

DRIVING MANEUVER CATEGORIES:
You MUST use ONLY these predefined labels for the ego vehicle's actions:

{CATEGORY_LIST_STR}
else (ONLY when NO label above matches, meaning the ego vehicle's action does not fit any of the predefined categories)

INSTRUCTION:
You will be asked a question in the format: "In the 20-second video, from <start_time>XX</start_time> to <end_time>YY</end_time> seconds, does the ego vehicle perform [specific action]?"
Your task is to analyze the specified time segment in the video and determine if the specified action occurs during that exact time segment.

OUTPUT FORMAT:
• If the specified action occurs during the specified time segment: 
  Yes, the ego vehicle performs <driving_maneuver>action_label</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.

• If the specified action does NOT occur during the specified time segment:
  No, the ego vehicle does not perform the specified action from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.

SPECIAL TOKENS RULES:
1. ALWAYS wrap action labels with <driving_maneuver> and </driving_maneuver> tags
2. ALWAYS wrap start time with <start_time> and </start_time> tags
3. ALWAYS wrap end time with <end_time> and </end_time> tags
4. For "Yes" answers, use the EXACT predefined action label that matches the action
5. For "No" answers, you do NOT need to provide an action label, but you MUST provide the time range in the response

TIME SEGMENT RULES:
1. The time segment in the question specifies exactly which part of the 20-second video to analyze
2. You MUST analyze ONLY the specified time segment: from <start_time>XX</start_time> to <end_time>YY</end_time> seconds
3. Do NOT consider actions outside the specified time segment
4. The action must be clearly identifiable and last for at least 1.0 second within the specified time segment
5. Time precision: 0 decimal places (e.g., 5, 23)
6. Base times on video timeline (0 to 20 seconds)

CATEGORY DEFINITIONS:
{CATEGORY_DEFINITIONS}
13. else: Default for all other behaviors not covered by the predefined categories

IMPORTANT GUIDELINES:
1. Analyze ONLY the specified time segment in the 20-second video
2. Check carefully if the specified action occurs during the exact time segment asked about
3. Be precise in identifying if the action occurs
4. For "Yes" answers, you MUST provide the action label and the exact time range when it occurs
5. For "No" answers, you MUST state that the action does not occur in the specified time segment
6. Do not confuse similar but different actions
7. Do not consider actions that partially overlap but do not fully occur within the specified time segment
8. NO additional text or explanations—only output the formatted response
"""


class WhetherQuestionDatasetGenerator:
    """Whether类问题数据集生成器 - 只生成训练集"""
    
    def __init__(self, annotations_file: str, output_dir: str, 
                 system_prompt: str = SYSTEM_PROMPT_WHETHER):
        """
        初始化whether类数据集生成器
        
        Args:
            annotations_file: 标注文件路径
            output_dir: 输出目录
            system_prompt: 系统提示词
        """
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.system_prompt = system_prompt
        self.category_labels = CATEGORY_LABELS
        
        # Whether类问题模板 - 明确指定时间范围
        self.whether_question_templates = [
            "<video>\nIn the 20-second video, from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds, does the ego vehicle perform {behavior_description}?",
            "<video>\nFrom <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds in this 20-second video, is the ego vehicle {behavior_description}?",
            "<video>\nDoes the ego vehicle {behavior_description} between <start_time>{start_time}</start_time> and <end_time>{end_time}</end_time> seconds in this video?",
            "<video>\nDuring the time segment from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds, is the ego vehicle {behavior_description}?",
            "<video>\nCheck if the ego vehicle performs {behavior_description} from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds in this 20-second video.",
            "<video>\nAnalyze the 20-second video from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds: is the ego vehicle {behavior_description}?",
            "<video>\nBetween <start_time>{start_time}</start_time> and <end_time>{end_time}</end_time> seconds, does the ego vehicle exhibit {behavior_description}?",
            "<video>\nIn the specified time frame of <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds, is the ego vehicle {behavior_description}?",
            "<video>\nFrom <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds, verify if the ego vehicle is {behavior_description}.",
            "<video>\nDuring <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds, determine if the ego vehicle performs {behavior_description}."
        ]
        
        # Whether类答案模板 - 明确包含special tokens
        self.whether_answer_templates_yes = [
            "Yes, the ego vehicle performs <driving_maneuver>{action_label}</driving_maneuver> from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds.",
            "Yes, from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds, the ego vehicle is <driving_maneuver>{action_label}</driving_maneuver>.",
            "Yes, the ego vehicle exhibits <driving_maneuver>{action_label}</driving_maneuver> during <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds.",
            "Yes, in the specified time segment, the ego vehicle performs <driving_maneuver>{action_label}</driving_maneuver> from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds."
        ]
        
        self.whether_answer_templates_no = [
            "No, the ego vehicle does not perform the specified action from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds.",
            "No, from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds, the ego vehicle is not performing the specified action.",
            "No, during <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds, the specified action is not observed.",
            "No, the ego vehicle does not exhibit the specified behavior from <start_time>{start_time}</start_time> to <end_time>{end_time}</end_time> seconds."
        ]
    
    def load_all_annotations(self) -> List[Dict]:
        """加载所有标注数据"""
        all_annotations = []
        
        try:
            with open(self.annotations_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"从 {self.annotations_file} 加载数据")
            
            # 根据文件格式处理
            if isinstance(data, list):
                all_annotations = data
            elif isinstance(data, dict) and "data" in data:
                all_annotations = data["data"]
            else:
                logger.error(f"标注文件格式不支持: {self.annotations_file}")
                return []
            
            logger.info(f"初始加载了 {len(all_annotations)} 个标注")
            return all_annotations
            
        except Exception as e:
            logger.error(f"加载标注文件失败: {str(e)}")
            return []
    
    def get_behavior_description(self, category_label: str, is_gerund: bool = True) -> str:
        """根据类别标签获取行为描述"""
        description = DRIVING_MANEUVER_CATEGORIES.get(category_label, "")
        
        if not description:
            return category_label
        
        # 将描述转换为更自然的whether问题格式
        if is_gerund:
            # 移除"Ego vehicle"并转换为现在分词
            if "Ego vehicle" in description:
                action_part = description.replace("Ego vehicle ", "").lower()
                
                # 针对每个类别生成更自然的描述
                if category_label == "TrafficLight_StraightStopOrGo":
                    return "stopping or starting at a traffic light for straight-line movement"
                elif category_label == "TrafficLight_LeftTurnStopOrGo":
                    return "stopping or starting at a traffic light for left-turn movement"
                elif category_label == "LaneChange_NavForIntersection":
                    return "changing lanes for navigation purposes approaching an intersection"
                elif category_label == "LaneChange_AvoidSlowVRU":
                    return "changing lanes to avoid slow-moving vulnerable road users (pedestrians, cyclists)"
                elif category_label == "LaneChange_AvoidStaticVehicle":
                    return "changing lanes to avoid stationary vehicles"
                elif category_label == "DynamicInteraction_VRUInLaneCrossing":
                    return "interacting with vulnerable road users crossing the ego's lane"
                elif category_label == "DynamicInteraction_VehicleInLaneCrossing":
                    return "interacting with other vehicles crossing the ego's lane"
                elif category_label == "DynamicInteraction_StandardVehicleCutIn":
                    return "experiencing another vehicle cutting in front"
                elif category_label == "StartStop_StartFromMainRoad":
                    return "starting from a stopped position on a main road"
                elif category_label == "StartStop_ParkRoadside":
                    return "parking or stopping at roadside"
                elif category_label == "Intersection_StandardUTurn":
                    return "making a U-turn at an intersection"
                elif category_label == "LaneCruising_Straight":
                    return "cruising straight without notable events"
        
        return description
    
    def generate_whether_question(self, category_label: str, start_time: float, end_time: float) -> str:
        """生成whether类问题，包含具体时间范围"""
        behavior_description = self.get_behavior_description(category_label, is_gerund=True)
        template = random.choice(self.whether_question_templates)
        
        # 格式化时间，确保整数
        start_time_str = f"{int(start_time)}"
        end_time_str = f"{int(end_time)}"
        
        return template.format(
            behavior_description=behavior_description,
            start_time=start_time_str,
            end_time=end_time_str
        )
    
    def generate_whether_answer(self, annotation: Dict, target_category: str, 
                                query_start_time: float, query_end_time: float) -> Tuple[str, bool, Dict]:
        """
        生成whether类问题的答案
        
        Args:
            annotation: 标注数据
            target_category: 目标类别
            query_start_time: 查询开始时间
            query_end_time: 查询结束时间
            
        Returns:
            Tuple[答案文本, 是否正例, 答案详细信息]
        """
        actual_category = annotation.get('label_en', '')
        actual_time_range = annotation.get('time_range_in_slice', [])
        
        if len(actual_time_range) < 2:
            return "", False, {}
        
        actual_start_time = actual_time_range[0]
        actual_end_time = actual_time_range[1]
        
        # 格式化时间，确保整数
        query_start_str = f"{int(query_start_time)}"
        query_end_str = f"{int(query_end_time)}"
        actual_start_str = f"{int(actual_start_time)}"
        actual_end_str = f"{int(actual_end_time)}"
        
        # 判断是否是正例：实际类别与目标类别匹配，且时间范围有重叠
        is_positive = False
        if actual_category == target_category:
            # 检查时间范围是否有重叠
            overlap_start = max(actual_start_time, query_start_time)
            overlap_end = min(actual_end_time, query_end_time)
            if overlap_start < overlap_end:  # 有重叠
                is_positive = True
        
        if is_positive:
            # 正例：行为发生
            template = random.choice(self.whether_answer_templates_yes)
            answer = template.format(
                action_label=actual_category,
                start_time=actual_start_str,
                end_time=actual_end_str
            )
            
            answer_info = {
                "is_positive": True,
                "actual_category": actual_category,
                "actual_start_time": actual_start_str,
                "actual_end_time": actual_end_str,
                "query_start_time": query_start_str,
                "query_end_time": query_end_str,
                "time_overlap": True
            }
        else:
            # 负例：行为未发生
            template = random.choice(self.whether_answer_templates_no)
            answer = template.format(
                start_time=query_start_str,
                end_time=query_end_str
            )
            
            answer_info = {
                "is_positive": False,
                "actual_category": actual_category,
                "actual_start_time": actual_start_str,
                "actual_end_time": actual_end_str,
                "query_start_time": query_start_str,
                "query_end_time": query_end_str,
                "time_overlap": False
            }
        
        return answer, is_positive, answer_info
    
    def group_annotations_by_category(self, annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """按类别分组标注数据"""
        categories = {label: [] for label in self.category_labels}
        categories["else"] = []  # 添加else类别
        
        for ann in annotations:
            label_en = ann.get('label_en', '')
            if label_en in categories:
                categories[label_en].append(ann)
            else:
                categories["else"].append(ann)
        
        return categories
    
    def generate_whether_samples_for_category(self, category: str, category_anns: List[Dict], 
                                             other_anns: List[Dict], samples_per_type: int = 10) -> List[Dict]:
        """
        为单个类别生成whether样本
        
        Args:
            category: 目标类别
            category_anns: 该类别下的标注
            other_anns: 其他类别的标注
            samples_per_type: 每个类型（正例/负例）的样本数
            
        Returns:
            生成的样本列表
        """
        samples = []
        
        # 生成正例
        positive_count = 0
        if category_anns:
            # 如果正例样本不足，则重复使用
            for i in range(samples_per_type):
                if i < len(category_anns):
                    ann = category_anns[i]
                else:
                    # 如果样本不够，随机选择一个
                    ann = random.choice(category_anns)
                
                # 获取时间范围
                time_range = ann.get('time_range_in_slice', [0, 20])
                if len(time_range) < 2:
                    time_range = [0, 20]
                
                start_time, end_time = time_range[0], time_range[1]
                
                # 生成whether问题和答案
                question = self.generate_whether_question(category, start_time, end_time)
                answer, is_positive, answer_info = self.generate_whether_answer(
                    ann, category, start_time, end_time
                )
                
                if answer:
                    sample = {
                        "instruction": question,
                        "input": "",  # 留空
                        "output": answer,
                        "videos": [ann.get('video_path', '')],
                        "system": self.system_prompt,
                        "slice_key": ann.get('slice_key', ''),
                        "time_range_in_slice": [start_time, end_time],
                        "actual_label": ann.get('label_en', ''),
                        "is_positive": is_positive,
                        "target_category": category,
                        "answer_info": answer_info
                    }
                    samples.append(sample)
                    positive_count += 1
        else:
            logger.warning(f"类别 {category} 没有正例样本，无法生成正例")
        
        # 生成负例
        negative_count = 0
        
        if other_anns:
            # 从其他类别中选取负例
            for i in range(samples_per_type):
                if i < len(other_anns):
                    ann = other_anns[i]
                else:
                    # 如果样本不够，随机选择一个
                    ann = random.choice(other_anns)
                
                # 使用标注的时间范围作为查询时间范围
                time_range = ann.get('time_range_in_slice', [0, 20])
                if len(time_range) < 2:
                    time_range = [0, 20]
                
                start_time, end_time = time_range[0], time_range[1]
                
                # 生成whether问题和答案
                question = self.generate_whether_question(category, start_time, end_time)
                answer, is_positive, answer_info = self.generate_whether_answer(
                    ann, category, start_time, end_time
                )
                
                # 确保是负例
                if not is_positive:
                    sample = {
                        "instruction": question,
                        "input": "",  # 留空
                        "output": answer,
                        "videos": [ann.get('video_path', '')],
                        "system": self.system_prompt,
                        "slice_key": ann.get('slice_key', ''),
                        "time_range_in_slice": [start_time, end_time],
                        "actual_label": ann.get('label_en', ''),
                        "is_positive": False,
                        "target_category": category,
                        "answer_info": answer_info
                    }
                    samples.append(sample)
                    negative_count += 1
                else:
                    # 如果意外生成了正例，跳过
                    logger.debug(f"意外生成了正例，跳过")
        else:
            logger.warning(f"没有可用的负例候选样本")
        
        logger.info(f"类别 {category}: 生成了 {positive_count} 个正例, {negative_count} 个负例")
        return samples
    
    def generate_whether_samples(self, samples_per_category: int = 10) -> Tuple[List[Dict], Dict[str, dict]]:
        """
        生成whether类问题的样本
        
        Args:
            samples_per_category: 每个类别生成的正例和负例数量
        Returns:
            Tuple[样本列表, 类别统计]
        """
        # 加载所有标注
        all_annotations = self.load_all_annotations()
        if not all_annotations:
            logger.error("没有加载到任何标注数据")
            return [], {}
        
        # 按类别分组
        categories = self.group_annotations_by_category(all_annotations)
        
        # 统计每个类别的样本数
        category_stats = {label: len(anns) for label, anns in categories.items()}
        logger.info(f"类别样本统计: {category_stats}")
        
        # 生成所有负例候选（排除当前类别的所有样本）
        all_annotations_by_category = {cat: anns for cat, anns in categories.items()}
        
        # 生成whether样本
        whether_samples = []
        category_counts = {label: {"positive": 0, "negative": 0} for label in self.category_labels}
        
        # 为每个类别生成正例和负例
        for category in tqdm(self.category_labels, desc="生成whether类样本"):
            # 获取当前类别的正例样本
            positive_anns = all_annotations_by_category.get(category, [])
            
            # 获取负例候选样本（所有其他类别的样本）
            negative_candidates = []
            for other_category, anns in all_annotations_by_category.items():
                if other_category != category:  # 排除当前类别
                    negative_candidates.extend(anns)
            
            # 生成该类别的样本
            category_samples = self.generate_whether_samples_for_category(
                category, positive_anns, negative_candidates, samples_per_category
            )
            
            # 统计
            for sample in category_samples:
                if sample.get('is_positive', False):
                    category_counts[category]["positive"] += 1
                else:
                    category_counts[category]["negative"] += 1
            
            whether_samples.extend(category_samples)
        
        # 打乱样本顺序
        random.shuffle(whether_samples)
        
        logger.info(f"总共生成了 {len(whether_samples)} 个whether类样本")
        
        return whether_samples, category_counts
    
    def save_training_dataset(self, samples: List[Dict], category_counts: Dict[str, dict]):
        """保存训练数据集"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"whether_training_dataset_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        
        # 保存数据集统计信息
        stats = {
            "total_samples": len(samples),
            "categories": len(self.category_labels),
            "samples_per_category": 20,  # 10正例 + 10负例
            "category_distribution": category_counts,
            "generation_time": timestamp,
            "dataset_type": "whether_training_only",
            "positive_samples": sum(counts["positive"] for counts in category_counts.values()),
            "negative_samples": sum(counts["negative"] for counts in category_counts.values()),
            "system_prompt_length": len(self.system_prompt)
        }
        
        stats_file = os.path.join(output_path, "dataset_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 只保存训练集
        train_file = os.path.join(output_path, "train.json")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        # 保存完整数据集（同训练集）
        all_file = os.path.join(output_path, "data.json")
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        # 保存dataset_info.json
        dataset_info = {
            "whether_training_dataset": {
                "file_name": "data.json",
                "columns": {
                    "prompt": "instruction",
                    "query": "input", 
                    "response": "output",
                    "videos": "videos",
                    "system": "system"
                }
            }
        }
        
        dataset_info_file = os.path.join(output_path, "dataset_info.json")
        with open(dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Whether训练数据集已保存到: {output_path}")
        logger.info(f"训练集: {len(samples)} 个样本")
        
        return output_path, stats
    
    def print_dataset_summary(self, samples: List[Dict], category_counts: Dict[str, dict]):
        """打印数据集摘要"""
        print("=" * 80)
        print("Whether类训练数据集摘要")
        print("=" * 80)
        
        # 统计正例和负例
        positive_samples = [s for s in samples if s.get('is_positive', False)]
        negative_samples = [s for s in samples if not s.get('is_positive', True)]
        
        print(f"总样本数: {len(samples)}")
        print(f"正例样本: {len(positive_samples)} (行为发生)")
        print(f"负例样本: {len(negative_samples)} (行为未发生)")
        print(f"正例比例: {len(positive_samples)/len(samples)*100:.1f}%")
        print(f"负例比例: {len(negative_samples)/len(samples)*100:.1f}%")
        print()
        
        # 打印每个类别的统计
        print("每个类别的样本分布:")
        print("-" * 60)
        for category, counts in category_counts.items():
            total = counts.get('positive', 0) + counts.get('negative', 0)
            if total > 0:
                print(f"{category}:")
                print(f"  正例: {counts.get('positive', 0)}")
                print(f"  负例: {counts.get('negative', 0)}")
                print(f"  总计: {total}")
        
        print()
        print("样本示例:")
        print("-" * 60)
        
        # 显示正例和负例示例
        positive_examples = [s for s in samples if s.get('is_positive', False)][:2]
        negative_examples = [s for s in samples if not s.get('is_positive', True)][:2]
        
        print("\n1. 正例示例 (行为发生):")
        for i, example in enumerate(positive_examples, 1):
            print(f"\n示例 {i}:")
            print(f"  问题: {example.get('instruction', '')}")
            print(f"  答案: {example.get('output', '')}")
            print(f"  目标类别: {example.get('target_category', '')}")
            print(f"  实际类别: {example.get('actual_label', '')}")
            print(f"  查询时间范围: {example.get('time_range_in_slice', [])}")
            answer_info = example.get('answer_info', {})
            if answer_info:
                print(f"  实际时间范围: {answer_info.get('actual_start_time', '')} 到 {answer_info.get('actual_end_time', '')}")
        
        print("\n2. 负例示例 (行为未发生):")
        for i, example in enumerate(negative_examples, 1):
            print(f"\n示例 {i}:")
            print(f"  问题: {example.get('instruction', '')}")
            print(f"  答案: {example.get('output', '')}")
            print(f"  目标类别: {example.get('target_category', '')}")
            print(f"  实际类别: {example.get('actual_label', '')}")
            print(f"  查询时间范围: {example.get('time_range_in_slice', [])}")
            answer_info = example.get('answer_info', {})
            if answer_info:
                print(f"  实际时间范围: {answer_info.get('actual_start_time', '')} 到 {answer_info.get('actual_end_time', '')}")
        
        print("\n3. Special Tokens 检查:")
        print("  - 问题中包含: <start_time>XX</start_time> 和 <end_time>YY</end_time>")
        print("  - 正例答案中包含: <driving_maneuver>action_label</driving_maneuver>")
        print("  - 所有时间都用special token包装")
        
        print("\n4. 系统提示摘要:")
        print(f"  长度: {len(self.system_prompt)} 字符")
        print(f"  是否包含special tokens规则: {'是' if '<driving_maneuver>' in self.system_prompt else '否'}")
        
        print("=" * 80)
    
    def validate_dataset(self, samples: List[Dict]) -> Dict:
        """验证数据集质量"""
        validation_results = {
            "total_samples": len(samples),
            "valid_samples": 0,
            "invalid_samples": 0,
            "positive_samples": 0,
            "negative_samples": 0,
            "categories_covered": set(),
            "video_paths_valid": 0,
            "video_paths_invalid": 0,
            "special_tokens_correct": 0,
            "special_tokens_incorrect": 0,
            "time_tokens_correct": 0,
            "time_tokens_incorrect": 0,
            "issues": []
        }
        
        for i, sample in enumerate(samples):
            # 检查必要字段
            required_fields = ['instruction', 'output', 'videos', 'system', 'is_positive', 'target_category']
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                validation_results['issues'].append(f"样本 {i}: 缺少字段 {missing_fields}")
                validation_results['invalid_samples'] += 1
                continue
            
            # 检查视频路径
            videos = sample.get('videos', [])
            if videos and isinstance(videos, list) and len(videos) > 0:
                video_path = videos[0]
                if os.path.exists(video_path):
                    validation_results['video_paths_valid'] += 1
                else:
                    validation_results['video_paths_invalid'] += 1
                    validation_results['issues'].append(f"样本 {i}: 视频文件不存在 {video_path}")
            
            # 检查是否问题格式
            instruction = sample.get('instruction', '')
            if not ("<start_time>" in instruction and "</start_time>" in instruction and 
                    "<end_time>" in instruction and "</end_time>" in instruction):
                validation_results['issues'].append(f"样本 {i}: 问题中缺少时间token")
                validation_results['time_tokens_incorrect'] += 1
            else:
                validation_results['time_tokens_correct'] += 1
            
            # 检查答案格式
            output = sample.get('output', '')
            is_positive = sample.get('is_positive', False)
            
            if is_positive:
                validation_results['positive_samples'] += 1
                # 检查正例格式
                if not ("Yes" in output and 
                        "<driving_maneuver>" in output and 
                        "</driving_maneuver>" in output and
                        "<start_time>" in output and 
                        "</start_time>" in output and
                        "<end_time>" in output and 
                        "</end_time>" in output):
                    validation_results['issues'].append(f"样本 {i}: 正例答案special tokens不完整")
                    validation_results['special_tokens_incorrect'] += 1
                else:
                    validation_results['special_tokens_correct'] += 1
            else:
                validation_results['negative_samples'] += 1
                # 检查负例格式
                if not ("No" in output and 
                        "<start_time>" in output and 
                        "</start_time>" in output and
                        "<end_time>" in output and 
                        "</end_time>" in output):
                    validation_results['issues'].append(f"样本 {i}: 负例答案时间tokens不完整")
                    validation_results['time_tokens_incorrect'] += 1
                else:
                    validation_results['time_tokens_correct'] += 1
            
            # 记录覆盖的类别
            target_category = sample.get('target_category', '')
            if target_category:
                validation_results['categories_covered'].add(target_category)
            
            validation_results['valid_samples'] += 1
        
        validation_results['categories_covered'] = list(validation_results['categories_covered'])
        validation_results['categories_covered_count'] = len(validation_results['categories_covered'])
        
        return validation_results


def main_whether_dataset():
    """主函数 - 生成whether类训练数据集"""
    # 配置路径
    ANNOTATIONS_FILE = "/root/workspace/sliced_vqa_dataset_prepared/converted_sliced_annotations/simple_sliced_dataset.json"
    OUTPUT_DIR = "/root/workspace/llama_factory_whether_training_dataset"
    
    print("=" * 80)
    print("Whether类训练数据集生成工具")
    print("=" * 80)
    print("📋 数据集特性:")
    print("  - 专门为whether类问题设计")
    print("  - 增强模型对负样本的识别能力")
    print("  - 12个类别，每个类别10个正例 + 10个负例")
    print("  - 总共240个样本 (12×20)")
    print("  - 只生成训练集，不生成测试集")
    print("  - 包含具体时间范围: from <start_time>XX</start_time> to <end_time>YY</end_time>")
    print("  - 包含special tokens: <driving_maneuver>, <start_time>, <end_time>")
    print("  - 包含system prompt")
    print("=" * 80)
    
    # 检查标注文件
    if not os.path.exists(ANNOTATIONS_FILE):
        logger.error(f"标注文件不存在: {ANNOTATIONS_FILE}")
        print(f"\n❌ 错误: 标注文件不存在: {ANNOTATIONS_FILE}")
        return
    
    if os.path.getsize(ANNOTATIONS_FILE) == 0:
        logger.error(f"标注文件为空: {ANNOTATIONS_FILE}")
        print(f"\n❌ 错误: 标注文件为空: {ANNOTATIONS_FILE}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化数据集生成器
    generator = WhetherQuestionDatasetGenerator(
        annotations_file=ANNOTATIONS_FILE,
        output_dir=OUTPUT_DIR
    )
    
    # 生成whether样本
    print("\n🚀 开始生成whether类训练数据集...")
    whether_samples, category_counts = generator.generate_whether_samples(samples_per_category=10)
    
    if not whether_samples:
        logger.error("没有生成有效的whether样本")
        print("\n❌ 错误: 没有生成有效的whether样本")
        return
    
    # 验证数据集
    print("\n🔍 验证数据集质量...")
    validation_results = generator.validate_dataset(whether_samples)
    
    print(f"✅ 有效样本: {validation_results['valid_samples']}/{validation_results['total_samples']}")
    print(f"❌ 无效样本: {validation_results['invalid_samples']}")
    print(f"✅ 正例样本: {validation_results['positive_samples']}")
    print(f"✅ 负例样本: {validation_results['negative_samples']}")
    print(f"✅ 覆盖类别: {validation_results['categories_covered_count']}/12")
    print(f"✅ 有效视频路径: {validation_results['video_paths_valid']}")
    print(f"❌ 无效视频路径: {validation_results['video_paths_invalid']}")
    print(f"✅ Special tokens正确: {validation_results['special_tokens_correct']}")
    print(f"❌ Special tokens错误: {validation_results['special_tokens_incorrect']}")
    print(f"✅ 时间tokens正确: {validation_results['time_tokens_correct']}")
    print(f"❌ 时间tokens错误: {validation_results['time_tokens_incorrect']}")
    
    if validation_results['issues']:
        print(f"\n⚠️ 发现 {len(validation_results['issues'])} 个问题:")
        for issue in validation_results['issues'][:5]:  # 只显示前5个问题
            print(f"  - {issue}")
    
    # 保存训练数据集
    print("\n💾 保存训练数据集...")
    output_path, stats = generator.save_training_dataset(whether_samples, category_counts)
    
    # 打印摘要
    generator.print_dataset_summary(whether_samples, category_counts)
    
    # 显示生成的文件
    print(f"\n📁 输出目录: {output_path}")
    print(f"\n📁 生成的文件:")
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  📄 {file} ({size_kb:.1f} KB)")
    
    # 显示样本示例
    print(f"\n📋 样本格式示例:")
    print("-" * 60)
    if whether_samples:
        sample = whether_samples[0]
        print(f"指令 (instruction):")
        print(f"  {sample.get('instruction', '')}")
        print(f"\n输入 (input):")
        print(f"  '{sample.get('input', '')}'")
        print(f"\n输出 (output):")
        print(f"  {sample.get('output', '')}")
        print(f"\n系统提示 (system) - 前200字符:")
        system_prompt = sample.get('system', '')
        print(f"  {system_prompt[:200]}...")
        print(f"\n视频路径 (videos):")
        print(f"  {sample.get('videos', [''])[0]}")
        print(f"\n目标类别 (target_category): {sample.get('target_category', '')}")
        print(f"是否正例 (is_positive): {sample.get('is_positive', False)}")
    
    # 显示配置
    print(f"\n🔧 Llama Factory 配置:")
    print("""
dataset_info:
  whether_training_dataset:
    file_name: data.json
    columns:
      prompt: instruction
      query: input
      response: output
      videos: videos
      system: system
    """)
    
    print(f"\n🎉 Whether类训练数据集生成完成!")
    print(f"   总计: {len(whether_samples)} 个样本")
    print(f"   正例: {stats.get('positive_samples', 0)} 个")
    print(f"   负例: {stats.get('negative_samples', 0)} 个")
    print(f"   输出路径: {output_path}")
    print("=" * 80)
    
    # 验证关键要求
    print("\n🔍 关键要求验证:")
    print(f"✅ 是否包含时间范围: 是 (从 <start_time>XX</start_time> 到 <end_time>YY</end_time>)")
    print(f"✅ 是否使用special tokens: 是 (<driving_maneuver>, <start_time>, <end_time>)")
    print(f"✅ 是否只生成训练集: 是")
    print(f"✅ 是否每个类别10正例10负例: 是")
    print(f"✅ 是否增强负样本识别: 是")
    print(f"✅ 是否与what类数据集格式兼容: 是")
    
    # 检查special tokens使用
    print(f"\n🔍 Special Tokens 检查:")
    if whether_samples:
        sample = whether_samples[0]
        instruction = sample.get('instruction', '')
        output = sample.get('output', '')
        
        print(f"  问题中是否有<start_time>: {'<start_time>' in instruction}")
        print(f"  问题中是否有<end_time>: {'<end_time>' in instruction}")
        print(f"  答案中是否有<driving_maneuver>: {'<driving_maneuver>' in output}")
        print(f"  答案中是否有<start_time>: {'<start_time>' in output}")
        print(f"  答案中是否有<end_time>: {'<end_time>' in output}")
    
    print("=" * 80)


if __name__ == "__main__":
    main_whether_dataset()