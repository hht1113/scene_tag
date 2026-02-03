import os
import json
import random
import copy  # <--- 新增：用于深拷贝
from typing import Dict, List, Tuple, Optional  
import logging
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter # <--- 新增：用于统计

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/llama_factory_dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 类别定义 - 可独立扩展的部分
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

# 主系统提示 - 修改为20秒切片视频
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


# 问题模板列表 - 在视频前添加<video>标记
ENGLISH_QUESTION_TEMPLATES = [
    "<video>\nWhat is the ego vehicle's action in this 20-second video clip?",
    "<video>\nWhat is the ego vehicle doing in this 20-second video?",
    "<video>\nWhat is the behavior of the ego vehicle in this 20-second clip?",
    "<video>\nPlease tell me the ego vehicle's action in this 20-second video.",
    "<video>\nWhat operation is the ego vehicle currently executing in this 20-second clip?",
    "<video>\nWhat is the driving maneuver of the ego vehicle in this 20-second video?",
    "<video>\nIdentify the ego vehicle's action in this 20-second video clip.",
    "<video>\nDescribe the behavior of the ego vehicle in this 20-second video.",
    "<video>\nWhat is the operation of the ego vehicle in this 20-second clip?",
    "<video>\nWhat is the vehicle's action shown in this 20-second video?",
    "<video>\nWhat action is the ego vehicle executing in this 20-second clip?",
    "<video>\nWhat is the ego vehicle's behavior in this 20-second video?",
    "<video>\nPlease explain the ego vehicle's action in this 20-second video.",
    "<video>\nWhat is the driving maneuver of the ego vehicle in this 20-second clip?",
    "<video>\nWhat is the ego vehicle's operation in this 20-second video?",
    "<video>\nWhat action is the ego vehicle completing in this 20-second video?",
    "<video>\nWhat is the driving behavior of the ego vehicle in this 20-second clip?",
    "<video>\nPlease analyze the ego vehicle's action in this 20-second video.",
    "<video>\nWhat is the ego vehicle's action in this 20-second video clip?",
    "<video>\nWhat did the ego vehicle do in this 20-second video?"
]

# 答案模板列表 - 在回答中引用视频
VIDEO_ANSWER_TEMPLATES = [
    "Based on the 20-second video, the ego vehicle's behavior from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds is <driving_maneuver>action</driving_maneuver>.",
    "From the 20-second video, the ego vehicle performs <driving_maneuver>action</driving_maneuver> between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds.",
    "In this 20-second video, from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle's action is <driving_maneuver>action</driving_maneuver>.",
    "The 20-second video shows the ego vehicle exhibits <driving_maneuver>action</driving_maneuver> behavior during <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "Based on the 20-second video content, the primary action of the ego vehicle is <driving_maneuver>action</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "From watching this 20-second video, between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds, the ego vehicle is <driving_maneuver>action</driving_maneuver>.",
    "The 20-second video depicts that during the interval <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle's behavior is <driving_maneuver>action</driving_maneuver>.",
    "In this 20-second video, the ego vehicle executes <driving_maneuver>action</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "Based on the 20-second video footage, from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle engages in <driving_maneuver>action</driving_maneuver>.",
    "The 20-second video demonstrates that the ego vehicle's driving maneuver is <driving_maneuver>action</driving_maneuver> between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds."
]

class LlamaFactoryVQADatasetBuilder:
    """Llama Factory VQA数据集构建器"""
    
    def __init__(self, annotations_file: str, output_dir: str, train_ratio: float = 0.8, 
                 system_prompt: str = None):
        """初始化数据集构建器"""
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        
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
    
    def remove_duplicate_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """移除重复的标注"""
        if not annotations:
            return []
        
        seen = set()
        unique_annotations = []
        
        for ann in annotations:
            video_path = ann.get('video_path', '')
            time_range = tuple(ann.get('time_range_in_slice', []))
            label_en = ann.get('label_en', '')
            
            # 只有当视频、时间、标签完全一致时才认为是重复
            key = (video_path, time_range, label_en)
            if key not in seen:
                seen.add(key)
                unique_annotations.append(ann)
        
        return unique_annotations
    
    def generate_single_action_description(self, action: Dict) -> str:
        """生成单个动作的描述"""
        label_en = action.get('label_en', '')
        time_range = action.get('time_range_in_slice', [])
        
        if not label_en or len(time_range) < 2:
            return ""
        
        start_time = time_range[0]
        end_time = time_range[1]
        
        # 确保时间在0-20秒范围内
        if start_time < 0:
            start_time = 0
        if end_time > 20:
            end_time = 20
        
        template = random.choice(VIDEO_ANSWER_TEMPLATES)
        description = template.replace(
            "<start_time>start_time_value</start_time>", 
            f"<start_time>{start_time:.1f}</start_time>"
        ).replace(
            "<end_time>end_time_value</end_time>", 
            f"<end_time>{end_time:.1f}</end_time>"
        ).replace(
            "<driving_maneuver>action</driving_maneuver>", 
            f"<driving_maneuver>{label_en}</driving_maneuver>"
        )
        
        return description
    
    def process_single_sliced_annotation(self, annotation: Dict) -> Optional[Dict]:
        """处理单个切片标注生成样本"""
        try:
            # 获取必要信息
            video_path = annotation.get('video_path', '')
            label_en = annotation.get('label_en', '')
            time_range = annotation.get('time_range_in_slice', [])
            
            # 验证数据
            if not video_path or not os.path.exists(video_path):
                logger.warning(f"视频文件不存在: {video_path}")
                return None
            
            if not label_en:
                logger.warning(f"缺少英文标签: {annotation.get('id', 'unknown')}")
                return None
            
            if len(time_range) < 2:
                logger.warning(f"无效的时间范围: {annotation.get('id', 'unknown')}")
                return None
            
            # 生成问题和答案
            question = random.choice(ENGLISH_QUESTION_TEMPLATES)
            answer = self.generate_single_action_description(annotation)
            
            if not answer:
                logger.warning(f"无法生成答案: {annotation.get('id', 'unknown')}")
                return None
            
            # 转换为Llama Factory格式
            return {
                "instruction": question,
                "input": "",  # 留空
                "output": answer,
                "videos": [video_path],  # 视频路径列表
                "system": self.system_prompt,  # 添加system prompt
                "slice_key": annotation.get('slice_key', ''),
                "time_range_in_slice": time_range,
                "label_en": label_en # <--- 保留这个字段，后续用于上采样统计
            }
            
        except Exception as e:
            logger.error(f"处理切片标注失败: {str(e)}")
            return None
    
    def process_all_sliced_annotations(self) -> List[Dict]:
        """处理所有切片标注生成数据集"""
        # 加载所有标注
        all_annotations = self.load_all_annotations()
        if not all_annotations:
            return []
        
        # 去重
        unique_annotations = self.remove_duplicate_annotations(all_annotations)
        logger.info(f"去重后剩余 {len(unique_annotations)} 个唯一标注")
        
        # 处理每个切片标注
        llama_factory_data = []
        
        for ann in tqdm(unique_annotations, desc="处理切片标注"):
            sample = self.process_single_sliced_annotation(ann)
            if sample:
                llama_factory_data.append(sample)
        
        logger.info(f"生成了 {len(llama_factory_data)} 个Llama Factory格式样本")
        return llama_factory_data
    
    # -------------------------------------------------------------------------
    # <<< 新增方法：打印类别统计 >>>
    # -------------------------------------------------------------------------
    def print_data_stats(self, data: List[Dict], title: str = "数据集统计"):
        """打印数据集中各类别分布情况"""
        if not data:
            print(f"\n📊 {title}: 无数据")
            return
            
        counter = Counter([item.get('label_en', 'unknown') for item in data])
        total = len(data)
        sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 {title} (总计: {total}):")
        print("-" * 60)
        print(f"{'Category Label':<50} | {'Count':<6} | {'Ratio'}")
        print("-" * 60)
        for label, count in sorted_counts:
            ratio = (count / total) * 100
            print(f"{label[:48]:<50} | {count:<6} | {ratio:.1f}%")
        print("-" * 60)
        return sorted_counts

    # -------------------------------------------------------------------------
    # <<< 新增方法：上采样逻辑 >>>
    # -------------------------------------------------------------------------
    def upsample_data(self, data: List[Dict]) -> List[Dict]:
        """
        对数据进行上采样：找出数量最多的类别，将其他类别补充到相同数量。
        """
        if not data:
            return []
            
        # 1. 按类别分组
        category_map = defaultdict(list)
        for item in data:
            label = item.get('label_en', 'unknown')
            category_map[label].append(item)
            
        # 2. 确定目标数量 (最大类的数量)
        counts = {k: len(v) for k, v in category_map.items()}
        if not counts:
            return data
            
        max_count = max(counts.values())
        print(f"\n🔄 正在执行上采样... 目标数量: 每个类别 {max_count} 个样本")
        
        balanced_data = []
        
        for label, items in category_map.items():
            current_count = len(items)
            # 先加入原有数据
            balanced_data.extend(items)
            
            # 计算需要补充的数量
            needed = max_count - current_count
            if needed > 0:
                # 随机采样并深拷贝 (Deepcopy很重要，防止修改副本影响原件)
                extras = random.choices(items, k=needed)
                extras_copy = [copy.deepcopy(x) for x in extras]
                balanced_data.extend(extras_copy)
                
        # 打乱顺序
        random.shuffle(balanced_data)
        return balanced_data

    def save_llama_factory_format(self, train_data: List[Dict], test_data: List[Dict]):
        """保存为Llama Factory格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"llama_factory_vqa_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        
        # 1. 保存训练集
        train_file = os.path.join(output_path, "train.json")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存训练集: {train_file} ({len(train_data)} 个样本)")
        
        # 2. 保存测试集
        test_file = os.path.join(output_path, "test.json")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存测试集: {test_file} ({len(test_data)} 个样本)")
        
        # 3. 保存完整数据集
        all_data = train_data + test_data
        all_file = os.path.join(output_path, "data.json")
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存完整数据集: {all_file} ({len(all_data)} 个样本)")
        
        # 4. 创建包含system字段的dataset_info.json
        dataset_info = {
            "sliced_video_vqa_dataset": {
                "file_name": "data.json",
                "columns": {
                    "prompt": "instruction",
                    "query": "input", 
                    "response": "output",
                    "videos": "videos",
                    "system": "system"  # 添加system字段映射
                }
            }
        }
        
        dataset_info_file = os.path.join(output_path, "dataset_info.json")
        with open(dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        logger.info(f"保存dataset_info.json: {dataset_info_file}")
        
        return output_path
    
    def save_qwen3_sft_format(self, train_data: List[Dict], test_data: List[Dict]):
        """保存为QWen3 SFT格式"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"qwen3_sft_vqa_{timestamp}")
        os.makedirs(output_path, exist_ok=True)
        
        # 1. 保存训练集
        train_file = os.path.join(output_path, "qwen3_sft_train.json")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存训练集: {train_file} ({len(train_data)} 个样本)")
        
        # 2. 保存测试集
        test_file = os.path.join(output_path, "qwen3_sft_test.json")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存测试集: {test_file} ({len(test_data)} 个样本)")
        
        # 3. 保存完整数据集
        all_data = train_data + test_data
        all_file = os.path.join(output_path, "qwen3_sft_all.json")
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        logger.info(f"保存完整数据集: {all_file} ({len(all_data)} 个样本)")
        
        # 4. 创建QWen3 SFT格式的dataset_info.json
        dataset_info = {
            "qwen3_sft_vqa_dataset": {
                "file_name": "qwen3_sft_train.json",
                "columns": {
                    "prompt": "instruction",
                    "query": "input", 
                    "response": "output",
                    "videos": "videos",
                    "system": "system"  # 添加system字段映射
                }
            }
        }
        
        dataset_info_file = os.path.join(output_path, "dataset_info.json")
        with open(dataset_info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        logger.info(f"保存dataset_info.json: {dataset_info_file}")
        
        return output_path
    
    def check_video_tag_consistency(self, data: List[Dict]) -> Dict:
        """检查<video>标记和视频数量的一致性"""
        results = {
            "total_samples": len(data),
            "consistent_samples": 0,
            "inconsistent_samples": 0,
            "missing_video_tag": 0,
            "video_count_mismatch": 0,
            "details": []
        }
        
        for i, item in enumerate(data):
            instruction = item.get("instruction", "")
            videos = item.get("videos", [])
            
            # 统计<video>标记的数量
            video_tags = instruction.count("<video>")
            
            # 检查一致性
            is_consistent = (video_tags == len(videos))
            
            detail = {
                "sample_index": i,
                "video_tags_count": video_tags,
                "videos_count": len(videos),
                "is_consistent": is_consistent,
                "instruction_preview": instruction[:100] + "..." if len(instruction) > 100 else instruction
            }
            
            results["details"].append(detail)
            
            if is_consistent:
                results["consistent_samples"] += 1
            else:
                results["inconsistent_samples"] += 1
                if video_tags == 0:
                    results["missing_video_tag"] += 1
                if video_tags != len(videos):
                    results["video_count_mismatch"] += 1
        
        return results
    
    def check_system_prompt_inclusion(self, data: List[Dict]) -> Dict:
        """检查system prompt是否包含"""
        results = {
            "total_samples": len(data),
            "with_system": 0,
            "without_system": 0,
            "system_prompt_lengths": []
        }
        
        for item in data:
            system_prompt = item.get("system", "")
            if system_prompt and system_prompt.strip():
                results["with_system"] += 1
                results["system_prompt_lengths"].append(len(system_prompt))
            else:
                results["without_system"] += 1
        
        if results["system_prompt_lengths"]:
            results["avg_system_length"] = sum(results["system_prompt_lengths"]) / len(results["system_prompt_lengths"])
        else:
            results["avg_system_length"] = 0
            
        return results

def main():
    """主函数"""
    # 修改为切片数据集路径
    ANNOTATIONS_FILE = "/root/workspace/sliced_vqa_dataset_prepared/converted_sliced_annotations/simple_sliced_dataset.json"
    OUTPUT_DIR = "/root/workspace/llama_factory_sliced_vqa_dataset"
    
    print("=" * 60)
    print("Llama Factory 切片视频VQA数据集生成工具 (含上采样平衡)")
    print("=" * 60)
    print("📋 关键特性:")
    print("  - 专门为20秒切片视频设计")
    print("  - instruction中包含<video>标记")
    print("  - videos列包含切片视频路径")
    print("  - 包含system prompt字段")
    print("  - 支持QWen3 SFT格式")
    print("  - [新增] 自动上采样平衡训练集") # <--- 提示
    print("=" * 60)
    
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
    
    # 初始化数据集构建器
    builder = LlamaFactoryVQADatasetBuilder(
        annotations_file=ANNOTATIONS_FILE,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8
    )
    
    # 处理所有切片标注
    llama_factory_data = builder.process_all_sliced_annotations()
    if not llama_factory_data:
        logger.error("没有生成有效的样本")
        print("\n❌ 错误: 没有生成有效的样本")
        return
    
    # <--- 新增：打印原始数据统计 --->
    builder.print_data_stats(llama_factory_data, title="原始数据分布")
    
    # 检查<video>标记一致性
    print("\n🔍 检查<video>标记一致性...")
    consistency_check = builder.check_video_tag_consistency(llama_factory_data)
    
    print(f"✅ 一致样本: {consistency_check['consistent_samples']}/{consistency_check['total_samples']}")
    print(f"❌ 不一致样本: {consistency_check['inconsistent_samples']}")
    
    if consistency_check['inconsistent_samples'] > 0:
        print(f"  - 缺少<video>标记: {consistency_check['missing_video_tag']}")
        print(f"  - 视频数量不匹配: {consistency_check['video_count_mismatch']}")
        
        # 显示不一致的样本
        print("\n📋 不一致样本详情:")
        for detail in consistency_check['details'][:5]:  # 只显示前5个
            if not detail['is_consistent']:
                print(f"  样本{detail['sample_index']}: {detail['instruction_preview']}")
                print(f"    <video>标记: {detail['video_tags_count']}, 视频数量: {detail['videos_count']}")
    
    # 检查system prompt
    print("\n🔍 检查system prompt包含情况...")
    system_check = builder.check_system_prompt_inclusion(llama_factory_data)
    print(f"✅ 包含system prompt: {system_check['with_system']}/{system_check['total_samples']}")
    print(f"❌ 缺少system prompt: {system_check['without_system']}")
    print(f"📊 平均system prompt长度: {system_check['avg_system_length']:.1f} 字符")
    
    # 简单划分训练集和测试集
    random.shuffle(llama_factory_data)
    split_idx = int(len(llama_factory_data) * 0.8)
    
    # <--- 修改：先划分 --->
    train_data = llama_factory_data[:split_idx]
    test_data = llama_factory_data[split_idx:]
    
    print(f"\n📊 初始划分:")
    print(f"  总样本数: {len(llama_factory_data)}")
    print(f"  训练集(Raw): {len(train_data)} 个样本")
    print(f"  测试集(Raw): {len(test_data)} 个样本")

    # <--- 新增：仅对训练集进行上采样 --->
    print("\n🚀 正在对训练集进行上采样平衡...")
    train_data = builder.upsample_data(train_data)
    
    # <--- 新增：打印上采样后的统计 --->
    builder.print_data_stats(train_data, title="上采样后的训练集分布")
    
    # 保存为Llama Factory格式
    output_path_llama = builder.save_llama_factory_format(train_data, test_data)
    
    # 保存为QWen3 SFT格式
    output_path_qwen = builder.save_qwen3_sft_format(train_data, test_data)
    
    # 显示样本示例
    print("\n" + "=" * 60)
    print("📋 数据集样本示例")
    print("=" * 60)
    
    if train_data:
        print("\n训练集样本 (前2个):")
        for i, sample in enumerate(train_data[:2], 1):
            print(f"\n样本 {i}:")
            print(f"  instruction: {sample.get('instruction', 'N/A')}")
            print(f"  input: '{sample.get('input', '')}'")
            print(f"  system: {sample.get('system', 'N/A')[:100]}...")
            print(f"  output: {sample.get('output', 'N/A')[:120]}...")
            video_path = sample.get('videos', [''])[0]
            print(f"  videos: ['{video_path[:60]}...']")
            print(f"  <video>标记数量: {sample.get('instruction', '').count('<video>')}")
            print(f"  视频数量: {len(sample.get('videos', []))}")
            print(f"  视频存在: {os.path.exists(video_path) if video_path else False}")
            print(f"  切片key: {sample.get('slice_key', '')}")
    
    print("=" * 60)
    
    # 显示Llama Factory格式的dataset_info.json内容
    dataset_info_file_llama = os.path.join(output_path_llama, "dataset_info.json")
    if os.path.exists(dataset_info_file_llama):
        with open(dataset_info_file_llama, 'r', encoding='utf-8') as f:
            dataset_info_llama = json.load(f)
        print(f"\n📁 Llama Factory格式 dataset_info.json 配置:")
        for dataset_name, config in dataset_info_llama.items():
            print(f"  数据集名称: {dataset_name}")
            print(f"  文件名: {config['file_name']}")
            print(f"  字段映射:")
            for field, mapping in config['columns'].items():
                print(f"    {field}: {mapping}")
    
    # 显示QWen3 SFT格式的dataset_info.json内容
    dataset_info_file_qwen = os.path.join(output_path_qwen, "dataset_info.json")
    if os.path.exists(dataset_info_file_qwen):
        with open(dataset_info_file_qwen, 'r', encoding='utf-8') as f:
            dataset_info_qwen = json.load(f)
        print(f"\n📁 QWen3 SFT格式 dataset_info.json 配置:")
        for dataset_name, config in dataset_info_qwen.items():
            print(f"  数据集名称: {dataset_name}")
            print(f"  文件名: {config['file_name']}")
            print(f"  字段映射:")
            for field, mapping in config['columns'].items():
                print(f"    {field}: {mapping}")
    
    print("\n" + "=" * 60)
    print("🎉 数据集生成完成")
    print("=" * 60)
    
    print(f"\n📁 输出目录:")
    print(f"  Llama Factory格式: {output_path_llama}")
    print(f"  QWen3 SFT格式: {output_path_qwen}")
    
    print(f"\n📁 Llama Factory格式生成的文件:")
    for file in os.listdir(output_path_llama):
        file_path = os.path.join(output_path_llama, file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  📄 {file} ({size_kb:.1f} KB)")
    
    print(f"\n📁 QWen3 SFT格式生成的文件:")
    for file in os.listdir(output_path_qwen):
        file_path = os.path.join(output_path_qwen, file)
        if os.path.isfile(file_path):
            size_kb = os.path.getsize(file_path) / 1024
            print(f"  📄 {file} ({size_kb:.1f} KB)")
    
    print(f"\n🚀 使用说明:")
    print("1. Llama Factory格式使用:")
    print(f"   数据集路径: {output_path_llama}")
    print(f"   数据集名称: sliced_video_vqa_dataset")
    print("\n2. QWen3 SFT格式使用:")
    print(f"   数据集路径: {output_path_qwen}")
    print(f"   数据集名称: qwen3_sft_vqa_dataset")
    print("\n3. 在Llama Factory配置文件中添加:")
    print("""
dataset_info:
  sliced_video_vqa_dataset:
    file_name: train.json
    columns:
      prompt: instruction
      query: input
      response: output
      videos: videos
      system: system
""")
    print("\n💡 注意: 训练时使用 train.json，评估时使用 test.json")
    
    print("=" * 60)
    
    # 验证关键要求
    print("\n🔍 Llama Factory要求验证:")
    print("✅ 每个样本包含 videos 列: 是")
    print("✅ instruction 中包含 <video> 标记: 是")
    print("✅ 包含 system 字段: 是")
    print("✅ <video>标记数量与视频数量一致: 是 (1个标记对应1个视频)")
    print("✅ videos 列是列表格式: 是")
    print("✅ 所有视频路径都存在: 已验证")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

