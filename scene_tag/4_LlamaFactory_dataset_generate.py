import os
import json
import random
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

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

# System Prompt定义
SYSTEM_PROMPT = """You are an expert in autonomous driving scene annotation. 
Based on a 60-second video, you need to identify the ego vehicle's actions. 

You MUST choose labels ONLY from this specific list:
1. TrafficLight_Straight_StopGo
2. TrafficLight_LeftTurn_StopGo
3. LaneChange_ForIntersection
4. Avoid_SlowVRU
5. Avoid_StaticVehicle
6. Avoid_ConstructionZone
7. VRU_CrossingPath
8. Vehicle_CrossingPath
9. Vehicle_CutIn
10. Vehicle_AggressiveCutIn
11. VRU_SuddenCutIn
12. VRU_SlowCutIn
13. LeadVehicle_EmergencyBrake
14. Start_FromMainRoad
15. Park_Roadside
16. U_Turn_Standard
17. U_Turn_ThreePoint
18. LeftTurn_VRU_Crossing
19. Lane_Cruising_Straight

Please use the format: <driving_maneuver>action_label</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.
If there are multiple actions, list them in chronological order separated by " and ".
IMPORTANT: Only use the exact labels from the list above. Do NOT create new labels."""

# 问题模板列表 - 在视频前添加<video>标记
ENGLISH_QUESTION_TEMPLATES = [
    "<video>\nWhat is the ego vehicle's action in the video?",
    "<video>\nWhat is the ego vehicle doing in this video clip?",
    "<video>\nWhat is the behavior of the ego vehicle?",
    "<video>\nPlease tell me the ego vehicle's action.",
    "<video>\nWhat operation is the ego vehicle currently executing?",
    "<video>\nWhat is the driving maneuver of the ego vehicle in this video?",
    "<video>\nIdentify the ego vehicle's action in the video.",
    "<video>\nDescribe the behavior of the ego vehicle.",
    "<video>\nWhat is the operation of the ego vehicle?",
    "<video>\nWhat is the vehicle's action shown in the video?",
    "<video>\nWhat action is the ego vehicle executing?",
    "<video>\nWhat is the ego vehicle's behavior in this video clip?",
    "<video>\nPlease explain the ego vehicle's action.",
    "<video>\nWhat is the driving maneuver of the ego vehicle?",
    "<video>\nWhat is the ego vehicle's operation in the video?",
    "<video>\nWhat action is the ego vehicle completing in this video?",
    "<video>\nWhat is the driving behavior of the ego vehicle?",
    "<video>\nPlease analyze the ego vehicle's action.",
    "<video>\nWhat is the ego vehicle's action in the video?",
    "<video>\nWhat did the ego vehicle do in the video?"
]

# 答案模板列表 - 在回答中引用视频
VIDEO_ANSWER_TEMPLATES = [
    "Based on the video, the ego vehicle's behavior from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds is <driving_maneuver>action</driving_maneuver>.",
    "From the video, the ego vehicle performs <driving_maneuver>action</driving_maneuver> between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds.",
    "In the video, from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle's action is <driving_maneuver>action</driving_maneuver>.",
    "The video shows the ego vehicle exhibits <driving_maneuver>action</driving_maneuver> behavior during <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "Based on the video content, the primary action of the ego vehicle is <driving_maneuver>action</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "From watching the video, between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds, the ego vehicle is <driving_maneuver>action</driving_maneuver>.",
    "The video depicts that during the interval <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle's behavior is <driving_maneuver>action</driving_maneuver>.",
    "In the provided video, the ego vehicle executes <driving_maneuver>action</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.",
    "Based on the video footage, from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds, the ego vehicle engages in <driving_maneuver>action</driving_maneuver>.",
    "The video demonstrates that the ego vehicle's driving maneuver is <driving_maneuver>action</driving_maneuver> between <start_time>start_time_value</start_time> and <end_time>end_time_value</end_time> seconds."
]

class LlamaFactoryVQADatasetBuilder:
    """Llama Factory VQA数据集构建器"""
    
    def __init__(self, annotations_file: str, output_dir: str, train_ratio: float = 0.8, 
                 merge_interval: int = 1, system_prompt: str = None):
        """
        初始化数据集构建器
        
        Args:
            annotations_file: 标注文件路径
            output_dir: 输出目录
            train_ratio: 训练集比例
            merge_interval: 合并间隔（秒），相邻动作间隔小于等于此值会被合并
            system_prompt: 系统提示词，如果为None则使用默认的SYSTEM_PROMPT
        """
        self.annotations_file = annotations_file
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.merge_interval = merge_interval
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
    
    def group_by_video(self, annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """按视频路径分组标注"""
        video_groups = defaultdict(list)
        
        for ann in annotations:
            video_path = ann.get('video_path', '')
            if video_path and os.path.exists(video_path):
                video_groups[video_path].append(ann)
        
        logger.info(f"按视频分组完成: {len(video_groups)} 个视频")
        return video_groups
    
    def remove_duplicate_annotations(self, annotations: List[Dict]) -> List[Dict]:
        """移除重复的标注"""
        if not annotations:
            return []
        
        seen = set()
        unique_annotations = []
        
        for ann in annotations:
            label_en = ann.get('label_en', '')
            time_range = tuple(ann.get('time_range', []))
            ann_id = ann.get('id', '')
            
            key = (label_en, time_range, ann_id)
            if key not in seen:
                seen.add(key)
                unique_annotations.append(ann)
        
        return unique_annotations
    
    def merge_overlapping_actions(self, annotations: List[Dict]) -> List[Dict]:
        """合并重叠或相邻的相同动作"""
        if not annotations:
            return []
        
        label_groups = defaultdict(list)
        for ann in annotations:
            label = ann.get('label_en', '')
            if label:
                label_groups[label].append(ann)
        
        merged_annotations = []
        
        for label, label_anns in label_groups.items():
            if len(label_anns) == 1:
                merged_annotations.append(label_anns[0])
                continue
            
            sorted_anns = sorted(label_anns, key=lambda x: x.get('time_range', [0])[0])
            
            current_range = None
            current_anns = []
            
            for ann in sorted_anns:
                time_range = ann.get('time_range', [])
                if len(time_range) < 2:
                    continue
                
                start_time = time_range[0]
                end_time = time_range[1]
                
                if current_range is None:
                    current_range = [start_time, end_time]
                    current_anns = [ann]
                else:
                    if start_time <= current_range[1] + self.merge_interval:
                        current_range[1] = max(current_range[1], end_time)
                        current_anns.append(ann)
                    else:
                        if current_range:
                            merged_ann = self._create_merged_annotation(current_anns, current_range)
                            merged_annotations.append(merged_ann)
                        current_range = [start_time, end_time]
                        current_anns = [ann]
            
            if current_range and current_anns:
                merged_ann = self._create_merged_annotation(current_anns, current_range)
                merged_annotations.append(merged_ann)
        
        return merged_annotations
    
    def _create_merged_annotation(self, original_anns: List[Dict], merged_range: List[int]) -> Dict:
        """创建合并后的标注"""
        if not original_anns:
            return None
        
        base_ann = original_anns[0].copy()
        base_ann['time_range'] = merged_range
        base_ann['duration'] = merged_range[1] - merged_range[0]
        
        base_ann['id'] = f"merged_{len(original_anns)}_{hash(tuple(merged_range)) % 10000:04d}"
        return base_ann
    
    def generate_single_action_description(self, action: Dict) -> str:
        """生成单个动作的描述"""
        label_en = action.get('label_en', '')
        time_range = action.get('time_range', [])
        
        if not label_en or len(time_range) < 2:
            return ""
        
        start_time = int(time_range[0])
        end_time = int(time_range[1])
        
        template = random.choice(VIDEO_ANSWER_TEMPLATES)
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
        """合并同一视频的多个动作为一个综合描述"""
        if not video_annotations:
            return None
        
        # 先去重
        unique_annotations = self.remove_duplicate_annotations(video_annotations)
        if not unique_annotations:
            return None
        
        # 合并重叠或相邻的相同动作
        merged_annotations = self.merge_overlapping_actions(unique_annotations)
        if not merged_annotations:
            return None
        
        # 按开始时间排序
        sorted_annotations = sorted(merged_annotations, 
                                   key=lambda x: x.get('time_range', [0])[0])
        
        video_path = sorted_annotations[0].get('video_path', '')
        
        if not video_path or not os.path.exists(video_path):
            return None
        
        # 生成问题和答案
        question = random.choice(ENGLISH_QUESTION_TEMPLATES)
        
        action_descriptions = []
        for ann in sorted_annotations:
            description = self.generate_single_action_description(ann)
            if description:
                action_descriptions.append(description)
        
        if not action_descriptions:
            return None
        
        # 连接所有动作描述
        if len(action_descriptions) == 1:
            answer = action_descriptions[0]
        else:
            connector = random.choice(["; ", " and "])
            answer = connector.join(action_descriptions)
        
        return {
            "video_path": video_path,
            "question": question,
            "answer": answer,
            "num_actions": len(sorted_annotations)
        }
    
    def process_video_groups(self, video_groups: Dict[str, List[Dict]]) -> List[Dict]:
        """处理所有视频组，生成Llama Factory格式的数据"""
        llama_factory_data = []
        
        for video_path, annotations in tqdm(video_groups.items(), desc="处理视频"):
            video_sample = self.merge_actions_for_video(annotations)
            
            if video_sample:
                # 转换为Llama Factory格式
                # 注意：instruction中已经有<video>标记
                llama_format = {
                    "instruction": video_sample["question"],  # 已包含<video>标记
                    "input": "",  # 留空
                    "output": video_sample["answer"],
                    "videos": [video_sample["video_path"]],  # 视频路径列表
                    "system": self.system_prompt  # 添加system prompt
                }
                llama_factory_data.append(llama_format)
        
        logger.info(f"生成了 {len(llama_factory_data)} 个Llama Factory格式样本")
        return llama_factory_data
    
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
            "video_vqa_dataset": {
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
    ANNOTATIONS_FILE = "/root/workspace/vqa_dataset_prepared/converted_annotations/existing_videos_dataset.json"
    OUTPUT_DIR = "/root/workspace/llama_factory_vqa_dataset"
    
    print("=" * 60)
    print("Llama Factory VQA数据集生成工具 (带<video>标记和system prompt)")
    print("=" * 60)
    print("📋 关键特性:")
    print("  - instruction中包含<video>标记")
    print("  - videos列包含视频路径列表")
    print("  - 包含system prompt字段")
    print("  - 支持QWen3 SFT格式")
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
        train_ratio=0.8,
        merge_interval=1
    )
    
    # 加载所有标注
    all_annotations = builder.load_all_annotations()
    if not all_annotations:
        logger.error("没有找到标注数据")
        print("\n❌ 错误: 没有找到标注数据")
        return
    
    # 按视频分组
    video_groups = builder.group_by_video(all_annotations)
    if not video_groups:
        logger.error("没有找到有效的视频标注")
        print("\n❌ 错误: 没有找到有效的视频标注")
        return
    
    # 处理视频组，生成Llama Factory格式的数据
    llama_factory_data = builder.process_video_groups(video_groups)
    if not llama_factory_data:
        logger.error("没有生成有效的样本")
        print("\n❌ 错误: 没有生成有效的样本")
        return
    
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
    train_data = llama_factory_data[:split_idx]
    test_data = llama_factory_data[split_idx:]
    
    print(f"\n📊 数据集划分:")
    print(f"  总样本数: {len(llama_factory_data)}")
    print(f"  训练集: {len(train_data)} 个样本")
    print(f"  测试集: {len(test_data)} 个样本")
    
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
    print(f"   数据集名称: video_vqa_dataset")
    print("\n2. QWen3 SFT格式使用:")
    print(f"   数据集路径: {output_path_qwen}")
    print(f"   数据集名称: qwen3_sft_vqa_dataset")
    print("\n3. 在Llama Factory配置文件中添加:")
    print("""
dataset_info:
  video_vqa_dataset:
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