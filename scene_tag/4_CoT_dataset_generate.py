import os
import json
import random
import time
import base64
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from tqdm import tqdm
import dashscope
from dashscope import MultiModalConversation
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/cot_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 用户问题列表
QUESTION_LIST = [
    "Analyze the driving scenario strictly.",
    "Classify the ego vehicle's behavior.",
    "Give me the hierarchical tag for this clip.",
    "Identify the current driving maneuver.",
    "Determine the scene category for the ego vehicle.",
    "Perform a scenario classification for this video.",
    "What is the standard tag for this driving situation?",
    "Annotate this clip with the correct scenario label.",
    "What is the car doing right now?",
    "Describe the current traffic situation and the ego car's action.",
    "Why did the car stop or maneuver like this?",
    "What is happening in front of the ego vehicle?",
    "Can you explain the ego vehicle's current behavior?",
    "Interpret the driving scene shown in the video.",
    "Look at the video and tell me what the scenario is.",
    "What kind of intersection or road event is this?"
]

class CotGenerator:
    """CoT生成器，使用完整的15帧序列"""
    
    def __init__(self, api_key: str, max_frames: int = 15):
        self.api_key = api_key
        self.model_name = "qwen-vl-plus"
        self.max_retries = 3
        self.retry_delay = 2
        self.max_frames = max_frames
        
        # 设置API密钥
        dashscope.api_key = api_key
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """将图片编码为base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"编码图片失败 {image_path}: {str(e)}")
            return None
    
    def generate_cot(self, frame_paths: List[str], true_label: str) -> Tuple[Optional[Dict], str]:
        """生成CoT分析（使用完整的15帧序列）"""
        if not frame_paths:
            return None, "没有图片帧"
        
        # 确保帧数不超过限制
        if len(frame_paths) > self.max_frames:
            logger.warning(f"帧数超过限制 ({len(frame_paths)} > {self.max_frames})，截取前{self.max_frames}帧")
            frame_paths = frame_paths[:self.max_frames]
        
        # 检查文件是否存在
        valid_frames = []
        for i, frame_path in enumerate(frame_paths):
            if os.path.exists(frame_path):
                valid_frames.append(frame_path)
            else:
                logger.warning(f"帧文件不存在: {frame_path}")
        
        if not valid_frames:
            return None, "所有帧文件都不存在"
        
        # 记录帧的时序信息
        total_frames = len(valid_frames)
        logger.info(f"使用{total_frames}帧进行分析，时间顺序: 帧1 → 帧{total_frames}")
        
        # 构建消息
        user_prompt = f"""Here are {total_frames} consecutive frames from a driving video clip, showing the complete scenario in strict temporal order (Frame 1 → Frame {total_frames}).
The GROUND TRUTH label for this scenario is: **"{true_label}"**

Please analyze the entire sequence of {total_frames} frames in chronological order to generate a comprehensive "Chain of Thought" analysis that strictly supports this label.
Your response must be a valid JSON object with the following fields:

1.  **"Observation"**: Describe the visual scene across all frames in chronological order. Focus on:
    * Road geometry and its changes throughout the sequence
    * Traffic control devices (lights, signs) and their state changes
    * Positions, movements, and trajectories of all relevant agents (vehicles, VRUs)
    * Temporal progression and dynamic changes between consecutive frames
    * Spatial relationships between objects and how they evolve
    * *Constraint*: Do not mention the label name. Describe what you see in the complete sequence.

2.  **"Reasoning"**: Connect the complete sequence of observations to the label:
    * Explain the causal relationships and decision-making process
    * Describe the temporal dynamics and key events in order
    * Identify critical moments and their timing in the sequence
    * Explain how the entire sequence justifies the label

3.  **"Tag"**: Exactly output: "{true_label}"

**Important Constraints:**
* Analyze all {total_frames} frames in chronological order
* Consider the complete temporal dynamics and causality
* Base analysis on visual evidence, but trust the ground truth
* The "Observation" must imply the "Tag" without explicitly stating it
* Output only valid JSON."""
        
        for attempt in range(self.max_retries):
            try:
                # 构建消息内容 - 按顺序添加所有图片
                message_content = []
                
                # 按顺序添加所有图片
                for i, frame_path in enumerate(valid_frames):
                    image_base64 = self.encode_image_to_base64(frame_path)
                    if image_base64:
                        message_content.append({
                            'image': f"data:image/jpeg;base64,{image_base64}"
                        })
                
                if not message_content:
                    return None, "无法编码任何图片"
                
                # 添加文本提示
                message_content.append({
                    'text': user_prompt
                })
                
                messages = [
                    {
                        'role': 'user',
                        'content': message_content
                    }
                ]
                
                logger.info(f"发送{len(valid_frames)}帧（完整序列）进行分析")
                
                # 增加最大tokens，因为15帧需要更详细的分析
                response = MultiModalConversation.call(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=3000
                )
                
                if response.status_code == 200:
                    cot_text = response.output.choices[0].message.content[0]['text']
                    
                    # 提取和解析JSON
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', cot_text, re.DOTALL)
                        if json_match:
                            cot_json = json.loads(json_match.group())
                        else:
                            cot_json = json.loads(cot_text)
                        
                        # 验证JSON结构
                        required_fields = ["Observation", "Reasoning", "Tag"]
                        for field in required_fields:
                            if field not in cot_json:
                                raise ValueError(f"缺少必要字段: {field}")
                        
                        if cot_json["Tag"] != true_label:
                            logger.warning(f"标签不匹配: 期望 {true_label}, 得到 {cot_json['Tag']}")
                            cot_json["Tag"] = true_label
                        
                        # 添加帧信息
                        cot_json["frames_used"] = len(valid_frames)
                        cot_json["frame_sequence"] = "完整时序序列"
                        cot_json["frame_count"] = len(frame_paths)
                        
                        return cot_json, ""
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"第{attempt+1}次尝试: JSON解析失败: {e}")
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                            continue
                        else:
                            return None, f"无法解析JSON响应: {str(e)}"
                else:
                    error_msg = f"API调用失败，状态码: {response.status_code}, 消息: {response.message}"
                    logger.error(f"第{attempt+1}次尝试: {error_msg}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return None, error_msg
                        
            except Exception as e:
                logger.error(f"第{attempt+1}次尝试: API调用失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return None, f"API调用失败: {str(e)}"
        
        return None, "所有尝试都失败"

class DatasetBuilder:
    """数据集构建器，创建指令微调数据集（使用抽帧图片）"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.processed_data = []
        self.failed_samples = []
        
    def load_samples(self, data_file: str = "simple_dataset.json") -> List[Dict]:
        """加载样本数据"""
        file_path = os.path.join(self.data_dir, "converted_annotations", data_file)
        
        if not os.path.exists(file_path):
            logger.error(f"数据文件不存在: {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        logger.info(f"加载了 {len(samples)} 个样本")
        return samples
    
    def build_finetuning_sample(self, sample: Dict, cot_result: Dict) -> Dict:
        """构建单条微调样本（使用完整的15帧序列）"""
        # 随机选择一个用户问题
        user_question = random.choice(QUESTION_LIST)
        
        # 获取帧路径
        frame_paths = sample.get("frame_paths", [])
        if not frame_paths:
            return None
        
        # 确保帧路径是绝对路径
        absolute_frame_paths = []
        for rel_path in frame_paths:
            abs_path = os.path.join(self.data_dir, rel_path)
            if os.path.exists(abs_path):
                absolute_frame_paths.append(abs_path)
        
        if not absolute_frame_paths:
            logger.warning(f"样本 {sample.get('id')} 没有有效的帧文件")
            return None
        
        # 保持完整的15帧序列
        MAX_FRAMES = 15
        if len(absolute_frame_paths) > MAX_FRAMES:
            logger.warning(f"帧数超过{MAX_FRAMES}，截取前{MAX_FRAMES}帧")
            selected_frames = absolute_frame_paths[:MAX_FRAMES]
        else:
            selected_frames = absolute_frame_paths
        
        logger.info(f"构建样本 {sample.get('id')}: 使用{len(selected_frames)}帧（完整时序）")
        
        # 构建dashscope格式的对话
        message_content = []
        for i, img_path in enumerate(selected_frames):
            # 编码图片为base64
            try:
                with open(img_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                message_content.append({
                    "image": f"data:image/jpeg;base64,{image_base64}"
                })
            except Exception as e:
                logger.error(f"编码图片失败 {img_path}: {str(e)}")
                continue
        
        if not message_content:
            logger.warning(f"无法编码任何图片: {sample.get('id')}")
            return None
        
        message_content.append({
            "text": user_question
        })
        
        conversations = [
            {
                "role": "user",
                "content": message_content
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "text": json.dumps(cot_result, ensure_ascii=False)
                    }
                ]
            }
        ]
        
        return {
            "id": sample.get("id", ""),
            "question": user_question,
            "frame_paths": selected_frames,
            "total_frames": len(absolute_frame_paths),
            "frames_used": len(selected_frames),
            "label_zh": sample.get("label_zh", ""),
            "label_en": sample.get("label_en", ""),
            "conversations": conversations,
            "cot": cot_result
        }
    
    def process_samples(self, samples: List[Dict], generator: CotGenerator, 
                       max_workers: int = 2, max_samples: int = None):
        """处理样本，生成CoT（使用完整的15帧序列）"""
        if max_samples:
            samples = samples[:max_samples]
        
        logger.info(f"开始处理 {len(samples)} 个样本，使用完整15帧时序序列")
        
        # 顺序处理，避免API限制
        for i, sample in enumerate(tqdm(samples, desc="生成CoT")):
            try:
                sample_id = sample.get("id", f"sample_{i}")
                label_en = sample.get("label_en", "")
                frame_paths = sample.get("frame_paths", [])
                
                if not frame_paths:
                    logger.warning(f"样本 {sample_id} 没有帧路径，跳过")
                    self.failed_samples.append({
                        "id": sample_id,
                        "reason": "没有帧路径"
                    })
                    continue
                
                # 将相对路径转换为绝对路径
                absolute_frame_paths = []
                for rel_path in frame_paths:
                    abs_path = os.path.join(self.data_dir, rel_path)
                    if os.path.exists(abs_path):
                        absolute_frame_paths.append(abs_path)
                
                if not absolute_frame_paths:
                    logger.warning(f"样本 {sample_id} 没有有效的帧文件，跳过")
                    self.failed_samples.append({
                        "id": sample_id,
                        "reason": "帧文件不存在"
                    })
                    continue
                
                # 记录帧的时序信息
                logger.info(f"处理样本 {i+1}/{len(samples)}: {sample_id} (包含{len(absolute_frame_paths)}帧，时间顺序: 1→{len(absolute_frame_paths)})")
                
                # 生成CoT
                start_time = time.time()
                cot_result, error = generator.generate_cot(absolute_frame_paths, label_en)
                end_time = time.time()
                
                logger.info(f"CoT生成耗时: {end_time - start_time:.2f}秒")
                
                if cot_result:
                    # 构建微调样本
                    finetuning_sample = self.build_finetuning_sample(sample, cot_result)
                    if finetuning_sample:
                        self.processed_data.append(finetuning_sample)
                        logger.info(f"成功生成CoT: {sample_id} (使用{len(absolute_frame_paths)}帧)")
                    else:
                        self.failed_samples.append({
                            "id": sample_id,
                            "reason": "无法构建微调样本"
                        })
                else:
                    self.failed_samples.append({
                        "id": sample_id,
                        "reason": error
                    })
                    logger.error(f"生成CoT失败 {sample_id}: {error}")
                
                # 避免API限制，增加延迟
                if i < len(samples) - 1:
                    wait_time = 3  # 3秒延迟，15帧处理更耗时
                    logger.info(f"等待{wait_time}秒后处理下一个样本...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                logger.error(f"处理样本失败 {sample.get('id', f'sample_{i}')}: {str(e)}")
                logger.error(traceback.format_exc())
                self.failed_samples.append({
                    "id": sample.get("id", f"sample_{i}"),
                    "reason": f"异常: {str(e)}"
                })
    
    def save_results(self):
        """保存结果"""
        # 创建输出目录
        output_path = os.path.join(self.output_dir, "finetuning_dataset")
        os.makedirs(output_path, exist_ok=True)
        
        # 1. 保存完整数据集
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_path, f"cot_dataset_{timestamp}.json")
        
        dataset = {
            "version": "3.0.0",
            "description": "Qwen-VL finetuning dataset with CoT reasoning (Frame Input)",
            "created": datetime.now().isoformat(),
            "statistics": {
                "total_samples": len(self.processed_data),
                "failed_samples": len(self.failed_samples),
                "success_rate": len(self.processed_data) / (len(self.processed_data) + len(self.failed_samples)) if self.processed_data else 0
            },
            "data": self.processed_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存完整数据集: {output_file} ({len(self.processed_data)} 个样本)")
        
        # 2. 保存失败样本
        if self.failed_samples:
            failed_file = os.path.join(output_path, f"failed_samples_{timestamp}.json")
            with open(failed_file, 'w', encoding='utf-8') as f:
                json.dump(self.failed_samples, f, ensure_ascii=False, indent=2)
            logger.info(f"保存失败样本: {failed_file} ({len(self.failed_samples)} 个)")
        
        # 3. 保存简化格式
        simple_data = []
        for sample in self.processed_data:
            simple_item = {
                "id": sample["id"],
                "question": sample["question"],
                "frames_used": sample["frames_used"],
                "total_frames": sample["total_frames"],
                "label_en": sample["label_en"],
                "cot": sample["cot"]
            }
            simple_data.append(simple_item)
        
        simple_file = os.path.join(output_path, f"simple_cot_dataset_{timestamp}.json")
        with open(simple_file, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存简化数据集: {simple_file}")
        
        return output_path

def main():
    """主函数"""
    DATA_DIR = "/root/workspace/vqa_dataset_prepared"
    OUTPUT_DIR = "/root/workspace/vqa_dataset_cot"
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    
    print("=" * 60)
    print("CoT指令数据集生成工具（抽帧图片版本）")
    print("=" * 60)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📦 输出目录: {OUTPUT_DIR}")
    
    # 检查API密钥
    if not API_KEY:
        logger.error("请设置DASHSCOPE_API_KEY环境变量")
        print("请设置环境变量: export DASHSCOPE_API_KEY='your-api-key'")
        return
    
    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        logger.error(f"数据目录不存在: {DATA_DIR}")
        print("请先运行标签转换和抽帧脚本")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化生成器和数据集构建器
    generator = CotGenerator(api_key=API_KEY, max_frames=15)
    builder = DatasetBuilder(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    
    # 加载样本
    samples = builder.load_samples("simple_dataset.json")
    if not samples:
        logger.error("没有找到样本数据")
        return
    
    # 处理样本（可以设置max_samples限制处理数量，用于测试）
    max_samples = None  # 设为None处理所有样本，或设为数字测试
    if max_samples:
        print(f"⚠️  测试模式: 只处理前 {max_samples} 个样本")
    
    builder.process_samples(
        samples=samples,
        generator=generator,
        max_workers=1,  # 顺序处理避免API限制
        max_samples=max_samples
    )
    
    # 保存结果
    output_path = builder.save_results()
    
    # 输出统计信息
    print("\n" + "=" * 60)
    print("🎉 CoT数据集生成完成（抽帧图片版本）")
    print("=" * 60)
    
    total_processed = len(builder.processed_data) + len(builder.failed_samples)
    success_count = len(builder.processed_data)
    fail_count = len(builder.failed_samples)
    
    print(f"📊 处理统计:")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ❌ 失败: {fail_count}")
    print(f"  📈 成功率: {success_count/total_processed*100:.1f}%" if total_processed > 0 else "0%")
    
    print(f"\n📁 输出目录: {output_path}")
    print("生成的文件:")
    for file in os.listdir(output_path):
        file_path = os.path.join(output_path, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)")
    
    print(f"\n📋 数据集格式示例:")
    if builder.processed_data:
        sample = builder.processed_data[0]
        print(f"\n样本ID: {sample['id']}")
        print(f"问题: {sample['question']}")
        print(f"使用帧数: {sample['frames_used']}/{sample['total_frames']}")
        print(f"标签: {sample['label_en']}")
        print(f"CoT:")
        cot = sample['cot']
        print(f"  Observation: {cot.get('Observation', '')[:100]}...")
        print(f"  Reasoning: {cot.get('Reasoning', '')[:100]}...")
        print(f"  Tag: {cot.get('Tag', '')}")
    
    print(f"\n🚀 特点:")
    print("✓ 使用抽帧后的图片，避免视频处理问题")
    print("✓ 精确控制帧序列，支持时间序列分析")
    print("✓ 自动选择关键帧，优化API使用效率")
    print("✓ 更稳定的处理流程")
    
    print("=" * 60)

if __name__ == "__main__":
    main()