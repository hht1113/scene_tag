#!/usr/bin/env python3
"""
Qwen3-VL-2B 视频VQA微调脚本
修复对话格式问题，与推理脚本完全一致
"""

import os
import json
import torch
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import AutoModelForImageTextToText, AutoProcessor, TrainingArguments, Trainer
import logging
from datetime import datetime
from tqdm import tqdm
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def extract_frames_from_video(video_path: str, num_frames: int = 60) -> List[str]:
    """
    从视频中提取帧（每秒1帧，最多60帧）
    返回base64编码的图片列表，与推理脚本完全一致
    """
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        return []
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        if fps <= 0:
            fps = 30
        
        # 计算要提取的帧索引（每秒1帧）
        frames_to_extract = []
        for i in range(min(num_frames, total_frames // fps)):
            frame_idx = i * fps
            if frame_idx < total_frames:
                frames_to_extract.append(frame_idx)
        
        # 提取帧
        frames_base64 = []
        for frame_idx in frames_to_extract:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame = blank_frame
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小
            h, w = frame.shape[:2]
            if h > 360 or w > 640:
                frame = cv2.resize(frame, (640, 360))
            
            pil_image = Image.fromarray(frame)
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            frames_base64.append(img_base64)
        
        cap.release()
        
        # 用空白帧补齐
        while len(frames_base64) < num_frames:
            blank_frame = np.zeros((360, 640, 3), dtype=np.uint8)
            pil_image = Image.fromarray(blank_frame.astype(np.uint8))
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            frames_base64.append(img_base64)
        
        return frames_base64[:num_frames]
        
    except Exception as e:
        logger.error(f"提取视频帧失败 {video_path}: {str(e)}")
        return []

def build_vqa_prompt(question: str, video_duration: int = 60) -> str:
    """
    构建视频VQA的prompt
    与推理脚本完全一致
    """
    prompt = f"""You are watching a {video_duration}-second video of driving scenarios. The video is sampled at 1 frame per second, showing {video_duration} consecutive seconds.

Question: {question}

Please analyze the ego vehicle's behavior and provide the following information:
1. Identify the driving maneuver(s) performed by the ego vehicle
2. Specify the start time and end time for each action (in seconds)
3. Use the format: <driving_maneuver>action_label</driving_maneuver> from <start_time>start_time_value</start_time> to <end_time>end_time_value</end_time> seconds.

If there are multiple actions, list them in chronological order separated by " and ".

Answer:"""
    
    return prompt

def prepare_conversation_format_inference(images_base64: List[str], prompt: str) -> List[Dict]:
    """
    准备符合Qwen-VL对话格式的数据（仅用户输入）
    与推理脚本的prepare_conversation_format函数完全一致
    """
    user_content = []
    
    for img_base64 in images_base64:
        user_content.append({
            "type": "image",
            "image": img_base64
        })
    
    user_content.append({
        "type": "text",
        "text": prompt
    })
    
    conversations = [
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    return conversations

def prepare_conversation_format_training(images_base64: List[str], prompt: str, answer: str) -> List[Dict]:
    """
    准备训练用的对话格式（用户输入 + 助手回复）
    训练时需要包含助手回复
    """
    user_content = []
    
    for img_base64 in images_base64:
        user_content.append({
            "type": "image",
            "image": img_base64
        })
    
    user_content.append({
        "type": "text",
        "text": prompt
    })
    
    conversations = [
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": answer
        }
    ]
    
    return conversations

class VideoVQADataset(Dataset):
    """视频VQA数据集类"""
    
    def __init__(self, data_path: str, processor, max_samples: int = None, num_frames: int = 8):
        self.processor = processor
        self.num_frames = num_frames
        self.samples = []
        
        logger.info(f"加载数据集: {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "data" in data:
            raw_samples = data["data"]
        else:
            raw_samples = data
        
        if max_samples and len(raw_samples) > max_samples:
            raw_samples = random.sample(raw_samples, max_samples)
        
        # 处理样本
        for i, sample in enumerate(tqdm(raw_samples, desc="处理样本")):
            try:
                video_path = sample["video_path"]
                question = sample["question"]
                answer = sample["answer"]
                video_duration = sample.get("video_duration", 60)
                
                # 提取帧
                images_base64 = extract_frames_from_video(video_path, self.num_frames)
                if len(images_base64) != self.num_frames:
                    logger.warning(f"样本 {i} 帧数不正确: {len(images_base64)} 帧，期望 {self.num_frames} 帧")
                    continue
                
                # 构建prompt
                prompt = build_vqa_prompt(question, video_duration)
                
                # 构建对话 - 训练时包含用户和助手消息
                conversation = prepare_conversation_format_training(images_base64, prompt, answer)
                
                self.samples.append({
                    "conversation": conversation,
                    "video_path": video_path,
                    "question": question,
                    "answer": answer
                })
                
            except Exception as e:
                logger.warning(f"处理样本 {i} 失败: {str(e)}")
                continue
        
        logger.info(f"成功加载 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 使用处理器处理对话
        try:
            # 应用聊天模板
            text = self.processor.apply_chat_template(
                item["conversation"],
                tokenize=False,
                add_generation_prompt=False
            )
            
            # 提取图片
            images = []
            for content in item["conversation"][0]["content"]:
                if content["type"] == "image":
                    # 解码base64图片
                    image_data = base64.b64decode(content["image"])
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    images.append(image)
            
            # 处理输入
            inputs = self.processor(
                text=[text],
                images=[images],
                return_tensors="pt",
                padding=True
            )
            
            # 处理标签
            with self.processor.tokenizer.as_target_tokenizer():
                labels = self.processor.tokenizer(
                    item["answer"],
                    padding=True,
                    return_tensors="pt",
                    max_length=512
                )
            
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0),
                "labels": labels["input_ids"].squeeze(0)
            }
            
        except Exception as e:
            logger.error(f"处理样本 {idx} 失败: {str(e)}")
            # 返回一个空样本
            return {
                "input_ids": torch.zeros(1, 10, dtype=torch.long),
                "attention_mask": torch.zeros(1, 10, dtype=torch.long),
                "pixel_values": torch.zeros(1, 3, 360, 640),
                "labels": torch.zeros(1, 10, dtype=torch.long)
            }

@dataclass
class DataCollatorForVideoVQA:
    """数据收集器"""
    processor: Any
    
    def __call__(self, features: List[Dict]) -> Dict[str, Any]:
        batch = {}
        
        # 分离输入特征
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        pixel_values = [f["pixel_values"] for f in features]
        labels = [f["labels"] for f in features]
        
        # 填充input_ids和attention_mask
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        
        # 填充labels
        batch_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        # 堆叠pixel_values
        batch_pixel_values = torch.stack(pixel_values)
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "pixel_values": batch_pixel_values,
            "labels": batch_labels
        }

def train_model():
    """训练模型主函数"""
    print("=" * 60)
    print("Qwen3-VL-2B 视频VQA训练 (修复对话格式版)")
    print("=" * 60)
    
    # 配置
    MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
    TRAIN_DATA = "/root/workspace/video_vqa_dataset/video_vqa_dataset_20251231_162049/train.json"
    OUTPUT_DIR = f"./qwen3_vl_video_vqa_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 检查数据
    if not os.path.exists(TRAIN_DATA):
        print(f"❌ 训练数据不存在: {TRAIN_DATA}")
        return
    
    # 1. 加载模型和处理器
    print("加载模型和处理器...")
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        print("✅ 模型和处理器加载成功")
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return
    
    # 2. 创建数据集
    print("\n创建数据集...")
    train_dataset = VideoVQADataset(
        data_path=TRAIN_DATA,
        processor=processor,
        max_samples=20,  # 限制样本数用于测试
        num_frames=8  # 减少帧数避免显存问题
    )
    
    if len(train_dataset) == 0:
        print("❌ 没有可用的训练数据")
        return
    
    print(f"数据集大小: {len(train_dataset)} 个样本")
    
    # 3. 创建数据收集器
    data_collator = DataCollatorForVideoVQA(processor=processor)
    
    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        dataloader_pin_memory=False,
        fp16=True,
        gradient_checkpointing=True,
    )
    
    # 5. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 6. 开始训练
    print(f"\n开始训练，共 {training_args.num_train_epochs} 个epoch")
    print(f"输出目录: {OUTPUT_DIR}")
    
    try:
        train_result = trainer.train()
        print(f"\n✅ 训练完成！最终损失: {train_result.training_loss:.4f}")
        
        # 保存模型
        trainer.save_model()
        processor.save_pretrained(OUTPUT_DIR)
        print(f"模型保存到: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

def test_conversation_format():
    """测试对话格式是否正确"""
    print("测试对话格式...")
    
    # 加载处理器
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    # 创建测试图片
    test_images = []
    for _ in range(8):
        test_image = Image.new('RGB', (640, 360), color='red')
        buffered = BytesIO()
        test_image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        test_images.append(img_base64)
    
    # 构建prompt
    question = "What is the ego vehicle's action in the video?"
    answer = "The ego vehicle performs <driving_maneuver>Single_lane_driving</driving_maneuver> from <start_time>0</start_time> to <end_time>60</end_time> seconds."
    
    prompt = build_vqa_prompt(question, 60)
    
    # 构建对话
    conversation = prepare_conversation_format_training(test_images, prompt, answer)
    
    print(f"对话长度: {len(conversation)}")
    print(f"用户消息内容类型: {[c['type'] for c in conversation[0]['content']]}")
    print(f"助手消息: {conversation[1]['content']}")
    
    # 测试处理器
    try:
        # 提取图片
        images = []
        for content in conversation[0]["content"]:
            if content["type"] == "image":
                # 解码base64图片
                image_data = base64.b64decode(content["image"])
                image = Image.open(BytesIO(image_data)).convert("RGB")
                images.append(image)
        
        # 应用聊天模板
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        print(f"处理后的文本长度: {len(text)} 字符")
        
        # 处理输入
        inputs = processor(
            text=[text],
            images=[images],
            return_tensors="pt",
            padding=True
        )
        
        print("✅ 对话格式测试通过")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        
    except Exception as e:
        print(f"❌ 对话格式测试失败: {e}")
        import traceback
        traceback.print_exc()

def validate_model_loading():
    """验证模型加载和基本推理"""
    print("\n验证模型加载...")
    
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            trust_remote_code=True
        )
        
        print("✅ 模型加载成功")
        
        # 测试推理
        test_images = []
        for _ in range(4):
            test_image = Image.new('RGB', (640, 360), color='blue')
            buffered = BytesIO()
            test_image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            test_images.append(img_base64)
        
        # 构建prompt
        question = "What is in the image?"
        prompt = build_vqa_prompt(question, 60)
        
        # 构建对话
        conversation = prepare_conversation_format_inference(test_images, prompt)
        
        # 提取图片
        images = []
        for content in conversation[0]["content"]:
            if content["type"] == "image":
                image_data = base64.b64decode(content["image"])
                image = Image.open(BytesIO(image_data)).convert("RGB")
                images.append(image)
        
        # 应用聊天模板
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 处理输入
        inputs = processor(
            text=[text],
            images=[images],
            return_tensors="pt",
            padding=True
        )
        
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Pixel values shape: {inputs['pixel_values'].shape}")
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                pixel_values=inputs["pixel_values"].to(model.device),
                max_new_tokens=50
            )
        
        decoded = processor.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 推理测试通过")
        print(f"生成结果: {decoded[:100]}...")
        
    except Exception as e:
        print(f"❌ 模型验证失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. 测试对话格式
    test_conversation_format()
    
    print("\n" + "=" * 60)
    
    # 2. 验证模型加载
    validate_model_loading()
    
    print("\n" + "=" * 60)
    
    # 3. 检查GPU
    if not torch.cuda.is_available():
        print("❌ 需要GPU进行训练")
        print("建议使用至少16GB显存的GPU")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🎮 GPU: {gpu_name}")
        print(f"   显存: {gpu_memory:.1f} GB")
        
        if gpu_memory < 16:
            print("⚠️  警告: 可能需要较多显存")
    
    # 询问是否继续
    response = input("\n是否开始训练？(y/n): ")
    if response.lower() == 'y':
        train_model()
    else:
        print("训练已取消")