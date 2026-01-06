import os
import json
import torch
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/video_vqa_inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoVQAInference:
    """视频VQA推理类"""
    
    def __init__(self, model_path: str, test_file: str, output_dir: str):
        """
        初始化推理类
        
        Args:
            model_path: 模型检查点路径
            test_file: 测试集文件路径
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.test_file = test_file
        self.output_dir = output_dir
        
        # 检查路径
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"测试文件不存在: {test_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设备设置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 模型和tokenizer
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model(self):
        """加载模型和tokenizer"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            from peft import PeftModel, PeftConfig
            
            logger.info("开始加载模型...")
            
            # 首先加载base model
            base_model_path = "Qwen/Qwen3-VL-2B-Instruct"
            logger.info(f"加载base model: {base_model_path}")
            
            # 加载配置
            config = AutoConfig.from_pretrained(
                base_model_path,
                trust_remote_code=True
            )
            
            # 设置最大截断序列为4096
            config.max_position_embeddings = 4096
            config.model_max_length = 4096
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                model_max_length=4096
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Tokenizer加载完成，pad_token: {self.tokenizer.pad_token}")
            
            # 加载base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                config=config,
                torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            # 加载LoRA权重
            logger.info(f"加载LoRA权重: {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            
            # 合并LoRA权重到基础模型
            logger.info("合并LoRA权重到基础模型...")
            self.model = self.model.merge_and_unload()
            
            self.model.eval()
            logger.info(f"模型加载完成，参数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        logger.info(f"加载测试数据: {self.test_file}")
        
        with open(self.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        logger.info(f"加载了 {len(test_data)} 个测试样本")
        
        # 检查数据结构
        if test_data and len(test_data) > 0:
            sample = test_data[0]
            logger.info(f"样本结构: {list(sample.keys())}")
            logger.info(f"样本示例 - instruction: {sample.get('instruction', '')[:50]}...")
            logger.info(f"样本示例 - output: {sample.get('output', '')[:50]}...")
            
        return test_data
    
    def generate_prompt(self, sample: Dict) -> str:
        """生成提示词"""
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        
        if input_text:
            return f"{instruction}\n{input_text}"
        else:
            return instruction
    
    def generate_response(self, prompt: str, videos: List[str], 
                         max_new_tokens: int = 512, 
                         temperature: float = 0.1) -> str:
        """
        生成响应
        
        注意：Qwen3-VL是多模态模型，但在这个版本中我们只处理文本
        视频信息在instruction中通过<video>标记表示
        """
        try:
            # 准备对话
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"应用聊天模板失败，使用原始提示: {e}")
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize输入
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=4096 - max_new_tokens
            ).to(self.device)
            
            input_length = inputs.input_ids.shape[1]
            logger.debug(f"输入token长度: {input_length}")
            
            # 检查是否超过最大长度
            if input_length > 4096 - max_new_tokens:
                logger.warning(f"输入长度 {input_length} 过长，可能被截断")
            
            # 生成参数
            generation_kwargs = {
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0.01,
                "top_p": 0.9 if temperature > 0.01 else None,
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # 生成
            with torch.no_grad():
                try:
                    # 使用generate方法
                    generated_ids = self.model.generate(**generation_kwargs)
                    
                    # 提取生成的文本
                    generated_ids = generated_ids[0, inputs.input_ids.shape[1]:]
                    response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # 清理响应
                    response = response.strip()
                    
                    # 移除可能的停止序列
                    stop_sequences = ["<|im_end|>", "</s>", "<|endoftext|>", "\n\n\n"]
                    for stop_seq in stop_sequences:
                        if response.endswith(stop_seq):
                            response = response[:-len(stop_seq)].strip()
                    
                    return response
                    
                except Exception as e:
                    logger.error(f"生成过程中出错: {str(e)}")
                    return f"生成错误: {str(e)}"
            
        except Exception as e:
            logger.error(f"生成响应失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error: {str(e)}"
    
    def batch_inference(self, test_data: List[Dict]) -> List[Dict]:
        """
        批量推理
        
        Args:
            test_data: 测试数据
            
        Returns:
            包含预测结果的数据列表
        """
        results = []
        
        logger.info(f"开始推理，共 {len(test_data)} 个样本")
        
        for i, sample in enumerate(tqdm(test_data, desc="推理进度")):
            try:
                # 提取数据
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                gt_answer = sample.get("output", "")
                video_paths = sample.get("videos", [])
                
                if not instruction:
                    logger.warning(f"样本 {i} 没有instruction字段")
                    continue
                
                # 检查视频路径
                video_exists = all(os.path.exists(v) for v in video_paths) if video_paths else False
                
                # 生成提示词
                prompt = self.generate_prompt(sample)
                
                # 检查<video>标记
                has_video_tag = "<video>" in instruction
                video_tag_count = instruction.count("<video>")
                video_count = len(video_paths) if video_paths else 0
                
                # 生成响应
                pred_answer = self.generate_response(
                    prompt, 
                    video_paths,
                    max_new_tokens=512,
                    temperature=0.1
                )
                
                # 构建结果
                result = {
                    "sample_id": i,
                    "instruction": instruction,
                    "input": input_text,
                    "ground_truth": gt_answer,
                    "prediction": pred_answer,
                    "video_paths": video_paths,
                    "video_exists": video_exists,
                    "has_video_tag": has_video_tag,
                    "video_tag_count": video_tag_count,
                    "video_count": video_count
                }
                
                results.append(result)
                
                # 每10个样本记录一次
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(test_data)} 个样本")
                    # 显示最近一个样本的示例
                    logger.info(f"  示例 {i} - 输入: {instruction[:50]}...")
                    logger.info(f"  示例 {i} - 预测: {pred_answer[:50]}...")
                
                # 清理显存
                if torch.cuda.is_available() and (i + 1) % 20 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"处理样本 {i} 失败: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                # 添加错误样本
                error_result = {
                    "sample_id": i,
                    "instruction": sample.get("instruction", ""),
                    "input": sample.get("input", ""),
                    "ground_truth": sample.get("output", ""),
                    "prediction": f"ERROR: {str(e)}",
                    "video_paths": sample.get("videos", []),
                    "video_exists": False,
                    "has_video_tag": False,
                    "video_tag_count": 0,
                    "video_count": 0,
                    "error": str(e)
                }
                results.append(error_result)
        
        successful_count = len([r for r in results if 'ERROR' not in r.get('prediction', '')])
        logger.info(f"推理完成，成功处理 {successful_count}/{len(test_data)} 个样本")
        return results
    
    def save_results(self, results: List[Dict]):
        """保存推理结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_file = os.path.join(self.output_dir, f"inference_results_{timestamp}.json")
        
        result_data = {
            "model_path": self.model_path,
            "test_file": self.test_file,
            "timestamp": timestamp,
            "total_samples": len(results),
            "successful_samples": len([r for r in results if 'ERROR' not in r.get('prediction', '')]),
            "failed_samples": len([r for r in results if 'ERROR' in r.get('prediction', '')]),
            "results": results
        }
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2, separators=(',', ': '))
        
        logger.info(f"详细结果已保存: {detailed_file}")
        
        # 计算基本统计
        successful_results = [r for r in results if 'ERROR' not in r.get('prediction', '')]
        if successful_results:
            # 计算平均长度
            avg_gt_len = sum(len(str(r.get('ground_truth', ''))) for r in successful_results) / len(successful_results)
            avg_pred_len = sum(len(str(r.get('prediction', ''))) for r in successful_results) / len(successful_results)
            
            # 统计视频标记一致性
            consistent_count = sum(1 for r in successful_results if r.get('video_tag_count', 0) == r.get('video_count', 0))
            video_exists_count = sum(1 for r in successful_results if r.get('video_exists', False))
            
            stats = {
                "avg_ground_truth_length": avg_gt_len,
                "avg_prediction_length": avg_pred_len,
                "video_tag_consistent_samples": consistent_count,
                "video_exists_samples": video_exists_count
            }
            
            stats_file = os.path.join(self.output_dir, f"stats_{timestamp}.txt")
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"模型路径: {self.model_path}\n")
                f.write(f"测试文件: {self.test_file}\n")
                f.write(f"时间戳: {timestamp}\n")
                f.write(f"总样本数: {len(results)}\n")
                f.write(f"成功推理: {len(successful_results)}\n")
                f.write(f"失败推理: {len(results) - len(successful_results)}\n")
                f.write(f"成功率: {len(successful_results)/len(results)*100:.2f}%\n")
                f.write(f"平均GT长度: {avg_gt_len:.2f} 字符\n")
                f.write(f"平均预测长度: {avg_pred_len:.2f} 字符\n")
                f.write(f"视频标记一致样本: {consistent_count}/{len(successful_results)} ({(consistent_count/len(successful_results))*100:.2f}%)\n")
                f.write(f"视频文件存在样本: {video_exists_count}/{len(successful_results)} ({(video_exists_count/len(successful_results))*100:.2f}%)\n")
            
            logger.info(f"统计信息已保存: {stats_file}")
        
        return detailed_file
    
    def run(self):
        """运行推理流程"""
        logger.info("=" * 60)
        logger.info("视频VQA推理开始")
        logger.info("=" * 60)
        logger.info(f"模型路径: {self.model_path}")
        logger.info(f"测试文件: {self.test_file}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"设备: {self.device}")
        logger.info(f"最大截断序列: 4096")
        logger.info("=" * 60)
        
        # 1. 加载模型
        logger.info("步骤 1/3: 加载模型...")
        self.load_model()
        
        # 2. 加载测试数据
        logger.info("步骤 2/3: 加载测试数据...")
        test_data = self.load_test_data()
        
        if not test_data:
            logger.error("测试数据为空，退出推理")
            return None, None
        
        # 3. 运行推理
        logger.info("步骤 3/3: 运行推理...")
        results = self.batch_inference(test_data)
        
        # 4. 保存结果
        logger.info("保存结果...")
        result_file = self.save_results(results)
        
        # 5. 打印总结
        self.print_summary(results, result_file)
        
        return results, result_file
    
    def print_summary(self, results: List[Dict], result_file: str):
        """打印总结信息"""
        successful_results = [r for r in results if 'ERROR' not in r.get('prediction', '')]
        failed_results = [r for r in results if 'ERROR' in r.get('prediction', '')]
        
        print("\n" + "=" * 60)
        print("推理完成总结")
        print("=" * 60)
        print(f"总样本数: {len(results)}")
        print(f"成功推理: {len(successful_results)}")
        print(f"失败推理: {len(failed_results)}")
        print(f"成功率: {len(successful_results)/len(results)*100:.2f}%")
        
        if successful_results:
            # 统计视频标记一致性
            consistent_count = sum(1 for r in successful_results if r.get('video_tag_count', 0) == r.get('video_count', 0))
            print(f"视频标记一致性: {consistent_count}/{len(successful_results)} ({(consistent_count/len(successful_results))*100:.2f}%)")
            
            # 视频文件存在性
            video_exists_count = sum(1 for r in successful_results if r.get('video_exists', False))
            print(f"视频文件存在: {video_exists_count}/{len(successful_results)} ({(video_exists_count/len(successful_results))*100:.2f}%)")
            
            # 平均长度
            avg_gt_len = sum(len(str(r.get('ground_truth', ''))) for r in successful_results) / len(successful_results)
            avg_pred_len = sum(len(str(r.get('prediction', ''))) for r in successful_results) / len(successful_results)
            print(f"平均GT长度: {avg_gt_len:.2f} 字符")
            print(f"平均预测长度: {avg_pred_len:.2f} 字符")
        
        print("\n" + "=" * 60)
        print("预测结果示例")
        print("=" * 60)
        
        for i, result in enumerate(successful_results[:3]):
            print(f"\n示例 {i + 1}:")
            print(f"  指令: {result.get('instruction', '')[:80]}...")
            print(f"  真实答案: {result.get('ground_truth', '')[:80]}...")
            print(f"  预测答案: {result.get('prediction', '')[:80]}...")
            print(f"  视频存在: {result.get('video_exists', False)}")
            print(f"  视频标记: {result.get('has_video_tag', False)}")
        
        print("\n" + "=" * 60)
        print("输出文件")
        print("=" * 60)
        print(f"结果文件: {result_file}")
        
        # 显示文件大小
        if os.path.exists(result_file):
            size_mb = os.path.getsize(result_file) / (1024 * 1024)
            print(f"文件大小: {size_mb:.2f} MB")
        
        # 查找统计文件
        import glob
        stats_files = glob.glob(os.path.join(self.output_dir, "stats_*.txt"))
        if stats_files:
            latest_stats = max(stats_files, key=os.path.getctime)
            print(f"统计文件: {latest_stats}")
        
        print("=" * 60)
        print("推理完成!")
        print("=" * 60)


def main():
    """主函数"""
    # 固定参数
    MODEL_PATH = "/root/workspace/LLaMA-Factory/saves/Qwen3-VL-2B-Instruct/lora/train_lora_2026-01-03-11-13-37-"
    TEST_FILE = "/root/workspace/llama_factory_vqa_dataset/llama_factory_vqa_20251231_210010/test.json"
    OUTPUT_DIR = "/root/workspace/video_vqa_inference_results"
    
    print("=" * 60)
    print("视频VQA推理脚本")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试文件: {TEST_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    
    # 检查测试文件是否存在
    if not os.path.exists(TEST_FILE):
        print(f"错误: 测试文件不存在: {TEST_FILE}")
        print("请检查测试文件路径")
        return
    
    # 检查模型路径
    if not os.path.exists(MODEL_PATH):
        print(f"警告: 模型路径不存在: {MODEL_PATH}")
        print("尝试加载base model和LoRA权重...")
    
    # 创建推理器
    try:
        inference = VideoVQAInference(
            model_path=MODEL_PATH,
            test_file=TEST_FILE,
            output_dir=OUTPUT_DIR
        )
        
        # 运行推理
        results, result_file = inference.run()
        
        if results and result_file:
            print(f"\n✅ 推理完成!")
            print(f"📁 结果文件: {result_file}")
            
            # 显示结果文件位置
            print(f"\n📊 您可以使用以下命令查看结果:")
            print(f"  cat {result_file} | head -n 100")
            print(f"  python -c \"import json; data=json.load(open('{result_file}')); print(f'总样本数: {len(data.get(\\\"results\\\", []))}')\"")
            
    except Exception as e:
        print(f"\n❌ 推理失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 设置最大分割大小
    torch.cuda.empty_cache()
    
    main()