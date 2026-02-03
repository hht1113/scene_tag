import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import time
import traceback
from tqdm import tqdm
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/sliced_label_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SlicedLabelConverter:
    """切片标签转换器，处理中英文标签映射（仅精确匹配）"""
    
    def __init__(self, mapping_file: str):
        self.mapping_file = mapping_file
        self.chinese_to_english = {}
        self.english_to_chinese = {}
        self.unmapped_labels = set()
        self.load_mapping()
    
    def load_mapping(self):
        """加载中英文对照表（精确匹配）"""
        try:
            if not os.path.exists(self.mapping_file):
                logger.warning(f"中英文对照表不存在: {self.mapping_file}")
                return
            
            df = pd.read_excel(self.mapping_file)
            logger.info(f"加载中英文对照表: {len(df)} 行")
            
            # 确保列名正确
            required_columns = ['中文标签', '英文标签']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"对照表缺少必要列: {col}")
                    return
            
            # 构建映射
            for _, row in df.iterrows():
                chinese = str(row['中文标签']).strip()
                english = str(row['英文标签']).strip()
                
                if chinese and english and chinese != 'nan' and english != 'nan':
                    self.chinese_to_english[chinese] = english
                    self.english_to_chinese[english] = chinese
                    logger.debug(f"映射: {chinese} -> {english}")
            
            logger.info(f"加载了 {len(self.chinese_to_english)} 个标签映射")
            
        except Exception as e:
            logger.error(f"加载标签映射失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    def convert_label(self, chinese_label: str) -> Tuple[str, bool]:
        """
        转换中文标签为英文标签
        返回: (英文标签, 是否成功映射)
        只进行精确匹配，不匹配则返回原中文标签
        """
        if not chinese_label:
            return "", False
        
        # 仅尝试完全匹配
        if chinese_label in self.chinese_to_english:
            return self.chinese_to_english[chinese_label], True
        else:
            # 记录未映射的标签
            self.unmapped_labels.add(chinese_label)
            logger.warning(f"未找到映射的标签: {chinese_label}")
            # 返回原中文标签
            return chinese_label, False

class SlicedAnnotationProcessor:
    """切片标注处理器，转换标签并准备视频VQA微调数据"""
    
    def __init__(self, annotations_dir: str, mapping_file: str, output_dir: str, 
                 slice_video_dir: str = None):
        self.annotations_dir = annotations_dir
        self.mapping_file = mapping_file
        self.output_dir = output_dir
        self.slice_video_dir = slice_video_dir
        self.label_converter = SlicedLabelConverter(mapping_file)
        
    def process_all_annotations(self) -> Dict:
        """处理所有切片标注文件"""
        # 查找所有标注文件
        annotation_files = []
        for file in os.listdir(self.annotations_dir):
            if file.endswith('.json') and file not in ['summary.json', 'all_sliced_annotations.json']:
                annotation_files.append(os.path.join(self.annotations_dir, file))
        
        logger.info(f"找到 {len(annotation_files)} 个切片标注文件")
        
        all_converted_data = []
        category_stats = {}
        
        for annotation_file in tqdm(annotation_files, desc="处理标注文件"):
            category_name = os.path.basename(annotation_file).replace('.json', '')
            category_data = self.process_single_file(annotation_file, category_name)
            
            if category_data:
                category_stats[category_name] = len(category_data)
                all_converted_data.extend(category_data)
        
        # 保存结果
        self.save_results(all_converted_data, category_stats)
        
        return {
            "total_samples": len(all_converted_data),
            "category_stats": category_stats,
            "unmapped_labels": list(self.label_converter.unmapped_labels)
        }
    
    def process_single_file(self, annotation_file: str, category_name: str) -> List[Dict]:
        """处理单个切片标注文件"""
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            
            converted_annotations = []
            
            for ann in tqdm(annotations, desc=f"处理 {category_name}", leave=False):
                converted = self.process_single_sliced_annotation(ann)
                if converted:
                    converted_annotations.append(converted)
            
            return converted_annotations
            
        except Exception as e:
            logger.error(f"处理文件失败 {annotation_file}: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def process_single_sliced_annotation(self, annotation: Dict) -> Optional[Dict]:
        """处理单个切片标注"""
        try:
            chinese_label = annotation.get('label', '')
            
            if not chinese_label:
                logger.warning(f"跳过无效标注: {annotation.get('id', 'unknown')}")
                return None
            
            # 转换标签
            english_label, mapped = self.label_converter.convert_label(chinese_label)
            
            # 获取切片视频路径 - 关键修改点！
            # 切片标注使用'slice_video_path'字段，而不是'original_video'
            video_path = annotation.get('slice_video_path', '')
            slice_key = annotation.get('slice_key', '')
            
            # 如果路径不存在，尝试从slice_key构造
            if not video_path or not os.path.exists(video_path):
                if slice_key:
                    # 尝试从slice_key构造路径
                    video_path = self._slice_key_to_video_path(slice_key)
                else:
                    video_path = ''
            
            # 验证视频是否存在
            video_exists = os.path.exists(video_path) if video_path else False
            
            # 如果视频不存在，尝试查找
            if not video_exists and video_path:
                # 尝试在切片视频目录中查找
                if self.slice_video_dir and os.path.exists(self.slice_video_dir):
                    # 从完整路径中提取文件名
                    filename = os.path.basename(video_path)
                    if filename:
                        # 在切片目录中递归查找文件
                        found_path = self._find_video_in_slice_dir(filename)
                        if found_path:
                            video_path = found_path
                            video_exists = True
                            logger.info(f"找到切片视频: {filename} -> {found_path}")
            
            if not video_exists and video_path:
                logger.warning(f"切片视频文件不存在: {video_path}")
            
            # 准备输出数据 - 适应切片视频结构
            result = {
                "id": annotation.get('id', ''),
                "slice_key": slice_key,
                "video_path": video_path,
                "video_exists": video_exists,
                "label_zh": chinese_label,
                "label_en": english_label,
                "label_mapped": mapped,
                "time_range_in_slice": annotation.get('time_range_in_slice', []),  # 在切片中的时间
                "slice_window": annotation.get('slice_window', []),  # 切片在原始视频中的窗口
                "duration_in_slice": annotation.get('duration_in_slice', 0),  # 在切片中的时长
                "original_info": {
                    "clip_path": annotation.get('clip_path', ''),
                    "original_bos_path": annotation.get('original_bos_path', ''),
                    "time_range_original": annotation.get('time_range_original', []),  # 原始视频中的时间
                    "source_row": annotation.get('source_row', 0)
                },
                "metadata": {
                    "slice_filename": annotation.get('slice_filename', ''),
                    "file_size": annotation.get('file_size', 0)
                }
            }
            
            # 添加视频文件信息
            if video_exists:
                try:
                    file_size = os.path.getsize(video_path)
                    result["metadata"]["video_size"] = file_size
                    result["metadata"]["video_size_mb"] = file_size / (1024 * 1024)
                except:
                    result["metadata"]["video_size"] = 0
                    result["metadata"]["video_size_mb"] = 0
            
            return result
            
        except Exception as e:
            logger.error(f"处理切片标注失败 {annotation.get('id', 'unknown')}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def _slice_key_to_video_path(self, slice_key: str) -> str:
        """从slice_key构造切片视频路径"""
        try:
            # slice_key格式: bos:/neolix-raw/.../video_0_20
            # 提取时间信息
            time_match = re.search(r"_(\d+)_(\d+)$", slice_key)
            if not time_match:
                return ""
            
            seg_start, seg_end = time_match.groups()
            
            # 构造切片文件名
            slice_filename = f"slice_{seg_start}_{seg_end}.mp4"
            
            # 从slice_key提取原始路径部分
            # 移除时间后缀
            base_key = re.sub(r"_\d+_\d+$", "", slice_key)
            
            # 移除开头的bos:前缀
            if base_key.startswith("bos:"):
                base_key = base_key[4:]
            
            # 移除开头的斜杠
            base_key = base_key.lstrip('/')
            
            # 去掉'neolix-raw/'前缀
            if base_key.startswith("neolix-raw/"):
                base_key = base_key[len("neolix-raw/"):]
            
            # 移除末尾的video
            if base_key.endswith("video"):
                base_key = base_key[:-5]
            
            # 构建完整路径
            if self.slice_video_dir:
                # 构造类似: sliced_videos/raw_clips/.../slices/slice_0_20.mp4
                video_path = os.path.join(
                    self.slice_video_dir,
                    base_key.rstrip('/'),
                    "slices",
                    slice_filename
                )
            else:
                # 默认路径
                video_path = os.path.join(
                    "/root/workspace/downloaded_videos_2fps/sliced_videos",
                    base_key.rstrip('/'),
                    "slices",
                    slice_filename
                )
            
            return video_path
            
        except Exception as e:
            logger.error(f"从slice_key构造路径失败 {slice_key}: {str(e)}")
            return ""
    
    def _find_video_in_slice_dir(self, filename: str) -> Optional[str]:
        """在切片视频目录中查找文件"""
        if not self.slice_video_dir or not os.path.exists(self.slice_video_dir):
            return None
        
        for root, dirs, files in os.walk(self.slice_video_dir):
            if filename in files:
                return os.path.join(root, filename)
        
        return None
    
    def save_results(self, all_data: List[Dict], category_stats: Dict):
        """保存处理结果"""
        # 创建输出目录
        output_annotations_dir = os.path.join(self.output_dir, "converted_sliced_annotations")
        os.makedirs(output_annotations_dir, exist_ok=True)
        
        # 统计视频存在情况
        video_exists_count = sum(1 for item in all_data if item.get("video_exists", False))
        video_missing_count = len(all_data) - video_exists_count
        
        # 统计映射情况
        mapped_count = sum(1 for item in all_data if item.get("label_mapped", False))
        unmapped_count = len(all_data) - mapped_count
        
        # 1. 保存完整数据集
        output_file = os.path.join(output_annotations_dir, "sliced_video_vqa_dataset.json")
        
        dataset = {
            "version": "1.0.0",
            "description": "Sliced Video VQA dataset with English labels (20-second clips)",
            "created": datetime.now().isoformat(),
            "source_annotations": self.annotations_dir,
            "slice_video_dir": self.slice_video_dir,
            "statistics": {
                "total_samples": len(all_data),
                "video_exists": video_exists_count,
                "video_missing": video_missing_count,
                "labels_mapped": mapped_count,
                "labels_unmapped": unmapped_count,
                "mapping_rate": mapped_count / len(all_data) if len(all_data) > 0 else 0,
                "categories": category_stats,
                "unmapped_labels": list(self.label_converter.unmapped_labels)
            },
            "data": all_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存完整切片数据集: {output_file} ({len(all_data)} 个样本)")
        logger.info(f"视频存在: {video_exists_count}, 视频缺失: {video_missing_count}")
        logger.info(f"标签映射: {mapped_count}, 未映射: {unmapped_count}")
        
        # 2. 按类别保存
        categories_data = {}
        for item in all_data:
            label_zh = item.get("label_zh", "unknown")
            if label_zh not in categories_data:
                categories_data[label_zh] = []
            categories_data[label_zh].append(item)
        
        for label_zh, items in categories_data.items():
            # 创建安全文件名
            safe_name = self._create_safe_filename(label_zh)
            category_file = os.path.join(
                output_annotations_dir, f"{safe_name}.json"
            )
            
            # 统计该类别的视频存在情况
            category_video_exists = sum(1 for item in items if item.get("video_exists", False))
            
            category_dataset = {
                "label_zh": label_zh,
                "label_en": items[0].get("label_en", "") if items else "",
                "count": len(items),
                "video_exists": category_video_exists,
                "video_missing": len(items) - category_video_exists,
                "data": items
            }
            
            with open(category_file, 'w', encoding='utf-8') as f:
                json.dump(category_dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存了 {len(categories_data)} 个类别文件")
        
        # 3. 保存简化版本（用于后续处理）
        simple_data = []
        for item in all_data:
            simple_item = {
                "id": item.get("id", ""),
                "slice_key": item.get("slice_key", ""),
                "video_path": item.get("video_path", ""),
                "video_exists": item.get("video_exists", False),
                "label_zh": item.get("label_zh", ""),
                "label_en": item.get("label_en", ""),
                "time_range_in_slice": item.get("time_range_in_slice", []),
                "slice_window": item.get("slice_window", []),
                "duration_in_slice": item.get("duration_in_slice", 0)
            }
            simple_data.append(simple_item)
        
        simple_file = os.path.join(output_annotations_dir, "simple_sliced_dataset.json")
        with open(simple_file, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存简化切片数据集: {simple_file}")
        
        # 4. 保存仅包含视频存在的样本
        existing_videos_data = [item for item in all_data if item.get("video_exists", False)]
        existing_file = os.path.join(output_annotations_dir, "existing_sliced_videos_dataset.json")
        
        existing_dataset = {
            "version": "1.0.0",
            "description": "Sliced Video VQA dataset with existing videos only",
            "created": datetime.now().isoformat(),
            "statistics": {
                "total_samples": len(existing_videos_data),
                "categories": {k: v for k, v in category_stats.items() 
                             if k in {item.get("label_zh") for item in existing_videos_data}}
            },
            "data": existing_videos_data
        }
        
        with open(existing_file, 'w', encoding='utf-8') as f:
            json.dump(existing_dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存仅包含存在视频的切片数据集: {existing_file} ({len(existing_videos_data)} 个样本)")
        
        # 5. 保存CSV格式（便于查看）
        csv_data = []
        for item in all_data:
            csv_item = {
                "id": item.get("id", ""),
                "slice_key": item.get("slice_key", ""),
                "video_path": item.get("video_path", ""),
                "video_exists": item.get("video_exists", False),
                "label_zh": item.get("label_zh", ""),
                "label_en": item.get("label_en", ""),
                "time_start": item.get("time_range_in_slice", [])[0] if item.get("time_range_in_slice") else 0,
                "time_end": item.get("time_range_in_slice", [])[1] if item.get("time_range_in_slice") else 0,
                "slice_start": item.get("slice_window", [])[0] if item.get("slice_window") else 0,
                "slice_end": item.get("slice_window", [])[1] if item.get("slice_window") else 0,
                "duration": item.get("duration_in_slice", 0)
            }
            csv_data.append(csv_item)
        
        csv_file = os.path.join(output_annotations_dir, "sliced_dataset.csv")
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"保存CSV格式数据集: {csv_file}")
        
        # 6. 保存统计信息
        stats = {
            "processing_time": datetime.now().isoformat(),
            "total_samples": len(all_data),
            "video_exists": video_exists_count,
            "video_missing": video_missing_count,
            "labels_mapped": mapped_count,
            "labels_unmapped": unmapped_count,
            "mapping_rate": mapped_count / len(all_data) if len(all_data) > 0 else 0,
            "categories": category_stats,
            "unmapped_labels": list(self.label_converter.unmapped_labels),
            "unmapped_count": len(self.label_converter.unmapped_labels)
        }
        
        stats_file = os.path.join(output_annotations_dir, "sliced_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存切片统计信息: {stats_file}")
        
        # 7. 保存标签映射统计
        label_mapping_stats = []
        for item in all_data:
            label_mapping_stats.append({
                "id": item.get("id", ""),
                "slice_key": item.get("slice_key", ""),
                "label_zh": item.get("label_zh", ""),
                "label_en": item.get("label_en", ""),
                "mapped": item.get("label_mapped", False)
            })
        
        mapping_stats_file = os.path.join(output_annotations_dir, "sliced_label_mapping_stats.json")
        with open(mapping_stats_file, 'w', encoding='utf-8') as f:
            json.dump(label_mapping_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存切片标签映射统计: {mapping_stats_file}")
    
    def _create_safe_filename(self, text: str) -> str:
        """创建安全的文件名"""
        # 替换非法字符
        safe = re.sub(r'[<>:"/\\|?*]', '_', text)
        safe = re.sub(r'\s+', '_', safe)
        safe = safe.strip('._')
        
        # 限制长度
        if len(safe) > 100:
            safe = safe[:100]
        
        return safe

def verify_sliced_annotations(annotations_dir: str):
    """验证切片标注文件"""
    print("\n" + "=" * 60)
    print("验证切片标注文件...")
    print("=" * 60)
    
    if not os.path.exists(annotations_dir):
        print(f"❌ 切片标注目录不存在: {annotations_dir}")
        return 0
    
    # 查找所有标注文件
    annotation_files = []
    for file in os.listdir(annotations_dir):
        if file.endswith('.json'):
            annotation_files.append(file)
    
    print(f"找到 {len(annotation_files)} 个标注文件")
    
    if annotation_files:
        print("\n标注文件列表:")
        for i, file in enumerate(annotation_files[:10], 1):
            file_path = os.path.join(annotations_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    count = len(data)
                else:
                    count = len(data.get('data', [])) if isinstance(data, dict) else 0
                print(f"  {i}. {file} ({count} 个标注)")
            except:
                print(f"  {i}. {file} (读取失败)")
        
        if len(annotation_files) > 10:
            print(f"  ... 还有 {len(annotation_files) - 10} 个文件")
    
    return len(annotation_files)

def main():
    """主函数"""
    # 输入路径
    ANNOTATIONS_DIR = "/root/workspace/sliced_vqa_annotations/sliced_annotations"
    MAPPING_FILE = "/root/workspace/LLaMA-Factory/data/中英对照表_12tags.xlsx"
    SLICE_VIDEO_DIR = "/root/workspace/downloaded_videos_2fps/sliced_videos"
    OUTPUT_DIR = "/root/workspace/sliced_vqa_dataset_prepared"
    
    print("=" * 60)
    print("🎯 切片视频VQA标注标签转换工具")
    print("=" * 60)
    print(f"📁 切片标注目录: {ANNOTATIONS_DIR}")
    print(f"📁 切片视频目录: {SLICE_VIDEO_DIR}")
    print(f"📄 中英文映射文件: {MAPPING_FILE}")
    print(f"📦 输出目录: {OUTPUT_DIR}")
    print("=" * 60)
    print("📋 功能说明:")
    print("  - 读取切片标注文件（20秒视频切片）")
    print("  - 使用中英文对照表转换为英文标签")
    print("  - 生成适合视频VQA模型训练的切片数据集")
    print("  - 适配切片视频的特殊数据结构")
    print("=" * 60)
    
    # 验证输入文件
    annotation_count = verify_sliced_annotations(ANNOTATIONS_DIR)
    if annotation_count == 0:
        logger.error(f"没有找到切片标注文件: {ANNOTATIONS_DIR}")
        print(f"\n❌ 错误: 切片标注目录不存在或为空: {ANNOTATIONS_DIR}")
        print("请先运行切片标注生成脚本生成标注文件")
        return
    
    if not os.path.exists(SLICE_VIDEO_DIR):
        logger.warning(f"切片视频目录不存在: {SLICE_VIDEO_DIR}")
        print(f"\n⚠️  警告: 切片视频目录不存在: {SLICE_VIDEO_DIR}")
        print("视频路径验证可能会失败")
    
    if not os.path.exists(MAPPING_FILE):
        logger.warning(f"映射文件不存在: {MAPPING_FILE}")
        logger.warning("将使用原中文标签作为英文标签")
        print(f"\n⚠️  警告: 映射文件不存在: {MAPPING_FILE}")
        print("将使用原中文标签作为英文标签")
        print("建议提供中英文对照表以获得更好的标签映射")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化处理器
    processor = SlicedAnnotationProcessor(
        annotations_dir=ANNOTATIONS_DIR,
        mapping_file=MAPPING_FILE,
        output_dir=OUTPUT_DIR,
        slice_video_dir=SLICE_VIDEO_DIR
    )
    
    # 处理所有标注
    start_time = time.time()
    result = processor.process_all_annotations()
    elapsed_time = time.time() - start_time
    
    # 输出结果
    print("\n" + "=" * 60)
    print("🎉 切片标注转换完成")
    print("=" * 60)
    print(f"⏱️  总耗时: {elapsed_time:.2f}秒")
    print(f"📊 处理结果:")
    print(f"  ✅ 总样本数: {result['total_samples']}")
    
    if result.get('category_stats'):
        print(f"  📂 类别统计 (前10个):")
        sorted_categories = sorted(result['category_stats'].items(), 
                                  key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:10]:
            print(f"    - {category}: {count}")
        if len(sorted_categories) > 10:
            print(f"    ... 还有 {len(sorted_categories) - 10} 个类别")
    
    if result.get('unmapped_labels'):
        print(f"  ⚠️  未映射标签: {len(result['unmapped_labels'])} 个")
        mapped_count = result['total_samples'] - len(result['unmapped_labels'])
        mapping_rate = mapped_count / result['total_samples'] if result['total_samples'] > 0 else 0
        print(f"    映射率: {mapping_rate*100:.1f}% ({mapped_count}/{result['total_samples']})")
        print(f"    未映射标签示例:")
        for label in list(result['unmapped_labels'])[:5]:  # 最多显示5个
            print(f"      - {label}")
        if len(result['unmapped_labels']) > 5:
            print(f"      ... 还有 {len(result['unmapped_labels']) - 5} 个")
    
    print(f"\n📁 输出目录: {OUTPUT_DIR}")
    print("目录结构:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  └── converted_sliced_annotations/    # 转换后的切片标注")
    print(f"      ├── sliced_video_vqa_dataset.json     # 完整切片数据集")
    print(f"      ├── simple_sliced_dataset.json        # 简化切片数据集")
    print(f"      ├── existing_sliced_videos_dataset.json  # 仅包含存在视频的切片数据集")
    print(f"      ├── sliced_dataset.csv               # CSV格式数据集")
    print(f"      ├── sliced_statistics.json           # 统计信息")
    print(f"      ├── sliced_label_mapping_stats.json  # 标签映射统计")
    print(f"      └── [类别].json                     # 按类别分的数据")
    
    print(f"\n📋 下一步建议:")
    print("1. 检查未映射的标签，更新中英文对照表")
    print("2. 查看视频存在情况，确保有足够的训练数据")
    print("3. 对切片视频进行抽帧: python /root/workspace/LLaMA-Factory/scene_tag/1.5_get_frames_squeeze.py -i /root/workspace/downloaded_videos_2fps/sliced_videos")
    print("4. 使用转换后的数据集进行视频VQA模型训练")
    
    # 显示生成的标注示例
    simple_file = os.path.join(OUTPUT_DIR, "converted_sliced_annotations", "simple_sliced_dataset.json")
    if os.path.exists(simple_file):
        with open(simple_file, 'r', encoding='utf-8') as f:
            simple_data = json.load(f)
        
        if simple_data:
            print(f"\n📝 切片标注示例 (前2个):")
            for i, item in enumerate(simple_data[:2], 1):
                print(f"\n  {i}. ID: {item.get('id', 'N/A')}")
                print(f"     slice_key: {item.get('slice_key', 'N/A')[:50]}...")
                print(f"     视频路径: {item.get('video_path', 'N/A')[:80]}...")
                print(f"     视频存在: {item.get('video_exists', False)}")
                print(f"     中文标签: {item.get('label_zh', 'N/A')}")
                print(f"     英文标签: {item.get('label_en', 'N/A')}")
                print(f"     时间范围(切片中): {item.get('time_range_in_slice', [])}")
                print(f"     切片窗口(原始视频): {item.get('slice_window', [])}")
                print(f"     时长(切片中): {item.get('duration_in_slice', 0)}s")
            print("\n" + "=" * 60)
    
    # 输出使用说明
    print(f"\n💡 使用说明:")
    print("1. 数据集已准备好，可以直接用于训练")
    print("2. 每个样本包含20秒切片视频的路径和标注")
    print("3. 使用'slice_key'可以唯一标识每个样本")
    print("4. 确保切片视频已下载并存储在正确位置")
    print("5. 训练前需要先对切片视频进行抽帧")
    print("=" * 60)

if __name__ == "__main__":
    main()

'''
确认参数：
    ANNOTATIONS_DIR = "/root/workspace/sliced_vqa_annotations/sliced_annotations"
    SLICE_VIDEO_DIR = "/root/workspace/downloaded_videos_for_segment/sliced_videos"
    
    OUTPUT_DIR = "/root/workspace/sliced_vqa_dataset_prepared"
    MAPPING_FILE是中英文对照表的路径
运行即可
'''