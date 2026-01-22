import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from datetime import datetime
import time
import traceback
import re
from tqdm import tqdm
from collections import defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/video_annotation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoAnnotationProcessor:
    def __init__(self, excel_path: str, video_base_dir: str, output_base_dir: str):
        self.excel_path = excel_path
        self.video_base_dir = video_base_dir
        self.output_base_dir = output_base_dir
        self.annotations = {}  # 按类别存储标注
        self.processed_annotations = set()  # 记录已处理的标注
        self.video_mapping = {}  # 记录BOS路径到本地路径的映射
        self.video_stats = {}  # 视频统计信息
    
    def load_excel_data(self) -> pd.DataFrame:
        """加载Excel数据"""
        try:
            df = pd.read_excel(self.excel_path)
            logger.info(f"Excel数据加载成功: {len(df)} 行")
            
            # 确保必要列存在
            required_columns = ['clip视频路径', '标签', 'T_start', 'T_end']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"缺少必要列: {missing_cols}")
            
            # 清理数据：移除路径中的换行符和多余空格
            if 'clip视频路径' in df.columns:
                # 记录清理前的示例
                sample_before = df['clip视频路径'].iloc[0] if len(df) > 0 else ""
                
                # 清理换行符和空格
                df['clip视频路径'] = df['clip视频路径'].astype(str).apply(
                    lambda x: x.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                )
                
                # 记录清理后的示例
                sample_after = df['clip视频路径'].iloc[0] if len(df) > 0 else ""
                
                logger.info(f"清理路径中的换行符: '{sample_before[:50]}...' -> '{sample_after[:50]}...'")
                
                # 统计清理情况
                paths_with_newlines = df['clip视频路径'].astype(str).apply(lambda x: '\n' in x or '\r' in x)
                if paths_with_newlines.any():
                    logger.warning(f"发现 {paths_with_newlines.sum()} 个路径包含换行符")
                    for idx, (_, row) in enumerate(df[paths_with_newlines].iterrows()):
                        if idx < 5:  # 只显示前5个示例
                            logger.warning(f"第{row.name}行: 原始路径包含换行符: {repr(row['clip视频路径'])}")
            
            # 清理标签列
            if '标签' in df.columns:
                df['标签'] = df['标签'].astype(str).str.strip()
            
            return df
            
        except Exception as e:
            logger.error(f"加载Excel失败: {str(e)}")
            raise
    
    def bos_to_local_path(self, bos_path: str) -> str:
        """
        将BOS路径转换为本地路径
        精确匹配：与脚本1的保存路径完全一致
        """
        try:
            # 移除开头的bos:前缀
            if bos_path.startswith("bos:"):
                bos_path = bos_path[4:]
            
            # 移除开头的斜杠
            bos_path = bos_path.lstrip('/')
            
            # 去掉'neolix-raw/'前缀
            if bos_path.startswith("neolix-raw/"):
                bos_path = bos_path[len("neolix-raw/"):]
            
            # 确保路径以/结尾
            if not bos_path.endswith('/'):
                bos_path += '/'
            
            # 添加video.mp4
            bos_path += "video.mp4"
            
            # 构建本地路径
            local_path = os.path.join(self.video_base_dir, bos_path)
            
            return local_path
            
        except Exception as e:
            logger.error(f"解析路径失败 {bos_path}: {str(e)}")
            return None
    
    def find_exact_match(self, bos_path: str) -> Optional[str]:
        """
        精确匹配BOS路径
        只接受完全匹配，不接受任何模糊匹配
        """
        local_path = self.bos_to_local_path(bos_path)
        
        if not local_path:
            return None
        
        # 检查文件是否存在
        if os.path.exists(local_path):
            return local_path
        
        return None
    
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
    
    def _create_annotation_id(self, bos_path: str, t_start: int, t_end: int, idx: int) -> str:
        """创建标注ID"""
        import hashlib
        # 使用BOS路径和时间戳创建哈希
        hash_input = f"{bos_path}_{t_start}_{t_end}_{idx}".encode('utf-8')
        hash_str = hashlib.md5(hash_input).hexdigest()[:8]
        
        # 从BOS路径中提取有用信息
        date_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", bos_path)
        date_str = date_match.group(1) if date_match else "unknown"
        
        # 创建ID
        annotation_id = f"anno_{idx:04d}_{date_str}_{t_start}s_{t_end}s_{hash_str}"
        return annotation_id
    
    def process_single_row(self, row: pd.Series, idx: int) -> Tuple[bool, str]:
        """处理单行数据"""
        try:
            # 获取原始视频路径
            bos_path = str(row['clip视频路径']).strip()
            # 额外的清理：确保移除所有空白字符
            bos_path = re.sub(r'\s+', '', bos_path)  # 移除所有空白字符（空格、换行、制表符等）

            label = str(row['标签']).strip()
            
            # 验证数据
            if pd.isna(bos_path) or bos_path == '':
                return False, f"第{idx}行: clip视频路径为空"
            
            if pd.isna(label) or label == '':
                return False, f"第{idx}行: 标签为空"
            
            # 检查T_start和T_end是否为NaN
            if pd.isna(row['T_start']) or pd.isna(row['T_end']):
                return False, f"第{idx}行: 时间戳为NaN"
            
            t_start = int(row['T_start'])
            t_end = int(row['T_end'])
            
            if t_start < 0 or t_end < 0 or t_start >= t_end:
                return False, f"第{idx}行: 时间范围无效 ({t_start}-{t_end})"
            
            # 获取精确匹配的视频路径
            if bos_path in self.video_mapping:
                local_video_path = self.video_mapping[bos_path]
                video_exists = True
            else:
                local_video_path = self.find_exact_match(bos_path)
                if local_video_path:
                    self.video_mapping[bos_path] = local_video_path
                    video_exists = True
                else:
                    # 尝试直接查找
                    local_path = self.bos_to_local_path(bos_path)
                    logger.warning(f"第{idx}行: 视频文件不存在，BOS路径: {bos_path}")
                    logger.warning(f"第{idx}行: 期望的本地路径: {local_path}")
                    video_exists = False
            
            # 生成标注ID
            annotation_id = self._create_annotation_id(bos_path, t_start, t_end, idx)
            
            # 检查是否已处理过
            if annotation_id in self.processed_annotations:
                logger.debug(f"跳过已处理的标注: {annotation_id}")
                return True, f"已处理: {annotation_id}"
            
            # 创建安全的类别名称
            safe_label = self._create_safe_filename(label)
            
            # 添加到标注
            if safe_label not in self.annotations:
                self.annotations[safe_label] = []
            
            annotation = {
                "id": annotation_id,
                "original_video": local_video_path if video_exists else None,
                "original_bos_path": bos_path,
                "label": label,
                "time_range": [t_start, t_end],
                "duration": t_end - t_start,
                "frame_count": int((t_end - t_start) * 30),  # 假设30fps
                "source_row": idx,
                "video_exists": video_exists,
                "file_size": os.path.getsize(local_video_path) if video_exists and local_video_path and os.path.exists(local_video_path) else 0
            }
            
            self.annotations[safe_label].append(annotation)
            self.processed_annotations.add(annotation_id)
            
            return True, f"成功添加标注: {annotation_id}"
                
        except Exception as e:
            logger.error(f"处理行 {idx} 失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"异常: {str(e)}"
    
    def analyze_data_statistics(self, df: pd.DataFrame) -> Dict:
        """分析数据统计信息"""
        stats = {
            "total_rows": len(df),
            "valid_rows": 0,
            "invalid_rows": 0,
            "invalid_reasons": defaultdict(int),
            "unique_videos": [],  # 使用列表而不是集合
            "labels_count": {},
            "video_existence": {"exists": 0, "not_exists": 0, "details": []},
            "time_range_stats": {
                "total_duration": 0,
                "avg_duration": 0,
                "min_duration": float('inf'),
                "max_duration": 0
            }
        }
        
        for idx, row in df.iterrows():
            try:
                bos_path = str(row['clip视频路径']).strip()
                label = str(row['标签']).strip()
                
                # 检查必要字段
                if pd.isna(bos_path) or bos_path == '':
                    stats["invalid_rows"] += 1
                    stats["invalid_reasons"]["空视频路径"] += 1
                    continue
                
                if pd.isna(label) or label == '':
                    stats["invalid_rows"] += 1
                    stats["invalid_reasons"]["空标签"] += 1
                    continue
                
                # 检查T_start和T_end是否为NaN
                if pd.isna(row['T_start']):
                    stats["invalid_rows"] += 1
                    stats["invalid_reasons"]["T_start为NaN"] += 1
                    continue
                    
                if pd.isna(row['T_end']):
                    stats["invalid_rows"] += 1
                    stats["invalid_reasons"]["T_end为NaN"] += 1
                    continue
                
                t_start = int(row['T_start'])
                t_end = int(row['T_end'])
                
                if t_start < 0 or t_end < 0:
                    stats["invalid_rows"] += 1
                    stats["invalid_reasons"]["时间戳为负"] += 1
                    continue
                
                if t_start >= t_end:
                    stats["invalid_rows"] += 1
                    stats["invalid_reasons"]["开始时间大于等于结束时间"] += 1
                    continue
                
                # 有效的行
                stats["valid_rows"] += 1
                
                # 统计唯一视频
                if bos_path not in stats["unique_videos"]:
                    stats["unique_videos"].append(bos_path)
                
                # 统计标签
                if label in stats["labels_count"]:
                    stats["labels_count"][label] += 1
                else:
                    stats["labels_count"][label] = 1
                
                # 统计时间范围
                duration = t_end - t_start
                stats["time_range_stats"]["total_duration"] += duration
                stats["time_range_stats"]["min_duration"] = min(stats["time_range_stats"]["min_duration"], duration)
                stats["time_range_stats"]["max_duration"] = max(stats["time_range_stats"]["max_duration"], duration)
                
                # 检查视频是否存在
                local_path = self.find_exact_match(bos_path)
                exists = local_path and os.path.exists(local_path)
                
                if exists:
                    stats["video_existence"]["exists"] += 1
                else:
                    stats["video_existence"]["not_exists"] += 1
                
                stats["video_existence"]["details"].append({
                    "row": idx,
                    "bos_path": bos_path,
                    "local_path": local_path,
                    "exists": exists,
                    "label": label
                })
                        
            except Exception as e:
                logger.debug(f"分析行 {idx} 失败: {str(e)}")
                stats["invalid_rows"] += 1
                stats["invalid_reasons"][str(type(e).__name__)] += 1
                continue
        
        # 计算平均值
        if stats["valid_rows"] > 0:
            stats["time_range_stats"]["avg_duration"] = stats["time_range_stats"]["total_duration"] / stats["valid_rows"]
        
        stats["unique_videos_count"] = len(stats["unique_videos"])
        
        return stats
    
    def save_annotations(self):
        """保存标注到JSON文件"""
        output_dir = os.path.join(self.output_base_dir, "annotations")
        os.makedirs(output_dir, exist_ok=True)
        
        total_annotations = 0
        category_stats = {}
        
        # 为每个类别保存单独的JSON文件
        for label, annotations in self.annotations.items():
            if not annotations:
                continue
                
            # 创建安全文件名
            safe_label_name = self._create_safe_filename(label)
            json_path = os.path.join(output_dir, f"{safe_label_name}.json")
            
            # 为每个标注添加索引
            for i, anno in enumerate(annotations):
                anno["index_in_category"] = i
            
            # 保存为JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存标注文件: {json_path} ({len(annotations)} 个标注)")
            total_annotations += len(annotations)
            category_stats[label] = len(annotations)
        
        # 保存汇总文件
        summary = {
            "total_categories": len(self.annotations),
            "total_annotations": total_annotations,
            "annotations_per_category": category_stats,
            "categories": list(self.annotations.keys()),
            "processing_time": datetime.now().isoformat()
        }
        
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存汇总文件: {summary_path}")
        
        # 保存合并的所有标注
        all_annotations = []
        for label, annotations in self.annotations.items():
            all_annotations.extend(annotations)
        
        all_annotations_path = os.path.join(output_dir, "all_annotations.json")
        with open(all_annotations_path, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存合并标注文件: {all_annotations_path} ({len(all_annotations)} 个标注)")
        
        return total_annotations
    
    def save_video_mapping(self):
        """保存视频路径映射"""
        mapping_path = os.path.join(self.output_base_dir, "video_mapping.json")
        
        mapping_data = {
            "total_mappings": len(self.video_mapping),
            "mappings": [
                {
                    "bos_path": bos_path,
                    "local_path": local_path,
                    "exists": os.path.exists(local_path) if local_path else False
                }
                for bos_path, local_path in self.video_mapping.items()
            ],
            "processing_time": datetime.now().isoformat()
        }
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存视频路径映射: {mapping_path}")
    
    def save_statistics(self, success_count: int, fail_count: int, 
                       data_stats: Dict, fail_details: List, elapsed_time: float):
        """保存详细的统计信息"""
        stats = {
            "processing_summary": {
                "total_processed": success_count + fail_count,
                "success_count": success_count,
                "fail_count": fail_count,
                "success_rate": success_count / (success_count + fail_count) if (success_count + fail_count) > 0 else 0,
                "categories_created": len(self.annotations),
                "annotations_created": sum(len(annos) for annos in self.annotations.values()),
                "processing_time": datetime.now().isoformat(),
                "duration_seconds": elapsed_time
            },
            "data_statistics": data_stats,
            "fail_details": [
                {"row": idx, "reason": reason} for idx, reason in fail_details[:100]
            ],
            "configuration": {
                "excel_path": self.excel_path,
                "video_base_dir": self.video_base_dir,
                "output_base_dir": self.output_base_dir
            }
        }
        
        stats_path = os.path.join(self.output_base_dir, "processing_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计信息已保存: {stats_path}")
        return stats
    
    def process_all(self, max_workers: int = 4):
        """处理所有数据"""
        # 加载数据
        df = self.load_excel_data()
        
        # 分析数据统计
        logger.info("分析数据统计...")
        stats = self.analyze_data_statistics(df)
        
        print("\n" + "=" * 60)
        print("📊 标注数据统计:")
        print("=" * 60)
        print(f"📄 原始标注行数: {stats['total_rows']}")
        print(f"✅ 有效标注数: {stats['valid_rows']}")
        print(f"❌ 无效标注数: {stats['invalid_rows']}")
        
        if stats['invalid_reasons']:
            print(f"📉 无效原因统计:")
            for reason, count in sorted(stats['invalid_reasons'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count}")
        
        print(f"📁 唯一BOS视频路径: {stats['unique_videos_count']}")
        
        # 显示标签统计
        if stats['labels_count']:
            print(f"\n📂 按类别统计:")
            sorted_labels = sorted(stats['labels_count'].items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_labels[:20]:
                print(f"  - {label}: {count} 个标注")
            if len(sorted_labels) > 20:
                print(f"  ... 还有 {len(sorted_labels) - 20} 个类别")
        
        # 显示时间统计
        print(f"\n⏱️  时间范围统计:")
        print(f"  - 总时长: {stats['time_range_stats']['total_duration']} 秒")
        print(f"  - 平均时长: {stats['time_range_stats']['avg_duration']:.2f} 秒")
        print(f"  - 最短时长: {stats['time_range_stats']['min_duration']} 秒")
        print(f"  - 最长时长: {stats['time_range_stats']['max_duration']} 秒")
        
        print(f"\n📹 视频文件匹配情况:")
        print(f"  ✅ 可找到的视频: {stats['video_existence']['exists']}")
        print(f"  ❌ 缺失的视频: {stats['video_existence']['not_exists']}")
        
        if stats['video_existence']['not_exists'] > 0:
            print(f"\n⚠️  警告: 有 {stats['video_existence']['not_exists']} 个标注没有找到匹配的视频!")
            print("可能的原因:")
            print("1. 视频未下载或下载不完整")
            print("2. Excel中的路径与下载的视频路径不匹配")
            print("3. 视频文件名不正确")
            
            # 显示不匹配的详细信息
            not_exists_details = [d for d in stats['video_existence']['details'] if not d['exists']]
            if not_exists_details:
                print(f"\n🔍 不匹配的示例 (前5个):")
                for i, detail in enumerate(not_exists_details[:5], 1):
                    print(f"\n  {i}. 行 {detail['row']}:")
                    print(f"     标签: {detail['label']}")
                    print(f"     BOS路径: {detail['bos_path']}")
                    print(f"     期望的本地路径: {detail['local_path']}")
        
        if stats['valid_rows'] == 0:
            logger.error("❌ 没有有效的标注数据！")
            return 0, 0
        
        logger.info(f"开始处理 {len(df)} 个标注行...")
        
        # 准备处理任务
        tasks = []
        for idx, row in df.iterrows():
            tasks.append((idx, row))
        
        # 使用线程池并行处理
        success_count = 0
        fail_count = 0
        fail_details = []
        
        with tqdm(total=len(tasks), desc="处理进度", unit="行") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_idx = {}
                for idx, row in tasks:
                    future = executor.submit(self.process_single_row, row, idx)
                    future_to_idx[future] = idx
                
                # 处理结果
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        success, message = future.result(timeout=30)
                        if success:
                            success_count += 1
                        else:
                            fail_count += 1
                            fail_details.append((idx, message))
                    except Exception as e:
                        fail_count += 1
                        fail_details.append((idx, f"处理异常: {str(e)}"))
                        logger.error(f"行 {idx} 处理异常: {str(e)}")
                    
                    pbar.update(1)
                    pbar.set_postfix_str(f"成功: {success_count}, 失败: {fail_count}")
        
        return success_count, fail_count, stats, fail_details

def verify_downloaded_videos(video_base_dir: str):
    """验证下载的视频文件"""
    print("\n" + "=" * 60)
    print("验证下载的视频文件...")
    print("=" * 60)
    
    if not os.path.exists(video_base_dir):
        print(f"❌ 视频目录不存在: {video_base_dir}")
        return 0
    
    # 查找所有视频文件
    video_files = []
    video_sizes = {}
    
    for root, dirs, files in os.walk(video_base_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                rel_path = os.path.relpath(video_path, video_base_dir)
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                video_files.append(rel_path)
                video_sizes[rel_path] = file_size
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    if video_files:
        # 计算总大小
        total_size = sum(video_sizes.values())
        avg_size = total_size / len(video_files) if video_files else 0
        
        print(f"总大小: {total_size:.2f} MB")
        print(f"平均大小: {avg_size:.2f} MB")
        
        print("\n视频文件示例:")
        for i, rel_path in enumerate(video_files[:5], 1):
            size = video_sizes[rel_path]
            print(f"  {i}. {rel_path} ({size:.2f} MB)")
        
        if len(video_files) > 5:
            print(f"  ... 还有 {len(video_files) - 5} 个文件")
    
    return len(video_files)

def main():
    """主函数"""
    EXCEL_PATH = "/root/workspace/人工标注视频数据_对比实验_12tags_.xlsx"
    VIDEO_BASE_DIR = "/root/workspace/downloaded_videos_2fps"
    OUTPUT_BASE_DIR = "/root/workspace/vqa_annotations_2fps"
    
    print("=" * 60)
    print("🎯 视频标注生成工具")
    print("=" * 60)
    print(f"📁 视频目录: {VIDEO_BASE_DIR}")
    print(f"📄 Excel文件: {EXCEL_PATH}")
    print(f"📦 输出目录: {OUTPUT_BASE_DIR}")
    print("=" * 60)
    print("📋 功能说明:")
    print("  - 读取Excel中的视频标注数据")
    print("  - 将BOS路径转换为本地路径")
    print("  - 按类别生成标注文档")
    print("  - 不进行视频切分，保留原始视频")
    print("=" * 60)
    
    # 验证下载的视频文件
    video_count = verify_downloaded_videos(VIDEO_BASE_DIR)
    if video_count == 0:
        logger.warning("未找到视频文件，但标注生成将继续进行")
    
    # 检查输入文件
    if not os.path.exists(EXCEL_PATH):
        logger.error(f"Excel文件不存在: {EXCEL_PATH}")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 初始化处理器
    processor = VideoAnnotationProcessor(
        excel_path=EXCEL_PATH,
        video_base_dir=VIDEO_BASE_DIR,
        output_base_dir=OUTPUT_BASE_DIR
    )
    
    # 处理所有数据
    start_time = time.time()
    success_count, fail_count, data_stats, fail_details = processor.process_all(max_workers=4)
    elapsed_time = time.time() - start_time
    
    # 保存结果
    if processor.annotations:
        total_annotations = processor.save_annotations()
        processor.save_video_mapping()
        
        # 生成统计信息
        stats = processor.save_statistics(success_count, fail_count, data_stats, fail_details, elapsed_time)
    else:
        total_annotations = 0
        logger.warning("没有生成任何标注数据")
        stats = processor.save_statistics(success_count, fail_count, data_stats, fail_details, elapsed_time)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("🎉 标注生成完成")
    print("=" * 60)
    print(f"⏱️  总耗时: {elapsed_time:.2f}秒")
    print(f"📊 总计处理: {success_count + fail_count} 行")
    print(f"✅ 成功: {success_count} 行")
    print(f"❌ 失败: {fail_count} 行")
    
    if success_count > 0:
        print(f"\n📁 输出目录: {OUTPUT_BASE_DIR}")
        print("目录结构:")
        print(f"  {OUTPUT_BASE_DIR}/")
        print(f"  ├── annotations/             # 标注文件")
        print(f"  │   ├── all_annotations.json  # 所有标注的合并文件")
        print(f"  │   ├── summary.json         # 汇总信息")
        print(f"  │   └── [类别].json         # 每个类别的标注")
        print(f"  ├── video_mapping.json       # 视频路径映射")
        print(f"  └── processing_statistics.json  # 处理统计")
        
        # 显示生成的类别
        if processor.annotations:
            print(f"\n📂 生成的标注类别 ({len(processor.annotations)} 个):")
            for label, annotations in sorted(processor.annotations.items(), 
                                          key=lambda x: len(x[1]), reverse=True)[:10]:
                print(f"  - {label}: {len(annotations)} 个标注")
            if len(processor.annotations) > 10:
                print(f"  ... 还有 {len(processor.annotations) - 10} 个类别")
        
        print(f"\n📋 下一步:")
        print("1. 检查标注文件: ls -la /root/workspace/vqa_annotations/annotations/")
        print("2. 查看标注统计: cat /root/workspace/vqa_annotations/processing_statistics.json | python -m json.tool")
        print("3. 使用标注文件进行模型训练")
        
        # 显示标注示例
        print(f"\n📝 标注示例:")
        for label, annotations in sorted(processor.annotations.items(), 
                                      key=lambda x: len(x[1]), reverse=True):
            if annotations:
                anno = annotations[0]
                print(f"  类别: {label}")
                print(f"    视频: {os.path.basename(anno.get('original_video', 'N/A'))}")
                print(f"    时间范围: {anno['time_range'][0]}s - {anno['time_range'][1]}s")
                print(f"    时长: {anno['duration']}s")
                break
    else:
        print(f"\n❌ 处理失败，没有生成任何标注数据")
        print("可能的原因:")
        print("1. Excel文件格式错误")
        print("2. 所有行都有数据问题")
        print("3. 没有有效的标注行")
        print(f"\n🔍 查看详细日志: tail -100 /root/workspace/video_annotation.log")
    
    print("=" * 60)

if __name__ == "__main__":
    main()