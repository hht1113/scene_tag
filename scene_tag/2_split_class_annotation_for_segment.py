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
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/sliced_video_annotation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SlicedVideoAnnotationProcessor:
    def __init__(self, slice_info_csv: str, slice_video_dir: str, output_base_dir: str):
        """
        初始化切片视频标注处理器
        
        参数:
            slice_info_csv: 切片信息CSV文件路径
            slice_video_dir: 切片视频目录路径
            output_base_dir: 输出目录路径
        """
        self.slice_info_csv = slice_info_csv
        self.slice_video_dir = slice_video_dir
        self.output_base_dir = output_base_dir
        self.annotations = {}  # 按类别存储标注
        self.processed_annotations = set()  # 记录已处理的标注
        self.slice_mapping = {}  # 记录slice_key到本地切片视频路径的映射
        self.slice_stats = {}  # 切片统计信息
        
    def load_slice_info(self) -> pd.DataFrame:
        """加载切片信息CSV文件"""
        try:
            if not os.path.exists(self.slice_info_csv):
                logger.error(f"切片信息CSV文件不存在: {self.slice_info_csv}")
                return pd.DataFrame()
            
            df = pd.read_csv(self.slice_info_csv)
            logger.info(f"切片信息CSV加载成功: {len(df)} 行")
            
            # 验证必要的列
            required_columns = ['slice_key', 'clip视频路径', '标签', 'T_start', 'T_end', 
                               'seg_start', 'seg_end', 't_start_new', 't_end_new', 'local_slice_path']
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                logger.error(f"切片信息CSV缺少必要列: {missing_cols}")
                return pd.DataFrame()
            
            # 清理数据
            if 'clip视频路径' in df.columns:
                df['clip视频路径'] = df['clip视频路径'].astype(str).str.strip()
            
            if '标签' in df.columns:
                df['标签'] = df['标签'].astype(str).str.strip()
            
            if 'local_slice_path' in df.columns:
                df['local_slice_path'] = df['local_slice_path'].astype(str).str.strip()
            
            return df
            
        except Exception as e:
            logger.error(f"加载切片信息CSV失败: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
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
    
    def _create_annotation_id(self, slice_key: str, idx: int) -> str:
        """创建标注ID"""
        # 使用slice_key创建哈希
        hash_input = f"{slice_key}_{idx}".encode('utf-8')
        hash_str = hashlib.md5(hash_input).hexdigest()[:8]
        
        # 提取时间信息
        time_match = re.search(r"(\d+)_(\d+)$", slice_key)
        if time_match:
            seg_start, seg_end = time_match.groups()
            time_str = f"{seg_start}s_{seg_end}s"
        else:
            time_str = "unknown"
        
        # 从slice_key中提取日期信息
        date_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", slice_key)
        date_str = date_match.group(1) if date_match else "unknown"
        
        # 创建ID
        annotation_id = f"slice_{idx:04d}_{date_str}_{time_str}_{hash_str}"
        return annotation_id
    
    def verify_slice_video_exists(self, local_slice_path: str) -> Tuple[bool, str, int]:
        """验证切片视频文件是否存在并获取信息"""
        if not local_slice_path or pd.isna(local_slice_path):
            return False, "路径为空", 0
        
        # 尝试几种可能的路径
        possible_paths = [
            local_slice_path,  # 原始路径
            os.path.join(self.slice_video_dir, local_slice_path),  # 相对于切片目录
            os.path.join(self.slice_video_dir, os.path.basename(local_slice_path)),  # 只取文件名
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    file_size = os.path.getsize(path)
                    if file_size > 1024:  # 至少1KB
                        return True, path, file_size
                except:
                    continue
        
        # 如果以上都不存在，尝试在切片目录中搜索
        try:
            # 从local_slice_path中提取文件名
            if os.path.basename(local_slice_path):
                filename = os.path.basename(local_slice_path)
                # 在整个切片目录中搜索
                for root, dirs, files in os.walk(self.slice_video_dir):
                    if filename in files:
                        found_path = os.path.join(root, filename)
                        file_size = os.path.getsize(found_path)
                        return True, found_path, file_size
        except:
            pass
        
        return False, local_slice_path, 0
    
    def process_single_slice(self, row: pd.Series, idx: int) -> Tuple[bool, str]:
        """处理单个切片"""
        try:
            # 获取必要信息
            slice_key = str(row['slice_key']).strip()
            label = str(row['标签']).strip()
            local_slice_path = str(row.get('local_slice_path', '')).strip()
            
            # 获取时间信息
            t_start_new = float(row.get('t_start_new', 0))
            t_end_new = float(row.get('t_end_new', 0))
            seg_start = float(row.get('seg_start', 0))
            seg_end = float(row.get('seg_end', 0))
            t_start = float(row.get('T_start', 0))
            t_end = float(row.get('T_end', 0))
            
            # 验证数据
            if pd.isna(slice_key) or slice_key == '':
                return False, f"第{idx}行: slice_key为空"
            
            if pd.isna(label) or label == '':
                return False, f"第{idx}行: 标签为空"
            
            if pd.isna(local_slice_path) or local_slice_path == '':
                return False, f"第{idx}行: 本地切片路径为空"
            
            # 检查时间信息
            if t_start_new < 0 or t_end_new < 0 or t_start_new >= t_end_new:
                logger.warning(f"第{idx}行: 相对时间范围无效 ({t_start_new}-{t_end_new})")
            
            if seg_end - seg_start != 20:
                logger.warning(f"第{idx}行: 切片长度不是20秒 ({seg_start}-{seg_end} = {seg_end-seg_start}s)")
            
            # 验证切片视频文件是否存在
            video_exists, verified_path, file_size = self.verify_slice_video_exists(local_slice_path)
            
            if not video_exists:
                logger.warning(f"第{idx}行: 切片视频文件不存在: {local_slice_path}")
                logger.warning(f"  尝试的路径: {verified_path}")
                # 不返回失败，但记录警告
            
            # 生成标注ID
            annotation_id = self._create_annotation_id(slice_key, idx)
            
            # 检查是否已处理过
            if annotation_id in self.processed_annotations:
                logger.debug(f"跳过已处理的标注: {annotation_id}")
                return True, f"已处理: {annotation_id}"
            
            # 创建安全的类别名称
            safe_label = self._create_safe_filename(label)
            
            # 添加到标注
            if safe_label not in self.annotations:
                self.annotations[safe_label] = []
            
            # 构建完整的标注信息
            annotation = {
                "id": annotation_id,
                "slice_key": slice_key,
                "slice_video_path": verified_path if video_exists else local_slice_path,
                "label": label,
                "time_range_original": [float(t_start), float(t_end)],  # 原始视频中的时间
                "time_range_in_slice": [float(t_start_new), float(t_end_new)],  # 切片视频中的相对时间
                "slice_window": [float(seg_start), float(seg_end)],  # 切片在原始视频中的时间窗口
                "duration_original": float(t_end - t_start),  # 原始时长
                "duration_in_slice": float(t_end_new - t_start_new),  # 切片中的时长
                "source_row": idx,
                "video_exists": video_exists,
                "file_size": file_size,
                "clip_path": str(row.get('clip视频路径', '')).strip(),
                "original_bos_path": f"{row.get('clip视频路径', '').strip()}video.mp4",
                "slice_filename": os.path.basename(verified_path if video_exists else local_slice_path)
            }
            
            self.annotations[safe_label].append(annotation)
            self.processed_annotations.add(annotation_id)
            
            # 记录映射
            self.slice_mapping[slice_key] = {
                "local_path": verified_path if video_exists else local_slice_path,
                "exists": video_exists,
                "file_size": file_size
            }
            
            return True, f"成功添加切片标注: {annotation_id}"
                
        except Exception as e:
            logger.error(f"处理切片 {idx} 失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False, f"异常: {str(e)}"
    
    def analyze_slice_statistics(self, df: pd.DataFrame) -> Dict:
        """分析切片数据统计信息"""
        stats = {
            "total_slices": len(df),
            "valid_slices": 0,
            "invalid_slices": 0,
            "invalid_reasons": defaultdict(int),
            "unique_labels": [],
            "labels_count": {},
            "video_existence": {"exists": 0, "not_exists": 0, "details": []},
            "time_stats": {
                "slice_lengths": [],
                "action_in_slice_durations": [],
                "avg_action_duration": 0,
                "min_action_duration": float('inf'),
                "max_action_duration": 0
            },
            "file_stats": {
                "total_size_mb": 0,
                "avg_size_mb": 0
            }
        }
        
        for idx, row in df.iterrows():
            try:
                slice_key = str(row['slice_key']).strip()
                label = str(row['标签']).strip()
                local_slice_path = str(row.get('local_slice_path', '')).strip()
                
                # 检查必要字段
                if pd.isna(slice_key) or slice_key == '':
                    stats["invalid_slices"] += 1
                    stats["invalid_reasons"]["空slice_key"] += 1
                    continue
                
                if pd.isna(label) or label == '':
                    stats["invalid_slices"] += 1
                    stats["invalid_reasons"]["空标签"] += 1
                    continue
                
                if pd.isna(local_slice_path) or local_slice_path == '':
                    stats["invalid_slices"] += 1
                    stats["invalid_reasons"]["空本地路径"] += 1
                    continue
                
                # 有效的切片
                stats["valid_slices"] += 1
                
                # 统计标签
                if label not in stats["unique_labels"]:
                    stats["unique_labels"].append(label)
                
                if label in stats["labels_count"]:
                    stats["labels_count"][label] += 1
                else:
                    stats["labels_count"][label] = 1
                
                # 检查切片视频是否存在
                video_exists, verified_path, file_size = self.verify_slice_video_exists(local_slice_path)
                
                if video_exists:
                    stats["video_existence"]["exists"] += 1
                    stats["file_stats"]["total_size_mb"] += file_size / (1024 * 1024)
                else:
                    stats["video_existence"]["not_exists"] += 1
                
                stats["video_existence"]["details"].append({
                    "row": idx,
                    "slice_key": slice_key,
                    "original_path": local_slice_path,
                    "verified_path": verified_path,
                    "exists": video_exists,
                    "label": label,
                    "file_size_mb": file_size / (1024 * 1024) if video_exists else 0
                })
                
                # 统计时间信息
                seg_start = float(row.get('seg_start', 0))
                seg_end = float(row.get('seg_end', 0))
                t_start_new = float(row.get('t_start_new', 0))
                t_end_new = float(row.get('t_end_new', 0))
                
                slice_length = seg_end - seg_start
                action_duration = t_end_new - t_start_new
                
                stats["time_stats"]["slice_lengths"].append(slice_length)
                stats["time_stats"]["action_in_slice_durations"].append(action_duration)
                stats["time_stats"]["min_action_duration"] = min(
                    stats["time_stats"]["min_action_duration"], action_duration
                )
                stats["time_stats"]["max_action_duration"] = max(
                    stats["time_stats"]["max_action_duration"], action_duration
                )
                        
            except Exception as e:
                logger.debug(f"分析切片 {idx} 失败: {str(e)}")
                stats["invalid_slices"] += 1
                stats["invalid_reasons"][str(type(e).__name__)] += 1
                continue
        
        # 计算统计量
        stats["unique_labels_count"] = len(stats["unique_labels"])
        
        if stats["valid_slices"] > 0:
            stats["time_stats"]["avg_action_duration"] = sum(
                stats["time_stats"]["action_in_slice_durations"]
            ) / stats["valid_slices"]
            
            if stats["video_existence"]["exists"] > 0:
                stats["file_stats"]["avg_size_mb"] = stats["file_stats"]["total_size_mb"] / stats["video_existence"]["exists"]
        
        return stats
    
    def save_annotations(self):
        """保存切片视频标注到JSON文件"""
        output_dir = os.path.join(self.output_base_dir, "sliced_annotations")
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
            
            logger.info(f"保存切片标注文件: {json_path} ({len(annotations)} 个标注)")
            total_annotations += len(annotations)
            category_stats[label] = len(annotations)
        
        # 保存汇总文件
        summary = {
            "total_categories": len(self.annotations),
            "total_annotations": total_annotations,
            "annotations_per_category": category_stats,
            "categories": list(self.annotations.keys()),
            "processing_time": datetime.now().isoformat(),
            "source_csv": self.slice_info_csv,
            "slice_video_dir": self.slice_video_dir
        }
        
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存切片标注汇总文件: {summary_path}")
        
        # 保存合并的所有标注
        all_annotations = []
        for label, annotations in self.annotations.items():
            all_annotations.extend(annotations)
        
        all_annotations_path = os.path.join(output_dir, "all_sliced_annotations.json")
        with open(all_annotations_path, 'w', encoding='utf-8') as f:
            json.dump(all_annotations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存合并切片标注文件: {all_annotations_path} ({len(all_annotations)} 个标注)")
        
        return total_annotations
    
    def save_slice_mapping(self):
        """保存切片视频路径映射"""
        mapping_path = os.path.join(self.output_base_dir, "slice_mapping.json")
        
        mapping_data = {
            "total_slices": len(self.slice_mapping),
            "slices_exist": sum(1 for m in self.slice_mapping.values() if m.get("exists", False)),
            "slices_missing": sum(1 for m in self.slice_mapping.values() if not m.get("exists", False)),
            "mappings": [
                {
                    "slice_key": slice_key,
                    "local_path": info["local_path"],
                    "exists": info.get("exists", False),
                    "file_size_mb": info.get("file_size", 0) / (1024 * 1024) if info.get("file_size") else 0
                }
                for slice_key, info in self.slice_mapping.items()
            ],
            "processing_time": datetime.now().isoformat()
        }
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"保存切片路径映射: {mapping_path}")
    
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
                "slice_info_csv": self.slice_info_csv,
                "slice_video_dir": self.slice_video_dir,
                "output_base_dir": self.output_base_dir
            }
        }
        
        stats_path = os.path.join(self.output_base_dir, "sliced_processing_statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"切片处理统计信息已保存: {stats_path}")
        return stats
    
    def process_all(self, max_workers: int = 4):
        """处理所有切片数据"""
        # 加载切片信息
        df = self.load_slice_info()
        
        if df.empty:
            logger.error("无法加载切片信息，处理终止")
            return 0, 0, {}, []
        
        # 分析数据统计
        logger.info("分析切片数据统计...")
        stats = self.analyze_slice_statistics(df)
        
        print("\n" + "=" * 60)
        print("📊 切片标注数据统计:")
        print("=" * 60)
        print(f"📄 总切片数: {stats['total_slices']}")
        print(f"✅ 有效切片: {stats['valid_slices']}")
        print(f"❌ 无效切片: {stats['invalid_slices']}")
        
        if stats['invalid_reasons']:
            print(f"📉 无效原因统计:")
            for reason, count in sorted(stats['invalid_reasons'].items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count}")
        
        print(f"🏷️  唯一标签类别: {stats['unique_labels_count']}")
        
        # 显示标签统计
        if stats['labels_count']:
            print(f"\n📂 按类别统计:")
            sorted_labels = sorted(stats['labels_count'].items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_labels[:20]:
                print(f"  - {label}: {count} 个切片")
            if len(sorted_labels) > 20:
                print(f"  ... 还有 {len(sorted_labels) - 20} 个类别")
        
        # 显示时间统计
        print(f"\n⏱️  时间统计:")
        print(f"  - 平均动作时长: {stats['time_stats']['avg_action_duration']:.2f} 秒")
        print(f"  - 最短动作时长: {stats['time_stats']['min_action_duration']:.2f} 秒")
        print(f"  - 最长动作时长: {stats['time_stats']['max_action_duration']:.2f} 秒")
        
        # 检查切片长度是否为20秒
        slice_lengths = stats['time_stats']['slice_lengths']
        if slice_lengths:
            avg_slice_length = sum(slice_lengths) / len(slice_lengths)
            print(f"  - 平均切片长度: {avg_slice_length:.2f} 秒")
            if abs(avg_slice_length - 20) > 0.1:
                print(f"  ⚠️  警告: 平均切片长度不是20秒!")
        
        print(f"\n📹 切片视频文件情况:")
        print(f"  ✅ 存在的切片视频: {stats['video_existence']['exists']}")
        print(f"  ❌ 缺失的切片视频: {stats['video_existence']['not_exists']}")
        
        if stats['video_existence']['not_exists'] > 0:
            print(f"\n⚠️  警告: 有 {stats['video_existence']['not_exists']} 个切片视频文件缺失!")
            print("可能的原因:")
            print("1. 切片视频未生成或生成失败")
            print("2. 切片视频保存路径与CSV中记录的不一致")
            print("3. 视频文件被移动或删除")
            
            # 显示不匹配的详细信息
            not_exists_details = [d for d in stats['video_existence']['details'] if not d['exists']]
            if not_exists_details:
                print(f"\n🔍 缺失文件示例 (前3个):")
                for i, detail in enumerate(not_exists_details[:3], 1):
                    print(f"\n  {i}. 行 {detail['row']}:")
                    print(f"     标签: {detail['label']}")
                    print(f"     slice_key: {detail['slice_key']}")
                    print(f"     原始路径: {detail['original_path']}")
                    print(f"     验证路径: {detail['verified_path']}")
        
        if stats['valid_slices'] == 0:
            logger.error("❌ 没有有效的切片数据！")
            return 0, 0, stats, []
        
        logger.info(f"开始处理 {len(df)} 个切片...")
        
        # 准备处理任务
        tasks = []
        for idx, row in df.iterrows():
            tasks.append((idx, row))
        
        # 使用线程池并行处理
        success_count = 0
        fail_count = 0
        fail_details = []
        
        with tqdm(total=len(tasks), desc="处理进度", unit="切片") as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_idx = {}
                for idx, row in tasks:
                    future = executor.submit(self.process_single_slice, row, idx)
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
                        logger.error(f"切片 {idx} 处理异常: {str(e)}")
                    
                    pbar.update(1)
                    pbar.set_postfix_str(f"成功: {success_count}, 失败: {fail_count}")
        
        return success_count, fail_count, stats, fail_details


def verify_sliced_videos(slice_video_dir: str):
    """验证切片视频文件"""
    print("\n" + "=" * 60)
    print("验证切片视频文件...")
    print("=" * 60)
    
    if not os.path.exists(slice_video_dir):
        print(f"❌ 切片视频目录不存在: {slice_video_dir}")
        return 0, 0
    
    # 查找所有切片视频文件
    slice_files = []
    slice_sizes = {}
    total_size_mb = 0
    
    for root, dirs, files in os.walk(slice_video_dir):
        for file in files:
            if file.endswith('.mp4'):
                slice_path = os.path.join(root, file)
                rel_path = os.path.relpath(slice_path, slice_video_dir)
                file_size_mb = os.path.getsize(slice_path) / (1024 * 1024)  # MB
                slice_files.append(rel_path)
                slice_sizes[rel_path] = file_size_mb
                total_size_mb += file_size_mb
    
    print(f"找到 {len(slice_files)} 个切片视频文件")
    
    if slice_files:
        # 计算平均大小
        avg_size_mb = total_size_mb / len(slice_files) if slice_files else 0
        
        print(f"总大小: {total_size_mb:.2f} MB")
        print(f"平均大小: {avg_size_mb:.2f} MB")
        
        print("\n切片视频文件示例:")
        for i, rel_path in enumerate(slice_files[:5], 1):
            size = slice_sizes[rel_path]
            # 提取切片时间信息
            time_match = re.search(r"slice_(\d+)_(\d+)\.mp4$", rel_path)
            if time_match:
                seg_start, seg_end = time_match.groups()
                time_info = f"{seg_start}-{seg_end}s"
            else:
                time_info = "未知时间"
            print(f"  {i}. {time_info} - {rel_path} ({size:.2f} MB)")
        
        if len(slice_files) > 5:
            print(f"  ... 还有 {len(slice_files) - 5} 个文件")
    
    return len(slice_files), total_size_mb


def main():
    """主函数 - 处理切片视频"""
    SLICE_INFO_CSV = "/root/workspace/downloaded_videos_for_segment/slice_info.csv"
    SLICE_VIDEO_DIR = "/root/workspace/downloaded_videos_for_segment/sliced_videos"
    OUTPUT_BASE_DIR = "/root/workspace/sliced_vqa_annotations"
    
    print("=" * 60)
    print("🎯 切片视频标注生成工具")
    print("=" * 60)
    print(f"📁 切片视频目录: {SLICE_VIDEO_DIR}")
    print(f"📄 切片信息CSV: {SLICE_INFO_CSV}")
    print(f"📦 输出目录: {OUTPUT_BASE_DIR}")
    print("=" * 60)
    print("📋 功能说明:")
    print("  - 读取切片信息CSV文件")
    print("  - 验证切片视频文件是否存在")
    print("  - 按类别生成切片视频标注文档")
    print("  - 为抽帧提供精确的切片视频路径")
    print("=" * 60)
    
    # 验证切片视频文件
    slice_count, total_size_mb = verify_sliced_videos(SLICE_VIDEO_DIR)
    if slice_count == 0:
        logger.warning("未找到切片视频文件，但标注生成将继续进行")
    
    # 检查切片信息CSV文件
    if not os.path.exists(SLICE_INFO_CSV):
        logger.error(f"切片信息CSV文件不存在: {SLICE_INFO_CSV}")
        logger.info(f"请先运行视频下载和切片脚本生成切片信息")
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 初始化处理器
    processor = SlicedVideoAnnotationProcessor(
        slice_info_csv=SLICE_INFO_CSV,
        slice_video_dir=SLICE_VIDEO_DIR,
        output_base_dir=OUTPUT_BASE_DIR
    )
    
    # 处理所有数据
    start_time = time.time()
    success_count, fail_count, data_stats, fail_details = processor.process_all(max_workers=4)
    elapsed_time = time.time() - start_time
    
    # 保存结果
    if processor.annotations:
        total_annotations = processor.save_annotations()
        processor.save_slice_mapping()
        
        # 生成统计信息
        stats = processor.save_statistics(success_count, fail_count, data_stats, fail_details, elapsed_time)
    else:
        total_annotations = 0
        logger.warning("没有生成任何切片标注数据")
        stats = processor.save_statistics(success_count, fail_count, data_stats, fail_details, elapsed_time)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("🎉 切片标注生成完成")
    print("=" * 60)
    print(f"⏱️  总耗时: {elapsed_time:.2f}秒")
    print(f"📊 总计处理: {success_count + fail_count} 个切片")
    print(f"✅ 成功: {success_count} 个切片")
    print(f"❌ 失败: {fail_count} 个切片")
    
    if success_count > 0:
        print(f"\n📁 输出目录: {OUTPUT_BASE_DIR}")
        print("目录结构:")
        print(f"  {OUTPUT_BASE_DIR}/")
        print(f"  ├── sliced_annotations/         # 切片标注文件")
        print(f"  │   ├── all_sliced_annotations.json  # 所有切片标注的合并文件")
        print(f"  │   ├── summary.json           # 汇总信息")
        print(f"  │   └── [类别].json           # 每个类别的切片标注")
        print(f"  ├── slice_mapping.json         # 切片视频路径映射")
        print(f"  └── sliced_processing_statistics.json  # 处理统计")
        
        # 显示生成的类别
        if processor.annotations:
            print(f"\n📂 生成的切片标注类别 ({len(processor.annotations)} 个):")
            for label, annotations in sorted(processor.annotations.items(), 
                                          key=lambda x: len(x[1]), reverse=True)[:10]:
                print(f"  - {label}: {len(annotations)} 个切片")
            if len(processor.annotations) > 10:
                print(f"  ... 还有 {len(processor.annotations) - 10} 个类别")
        
        print(f"\n📋 下一步:")
        print("1. 检查切片标注文件: ls -la /root/workspace/sliced_vqa_annotations/sliced_annotations/")
        print("2. 查看切片标注统计: cat /root/workspace/sliced_vqa_annotations/sliced_processing_statistics.json | python -m json.tool")
        print("3. 对切片视频进行抽帧: python /root/workspace/LLaMA-Factory/scene_tag/1.5_get_frames_squeeze.py -i /root/workspace/downloaded_videos_2fps/sliced_videos")
        print("4. 使用切片标注进行模型训练")
        
        # 显示切片标注示例
        print(f"\n📝 切片标注示例:")
        for label, annotations in sorted(processor.annotations.items(), 
                                      key=lambda x: len(x[1]), reverse=True):
            if annotations:
                anno = annotations[0]
                print(f"  类别: {label}")
                print(f"    切片视频: {os.path.basename(anno.get('slice_video_path', 'N/A'))}")
                print(f"    切片窗口: {anno['slice_window'][0]}s-{anno['slice_window'][1]}s (原始视频)")
                print(f"    动作时间: {anno['time_range_in_slice'][0]}s-{anno['time_range_in_slice'][1]}s (切片中)")
                print(f"    文件存在: {anno.get('video_exists', False)}")
                break
    else:
        print(f"\n❌ 处理失败，没有生成任何切片标注数据")
        print("可能的原因:")
        print("1. 切片信息CSV文件格式错误")
        print("2. 所有行都有数据问题")
        print("3. 没有有效的切片行")
        print(f"\n🔍 查看详细日志: tail -100 /root/workspace/sliced_video_annotation.log")
    
    print("=" * 60)


if __name__ == "__main__":
    main()