#!/usr/bin/env python3
"""
从Excel文件读取路口数据集，下载指定车号和日期下的front_left_1摄像头视频文件
Excel字段格式: X6S5009_2025-10-22_14-22-20
其中X6S5009是car_id，2025-10-22_14-22-20是date_dir
"""

import os
import sys
import logging
import traceback
import re
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
from datetime import datetime

from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient
from baidubce.exception import BceError

# 尝试导入pandas和openpyxl
try:
    import pandas as pd
except ImportError:
    print("错误: 需要安装pandas库，请运行: pip install pandas")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print("错误: 需要安装openpyxl库，请运行: pip install openpyxl")
    sys.exit(1)

# 从环境变量读取凭证
BOS_AK = os.environ.get('BOS_AK', 'ALTAKZ49HCOHFffGHKawumDZRy')
BOS_SK = os.environ.get('BOS_SK', '7b7ec8e3832148adaa0a1ccecdf65cf4')
BOS_HOST = os.environ.get('BOS_HOST', 'bj.bcebos.com')
BUCKET_NAME = "neolix-raw"  # 固定bucket名称
MAX_THREADS = 8  # 并行下载线程数

# Excel文件路径
EXCEL_FILE = "/root/workspace/LLaMA-Factory/data/路口数据集.xlsx"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/root/workspace/download_junction.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_excel_field(field_value: str) -> Optional[Tuple[str, str]]:
    """
    解析Excel字段，格式: X6S5009_2025-10-22_14-22-20
    返回: (car_id, date_dir) 或 None
    
    Args:
        field_value: Excel中的字段值
        
    Returns:
        (car_id, date_dir) 元组，如果解析失败返回None
    """
    if pd.isna(field_value):
        return None
    
    field_str = str(field_value).strip()
    if not field_str:
        return None
    
    # 使用正则表达式匹配格式: X6S5009_2025-10-22_14-22-20
    # 第一个下划线之前是car_id，之后是date_dir
    pattern = r'^([^_]+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})$'
    match = re.match(pattern, field_str)
    
    if match:
        car_id = match.group(1)
        date_dir = match.group(2)
        return (car_id, date_dir)
    else:
        logger.warning(f"无法解析字段格式: {field_str}")
        return None


def read_junction_data_from_excel(excel_path: str) -> List[Tuple[str, str]]:
    """
    从Excel文件读取路口数据集
    
    Args:
        excel_path: Excel文件路径
        
    Returns:
        [(car_id, date_dir), ...] 列表
    """
    car_date_pairs = []
    
    try:
        # 读取Excel文件
        logger.info(f"正在读取Excel文件: {excel_path}")
        df = pd.read_excel(excel_path)
        
        logger.info(f"Excel文件列名: {df.columns.tolist()}")
        logger.info(f"Excel文件总行数: {len(df)}")
        
        # 遍历所有列，查找包含数据的列
        for col_name in df.columns:
            logger.info(f"处理列: {col_name}")
            for idx, value in df[col_name].items():
                parsed = parse_excel_field(value)
                if parsed:
                    car_id, date_dir = parsed
                    car_date_pairs.append((car_id, date_dir))
                    logger.debug(f"  行{idx}: {value} -> car_id={car_id}, date_dir={date_dir}")
        
        logger.info(f"成功解析 {len(car_date_pairs)} 个car_id和date_dir组合")
        
        # 去重
        unique_pairs = list(set(car_date_pairs))
        logger.info(f"去重后剩余 {len(unique_pairs)} 个唯一组合")
        
        return unique_pairs
        
    except Exception as e:
        logger.error(f"读取Excel文件时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return []


class BOSVideoDownloader:
    def __init__(self):
        if not BOS_AK or not BOS_SK:
            raise ValueError("请设置BOS_AK和BOS_SK环境变量")
        
        config = BceClientConfiguration(
            credentials=BceCredentials(BOS_AK, BOS_SK),
            endpoint=BOS_HOST
        )
        self.bos_client = BosClient(config)
        self.downloaded_files: Set[str] = set()  # 记录已下载文件，避免重复下载
        self._load_downloaded_cache()
    
    def _load_downloaded_cache(self):
        """加载已下载文件缓存"""
        cache_file = Path("/root/workspace/download_cache.txt")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.downloaded_files.add(line)
            logger.info(f"已加载 {len(self.downloaded_files)} 个已下载文件记录")
        else:
            logger.info("未找到下载缓存文件，将重新下载所有文件")
    
    def _save_downloaded_cache(self, bos_path: str):
        """保存已下载文件记录"""
        cache_file = Path("/root/workspace/download_cache.txt")
        with open(cache_file, 'a') as f:
            f.write(f"{bos_path}\n")
        self.downloaded_files.add(bos_path)
    
    def list_clips_for_date(self, car_id: str, date_dir: str) -> List[str]:
        """列出指定日期下的所有clips目录"""
        clips = []
        prefix = f"raw_clips/{car_id}/{date_dir}/clips/"
        
        try:
            marker = None
            while True:
                response = self.bos_client.list_objects(
                    bucket_name=BUCKET_NAME,
                    prefix=prefix,
                    delimiter='/',  # 使用分隔符模拟目录
                    marker=marker,
                    max_keys=1000
                )
                
                # 处理返回的目录（common prefixes）
                if hasattr(response, 'common_prefixes') and response.common_prefixes:
                    for common_prefix in response.common_prefixes:
                        # 提取clip目录名
                        clip_dir = common_prefix.prefix
                        # 移除前缀和斜杠
                        clip_dir = clip_dir.replace(prefix, '').rstrip('/')
                        if clip_dir and self._is_valid_clip_dir(clip_dir):
                            clips.append(clip_dir)
                
                # 检查是否还有更多结果
                if response.is_truncated:
                    marker = response.next_marker
                else:
                    break
                    
        except Exception as e:
            logger.error(f"列出日期 {date_dir} 的clips目录时出错: {str(e)}")
        
        return sorted(clips)
    
    def _is_valid_clip_dir(self, dir_name: str) -> bool:
        """检查clip目录名是否符合格式"""
        # 格式: 20251223161022_00076
        pattern = r'^\d{14}_\d{5}$'
        return bool(re.match(pattern, dir_name))
    
    def find_front_left_videos(self, car_id: str, date_dir: str, clip_dir: str) -> List[str]:
        """查找指定clip下的front_left_1视频文件"""
        videos = []
        video_path = f"raw_clips/{car_id}/{date_dir}/clips/{clip_dir}/sensor/camera/front_left_1/image/video/video.mp4"
        
        # 检查文件是否存在
        if self.file_exists(BUCKET_NAME, video_path):
            videos.append(video_path)
        
        return videos
    
    def file_exists(self, bucket: str, key: str) -> bool:
        """检查文件是否存在"""
        try:
            # 尝试获取文件元数据，如果成功则文件存在
            self.bos_client.get_object_meta_data(bucket_name=bucket, key=key)
            return True
        except Exception as e:
            # 文件不存在或其他错误
            return False
    
    def download_file(self, bos_key: str, local_base_path: str) -> bool:
        """下载单个文件"""
        local_path = os.path.join(local_base_path, bos_key)
        
        # 检查是否已下载
        if bos_key in self.downloaded_files:
            if os.path.exists(local_path):
                logger.info(f"文件已下载: {local_path}")
                return True
            else:
                # 缓存中有记录但本地文件不存在，移除缓存记录
                self.downloaded_files.discard(bos_key)
        
        try:
            # 确保本地目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 检查文件是否存在
            if not self.file_exists(BUCKET_NAME, bos_key):
                logger.error(f"文件在BOS上不存在: {bos_key}")
                return False
            
            # 下载文件
            logger.info(f"开始下载: {bos_key}")
            logger.info(f"保存到: {local_path}")
            
            # 使用get_object_to_file下载
            self.bos_client.get_object_to_file(
                bucket_name=BUCKET_NAME,
                key=bos_key,
                file_name=local_path
            )
            
            # 验证下载结果
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                if file_size > 0:
                    logger.info(f"✅ 下载成功: {local_path} ({file_size:,} bytes)")
                    
                    # 保存到下载记录
                    self._save_downloaded_cache(bos_key)
                    return True
                else:
                    logger.error(f"❌ 下载的文件为空: {local_path}")
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    return False
            else:
                logger.error(f"❌ 下载后文件不存在: {local_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ 下载失败 {bos_key}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 清理可能不完整的文件
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass
            return False
    
    def download_videos_for_car_date(
        self, 
        car_id: str, 
        date_dir: str,
        local_base_path: str, 
        max_videos: Optional[int] = None
    ) -> Dict[str, int]:
        """
        下载指定车号和日期下的front_left_1视频
        
        Args:
            car_id: 车号
            date_dir: 日期目录，格式: 2025-10-22_14-22-20
            local_base_path: 本地保存的基础路径
            max_videos: 最大下载数量（None表示不限制）
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_clips': 0,
            'total_videos_found': 0,
            'total_videos_downloaded': 0,
            'total_videos_skipped': 0,
            'total_videos_failed': 0
        }
        
        try:
            logger.info(f"处理车号: {car_id}, 日期目录: {date_dir}")
            
            # 获取该日期下的所有clips
            clips = self.list_clips_for_date(car_id, date_dir)
            stats['total_clips'] = len(clips)
            
            logger.info(f"  在日期 {date_dir} 下找到 {len(clips)} 个clips")
            
            if not clips:
                logger.warning(f"  车号 {car_id} 日期 {date_dir} 下未找到任何clips")
                return stats
            
            # 遍历每个clip
            downloaded_count = 0
            for clip_idx, clip_dir in enumerate(clips, 1):
                if max_videos and downloaded_count >= max_videos:
                    break
                
                # 查找front_left_1视频
                videos = self.find_front_left_videos(car_id, date_dir, clip_dir)
                stats['total_videos_found'] += len(videos)
                
                for video_path in videos:
                    if max_videos and downloaded_count >= max_videos:
                        break
                    
                    logger.info(f"  [{downloaded_count + 1}{f'/{max_videos}' if max_videos else ''}] 处理视频: {clip_dir}/front_left_1")
                    
                    # 下载视频
                    success = self.download_file(video_path, local_base_path)
                    
                    if success:
                        downloaded_count += 1
                        stats['total_videos_downloaded'] += 1
                    else:
                        stats['total_videos_failed'] += 1
                    
                    if max_videos:
                        logger.info(f"  进度: 已下载 {downloaded_count}/{max_videos} 个视频")
            
            logger.info(f"  完成: 找到 {stats['total_videos_found']} 个视频, 下载 {stats['total_videos_downloaded']} 个")
            
            return stats
            
        except Exception as e:
            logger.error(f"下载过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            return stats
    
    def download_junction_videos(
        self,
        car_date_pairs: List[Tuple[str, str]],
        local_base_path: str,
        max_videos_per_pair: Optional[int] = None,
        max_total_videos: Optional[int] = None
    ) -> Dict[str, int]:
        """
        批量下载路口数据集中的视频
        
        Args:
            car_date_pairs: [(car_id, date_dir), ...] 列表
            local_base_path: 本地保存的基础路径
            max_videos_per_pair: 每个car_id+date_dir组合的最大下载数量（None表示不限制）
            max_total_videos: 总的最大下载数量（None表示不限制）
        
        Returns:
            统计信息字典
        """
        total_stats = {
            'total_pairs': len(car_date_pairs),
            'processed_pairs': 0,
            'total_clips': 0,
            'total_videos_found': 0,
            'total_videos_downloaded': 0,
            'total_videos_skipped': 0,
            'total_videos_failed': 0
        }
        
        total_downloaded = 0
        
        try:
            for pair_idx, (car_id, date_dir) in enumerate(car_date_pairs, 1):
                # 检查总下载数量限制
                if max_total_videos and total_downloaded >= max_total_videos:
                    logger.info(f"已达到总下载数量限制 ({max_total_videos})，停止下载")
                    break
                
                logger.info("=" * 60)
                logger.info(f"[{pair_idx}/{len(car_date_pairs)}] 处理: car_id={car_id}, date_dir={date_dir}")
                logger.info("=" * 60)
                
                # 计算该组合还能下载多少个
                remaining_total = None
                if max_total_videos:
                    remaining_total = max_total_videos - total_downloaded
                
                # 确定该组合的最大下载数量
                max_for_this_pair = max_videos_per_pair
                if max_total_videos and remaining_total:
                    if max_videos_per_pair:
                        max_for_this_pair = min(max_videos_per_pair, remaining_total)
                    else:
                        max_for_this_pair = remaining_total
                
                # 下载该组合的视频
                stats = self.download_videos_for_car_date(
                    car_id=car_id,
                    date_dir=date_dir,
                    local_base_path=local_base_path,
                    max_videos=max_for_this_pair
                )
                
                # 累计统计
                total_stats['processed_pairs'] += 1
                total_stats['total_clips'] += stats['total_clips']
                total_stats['total_videos_found'] += stats['total_videos_found']
                total_stats['total_videos_downloaded'] += stats['total_videos_downloaded']
                total_stats['total_videos_failed'] += stats['total_videos_failed']
                
                total_downloaded += stats['total_videos_downloaded']
                
                logger.info(f"累计已下载: {total_downloaded}{f'/{max_total_videos}' if max_total_videos else ''} 个视频")
            
            logger.info("=" * 60)
            logger.info("批量下载完成!")
            logger.info(f"处理的car_id+date_dir组合数: {total_stats['processed_pairs']}/{total_stats['total_pairs']}")
            logger.info(f"clips总数: {total_stats['total_clips']}")
            logger.info(f"找到的视频总数: {total_stats['total_videos_found']}")
            logger.info(f"成功下载: {total_stats['total_videos_downloaded']}")
            logger.info(f"下载失败: {total_stats['total_videos_failed']}")
            logger.info("=" * 60)
            
            return total_stats
            
        except Exception as e:
            logger.error(f"批量下载过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            return total_stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='从Excel文件读取路口数据集，下载front_left_1摄像头视频')
    parser.add_argument('--excel-file', type=str, default=EXCEL_FILE,
                       help=f'Excel文件路径，默认: {EXCEL_FILE}')
    parser.add_argument('--max-videos-per-pair', type=int, default=None,
                       help='每个car_id+date_dir组合的最大下载数量，默认: 不限制')
    parser.add_argument('--max-total-videos', type=int, default=None,
                       help='总的最大下载数量，默认: 不限制')
    parser.add_argument('--local-path', type=str, default='/mnt/pfs/houhaotian/junction_videos',
                       help='本地保存基础路径，默认: /mnt/pfs/houhaotian/junction_videos')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("开始下载路口数据集视频")
    logger.info(f"Excel文件: {args.excel_file}")
    logger.info(f"每个组合最大下载数量: {args.max_videos_per_pair if args.max_videos_per_pair else '不限制'}")
    logger.info(f"总最大下载数量: {args.max_total_videos if args.max_total_videos else '不限制'}")
    logger.info(f"本地保存路径: {args.local_path}")
    logger.info("=" * 60)
    
    try:
        # 1. 从Excel读取数据
        car_date_pairs = read_junction_data_from_excel(args.excel_file)
        
        if not car_date_pairs:
            logger.error("未能从Excel文件中读取到任何有效数据")
            sys.exit(1)
        
        logger.info(f"从Excel读取到 {len(car_date_pairs)} 个car_id+date_dir组合")
        
        # 2. 初始化下载器
        downloader = BOSVideoDownloader()
        
        # 3. 开始批量下载
        stats = downloader.download_junction_videos(
            car_date_pairs=car_date_pairs,
            local_base_path=args.local_path,
            max_videos_per_pair=args.max_videos_per_pair,
            max_total_videos=args.max_total_videos
        )
        
        # 4. 输出总结
        logger.info("=" * 60)
        logger.info("下载统计:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        if stats['total_videos_downloaded'] > 0:
            logger.info("✅ 下载任务完成!")
        else:
            logger.info("⚠️  没有下载任何新视频")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()


'''
# 基本用法：下载Excel中所有路口数据集的视频（不限制数量）
python 7_download_from_bos_junction.py

# 限制每个组合最多下载10个视频
python 7_download_from_bos_junction.py --max-videos-per-pair 10

# 限制总共最多下载100个视频
python 7_download_from_bos_junction.py --max-total-videos 100

# 同时限制每个组合和总数
python 7_download_from_bos_junction.py --max-videos-per-pair 5 --max-total-videos 50

# 指定Excel文件路径和本地保存路径
python 7_download_from_bos_junction.py --excel-file /path/to/excel.xlsx --local-path /mnt/pfs/houhaotian/my_videos
'''
