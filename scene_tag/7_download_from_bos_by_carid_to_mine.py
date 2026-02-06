#!/usr/bin/env python3
"""
下载指定车号下front_left_1摄像头的视频文件
保持原有路径结构，指定下载个数，避免重复下载
"""

import os
import sys
import logging
import traceback
import re
from pathlib import Path
from typing import List, Set, Dict, Optional
from datetime import datetime

from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient
from baidubce.exception import BceError

# 从环境变量读取凭证
BOS_AK = os.environ.get('BOS_AK', 'ALTAKZ49HCOHFffGHKawumDZRy')
BOS_SK = os.environ.get('BOS_SK', '7b7ec8e3832148adaa0a1ccecdf65cf4')
BOS_HOST = os.environ.get('BOS_HOST', 'bj.bcebos.com')
BUCKET_NAME = "neolix-raw"  # 固定bucket名称
MAX_THREADS = 8  # 并行下载线程数

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/root/workspace/download.log')
    ]
)
logger = logging.getLogger(__name__)

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
    
    def list_dates_for_car(self, car_id: str) -> List[str]:
        """列出指定车号下的所有日期目录"""
        dates = []
        prefix = f"raw_clips/{car_id}/"
        
        logger.info(f"正在搜索车号 {car_id} 的日期目录...")
        
        try:
            # 使用list_objects列出车号下的所有前缀
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
                        # 提取日期目录名，格式如：2025-12-23_16-09-31/
                        date_dir = common_prefix.prefix
                        # 移除前缀和斜杠
                        date_dir = date_dir.replace(prefix, '').rstrip('/')
                        if date_dir and self._is_valid_date_dir(date_dir):
                            dates.append(date_dir)
                
                # 检查是否还有更多结果
                if response.is_truncated:
                    marker = response.next_marker
                else:
                    break
                    
        except Exception as e:
            logger.error(f"列出日期目录时出错: {str(e)}")
        
        logger.info(f"找到 {len(dates)} 个日期目录: {dates[:5]}{'...' if len(dates) > 5 else ''}")
        return sorted(dates)
    
    def _is_valid_date_dir(self, dir_name: str) -> bool:
        """检查目录名是否符合日期格式"""
        # 格式: 2025-12-23_16-09-31
        pattern = r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$'
        return bool(re.match(pattern, dir_name))
    
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
    
    def download_car_videos(
        self, 
        car_id: str, 
        local_base_path: str, 
        max_videos: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, int]:
        """
        下载指定车号下的front_left_1视频
        
        Args:
            car_id: 车号
            local_base_path: 本地保存的基础路径
            max_videos: 最大下载数量
            start_date: 开始日期（可选，格式: 2025-12-23_16-09-31）
            end_date: 结束日期（可选，格式: 2025-12-23_16-09-31）
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_dates': 0,
            'total_clips': 0,
            'total_videos_found': 0,
            'total_videos_downloaded': 0,
            'total_videos_skipped': 0,
            'total_videos_failed': 0
        }
        
        try:
            # 1. 获取所有日期目录
            all_dates = self.list_dates_for_car(car_id)
            if not all_dates:
                logger.error(f"车号 {car_id} 下未找到任何日期目录")
                return stats
            
            # 2. 过滤日期范围
            filtered_dates = []
            for date_dir in all_dates:
                if start_date and date_dir < start_date:
                    continue
                if end_date and date_dir > end_date:
                    continue
                filtered_dates.append(date_dir)
            
            logger.info(f"车号 {car_id} 找到 {len(all_dates)} 个日期目录")
            if start_date or end_date:
                logger.info(f"经过日期过滤后剩 {len(filtered_dates)} 个日期目录")
            
            stats['total_dates'] = len(filtered_dates)
            
            # 3. 遍历每个日期目录
            downloaded_count = 0
            for date_idx, date_dir in enumerate(filtered_dates, 1):
                if downloaded_count >= max_videos:
                    break
                
                logger.info(f"[{date_idx}/{len(filtered_dates)}] 处理日期: {date_dir}")
                
                # 获取该日期下的所有clips
                clips = self.list_clips_for_date(car_id, date_dir)
                stats['total_clips'] += len(clips)
                
                logger.info(f"  在日期 {date_dir} 下找到 {len(clips)} 个clips")
                
                # 遍历每个clip
                for clip_idx, clip_dir in enumerate(clips, 1):
                    if downloaded_count >= max_videos:
                        break
                    
                    # 查找front_left_1视频
                    videos = self.find_front_left_videos(car_id, date_dir, clip_dir)
                    stats['total_videos_found'] += len(videos)
                    
                    for video_path in videos:
                        if downloaded_count >= max_videos:
                            break
                        
                        logger.info(f"  [{downloaded_count + 1}/{max_videos}] 处理视频: {clip_dir}/front_left_1")
                        
                        # 下载视频
                        success = self.download_file(video_path, local_base_path)
                        
                        if success:
                            downloaded_count += 1
                            stats['total_videos_downloaded'] += 1
                        else:
                            stats['total_videos_failed'] += 1
                        
                        logger.info(f"  进度: 已下载 {downloaded_count}/{max_videos} 个视频")
            
            # 统计跳过的文件
            stats['total_videos_skipped'] = len(self.downloaded_files) - stats['total_videos_downloaded']
            
            logger.info("=" * 60)
            logger.info("下载完成!")
            logger.info(f"日期目录总数: {stats['total_dates']}")
            logger.info(f"clips总数: {stats['total_clips']}")
            logger.info(f"找到的视频总数: {stats['total_videos_found']}")
            logger.info(f"成功下载: {stats['total_videos_downloaded']}")
            logger.info(f"跳过下载(已存在): {stats['total_videos_skipped']}")
            logger.info(f"下载失败: {stats['total_videos_failed']}")
            logger.info("=" * 60)
            
            return stats
            
        except Exception as e:
            logger.error(f"下载过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            return stats

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='下载指定车号下的front_left_1摄像头视频')
    parser.add_argument('--car-id', type=str, required=True, 
                       help='车号，例如: X6S5868')
    parser.add_argument('--max-videos', type=int, default=10,
                       help='最大下载数量，默认: 10')
    parser.add_argument('--local-path', type=str, default='/root/workspace/digged_videos',
                       help='本地保存基础路径，默认: /root/workspace/digged_videos')
    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期(可选)，格式: 2025-12-23_16-09-31')
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期(可选)，格式: 2025-12-23_16-09-31')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("开始下载车号视频")
    logger.info(f"车号: {args.car_id}")
    logger.info(f"最大下载数量: {args.max_videos}")
    logger.info(f"本地保存路径: {args.local_path}")
    if args.start_date:
        logger.info(f"开始日期: {args.start_date}")
    if args.end_date:
        logger.info(f"结束日期: {args.end_date}")
    logger.info("=" * 60)
    
    try:
        # 初始化下载器
        downloader = BOSVideoDownloader()
        
        # 开始下载
        stats = downloader.download_car_videos(
            car_id=args.car_id,
            local_base_path=args.local_path,
            max_videos=args.max_videos,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # 输出总结
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
# 基本用法：下载车号X6S5868的最多10个视频
python download_videos.py --car-id X6S5868 --max-videos 10

# 指定更多数量
python download_videos.py --car-id X6S5868 --max-videos 50

# 指定日期范围
python download_videos.py --car-id X6S5868 --max-videos 20 --start-date 2025-12-23_14-00-00 --end-date 2025-12-23_16-00-00

# 指定本地保存路径
python /root/workspace/LLaMA-Factory/scene_tag/7_download_from_bos_by_carid_to_mine.py --car-id X6S5868 --max-videos 10000 --local-path /mnt/pfs/houhaotian/my_videos
'''