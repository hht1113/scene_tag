import os
import pandas as pd
import sys
import time
import threading
import subprocess
from queue import Queue
from typing import List, Set, Dict
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient
from tqdm import tqdm
import logging
from pathlib import Path
import traceback
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/video_download_and_slice.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 从环境变量读取凭证
BOS_AK = os.environ.get('BOS_AK','ALTAKTdWjuZ7BdKtuKV8oWQeSn')
BOS_SK = os.environ.get('BOS_SK','68c709fbd2fc43708c12192175150673')
BOS_HOST = os.environ.get('BOS_HOST', 'bj.bcebos.com')
MAX_THREADS = 4  # 并行下载线程数
VIDEO_TOTAL_LENGTH = 60  # 假设视频总长为60秒
SLICE_LENGTH = 20  # 切片长度20秒

class VideoSlicer:
    """视频切片工具类"""
    
    @staticmethod
    def slice_video(video_path: str, seg_start: int, seg_end: int, output_path: str) -> bool:
        """
        对视频进行切片
        使用FFmpeg从seg_start到seg_end截取视频片段
        
        参数:
            video_path: 输入视频路径
            seg_start: 切片开始时间(秒)
            seg_end: 切片结束时间(秒)
            output_path: 输出切片路径
            
        返回:
            bool: 是否切片成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 计算持续时间
            duration = seg_end - seg_start
            
            # 构建FFmpeg命令
            cmd = [
                'ffmpeg',
                '-i', video_path,  # 输入文件
                '-ss', str(seg_start),  # 开始时间
                '-t', str(duration),  # 持续时间
                '-c', 'copy',  # 直接复制编码，不重新编码
                '-avoid_negative_ts', '1',  # 避免负时间戳
                '-y',  # 覆盖输出文件
                output_path
            ]
            
            logger.info(f"执行切片命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode != 0:
                logger.error(f"切片失败: {result.stderr}")
                return False
            
            # 检查输出文件是否存在且大小合理
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                if file_size > 1024:  # 至少1KB
                    logger.info(f"切片成功: {output_path} ({file_size} bytes)")
                    
                    # 验证视频文件是否可以打开
                    try:
                        # 快速验证视频文件
                        probe_cmd = [
                            'ffprobe',
                            '-v', 'error',
                            '-select_streams', 'v:0',
                            '-show_entries', 'stream=duration',
                            '-of', 'default=noprint_wrappers=1:nokey=1',
                            output_path
                        ]
                        probe_result = subprocess.run(
                            probe_cmd,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        if probe_result.returncode == 0:
                            actual_duration = float(probe_result.stdout.strip())
                            logger.debug(f"切片时长: {actual_duration}秒")
                        return True
                    except Exception as e:
                        logger.warning(f"无法验证切片视频: {e}")
                        return True
                else:
                    logger.error(f"切片文件过小: {output_path} ({file_size} bytes)")
                    try:
                        os.remove(output_path)
                    except:
                        pass
                    return False
            else:
                logger.error(f"切片输出文件不存在: {output_path}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"切片超时: {video_path}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return False
        except Exception as e:
            logger.error(f"切片异常: {video_path}, 错误: {str(e)}")
            logger.error(traceback.format_exc())
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return False
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """获取视频时长"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0.0


class BOSExcelDownloader:
    def __init__(self):
        if not BOS_AK or not BOS_SK:
            raise ValueError("请设置BOS_AK和BOS_SK环境变量")
        
        config = BceClientConfiguration(
            credentials=BceCredentials(BOS_AK, BOS_SK),
            endpoint=BOS_HOST
        )
        self.bos_client = BosClient(config)
        self.downloaded_files = set()  # 记录已下载文件，避免重复下载
        self.sliced_videos = set()  # 记录已切片视频
        self.slice_info = []  # 存储切片信息
        self._load_downloaded_cache()

        # 添加超时设置
        config.connection_timeout_in_mills = 30000  # 连接超时30秒
        config.socket_timeout_in_mills = 300000     # 数据超时5分钟
        self.bos_client = BosClient(config)
    
    def _load_downloaded_cache(self):
        """加载已下载文件缓存"""
        cache_file = Path("/root/workspace/download_cache.txt")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SLICE:"):
                        self.sliced_videos.add(line[6:])  # 移除"SLICE:"前缀
                    else:
                        self.downloaded_files.add(line)
    
    def _save_downloaded_cache(self, bos_path: str, is_slice: bool = False):
        """保存已下载文件记录"""
        cache_file = Path("/root/workspace/download_cache.txt")
        with open(cache_file, 'a') as f:
            if is_slice:
                f.write(f"SLICE:{bos_path}\n")
                self.sliced_videos.add(bos_path)
            else:
                f.write(f"{bos_path}\n")
                self.downloaded_files.add(bos_path)
    
    def parse_bos_path(self, bos_path: str):
        """解析BOS路径，提取bucket和key"""
        # 移除开头的bos:前缀
        if bos_path.startswith("bos:"):
            bos_path = bos_path[4:]
        
        # 移除开头的斜杠
        bos_path = bos_path.lstrip('/')
        
        # 分割bucket和key
        parts = bos_path.split('/', 1)
        if len(parts) < 2:
            raise ValueError(f"无效的BOS路径格式: {bos_path}")
        
        bucket = parts[0]
        key = parts[1]
        
        return bucket, key
    
    def download_file(self, bucket: str, key: str, local_path: str) -> bool:
        """下载单个文件，带详细的错误处理"""
        try:
            # 确保本地目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 检查文件是否已存在且完整
            file_size = self.get_file_size(bucket, key)
            
            if file_size <= 0:
                logger.warning(f"文件大小无效或无法获取: {bucket}/{key} (大小: {file_size})")
                skip_size_check = True
            else:
                skip_size_check = False
            
            # 检查是否需要下载
            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                
                if not skip_size_check and local_size == file_size:
                    logger.info(f"文件已存在且完整: {local_path}")
                    return True
                elif skip_size_check and local_size > 1024:
                    logger.info(f"文件已存在且大小合理: {local_path} ({local_size} bytes)")
                    return True
                else:
                    logger.warning(f"文件大小不匹配，重新下载: {local_path} (本地: {local_size}, 远程: {file_size})")
                    try:
                        os.remove(local_path)
                    except:
                        pass
            
            # 下载文件
            logger.info(f"开始下载: {bucket}/{key} -> {local_path}")
            
            # 使用流式下载
            try:
                self.bos_client.get_object_to_file(
                    bucket_name=bucket,
                    key=key,
                    file_name=local_path
                )
            except Exception as e:
                logger.error(f"下载失败 {bucket}/{key}: {str(e)}")
                if os.path.exists(local_path):
                    try:
                        os.remove(local_path)
                    except:
                        pass
                return False
            
            # 验证下载完整性
            if os.path.exists(local_path):
                downloaded_size = os.path.getsize(local_path)
                
                if not skip_size_check and downloaded_size != file_size:
                    logger.error(f"文件大小不匹配: {local_path} (期望: {file_size}, 实际: {downloaded_size})")
                    try:
                        os.remove(local_path)
                    except:
                        pass
                    return False
                elif downloaded_size == 0:
                    logger.error(f"下载的文件为空: {local_path}")
                    try:
                        os.remove(local_path)
                    except:
                        pass
                    return False
                else:
                    logger.info(f"下载完成: {local_path} ({downloaded_size} bytes)")
                    return True
            else:
                logger.error(f"下载后文件不存在: {local_path}")
                return False
                
        except Exception as e:
            logger.error(f"下载失败 {bucket}/{key}: {str(e)}")
            logger.error(traceback.format_exc())
            
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except:
                    pass
            return False
    
    def get_file_size(self, bucket: str, key: str) -> int:
        """获取文件大小"""
        try:
            response = self.bos_client.get_object_meta_data(bucket_name=bucket, key=key)
            
            if hasattr(response, 'metadata') and 'content-length' in response.metadata:
                file_size = int(response.metadata['content-length'])
                return file_size
            else:
                file_size = 0
                if hasattr(response, 'content_length'):
                    file_size = int(response.content_length)
                elif hasattr(response, 'metadata') and hasattr(response.metadata, 'get'):
                    file_size = int(response.metadata.get('content-length', 0))
                return file_size
        except Exception as e:
            logger.warning(f"无法获取文件大小 {bucket}/{key}: {str(e)}")
            return 0
    
    def process_excel(self, excel_path: str, download_dir: str) -> Dict:
        """
        处理Excel文件，提取视频下载和切片信息
        
        返回:
            dict: 包含视频任务和切片任务
        """
        try:
            # 读取Excel文件
            df = pd.read_excel(excel_path)
            logger.info(f"Excel文件加载成功，共 {len(df)} 行")
            
            # 验证必要的列
            required_columns = ['clip视频路径', '标签', 'T_start', 'T_end']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Excel文件中缺少必要的列: {missing_columns}")
                return {"video_tasks": [], "slice_tasks": []}
            
            # 计算切片时间窗口
            # seg_start: 最大不超过(60-20)=40，最小为0
            df['seg_start'] = df['T_start'].apply(
                lambda x: max(0, min(x - 2, VIDEO_TOTAL_LENGTH - SLICE_LENGTH))
            )
            # seg_end = seg_start + 20
            df['seg_end'] = df['seg_start'] + SLICE_LENGTH
            # t_start_new = T_start - seg_start
            df['t_start_new'] = df['T_start'] - df['seg_start']
            # t_end_new = T_end - seg_start
            df['t_end_new'] = df['T_end'] - df['seg_start']
            
            # 添加主键列: BOS路径+segment_start+segment_end
            df['slice_key'] = df.apply(
                lambda row: f"{row['clip视频路径'].strip('/')}_{int(row['seg_start'])}_{int(row['seg_end'])}", 
                axis=1
            )
            
            # 提取不重复的原始视频下载任务
            video_tasks = []
            slice_tasks_by_video = {}  # 按视频分组切片任务
            
            for idx, row in df.iterrows():
                clip_path = str(row['clip视频路径']).strip()
                
                if pd.isna(clip_path) or clip_path == '':
                    logger.warning(f"第{idx}行: clip视频路径为空，跳过")
                    continue
                
                # 确保路径以斜杠结尾
                if not clip_path.endswith('/'):
                    clip_path += '/'
                
                # 原始视频的BOS路径
                bos_path = clip_path + "video.mp4"
                
                # 生成本地保存路径
                try:
                    bucket, key = self.parse_bos_path(bos_path)
                    
                    # 原始视频的本地路径
                    relative_path = key
                    local_video_path = os.path.join(download_dir, "original_videos", relative_path)
                    
                    # 切片视频的本地路径
                    slice_filename = f"slice_{int(row['seg_start'])}_{int(row['seg_end'])}.mp4"
                    slice_relative_path = key.replace("video.mp4", f"slices/{slice_filename}")
                    local_slice_path = os.path.join(download_dir, "sliced_videos", slice_relative_path)
                    
                    # 添加原始视频下载任务（如果尚未添加）
                    video_task_key = f"{bucket}/{key}"
                    if video_task_key not in {f"{b}/{k}" for b, k, _, _ in video_tasks}:
                        video_tasks.append((bucket, key, local_video_path, bos_path))
                    
                    # 添加切片任务
                    slice_task = {
                        'row_index': idx,
                        'slice_key': row['slice_key'],
                        'clip_path': clip_path,
                        'label': row['标签'],
                        't_start': row['T_start'],
                        't_end': row['T_end'],
                        'seg_start': row['seg_start'],
                        'seg_end': row['seg_end'],
                        't_start_new': row['t_start_new'],
                        't_end_new': row['t_end_new'],
                        'local_video_path': local_video_path,
                        'local_slice_path': local_slice_path
                    }
                    
                    # 按原始视频路径分组切片任务
                    if bos_path not in slice_tasks_by_video:
                        slice_tasks_by_video[bos_path] = []
                    slice_tasks_by_video[bos_path].append(slice_task)
                    
                except Exception as e:
                    logger.error(f"第{idx}行: 处理失败 {clip_path}: {str(e)}")
                    continue
            
            # 保存切片信息到CSV
            slice_info_df = df[[
                'slice_key', 'clip视频路径', '标签', 'T_start', 'T_end',
                'seg_start', 'seg_end', 't_start_new', 't_end_new'
            ]].copy()
            
            # 添加本地路径
            slice_info_df['local_video_path'] = ""
            slice_info_df['local_slice_path'] = ""
            
            for idx, row in df.iterrows():
                clip_path = str(row['clip视频路径']).strip()
                if not clip_path.endswith('/'):
                    clip_path += '/'
                bos_path = clip_path + "video.mp4"
                
                try:
                    bucket, key = self.parse_bos_path(bos_path)
                    relative_path = key
                    local_video_path = os.path.join(download_dir, "original_videos", relative_path)
                    
                    slice_filename = f"slice_{int(row['seg_start'])}_{int(row['seg_end'])}.mp4"
                    slice_relative_path = key.replace("video.mp4", f"slices/{slice_filename}")
                    local_slice_path = os.path.join(download_dir, "sliced_videos", slice_relative_path)
                    
                    slice_info_df.at[idx, 'local_video_path'] = local_video_path
                    slice_info_df.at[idx, 'local_slice_path'] = local_slice_path
                except:
                    pass
            
            # 保存切片信息
            slice_info_path = os.path.join(download_dir, "slice_info.csv")
            os.makedirs(os.path.dirname(slice_info_path), exist_ok=True)
            slice_info_df.to_csv(slice_info_path, index=False, encoding='utf-8-sig')
            logger.info(f"切片信息已保存到: {slice_info_path}")
            
            # 也保存为JSON，便于后续处理
            slice_info_json_path = os.path.join(download_dir, "slice_info.json")
            slice_info_list = []
            for bos_path, tasks in slice_tasks_by_video.items():
                for task in tasks:
                    slice_info_list.append(task)
            
            with open(slice_info_json_path, 'w', encoding='utf-8') as f:
                json.dump(slice_info_list, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON格式切片信息已保存到: {slice_info_json_path}")
            logger.info(f"找到 {len(video_tasks)} 个唯一原始视频需要下载")
            logger.info(f"生成 {len(slice_info_list)} 个切片任务")
            
            # 输出示例
            for i, (bucket, key, local_path, bos_path) in enumerate(video_tasks[:3], 1):
                logger.info(f"示例原始视频任务 {i}:")
                logger.info(f"  BOS路径: {bos_path}")
                logger.info(f"  本地路径: {local_path}")
                logger.info(f"  相关切片数: {len(slice_tasks_by_video.get(bos_path, []))}")
            
            for i, slice_task in enumerate(slice_info_list[:3], 1):
                logger.info(f"示例切片任务 {i}:")
                logger.info(f"  主键: {slice_task['slice_key']}")
                logger.info(f"  时间窗口: {slice_task['seg_start']}-{slice_task['seg_end']}s")
                logger.info(f"  动作时间(相对): {slice_task['t_start_new']}-{slice_task['t_end_new']}s")
                logger.info(f"  切片保存路径: {slice_task['local_slice_path']}")
            
            return {
                "video_tasks": video_tasks,
                "slice_tasks_by_video": slice_tasks_by_video,
                "slice_info_path": slice_info_path,
                "slice_info_json_path": slice_info_json_path
            }
            
        except Exception as e:
            logger.error(f"读取Excel文件失败: {str(e)}")
            logger.error(traceback.format_exc())
            return {"video_tasks": [], "slice_tasks": []}


class DownloadSliceWorker(threading.Thread):
    """下载和切片工作线程"""
    def __init__(self, downloader: BOSExcelDownloader, video_queue: Queue, 
                 slice_queue: Queue, result_queue: Queue, 
                 slice_tasks_by_video: Dict, download_dir: str):
        threading.Thread.__init__(self)
        self.downloader = downloader
        self.video_queue = video_queue
        self.slice_queue = slice_queue
        self.result_queue = result_queue
        self.slice_tasks_by_video = slice_tasks_by_video
        self.download_dir = download_dir
        self.daemon = True
        self.slicer = VideoSlicer()
    
    def run(self):
        while True:
            task = self.video_queue.get()
            if task is None:  # 退出信号
                break
            
            bucket, key, local_video_path, bos_path = task
            video_success = False
            
            # 检查原始视频是否已下载
            if bos_path in self.downloader.downloaded_files:
                if os.path.exists(local_video_path):
                    file_size = os.path.getsize(local_video_path)
                    if file_size > 1024 * 1024:  # 至少1MB
                        logger.info(f"原始视频已存在: {local_video_path} ({file_size} bytes)")
                        video_success = True
                    else:
                        logger.warning(f"原始视频已存在但过小，重新下载: {local_video_path} ({file_size} bytes)")
            
            # 下载原始视频
            if not video_success:
                video_success = self.downloader.download_file(bucket, key, local_video_path)
                if video_success:
                    self.downloader._save_downloaded_cache(bos_path, is_slice=False)
                    self.result_queue.put(("video", True, bos_path, "下载完成"))
                else:
                    self.result_queue.put(("video", False, bos_path, "下载失败"))
                    self.video_queue.task_done()
                    continue
            
            # 下载成功后，处理该视频的所有切片
            if bos_path in self.slice_tasks_by_video:
                slice_tasks = self.slice_tasks_by_video[bos_path]
                for slice_task in slice_tasks:
                    slice_key = slice_task['slice_key']
                    local_slice_path = slice_task['local_slice_path']
                    
                    # 检查切片是否已存在
                    if f"SLICE:{slice_key}" in self.downloader.sliced_videos:
                        if os.path.exists(local_slice_path):
                            file_size = os.path.getsize(local_slice_path)
                            if file_size > 1024:  # 至少1KB
                                logger.info(f"切片已存在: {local_slice_path}")
                                self.result_queue.put(("slice", True, slice_key, "已存在"))
                                continue
                    
                    # 执行切片
                    seg_start = slice_task['seg_start']
                    seg_end = slice_task['seg_end']
                    
                    logger.info(f"开始切片: {local_video_path} -> {local_slice_path} ({seg_start}-{seg_end}s)")
                    
                    slice_success = self.slicer.slice_video(
                        local_video_path, 
                        seg_start, 
                        seg_end, 
                        local_slice_path
                    )
                    
                    if slice_success:
                        self.downloader._save_downloaded_cache(slice_key, is_slice=True)
                        self.result_queue.put(("slice", True, slice_key, "切片完成"))
                    else:
                        self.result_queue.put(("slice", False, slice_key, "切片失败"))
            else:
                logger.warning(f"视频没有对应的切片任务: {bos_path}")
            
            self.video_queue.task_done()


def main():
    """主函数"""
    EXCEL_PATH = "/root/workspace/LLaMA-Factory/data/人工标注视频数据_对比实验_12tags_split.xlsx"
    DOWNLOAD_DIR = "/root/workspace/downloaded_videos_for_segment"
    
    print("=" * 60)
    print("BOS视频下载与切片工具")
    print("=" * 60)
    print(f"Excel文件: {EXCEL_PATH}")
    print(f"下载目录: {DOWNLOAD_DIR}")
    print(f"视频总长: {VIDEO_TOTAL_LENGTH}秒")
    print(f"切片长度: {SLICE_LENGTH}秒")
    print(f"线程数: {MAX_THREADS}")
    print("=" * 60)
    
    # 检查Excel文件
    if not os.path.exists(EXCEL_PATH):
        logger.error(f"Excel文件不存在: {EXCEL_PATH}")
        sys.exit(1)
    
    # 初始化下载器
    try:
        downloader = BOSExcelDownloader()
    except Exception as e:
        logger.error(f"初始化BOS客户端失败: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # 解析Excel文件
    logger.info("正在解析Excel文件...")
    result = downloader.process_excel(EXCEL_PATH, DOWNLOAD_DIR)
    
    video_tasks = result.get("video_tasks", [])
    slice_tasks_by_video = result.get("slice_tasks_by_video", {})
    slice_info_path = result.get("slice_info_path", "")
    
    if not video_tasks:
        logger.error("没有找到需要下载的视频")
        sys.exit(1)
    
    # 创建下载队列
    video_queue = Queue()
    result_queue = Queue()
    
    for task in video_tasks:
        video_queue.put(task)
    
    # 启动工作线程
    workers = []
    for i in range(min(MAX_THREADS, len(video_tasks))):
        worker = DownloadSliceWorker(
            downloader, 
            video_queue, 
            Queue(),  # 不需要单独的切片队列
            result_queue, 
            slice_tasks_by_video,
            DOWNLOAD_DIR
        )
        worker.start()
        workers.append(worker)
        logger.info(f"启动下载切片线程 {i+1}")
    
    # 显示进度
    total_videos = len(video_tasks)
    total_slices = sum(len(tasks) for tasks in slice_tasks_by_video.values())
    
    video_success = 0
    video_fail = 0
    slice_success = 0
    slice_fail = 0
    failed_videos = []
    failed_slices = []
    
    with tqdm(total=total_videos + total_slices, desc="整体进度", unit="任务") as pbar:
        completed = 0
        
        while completed < (total_videos + total_slices):
            if not result_queue.empty():
                task_type, success, key, message = result_queue.get()
                
                if task_type == "video":
                    if success:
                        video_success += 1
                    else:
                        video_fail += 1
                        failed_videos.append(key)
                else:  # slice
                    if success:
                        slice_success += 1
                    else:
                        slice_fail += 1
                        failed_slices.append(key)
                
                completed += 1
                pbar.update(1)
                pbar.set_postfix_str(f"视频: {video_success}/{video_fail} 切片: {slice_success}/{slice_fail}")
            else:
                time.sleep(0.1)
        
        # 停止工作线程
        for _ in range(len(workers)):
            video_queue.put(None)
        for worker in workers:
            worker.join()
    
    # 输出摘要
    print("\n" + "=" * 60)
    print("下载和切片任务完成")
    print("=" * 60)
    print(f"原始视频: {total_videos} 个")
    print(f"  成功: {video_success}")
    print(f"  失败: {video_fail}")
    print(f"切片视频: {total_slices} 个")
    print(f"  成功: {slice_success}")
    print(f"  失败: {slice_fail}")
    print(f"原始视频保存目录: {os.path.join(DOWNLOAD_DIR, 'original_videos')}")
    print(f"切片视频保存目录: {os.path.join(DOWNLOAD_DIR, 'sliced_videos')}")
    print(f"切片信息文件: {slice_info_path}")
    
    # 统计文件大小
    total_original_size = 0
    original_files = []
    for root, dirs, files in os.walk(os.path.join(DOWNLOAD_DIR, "original_videos")):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_original_size += file_size
                original_files.append(file_path)
    
    total_slice_size = 0
    slice_files = []
    for root, dirs, files in os.walk(os.path.join(DOWNLOAD_DIR, "sliced_videos")):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_slice_size += file_size
                slice_files.append(file_path)
    
    print(f"\n文件统计:")
    print(f"原始视频文件数: {len(original_files)}")
    print(f"原始视频总大小: {total_original_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"切片视频文件数: {len(slice_files)}")
    print(f"切片视频总大小: {total_slice_size / 1024 / 1024 / 1024:.2f} GB")
    
    # 输出失败的文件列表
    if video_fail > 0 or slice_fail > 0:
        print(f"\n失败任务列表:")
        if video_fail > 0:
            print(f"失败原始视频 ({video_fail} 个):")
            for i, bos_path in enumerate(failed_videos[:5], 1):
                print(f"  {i}. {bos_path}")
            if len(failed_videos) > 5:
                print(f"  ... 还有 {len(failed_videos) - 5} 个")
        
        if slice_fail > 0:
            print(f"失败切片 ({slice_fail} 个):")
            for i, slice_key in enumerate(failed_slices[:5], 1):
                print(f"  {i}. {slice_key}")
            if len(failed_slices) > 5:
                print(f"  ... 还有 {len(failed_slices) - 5} 个")
        
        # 保存失败文件列表
        fail_log_path = "/root/workspace/download_and_slice_failures.txt"
        with open(fail_log_path, 'w', encoding='utf-8') as f:
            f.write("=== 失败原始视频 ===\n")
            for bos_path in failed_videos:
                f.write(f"{bos_path}\n")
            f.write("\n=== 失败切片 ===\n")
            for slice_key in failed_slices:
                f.write(f"{slice_key}\n")
        print(f"失败文件列表已保存到: {fail_log_path}")
    
    print("=" * 60)
    print("\n下一步操作:")
    print(f"1. 切片信息文件: {slice_info_path}")
    print(f"2. 切片视频目录: {os.path.join(DOWNLOAD_DIR, 'sliced_videos')}")
    print(f"3. 可以直接使用切片视频进行抽帧处理")
    print(f"4. 每个切片视频的主键格式: BOS路径_seg_start_seg_end")


if __name__ == "__main__":
    main()

'''
1. 修改EXCEL_PATH为excel文件的路径
2. 修改DOWNLOAD_DIR为想要保存视频的路径
3. 运行
'''
