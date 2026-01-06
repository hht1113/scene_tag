import os
import pandas as pd
import sys
import time
import threading
from queue import Queue
from typing import List, Set
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient
from tqdm import tqdm
import logging
from pathlib import Path
import hashlib
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/video_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 从环境变量读取凭证
BOS_AK = os.environ.get('BOS_AK','ALTAKTdWjuZ7BdKtuKV8oWQeSn')
BOS_SK = os.environ.get('BOS_SK','68c709fbd2fc43708c12192175150673')
BOS_HOST = os.environ.get('BOS_HOST', 'bj.bcebos.com')
MAX_THREADS = 8  # 并行下载线程数

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
        self._load_downloaded_cache()
    
    def _load_downloaded_cache(self):
        """加载已下载文件缓存"""
        cache_file = Path("/root/workspace/download_cache.txt")
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                self.downloaded_files = set(line.strip() for line in f)
        else:
            logger.info("未找到下载缓存文件，将重新下载所有文件")
    
    def _save_downloaded_cache(self, bos_path: str):
        """保存已下载文件记录"""
        cache_file = Path("/root/workspace/download_cache.txt")
        with open(cache_file, 'a') as f:
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
    
    def get_file_size(self, bucket: str, key: str) -> int:
        """获取文件大小 - 修正方法名"""
        try:
            # 正确的BOS SDK方法是get_object_meta_data
            response = self.bos_client.get_object_meta_data(bucket_name=bucket, key=key)
            
            # 从响应中获取文件大小
            if hasattr(response, 'metadata') and 'content-length' in response.metadata:
                file_size = int(response.metadata['content-length'])
                logger.debug(f"获取文件大小成功: {bucket}/{key} - {file_size} bytes")
                return file_size
            else:
                # 尝试其他可能的属性
                file_size = 0
                if hasattr(response, 'content_length'):
                    file_size = int(response.content_length)
                elif hasattr(response, 'metadata') and hasattr(response.metadata, 'get'):
                    file_size = int(response.metadata.get('content-length', 0))
                
                logger.debug(f"获取文件大小: {bucket}/{key} - {file_size} bytes (通过备用方法)")
                return file_size
                
        except Exception as e:
            logger.warning(f"无法获取文件大小 {bucket}/{key}: {str(e)}")
            # 尝试另一种方法：通过head_object
            try:
                # 尝试使用head_object（如果可用）
                response = self.bos_client.head_object(bucket_name=bucket, key=key)
                if hasattr(response, 'content_length'):
                    file_size = int(response.content_length)
                    logger.debug(f"通过head_object获取文件大小: {bucket}/{key} - {file_size} bytes")
                    return file_size
            except Exception as e2:
                logger.warning(f"head_object也失败: {str(e2)}")
            
            return 0
    
    def download_file(self, bucket: str, key: str, local_path: str) -> bool:
        """下载单个文件，带详细的错误处理"""
        try:
            # 确保本地目录存在
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 检查文件是否已存在且完整
            file_size = self.get_file_size(bucket, key)
            
            if file_size <= 0:
                logger.warning(f"文件大小无效或无法获取: {bucket}/{key} (大小: {file_size})")
                # 仍然尝试下载，但跳过完整性检查
                skip_size_check = True
            else:
                skip_size_check = False
            
            # 检查是否需要下载
            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                
                if not skip_size_check and local_size == file_size:
                    logger.info(f"文件已存在且完整: {local_path}")
                    return True
                elif skip_size_check and local_size > 1024:  # 至少1KB
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
                # 使用get_object_to_file方法
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
    
    def process_excel(self, excel_path: str, download_dir: str) -> List[tuple]:
        """处理Excel文件，提取视频下载信息"""
        try:
            # 读取Excel文件
            df = pd.read_excel(excel_path)
            logger.info(f"Excel文件加载成功，共 {len(df)} 行")
            
            # 验证必要的列
            required_columns = ['clip视频路径', '标签', 'T_start', 'T_end']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Excel文件中缺少必要的列: {missing_columns}")
                return []
            
            # 提取不重复的视频路径
            video_tasks = []
            seen_paths = set()
            
            for idx, row in df.iterrows():
                clip_path = str(row['clip视频路径']).strip()
                
                if pd.isna(clip_path) or clip_path == '':
                    logger.warning(f"第{idx}行: clip视频路径为空，跳过")
                    continue
                
                # 确保路径以斜杠结尾
                if not clip_path.endswith('/'):
                    clip_path += '/'
                
                # 添加video.mp4
                bos_path = clip_path + "video.mp4"
                
                if bos_path in seen_paths:
                    logger.debug(f"跳过重复路径: {bos_path}")
                    continue
                
                seen_paths.add(bos_path)
                
                # 解析BOS路径
                try:
                    bucket, key = self.parse_bos_path(bos_path)
                    
                    # 生成本地保存路径
                    # 移除bucket部分，保留后面的路径结构
                    relative_path = key
                    local_path = os.path.join(download_dir, relative_path)
                    
                    video_tasks.append((bucket, key, local_path, bos_path))
                    logger.debug(f"添加下载任务: {bos_path}")
                    
                except Exception as e:
                    logger.error(f"第{idx}行: 解析路径失败 {clip_path}: {str(e)}")
                    continue
            
            logger.info(f"找到 {len(video_tasks)} 个唯一视频需要下载")
            
            # 输出前5个任务作为示例
            for i, (bucket, key, local_path, bos_path) in enumerate(video_tasks[:5], 1):
                logger.info(f"示例任务 {i}: {bos_path}")
                logger.info(f"   Bucket: {bucket}")
                logger.info(f"   Key: {key}")
                logger.info(f"   本地路径: {local_path}")
            
            return video_tasks
            
        except Exception as e:
            logger.error(f"读取Excel文件失败: {str(e)}")
            logger.error(traceback.format_exc())
            return []

class DownloadWorker(threading.Thread):
    """下载工作线程"""
    def __init__(self, downloader: BOSExcelDownloader, queue: Queue, result_queue: Queue):
        threading.Thread.__init__(self)
        self.downloader = downloader
        self.queue = queue
        self.result_queue = result_queue
        self.daemon = True
    
    def run(self):
        while True:
            task = self.queue.get()
            if task is None:  # 退出信号
                break
            
            bucket, key, local_path, bos_path = task
            
            # 检查是否已下载
            if bos_path in self.downloader.downloaded_files:
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    if file_size > 1024:  # 至少1KB
                        logger.info(f"文件已存在: {local_path} ({file_size} bytes)")
                        self.result_queue.put((True, bos_path, "已存在"))
                        self.queue.task_done()
                        continue
                    else:
                        logger.warning(f"文件已存在但过小，重新下载: {local_path} ({file_size} bytes)")
            
            # 下载文件
            success = self.downloader.download_file(bucket, key, local_path)
            
            if success:
                self.downloader._save_downloaded_cache(bos_path)
                self.result_queue.put((True, bos_path, "下载完成"))
            else:
                self.result_queue.put((False, bos_path, "下载失败"))
            
            self.queue.task_done()

def test_bos_connection():
    """测试BOS连接"""
    print("测试BOS连接...")
    
    try:
        if not BOS_AK or not BOS_SK:
            print("错误: 请设置BOS_AK和BOS_SK环境变量")
            print("export BOS_AK=your_access_key")
            print("export BOS_SK=your_secret_key")
            return False
        
        config = BceClientConfiguration(
            credentials=BceCredentials(BOS_AK, BOS_SK),
            endpoint=BOS_HOST
        )
        client = BosClient(config)
        
        # 测试简单的BOS操作
        # 尝试列出bucket（需要权限）
        response = client.list_buckets()
        print(f"✓ BOS连接成功")
        print(f"  可用bucket数量: {len(response.buckets)}")
        
        return True
    except Exception as e:
        print(f"✗ BOS连接失败: {str(e)}")
        return False

def main():
    """主函数"""
    EXCEL_PATH = "/root/workspace/人工标注视频数据.xlsx"
    DOWNLOAD_DIR = "/root/workspace/downloaded_videos"
    
    print("=" * 60)
    print("BOS视频下载工具 (Excel驱动)")
    print("=" * 60)
    
    # 测试BOS连接
    if not test_bos_connection():
        print("\n请检查:")
        print("1. BOS_AK和BOS_SK环境变量是否正确设置")
        print("2. 网络连接是否正常")
        print("3. BOS服务是否可用")
        sys.exit(1)
    
    print(f"\nExcel文件: {EXCEL_PATH}")
    print(f"下载目录: {DOWNLOAD_DIR}")
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
    video_tasks = downloader.process_excel(EXCEL_PATH, DOWNLOAD_DIR)
    
    if not video_tasks:
        logger.error("没有找到需要下载的视频")
        sys.exit(1)
    
    # 创建下载队列
    download_queue = Queue()
    result_queue = Queue()
    
    for task in video_tasks:
        download_queue.put(task)
    
    # 启动工作线程
    workers = []
    for i in range(min(MAX_THREADS, len(video_tasks))):
        worker = DownloadWorker(downloader, download_queue, result_queue)
        worker.start()
        workers.append(worker)
        logger.info(f"启动下载线程 {i+1}")
    
    # 显示下载进度
    total_files = len(video_tasks)
    success_count = 0
    fail_count = 0
    failed_files = []
    
    with tqdm(total=total_files, desc="下载进度", unit="文件") as pbar:
        completed = 0
        
        while completed < total_files:
            if not result_queue.empty():
                success, bos_path, message = result_queue.get()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_files.append(bos_path)
                    logger.warning(f"下载失败: {bos_path} - {message}")
                
                completed += 1
                pbar.update(1)
                pbar.set_postfix_str(f"成功: {success_count}, 失败: {fail_count}")
            else:
                time.sleep(0.1)
        
        # 停止工作线程
        for _ in range(len(workers)):
            download_queue.put(None)
        for worker in workers:
            worker.join()
    
    # 输出摘要
    print("\n" + "=" * 60)
    print("下载任务完成")
    print("=" * 60)
    print(f"总计: {total_files} 个视频")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"视频保存目录: {DOWNLOAD_DIR}")
    
    # 统计已下载的文件大小
    total_size = 0
    downloaded_files = []
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                downloaded_files.append(file_path)
    
    print(f"已下载文件数: {len(downloaded_files)}")
    print(f"总大小: {total_size / 1024 / 1024:.2f} MB")
    
    # 输出失败的文件列表
    if fail_count > 0:
        print(f"\n失败文件 ({fail_count} 个):")
        for i, bos_path in enumerate(failed_files[:10], 1):  # 只显示前10个
            print(f"  {i}. {bos_path}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")
        
        # 保存失败文件列表
        fail_log_path = "/root/workspace/download_failures.txt"
        with open(fail_log_path, 'w', encoding='utf-8') as f:
            for bos_path in failed_files:
                f.write(f"{bos_path}\n")
        print(f"失败文件列表已保存到: {fail_log_path}")
    
    # 创建成功文件列表
    success_log_path = "/root/workspace/download_success.txt"
    with open(success_log_path, 'w', encoding='utf-8') as f:
        for bos_path in downloader.downloaded_files:
            f.write(f"{bos_path}\n")
    print(f"成功文件列表已保存到: {success_log_path}")
    
    print("=" * 60)
    print("\n下一步操作:")
    print(f"1. 检查下载的视频文件: ls -lh {DOWNLOAD_DIR}")
    print("2. 运行视频片段截取脚本: python process_clips.py")

def create_test_excel():
    """创建一个测试Excel文件用于验证"""
    import pandas as pd
    
    # 创建测试数据
    test_data = [
        {
            "clip视频路径": "bos:/neolix-raw/raw_clips/X6S5006/2025-12-17_15-35-02/clips/20251217153545_00018/sensor/camera/front_left_1/image/video/",
            "车号+任务号路径": "/X6S5006/2025-12-17_15-35-02/",
            "标签": "路口通行.路口掉头.普通掉头",
            "T_start": 15,
            "T_end": 30
        },
        {
            "clip视频路径": "bos:/neolix-raw/raw_clips/X6S5006/2025-12-17_15-35-02/clips/20251217153545_00018/sensor/camera/front_left_1/image/video/",
            "车号+任务号路径": "",
            "标签": "变道绕行.车道内避让.静态障碍车避让",
            "T_start": 25,
            "T_end": 40
        }
    ]
    
    df = pd.DataFrame(test_data)
    test_path = "/root/workspace/test_videos.xlsx"
    df.to_excel(test_path, index=False)
    print(f"测试Excel文件已创建: {test_path}")
    return test_path

if __name__ == "__main__":
    # 检查参数
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("创建测试Excel文件...")
        test_excel = create_test_excel()
        print("使用测试文件运行下载脚本...")
        EXCEL_PATH = test_excel
    else:
        EXCEL_PATH = "/root/workspace/人工标注视频数据.xlsx"
    
    main()