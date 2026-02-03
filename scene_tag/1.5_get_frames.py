import os
import sys
import cv2
from pathlib import Path
import logging
import traceback
from tqdm import tqdm
import argparse
from datetime import datetime
import json
import subprocess
import tempfile
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/video_downsample.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoDownsampler:
    """视频降采样工具，从10fps降为2fps，不调整分辨率"""
    
    def __init__(self, target_fps=2, original_fps=10):
        """
        初始化降采样器
        
        Args:
            target_fps: 目标帧率 (默认2fps)
            original_fps: 原始帧率 (默认10fps)
        """
        self.target_fps = target_fps
        self.original_fps = original_fps
        
        # 计算抽帧间隔
        self.downsample_ratio = max(1, int(original_fps / target_fps))
        
        logger.info(f"视频降采样配置:")
        logger.info(f"  原始帧率: {original_fps}fps")
        logger.info(f"  目标帧率: {target_fps}fps")
        # logger.info(f"  目标分辨率: {target_resolution[0]}x{target_resolution[1]}")
        logger.info(f"  抽帧间隔: 每{self.downsample_ratio}帧抽取1帧")
    
    # def _resize_frame(self, frame):
    #     """调整帧分辨率为目标分辨率"""
    #     if frame is None:
    #         return None
            
    #     # 获取原始分辨率
    #     h, w = frame.shape[:2]
    #     target_w, target_h = self.target_resolution
        
    #     # 检查是否需要调整大小
    #     if w == target_w and h == target_h:
    #         return frame
            
    #     # 调整大小
    #     resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    #     return resized_frame
    
    def get_video_info(self, video_path):
        """获取视频信息"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"无法打开视频: {video_path}")
                return None
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 如果frame_count为0，尝试计算
            if frame_count <= 0:
                # 通过读取所有帧来计数
                frame_count = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                # 重置视频位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 计算实际时长
            duration = frame_count / fps if fps > 0 else 0
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
            }
        except Exception as e:
            logger.warning(f"获取视频信息失败 {video_path}: {str(e)}")
            return None
        finally:
            if 'cap' in locals():
                cap.release()
    
    def _try_fourcc_codes(self, width, height, fps, output_path):
        """尝试多种编码器"""
        fourcc_options = [
            cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
            cv2.VideoWriter_fourcc(*'avc1'),  # H.264/AVC
            cv2.VideoWriter_fourcc(*'X264'),  # x264
            cv2.VideoWriter_fourcc(*'h264'),  # H.264
            cv2.VideoWriter_fourcc(*'XVID'),  # Xvid (生成.avi)
        ]
        
        for fourcc in fourcc_options:
            try:
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    logger.debug(f"使用编码器成功: {fourcc}")
                    return out
                out.release()
            except:
                continue
        return None
    
    def downsample_video_opencv_robust(self, input_path, output_path=None):
        """
        使用OpenCV进行视频降采样 - 健壮版本
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径 (默认覆盖原文件)
        """
        if output_path is None:
            output_path = input_path
        
        # 检查输入文件
        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在: {input_path}")
            return False
        
        # 获取视频信息
        video_info = self.get_video_info(input_path)
        if not video_info:
            logger.error(f"无法读取视频信息: {input_path}")
            return False
        
        original_size = video_info['file_size']
        # target_w, target_h = self.target_resolution
        
        logger.info(f"处理视频: {input_path}")
        logger.info(f"  原始分辨率: {video_info['width']}x{video_info['height']}")
        # logger.info(f"  目标分辨率: {target_w}x{target_h}")
        logger.info(f"  原始帧率: {video_info['fps']:.2f}fps")
        logger.info(f"  目标帧率: {self.target_fps}fps")
        logger.info(f"  原始帧数: {video_info['frame_count']}")
        logger.info(f"  目标帧数: ~{int(video_info['frame_count'] / self.downsample_ratio)}")
        logger.info(f"  文件大小: {original_size / 1024 / 1024:.2f} MB")
        
        # 检查是否已经是目标帧率和分辨率
        if abs(video_info['fps'] - self.target_fps) < 0.1:
            logger.info(f"视频已经是{self.target_fps}fps，跳过处理: {input_path}")
            return True
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='video_downsample_')
        temp_output = os.path.join(temp_dir, Path(input_path).name)
        
        # 打开视频
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            logger.error(f"无法打开视频: {input_path}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
        
        # 尝试多种编码器，使用原始分辨率
        out = self._try_fourcc_codes(
            video_info['width'],  # 使用原始宽度
            video_info['height'],  # 使用原始高度
            self.target_fps, 
            temp_output
        )
        
        if out is None:
            logger.error(f"无法创建视频写入器，尝试所有编码器都失败: {temp_output}")
            cap.release()
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
        
        try:
            frame_idx = 0
            saved_frame_idx = 0
            
            # 计算目标帧数
            target_frame_count = int(video_info['frame_count'] / self.downsample_ratio)
            pbar = tqdm(total=target_frame_count, 
                       desc=f"处理 {Path(input_path).name}", 
                       unit="帧")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 每downsample_ratio帧保存一帧
                if frame_idx % self.downsample_ratio == 0:
                    # 调整分辨率
                    # resized_frame = self._resize_frame(frame)
                    resized_frame = frame
                    if resized_frame is not None:
                        out.write(resized_frame)
                        saved_frame_idx += 1
                        pbar.update(1)
                
                frame_idx += 1
            
            pbar.close()
            
            # 释放资源
            cap.release()
            out.release()
            
            logger.info(f"处理完成: 读取{frame_idx}帧, 保存{saved_frame_idx}帧")
            
            # 检查输出文件
            if not os.path.exists(temp_output):
                logger.error(f"临时输出文件不存在: {temp_output}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            output_size = os.path.getsize(temp_output)
            if output_size == 0:
                logger.error(f"输出文件为空: {temp_output}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 替换原文件
            if input_path == output_path and os.path.exists(input_path):
                os.remove(input_path)
            
            # 移动文件
            shutil.move(temp_output, output_path)
            
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 验证输出文件
            if os.path.exists(output_path):
                processed_size = os.path.getsize(output_path)
                if processed_size > 0:
                    compression_ratio = 1 - (processed_size / original_size)
                    logger.info(f"处理完成: {output_path}")
                    logger.info(f"  输出大小: {processed_size / 1024 / 1024:.2f} MB")
                    logger.info(f"  压缩比例: {compression_ratio:.1%}")
                    
                    # 验证分辨率
                    processed_info = self.get_video_info(output_path)
                    if processed_info:
                        logger.info(f"  输出分辨率: {processed_info['width']}x{processed_info['height']}")
                        logger.info(f"  输出帧率: {processed_info['fps']:.2f}fps")
                        logger.info(f"  输出帧数: {processed_info['frame_count']}")
                    
                    return True
                else:
                    logger.error(f"输出文件为空: {output_path}")
                    return False
            else:
                logger.error(f"输出文件不存在: {output_path}")
                return False
            
        except Exception as e:
            logger.error(f"处理视频失败 {input_path}: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 清理
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals() and out.isOpened():
                out.release()
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
    
    def downsample_video_ffmpeg_simple(self, input_path, output_path=None):
        """
        使用简单的FFmpeg命令进行视频降采样
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
        """
        if output_path is None:
            output_path = input_path
        
        # 检查输入文件
        if not os.path.exists(input_path):
            logger.error(f"输入文件不存在: {input_path}")
            return False
        
        # 获取视频信息
        video_info = self.get_video_info(input_path)
        if not video_info:
            logger.error(f"无法读取视频信息: {input_path}")
            return False
        
        original_size = video_info['file_size']
        # target_w, target_h = self.target_resolution
        
        logger.info(f"使用FFmpeg处理视频: {input_path}")
        logger.info(f"  原始分辨率: {video_info['width']}x{video_info['height']}")
        # logger.info(f"  目标分辨率: {target_w}x{target_h}")
        logger.info(f"  原始帧率: {video_info['fps']:.2f}fps")
        logger.info(f"  目标帧率: {self.target_fps}fps")
        logger.info(f"  文件大小: {original_size / 1024 / 1024:.2f} MB")
        
        # 检查是否已经是目标帧率
        if abs(video_info['fps'] - self.target_fps) < 0.1:
            logger.info(f"视频已经是{self.target_fps}fps，跳过处理: {input_path}")
            return True 
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='video_ffmpeg_')
        temp_output = os.path.join(temp_dir, Path(output_path).name)
        
        try:
            # 构建FFmpeg命令，添加缩放滤镜
            # 命令结构: 先调整帧率，再调整分辨率
            cmd_with_audio = [
                'ffmpeg',
                '-i', input_path,
                '-r', str(self.target_fps),  # 设置输出帧率
                '-c:v', 'libx264',           # 视频编码
                '-preset', 'fast',           # 编码速度
                '-crf', '23',               # 质量
                '-c:a', 'copy',             # 复制音频
                '-y',                       # 覆盖
                temp_output
            ]
            
            cmd_no_audio = [
                'ffmpeg',
                '-i', input_path,
                '-r', str(self.target_fps),
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-an',                      # 不处理音频
                '-y',
                temp_output
            ]
            
            cmds_to_try = [cmd_with_audio, cmd_no_audio]
            success = False
            
            for cmd in cmds_to_try:
                logger.debug(f"尝试FFmpeg命令: {' '.join(cmd)}")
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5分钟超时
                        check=False
                    )
                    
                    if result.returncode == 0:
                        success = True
                        break
                    else:
                        logger.warning(f"FFmpeg命令失败，尝试下一个命令: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg处理超时")
                    break
                except Exception as e:
                    logger.warning(f"FFmpeg执行异常: {str(e)}")
                    continue
            
            if not success:
                logger.error(f"所有FFmpeg命令都失败: {input_path}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            # 检查输出文件
            if not os.path.exists(temp_output):
                logger.error(f"临时输出文件不存在: {temp_output}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            output_size = os.path.getsize(temp_output)
            if output_size == 0:
                logger.error(f"输出文件为空: {temp_output}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False
            
            # 确保输出目录存在
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 替换原文件
            if input_path == output_path and os.path.exists(input_path):
                os.remove(input_path)
            
            # 移动文件
            shutil.move(temp_output, output_path)
            
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 验证输出
            if os.path.exists(output_path):
                processed_size = os.path.getsize(output_path)
                compression_ratio = 1 - (processed_size / original_size)
                logger.info(f"FFmpeg处理完成: {output_path}")
                logger.info(f"  输出大小: {processed_size / 1024 / 1024:.2f} MB")
                logger.info(f"  压缩比例: {compression_ratio:.1%}")
                
                # 验证分辨率
                processed_info = self.get_video_info(output_path)
                if processed_info:
                    logger.info(f"  输出分辨率: {processed_info['width']}x{processed_info['height']}")
                    logger.info(f"  输出帧率: {processed_info['fps']:.2f}fps")
                    
                return True
            else:
                logger.error(f"输出文件不存在: {output_path}")
                return False
            
        except Exception as e:
            logger.error(f"FFmpeg处理失败 {input_path}: {str(e)}")
            logger.error(traceback.format_exc())
            
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
    
    def process_video(self, input_path, output_path=None, method='auto'):
        """
        处理单个视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            method: 处理方法 ('opencv', 'ffmpeg', 'auto')
        """
        if method == 'auto':
            # 检查FFmpeg是否可用
            try:
                subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              check=True)
                method = 'ffmpeg'
                logger.debug("检测到FFmpeg可用")
            except:
                method = 'opencv'
                logger.debug("FFmpeg不可用，使用OpenCV")
        
        if method == 'ffmpeg':
            return self.downsample_video_ffmpeg_simple(input_path, output_path)
        else:
            return self.downsample_video_opencv_robust(input_path, output_path)
    
    def process_directory(self, directory, pattern='**/*.mp4', 
                         output_dir=None, method='auto'):
        """
        处理目录下的所有视频
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            output_dir: 输出目录（None则覆盖原文件）
            method: 处理方法
        
        Returns:
            dict: 处理结果统计
        """
        directory = Path(directory)
        
        # 查找视频文件
        video_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            logger.warning(f"在目录 {directory} 中未找到MP4视频文件")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"找到 {len(video_files)} 个MP4视频文件")
        
        # 处理每个视频
        stats = {
            'total': len(video_files),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'failed_files': []
        }
        
        for video_path in tqdm(video_files, desc="处理视频", unit="个"):
            try:
                video_info = self.get_video_info(video_path)
                if not video_info:
                    logger.error(f"无法读取视频信息，跳过: {video_path}")
                    stats['failed'] += 1
                    stats['failed_files'].append(video_path)
                    continue
                                
                # 检查是否已经是目标帧率
                if abs(video_info['fps'] - self.target_fps) < 0.1:
                    logger.debug(f"视频已经是{self.target_fps}fps，跳过: {video_path}")
                    stats['skipped'] += 1
                    continue
                
                if output_dir:
                    # 构建相对路径
                    rel_path = Path(video_path).relative_to(directory)
                    output_path = Path(output_dir) / rel_path
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_path = video_path
                
                # 处理视频
                success = self.process_video(video_path, str(output_path), method)
                
                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['failed_files'].append(video_path)
                    
            except Exception as e:
                logger.error(f"处理视频失败 {video_path}: {str(e)}")
                stats['failed'] += 1
                stats['failed_files'].append(video_path)
        
        return stats

def find_videos_from_excel(excel_path, base_dir):
    """从Excel文件中查找需要处理的视频路径"""
    try:
        import pandas as pd
        
        # 读取Excel文件
        df = pd.read_excel(excel_path)
        logger.info(f"Excel文件加载成功，共 {len(df)} 行")
        
        # 验证必要的列
        if 'clip视频路径' not in df.columns:
            logger.error("Excel文件中缺少'clip视频路径'列")
            return []
        
        # 提取视频路径
        video_paths = []
        for idx, row in df.iterrows():
            clip_path = str(row['clip视频路径']).strip()
            
            if pd.isna(clip_path) or clip_path == '':
                continue
            
            # 构建本地视频路径
            if clip_path.startswith("bos:"):
                clip_path = clip_path[4:]
            
            # 移除开头的斜杠
            clip_path = clip_path.lstrip('/')
            
            # 分割路径获取相对部分
            parts = clip_path.split('/', 1)
            if len(parts) < 2:
                continue
            
            relative_path = parts[1]  # 移除bucket部分
            
            # 添加video.mp4
            video_filename = os.path.join(relative_path, "video.mp4")
            local_path = os.path.join(base_dir, video_filename)
            
            if os.path.exists(local_path):
                video_paths.append(local_path)
            else:
                logger.warning(f"视频文件不存在: {local_path}")
        
        logger.info(f"从Excel中找到 {len(video_paths)} 个视频文件")
        return video_paths
        
    except Exception as e:
        logger.error(f"读取Excel文件失败: {str(e)}")
        return []

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视频降采样工具 - 从10fps降到2fps，保持原始分辨率')
    parser.add_argument('--input', '-i', required=True, 
                       help='输入视频文件或目录路径')
    parser.add_argument('--output', '-o', 
                       help='输出目录路径（不指定则覆盖原文件）')
    parser.add_argument('--excel', '-e', 
                       help='Excel文件路径（用于查找视频）')
    parser.add_argument('--method', '-m', default='auto',
                       choices=['auto', 'opencv', 'ffmpeg'],
                       help='处理方法：auto/opencv/ffmpeg')
    parser.add_argument('--target-fps', type=float, default=2.0,
                       help='目标帧率（默认2fps）')
    parser.add_argument('--original-fps', type=float, default=10.0,
                       help='原始帧率（默认10fps）')

    args = parser.parse_args()
    
    print("=" * 60)
    print("视频降采样工具 - 从10fps降到2fps，保持原始分辨率")
    print("=" * 60)
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入路径不存在: {input_path}")
        sys.exit(1)
    
    # 创建降采样器
    downsampler = VideoDownsampler(
        target_fps=args.target_fps,
        original_fps=args.original_fps,
    )
    
    # 收集视频文件
    video_files = []
    
    if args.excel:
        # 从Excel文件查找视频
        base_dir = args.input if input_path.is_dir() else input_path.parent
        video_files = find_videos_from_excel(args.excel, base_dir)
        
        if not video_files:
            logger.error("从Excel文件中未找到视频，请检查路径")
            sys.exit(1)
    elif input_path.is_file():
        # 单个视频文件
        if input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
            video_files = [str(input_path)]
        else:
            logger.error(f"不支持的文件格式: {input_path.suffix}")
            sys.exit(1)
    else:
        # 目录处理
        output_dir = args.output
        
        # 处理目录
        stats = downsampler.process_directory(
            directory=str(input_path),
            output_dir=output_dir,
            method=args.method
        )
        
        # 输出结果
        print("\n" + "=" * 60)
        print("处理完成")
        print("=" * 60)
        print(f"总计: {stats['total']} 个视频")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        print(f"跳过: {stats['skipped']}")
        
        if stats['failed_files']:
            print(f"\n失败文件 ({len(stats['failed_files'])} 个):")
            for i, file_path in enumerate(stats['failed_files'][:10], 1):
                print(f"  {i}. {file_path}")
            if len(stats['failed_files']) > 10:
                print(f"  ... 还有 {len(stats['failed_files']) - 10} 个失败文件")
            
            # 保存失败文件列表
            fail_log_path = "/root/workspace/downsample_failures.txt"
            with open(fail_log_path, 'w', encoding='utf-8') as f:
                for file_path in stats['failed_files']:
                    f.write(f"{file_path}\n")
            print(f"失败文件列表已保存到: {fail_log_path}")
        
        return
    
    # 处理从Excel找到的视频文件
    logger.info(f"开始处理 {len(video_files)} 个视频文件...")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    failed_files = []
    
    for video_path in tqdm(video_files, desc="处理视频", unit="个"):
        try:
            # 检查是否已经是目标帧率
            video_info = downsampler.get_video_info(video_path)
            if not video_info:
                logger.error(f"无法读取视频信息，跳过: {video_path}")
                fail_count += 1
                failed_files.append(video_path)
                continue
            
            if abs(video_info['fps'] - args.target_fps) < 0.1:
                logger.debug(f"视频已经是{args.target_fps}fps，跳过: {video_path}")
                skipped_count += 1
                continue
            
            output_path = None
            if args.output:
                # 构建相对路径
                rel_path = Path(video_path).relative_to(input_path)
                output_path = Path(args.output) / rel_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            success = downsampler.process_video(
                video_path, 
                str(output_path) if output_path else None,
                args.method
            )
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_files.append(video_path)
                
        except Exception as e:
            logger.error(f"处理视频失败 {video_path}: {str(e)}")
            fail_count += 1
            failed_files.append(video_path)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"总计: {len(video_files)} 个视频")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"跳过: {skipped_count}")
    
    if failed_files:
        print(f"\n失败文件 ({len(failed_files)} 个):")
        for i, file_path in enumerate(failed_files[:10], 1):
            print(f"  {i}. {file_path}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个失败文件")
        
        # 保存失败文件列表
        fail_log_path = "/root/workspace/downsample_failures.txt"
        with open(fail_log_path, 'w', encoding='utf-8') as f:
            for file_path in failed_files:
                f.write(f"{file_path}\n")
        print(f"失败文件列表已保存到: {fail_log_path}")
    
    # 保存处理记录
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'input': str(input_path),
        'output': args.output,
        'method': args.method,
        'target_fps': args.target_fps,
        'original_fps': args.original_fps,
        'total_videos': len(video_files),
        'success': success_count,
        'failed': fail_count,
        'skipped': skipped_count,
        'failed_files': failed_files
    }
    
    log_path = "/root/workspace/downsample_log.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"处理日志已保存到: {log_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()

# 处理单个视频
# python video_downsampler.py -i /path/to/input/video.mp4
# 处理整个目录:
# python video_downsampler.py -i /path/to/input/folder target_fps默认是2