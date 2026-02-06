import os
import subprocess
from pathlib import Path
import time

def is_video_processed(video_path, output_dir, filename):
    """检查视频是否已经处理过（三个切片都存在）"""
    for i in range(1, 4):
        segment_path = output_dir / f"{filename}_segment_{i:03d}.mp4"
        if not segment_path.exists() or segment_path.stat().st_size < 1024:
            return False
    return True

def process_videos():
    # 设置路径
    input_base = Path("/mnt/pfs/houhaotian/junction_videos")
    output_base = Path("/mnt/pfs/houhaotian/junction_videos_segment")
    
    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.MP4', '.AVI', '.MOV', '.MKV'}
    
    # 递归查找所有视频文件
    video_files = []
    for root, dirs, files in os.walk(input_base):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in video_extensions:
                video_files.append(file_path)
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    total_segments = 0
    skipped_videos = 0
    processed_videos = 0
    start_time = time.time()
    
    for idx, video_path in enumerate(video_files, 1):
        # 计算相对路径
        relative_path = video_path.relative_to(input_base)
        
        # 创建对应的输出目录结构
        output_dir = output_base / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取文件名（不带扩展名）
        filename = video_path.stem
        
        # 检查是否已处理
        if is_video_processed(video_path, output_dir, filename):
            skipped_videos += 1
            print(f"[{idx}/{len(video_files)}] 跳过已处理: {relative_path}")
            continue
        
        print(f"[{idx}/{len(video_files)}] 处理: {relative_path}")
        processed_videos += 1
        
        # 切割视频为三个20秒片段
        for i, start_sec in enumerate([0, 20, 40], 1):
            # 使用原文件名加上段号
            segment_name = f"{filename}_segment_{i:03d}.mp4"
            segment_path = output_dir / segment_name
            
            # 构建切割命令
            cut_cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-ss', f'00:00:{start_sec:02d}',
                '-t', '20',
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-loglevel', 'error',
                '-y',
                str(segment_path)
            ]
            
            try:
                # 执行切割
                subprocess.run(cut_cmd, check=True, capture_output=True)
                
                # 检查文件是否成功创建
                if segment_path.exists() and segment_path.stat().st_size > 0:
                    # 对切割后的视频进行2FPS抽帧
                    temp_path = output_dir / f"{filename}_segment_{i:03d}_temp.mp4"
                    
                    # 构建抽帧命令
                    extract_cmd = [
                        'ffmpeg',
                        '-i', str(segment_path),
                        '-vf', 'fps=2',
                        '-c:a', 'copy',
                        '-loglevel', 'error',
                        '-y',
                        str(temp_path)
                    ]
                    
                    try:
                        # 执行抽帧
                        subprocess.run(extract_cmd, check=True, capture_output=True)
                        
                        # 覆盖原文件
                        if temp_path.exists() and temp_path.stat().st_size > 0:
                            segment_path.unlink()
                            temp_path.rename(segment_path)
                            total_segments += 1
                        
                    except subprocess.CalledProcessError as e:
                        print(f"  警告: 片段{i}抽帧失败 - 保留原始切片")
                        total_segments += 1
                
            except subprocess.CalledProcessError as e:
                # 捕获ffmpeg错误，但继续处理其他片段
                if "Invalid data found" in str(e.stderr) or "moov atom not found" in str(e.stderr):
                    print(f"  警告: 视频文件可能损坏 - {video_path.name}")
                elif "Invalid argument" in str(e.stderr) or "position out of range" in str(e.stderr):
                    print(f"  警告: 片段{i}超出视频长度，跳过此片段")
                else:
                    print(f"  警告: 片段{i}切割失败 - {str(e.stderr)[:100]}")
            except Exception as e:
                print(f"  警告: 处理片段{i}时发生未知错误 - {str(e)[:100]}")
    
    elapsed_time = time.time() - start_time
    print(f"\n处理完成！")
    print(f"找到视频文件: {len(video_files)} 个")
    print(f"已跳过视频: {skipped_videos} 个")
    print(f"新处理视频: {processed_videos} 个")
    print(f"生成视频片段: {total_segments} 个")
    print(f"总耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    process_videos()