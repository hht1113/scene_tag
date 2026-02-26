"""
只对新增视频进行切片和抽帧，从文件列表读取视频路径，避免扫描全目录。
用法: python 7.5_segment_get_frames_newonly.py --video_list /root/workspace/new_junction_videos.txt
"""
import os
import subprocess
from pathlib import Path
import time
import argparse


def is_video_processed(output_dir, filename):
    """检查视频是否已经处理过（三个切片都存在）"""
    for i in range(1, 4):
        segment_path = output_dir / f"{filename}_segment_{i:03d}.mp4"
        if not segment_path.exists() or segment_path.stat().st_size < 1024:
            return False
    return True


def process_videos(video_list_path, input_base, output_base):
    input_base = Path(input_base)
    output_base = Path(output_base)

    # 从文件列表读取视频路径（不再扫描目录）
    video_files = []
    with open(video_list_path, 'r') as f:
        for line in f:
            p = line.strip()
            if p and os.path.exists(p):
                video_files.append(Path(p))

    print(f"从列表加载 {len(video_files)} 个视频文件")

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
        if is_video_processed(output_dir, filename):
            skipped_videos += 1
            if skipped_videos % 1000 == 0:
                print(f"[{idx}/{len(video_files)}] 已跳过 {skipped_videos} 个已处理视频")
            continue

        print(f"[{idx}/{len(video_files)}] 处理: {relative_path}")
        processed_videos += 1

        # 切割视频为三个20秒片段
        for i, start_sec in enumerate([0, 20, 40], 1):
            segment_name = f"{filename}_segment_{i:03d}.mp4"
            segment_path = output_dir / segment_name

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
                subprocess.run(cut_cmd, check=True, capture_output=True)

                if segment_path.exists() and segment_path.stat().st_size > 0:
                    temp_path = output_dir / f"{filename}_segment_{i:03d}_temp.mp4"

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
                        subprocess.run(extract_cmd, check=True, capture_output=True)
                        if temp_path.exists() and temp_path.stat().st_size > 0:
                            segment_path.unlink()
                            temp_path.rename(segment_path)
                            total_segments += 1
                    except subprocess.CalledProcessError:
                        print(f"  警告: 片段{i}抽帧失败 - 保留原始切片")
                        total_segments += 1

            except subprocess.CalledProcessError as e:
                stderr = str(e.stderr) if e.stderr else ""
                if "Invalid data found" in stderr or "moov atom not found" in stderr:
                    print(f"  警告: 视频文件可能损坏 - {video_path.name}")
                elif "Invalid argument" in stderr or "position out of range" in stderr:
                    print(f"  警告: 片段{i}超出视频长度，跳过此片段")
                else:
                    print(f"  警告: 片段{i}切割失败 - {stderr[:100]}")
            except Exception as e:
                print(f"  警告: 处理片段{i}时发生未知错误 - {str(e)[:100]}")

        # 每处理100个打印进度
        if processed_videos % 100 == 0:
            elapsed = time.time() - start_time
            speed = processed_videos / elapsed if elapsed > 0 else 0
            remaining = (len(video_files) - idx) / speed if speed > 0 else 0
            print(f"  进度: 已处理 {processed_videos}, 跳过 {skipped_videos}, "
                  f"速度 {speed:.1f} 个/秒, 预计剩余 {remaining/60:.0f} 分钟")

    elapsed_time = time.time() - start_time
    print(f"\n处理完成！")
    print(f"视频文件总数: {len(video_files)} 个")
    print(f"已跳过视频: {skipped_videos} 个")
    print(f"新处理视频: {processed_videos} 个")
    print(f"生成视频片段: {total_segments} 个")
    print(f"总耗时: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从文件列表读取新增视频，进行切片和抽帧')
    parser.add_argument('--video_list', type=str, default='/root/workspace/new_junction_videos.txt',
                        help='新增视频路径列表文件')
    parser.add_argument('--input_base', type=str, default='/mnt/pfs/houhaotian/junction_videos',
                        help='输入视频的基础路径')
    parser.add_argument('--output_base', type=str, default='/mnt/pfs/houhaotian/junction_videos_segment',
                        help='输出切片的基础路径')
    args = parser.parse_args()

    process_videos(args.video_list, args.input_base, args.output_base)
