#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-235B vLLM 模型部署脚本

在本机启动 vLLM OpenAI 兼容 API 服务，服务会一直运行直到 Ctrl+C 停止。
部署完成后可通过 12_distillation.py 调用服务进行视频标注。

用法:
    # 默认配置（自动检测 GPU 数量）
    python 11_deploy.py

    # 指定 GPU 数量和端口
    python 11_deploy.py --tp 4 --port 8080

    # 自定义模型路径
    python 11_deploy.py \
        --model_path /path/to/other/model \
        --tp 8 --port 8000

    # 启动后，在另一个终端运行标注脚本:
    python 12_distillation.py \
        --api_base http://localhost:8000/v1 \
        --video_list /mnt/pfs/houhaotian/sampled_1000_videos.txt \
        --output results/annotations.json
"""

import os
import sys
import json
import argparse
import torch


# ==================== 默认配置 ====================
DEFAULT_MODEL_PATH = "/mnt/pfs/houhaotian/models/Qwen3-VL-235B-A22B-Instruct-FP8"
DEFAULT_MODEL_NAME = "qwen3-vl-235b"
DEFAULT_PORT = 8000
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_GPU_MEM_UTIL = 0.95
DEFAULT_MAX_NUM_SEQS = 4
DEFAULT_MAX_IMAGES_PER_PROMPT = 40


# ==================== GPU 检测 ====================
def get_tensor_parallel_size(tp_arg: str) -> int:
    """解析 tensor parallel 参数，支持 'auto' 自动检测"""
    if tp_arg.lower() == "auto":
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError(
                "未检测到可用 GPU!\n"
                "Qwen3-VL-235B 需要至少 4 块 GPU (FP8) 或 8 块 GPU (FP16)。"
            )
        print(f"自动检测到 {gpu_count} 块 GPU，将全部用于张量并行推理")
        return gpu_count
    else:
        tp = int(tp_arg)
        available = torch.cuda.device_count()
        if tp > available:
            raise RuntimeError(
                f"请求 {tp} 块 GPU 但仅检测到 {available} 块可用 GPU"
            )
        return tp


# ==================== 命令行参数 ====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="部署 Qwen3-VL-235B vLLM API 服务（前台运行，Ctrl+C 停止）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 默认配置启动（自动检测 GPU）
  python 11_deploy.py

  # FP8 模型，4 卡部署
  python 11_deploy.py --tp 4

  # 自定义端口
  python 11_deploy.py --port 8080

  # 完全自定义
  python 11_deploy.py \\
      --model_path /path/to/model \\
      --tp 8 \\
      --port 8000 \\
      --max_model_len 16384 \\
      --gpu_memory_utilization 0.90

启动成功后，在另一个终端运行:
  python 12_distillation.py \\
      --api_base http://localhost:8000/v1 \\
      --video_list /mnt/pfs/houhaotian/sampled_1000_videos.txt
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"模型路径（默认: {DEFAULT_MODEL_PATH}）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"API 中的模型名称（默认: {DEFAULT_MODEL_NAME}）",
    )
    parser.add_argument(
        "--tp",
        type=str,
        default="auto",
        help="张量并行 GPU 数量（默认 auto 自动检测，FP8 可用 4 卡，FP16 需 8 卡）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"API 服务端口（默认: {DEFAULT_PORT}）",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"最大序列长度（默认: {DEFAULT_MAX_MODEL_LEN}）",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=DEFAULT_GPU_MEM_UTIL,
        help=f"GPU 显存利用率（默认: {DEFAULT_GPU_MEM_UTIL}）",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=DEFAULT_MAX_NUM_SEQS,
        help=f"最大并发序列数（默认: {DEFAULT_MAX_NUM_SEQS}）",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_PROMPT,
        help=f"每个请求最大图片数（默认: {DEFAULT_MAX_IMAGES_PER_PROMPT}）",
    )

    return parser.parse_args()


# ==================== 主入口 ====================
def main():
    args = parse_args()

    # 1. 检测 GPU
    tp_size = get_tensor_parallel_size(args.tp)

    # 2. 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"警告: 模型路径 '{args.model_path}' 不存在，vLLM 可能会尝试从 HuggingFace 下载")

    # 3. 构造 vLLM 启动命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--served-model-name", args.model_name,
        "--tensor-parallel-size", str(tp_size),
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--max-num-seqs", str(args.max_num_seqs),
        "--limit-mm-per-prompt", json.dumps({"image": args.max_images}),
        "--allowed-local-media-path", "/",
    ]

    # 4. 打印配置信息
    print("=" * 60)
    print("  Qwen3-VL-235B vLLM API 服务部署")
    print("=" * 60)
    print(f"  模型路径:       {args.model_path}")
    print(f"  API 模型名称:   {args.model_name}")
    print(f"  张量并行:       {tp_size} GPUs")
    print(f"  服务端口:       {args.port}")
    print(f"  最大序列长度:   {args.max_model_len}")
    print(f"  最大并发请求:   {args.max_num_seqs}")
    print(f"  GPU 显存利用率: {args.gpu_memory_utilization}")
    print(f"  每请求最大图片: {args.max_images}")
    print(f"{'=' * 60}")
    print(f"  API 地址:       http://localhost:{args.port}/v1")
    print(f"  健康检查:       http://localhost:{args.port}/health")
    print(f"{'=' * 60}")
    print(f"\n服务启动后，在另一个终端运行标注脚本:")
    print(f"  python 12_distillation.py \\")
    print(f"      --api_base http://localhost:{args.port}/v1 \\")
    print(f"      --video_list /mnt/pfs/houhaotian/sampled_1000_videos.txt \\")
    print(f"      --output results/annotations.json")
    print(f"\n按 Ctrl+C 停止服务\n")
    print(f"启动命令:\n  {' '.join(cmd)}\n")
    print("=" * 60)
    print()

    # 5. 用 os.execvp 替换当前进程为 vLLM 服务（前台运行）
    #    这样 Ctrl+C 直接由 vLLM 处理，信号传递自然
    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
