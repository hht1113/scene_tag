#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 模型部署脚本 - 支持多模型预设

在本机启动 vLLM OpenAI 兼容 API 服务，服务会一直运行直到 Ctrl+C 停止。
部署完成后可通过 12_distillation.py 调用服务进行视频标注。

支持的模型预设:
    - qwen3-vl-235b:   Qwen3-VL-235B-A22B-Instruct-FP8 (视觉语言模型)
    - qwen3.5-122b:    Qwen3.5-122B-A10B-FP8 (原生多模态 MoE 模型)

用法:
    # 默认部署 Qwen3-VL-235B
    python 11_deploy.py

    # 部署 Qwen3.5-122B
    python 11_deploy.py --preset qwen3.5-122b

    # 指定 GPU 数量和端口
    python 11_deploy.py --preset qwen3-vl-235b --tp 4 --port 8080

    # 自定义模型路径 (覆盖预设)
    python 11_deploy.py --model_path /path/to/model --tp 8 --port 8000

    # 查看所有可用预设
    python 11_deploy.py --list_presets
"""

import os
import sys
import json
import argparse
import torch


# ==================== 模型预设配置 ====================
MODEL_PRESETS = {
    "qwen3-vl-235b": {
        "model_path": "/mnt/pfs/houhaotian/models/Qwen3-VL-235B-A22B-Instruct-FP8",
        "model_name": "qwen3-vl-235b",
        "is_multimodal": True,
        "max_model_len": 32768,
        "max_num_seqs": 4,
        "gpu_mem_util": 0.95,
        "max_images": 40,
        "description": "Qwen3-VL-235B 视觉语言模型 (FP8, MoE A22B)",
    },
    "qwen3.5-122b": {
        "model_path": "/mnt/pfs/qwen3.5/Qwen3.5-122B-A10B-FP8",
        "model_name": "qwen3.5-122b",
        "is_multimodal": True,
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "gpu_mem_util": 0.95,
        "max_images": 40,
        "extra_args": [
            "--mm-encoder-tp-mode", "data",
            "--mm-processor-cache-type", "shm",
        ],
        "description": "Qwen3.5-122B 原生多模态 MoE 模型 (FP8, A10B)",
    },
}

DEFAULT_PRESET = "qwen3-vl-235b"
DEFAULT_PORT = 8000


# ==================== GPU 检测 ====================
def get_tensor_parallel_size(tp_arg: str) -> int:
    """解析 tensor parallel 参数，支持 'auto' 自动检测"""
    if tp_arg.lower() == "auto":
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError(
                "未检测到可用 GPU!\n"
                "大模型推理需要多块 GPU，请确认 GPU 可用。"
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


def list_presets():
    """打印所有可用的模型预设"""
    print("=" * 60)
    print("  可用模型预设")
    print("=" * 60)
    for name, cfg in MODEL_PRESETS.items():
        marker = " (默认)" if name == DEFAULT_PRESET else ""
        print(f"\n  [{name}]{marker}")
        print(f"    描述:       {cfg['description']}")
        print(f"    路径:       {cfg['model_path']}")
        print(f"    多模态:     {'是' if cfg['is_multimodal'] else '否'}")
        print(f"    最大序列长: {cfg['max_model_len']}")
        print(f"    最大并发:   {cfg['max_num_seqs']}")
    print(f"\n{'=' * 60}")


# ==================== 命令行参数 ====================
def parse_args():
    preset_names = list(MODEL_PRESETS.keys())
    parser = argparse.ArgumentParser(
        description="部署 vLLM API 服务（前台运行，Ctrl+C 停止）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用预设: {', '.join(preset_names)}

使用示例:
  # 默认预设 ({DEFAULT_PRESET})
  python 11_deploy.py

  # 部署 Qwen3.5-122B
  python 11_deploy.py --preset qwen3.5-122b

  # FP8 模型，4 卡部署
  python 11_deploy.py --tp 4

  # 自定义端口
  python 11_deploy.py --port 8080

  # 自定义模型路径 (覆盖预设)
  python 11_deploy.py --model_path /path/to/model --tp 8

  # 查看所有预设
  python 11_deploy.py --list_presets

启动成功后，在另一个终端运行:
  python 12_distillation.py \\
      --api_base http://localhost:8000/v1 \\
      --video_list /mnt/pfs/houhaotian/sampled_1000_videos.txt
        """,
    )

    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        choices=preset_names,
        help=f"模型预设名称（默认: {DEFAULT_PRESET}，可选: {', '.join(preset_names)}）",
    )
    parser.add_argument(
        "--list_presets",
        action="store_true",
        help="列出所有可用的模型预设并退出",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="自定义模型路径（覆盖预设中的路径）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="API 中的模型名称（覆盖预设中的名称）",
    )
    parser.add_argument(
        "--tp",
        type=str,
        default="auto",
        help="张量并行 GPU 数量（默认 auto 自动检测）",
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
        default=None,
        help="最大序列长度（覆盖预设值）",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="GPU 显存利用率（覆盖预设值）",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=None,
        help="最大并发序列数（覆盖预设值）",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="每个请求最大图片数（仅多模态模型有效，覆盖预设值）",
    )

    return parser.parse_args()


# ==================== 主入口 ====================
def main():
    args = parse_args()

    if args.list_presets:
        list_presets()
        return

    preset = MODEL_PRESETS[args.preset]

    model_path = args.model_path or preset["model_path"]
    model_name = args.model_name or preset["model_name"]
    max_model_len = args.max_model_len or preset["max_model_len"]
    gpu_mem_util = args.gpu_memory_utilization or preset["gpu_mem_util"]
    max_num_seqs = args.max_num_seqs or preset["max_num_seqs"]
    max_images = args.max_images if args.max_images is not None else preset["max_images"]
    is_multimodal = preset["is_multimodal"]

    tp_size = get_tensor_parallel_size(args.tp)

    if not os.path.exists(model_path):
        print(f"警告: 模型路径 '{model_path}' 不存在，vLLM 可能会尝试从 HuggingFace 下载")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", model_name,
        "--tensor-parallel-size", str(tp_size),
        "--port", str(args.port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--trust-remote-code",
        "--max-num-seqs", str(max_num_seqs),
    ]

    if is_multimodal:
        cmd.extend([
            "--limit-mm-per-prompt", json.dumps({"image": max_images}),
            "--allowed-local-media-path", "/",
        ])

    extra_args = preset.get("extra_args", [])
    if extra_args:
        cmd.extend(extra_args)

    print("=" * 60)
    print(f"  vLLM API 服务部署 [{args.preset}]")
    print("=" * 60)
    print(f"  预设:           {args.preset} - {preset['description']}")
    print(f"  模型路径:       {model_path}")
    print(f"  API 模型名称:   {model_name}")
    print(f"  多模态:         {'是' if is_multimodal else '否 (纯文本)'}")
    print(f"  张量并行:       {tp_size} GPUs")
    print(f"  服务端口:       {args.port}")
    print(f"  最大序列长度:   {max_model_len}")
    print(f"  最大并发请求:   {max_num_seqs}")
    print(f"  GPU 显存利用率: {gpu_mem_util}")
    if is_multimodal:
        print(f"  每请求最大图片: {max_images}")
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

    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
