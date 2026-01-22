#!/bin/bash

# 创建保存结果的目录（如果不存在）
mkdir -p /root/workspace/LLaMA-Factory/infer_results

cd /root/workspace/LLaMA-Factory

python scripts/vllm_infer.py \
    --model_name_or_path "Qwen/Qwen3-VL-4B-Instruct" \
    --adapter_name_or_path "/root/workspace/LLaMA-Factory/saves/Qwen3-VL-4B-Instruct/lora/train_2026-01-16-23-21-37/checkpoint-125" \
    --dataset "qwen3_sft_test_dataset_2fps" \
    --template "qwen3_vl_nothink" \
    --save_name "/root/workspace/LLaMA-Factory/infer_results/12tags_Qwen3-VL-4B-2epoch.jsonl" \
    --cutoff_len 45000 \
    --max_new_tokens 512 \
    --batch_size 1 \
    --video_fps 2.0 \
    --video_maxlen 120 \
    --image_max_pixels 65536 \
    # --image_max_pixels 65536  # 256 * 256=65536
    # --adapter_name_or_path "/root/workspace/LLaMA-Factory/saves/Qwen3-VL-4B-Instruct/lora/train_2026-01-07-13-45-17" \

# video_fps是抽帧的目标帧率 不输入默认是2
# video_maxlens是最大帧数 不输入默认是128
# cutoff_len 对于1fps，应该要在25000；2fps要在45000
# image_max_pixels是单张图像的最大像素数，默认65536（256*256），和训练保持一致
