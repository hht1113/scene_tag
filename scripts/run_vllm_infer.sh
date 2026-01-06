#!/bin/bash

# 创建保存结果的目录（如果不存在）
mkdir -p /root/workspace/LLaMA-Factory/infer_results

cd /root/workspace/LLaMA-Factory

python scripts/vllm_infer.py \
    --model_name_or_path "Qwen/Qwen3-VL-4B-Instruct" \
    --adapter_name_or_path "/root/workspace/LLaMA-Factory/saves/Qwen3-VL-4B-Instruct/lora/train_2026-01-05-19-54-24" \
    --dataset "qwen3_sft_test_dataset" \
    --template "qwen3_vl_nothink" \
    --save_name "/root/workspace/LLaMA-Factory/infer_results/Qwen3-VL-4B-new_label_new_prompt.jsonl" \
    --cutoff_len 8000 \
    --max_new_tokens 512 \
    --batch_size 1 \
    --video_fps 1.0 \
    --video_maxlen 60 \
    --image_max_pixels 200704 \
    # --image_max_pixels 200704  # 448 * 448=200704，直接计算结果
    # --adapter_name_or_path "/root/workspace/LLaMA-Factory/saves/Qwen3-VL-4B-Instruct/lora/train_2026-01-05-19-54-24" \