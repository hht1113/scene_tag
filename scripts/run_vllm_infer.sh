#!/bin/bash

# 创建保存结果的目录（如果不存在）
mkdir -p /root/workspace/LLaMA-Factory/infer_results

cd /root/workspace/LLaMA-Factory

python scripts/vllm_infer.py \
    --model_name_or_path "/mnt/pfs/qwen3/Qwen3-VL-32B-Instruct" \
    --dataset "qwen3_sft_test_dataset_segment_upsample" \
    --template "qwen3_vl_nothink" \
    --save_name "/root/workspace/LLaMA-Factory/infer_results/12tags_Qwen3-VL-32B_SFT_segment_upsample.jsonl" \
    --cutoff_len 15000 \
    --max_new_tokens 512 \
    --batch_size 1 \
    --video_fps 2.0 \
    --video_maxlen 40 \
    --image_max_pixels 65536

# 说明：
# 全参微调的模型权重直接保存在 output_dir，不需要 adapter_name_or_path
# video_fps 是抽帧的目标帧率，不输入默认是 2
# video_maxlen 是最大帧数，不输入默认是 128
# cutoff_len 60s视频，对于1fps，应该要在25000；2fps要在45000
# image_max_pixels 是单张图像的最大像素数，默认65536（256*256），和训练保持一致
# --adapter_name_or_path "/root/workspace/LLaMA-Factory/saves/Qwen3-VL-8B-Instruct/full/sft_segment_upsample_8gpu/checkpoint-58" \