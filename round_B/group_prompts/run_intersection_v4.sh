#!/bin/bash
# Round-B 组7路口交互 — 多模态+thinking 模式推理
# 使用 Qwen3.5-9B + 示意图

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=1 python3 round_b_label_infer_v4_multimodal.py \
    --model /mnt/pfs/qwen3.5/Qwen3.5-9B \
    --input /mnt/pfs/chenruize/dataset/round_a_results_v2_final.jsonl \
    --output ../../round_B/round_b_results_intersection_v4_thinking.json \
    --prompt IntersectionInteraction_v2.txt \
    --images intersection_interaction_vehicle.png \
             intersection_interaction_vru.png \
    --enable-thinking \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --max-tokens 4096 \
    --temperature 0.6 \
    --batch-size 8 \
    --gpu-memory-utilization 0.92 \
    --max-records 2000
