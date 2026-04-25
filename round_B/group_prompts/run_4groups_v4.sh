#!/bin/bash
# Round-B 4组推理 — Qwen3-VL-8B + 示意图 + thinking
# 01_DynamicInteraction, 04_Intersection, 06_LaneChange, 07_IntersectionInteraction
# GPU1 跑 01+04（串行），GPU3 跑 06+07（串行），两组并行

cd "$(dirname "$0")"

INPUT=/tmp/round_a_second_half.jsonl
MODEL=/mnt/pfs/qwen3/Qwen3-VL-8B-Instruct
OUTDIR=/mnt/pfs/houhaotian/pnc
IMAGES="intersection_interaction_vehicle.png intersection_interaction_vru.png"
COMMON="--enable-thinking --tensor-parallel-size 1 --max-model-len 16384 --max-tokens 4096 --temperature 0.6 --batch-size 8 --gpu-memory-utilization 0.92"

# GPU 1: 01_DynamicInteraction -> 04_Intersection
(
  CUDA_VISIBLE_DEVICES=1 python3 round_b_label_infer_v4_multimodal.py \
    --model $MODEL --input $INPUT \
    --output $OUTDIR/round_b_v4_01_DynamicInteraction.json \
    --prompt DynamicInteraction_v2.txt \
    --images $IMAGES $COMMON \
  && echo "[GPU1] 01_DynamicInteraction 完成" \
  && CUDA_VISIBLE_DEVICES=1 python3 round_b_label_infer_v4_multimodal.py \
    --model $MODEL --input $INPUT \
    --output $OUTDIR/round_b_v4_04_Intersection.json \
    --prompt Intersection_v2.txt \
    --images $IMAGES $COMMON \
  && echo "[GPU1] 04_Intersection 完成"
) > /tmp/round_b_v4_gpu1.log 2>&1 &
echo "GPU1 PID: $!"

# GPU 3: 06_LaneChange -> 07_IntersectionInteraction
(
  CUDA_VISIBLE_DEVICES=3 python3 round_b_label_infer_v4_multimodal.py \
    --model $MODEL --input $INPUT \
    --output $OUTDIR/round_b_v4_06_LaneChange.json \
    --prompt LaneChange_v2.txt \
    --images $IMAGES $COMMON \
  && echo "[GPU3] 06_LaneChange 完成" \
  && CUDA_VISIBLE_DEVICES=3 python3 round_b_label_infer_v4_multimodal.py \
    --model $MODEL --input $INPUT \
    --output $OUTDIR/round_b_v4_07_IntersectionInteraction.json \
    --prompt IntersectionInteraction_v2.txt \
    --images $IMAGES $COMMON \
  && echo "[GPU3] 07_IntersectionInteraction 完成"
) > /tmp/round_b_v4_gpu3.log 2>&1 &
echo "GPU3 PID: $!"

echo "4组推理已启动（GPU1: 01+04, GPU3: 06+07）"
echo "查看进度: tail -f /tmp/round_b_v4_gpu1.log /tmp/round_b_v4_gpu3.log"
