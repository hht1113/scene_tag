#!/bin/bash
# Round-B 4组串行推理 — Qwen3-VL-8B + thinking，单GPU
#
# 图片搭配：
#   01_DynamicInteraction: cutin.png（1张）
#   04_Intersection:       cutin.png + intersection_interaction_vehicle.png + intersection_interaction_vru.png（3张）
#   06_LaneChange:         cutin.png + intersection_interaction_vehicle.png + intersection_interaction_vru.png（3张）
#   07_IntersectionInteraction: intersection_interaction_vehicle.png + intersection_interaction_vru.png（2张）

cd "$(dirname "$0")"

GPU=3
INPUT=/tmp/round_a_second_half.jsonl
MODEL=/mnt/pfs/qwen3/Qwen3-VL-8B-Instruct
OUTDIR=/mnt/pfs/houhaotian/pnc
COMMON="--enable-thinking --tensor-parallel-size 1 --max-model-len 16384 --max-tokens 4096 --temperature 0.6 --batch-size 8 --gpu-memory-utilization 0.92"

echo "[$(date '+%H:%M:%S')] === 01_DynamicInteraction (1张图: cutin) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 round_b_label_infer_v4_multimodal.py \
    --model $MODEL --input $INPUT \
    --output $OUTDIR/round_b_v4_01_DynamicInteraction.json \
    --prompt DynamicInteraction_v2.txt \
    --images cutin.png \
    $COMMON
echo "[$(date '+%H:%M:%S')] 01 完成"

echo "[$(date '+%H:%M:%S')] === 04_Intersection (3张图: cutin + vehicle + vru) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 round_b_label_infer_v4_multimodal.py \
    --model $MODEL --input $INPUT \
    --output $OUTDIR/round_b_v4_04_Intersection.json \
    --prompt Intersection_v2.txt \
    --images cutin.png intersection_interaction_vehicle.png intersection_interaction_vru.png \
    $COMMON
echo "[$(date '+%H:%M:%S')] 04 完成"

echo "[$(date '+%H:%M:%S')] === 06_LaneChange (3张图: cutin + vehicle + vru) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 round_b_label_infer_v4_multimodal.py \
    --model $MODEL --input $INPUT \
    --output $OUTDIR/round_b_v4_06_LaneChange.json \
    --prompt LaneChange_v2.txt \
    --images cutin.png intersection_interaction_vehicle.png intersection_interaction_vru.png \
    $COMMON
echo "[$(date '+%H:%M:%S')] 06 完成"

echo "[$(date '+%H:%M:%S')] === 07_IntersectionInteraction (2张图: vehicle + vru) ==="
CUDA_VISIBLE_DEVICES=$GPU python3 round_b_label_infer_v4_multimodal.py \
    --model $MODEL --input $INPUT \
    --output $OUTDIR/round_b_v4_07_IntersectionInteraction.json \
    --prompt IntersectionInteraction_v2.txt \
    --images intersection_interaction_vehicle.png intersection_interaction_vru.png \
    $COMMON
echo "[$(date '+%H:%M:%S')] 07 完成"

echo "[$(date '+%H:%M:%S')] === 全部4组完成 ==="
ls -lh $OUTDIR/round_b_v4_*.json
