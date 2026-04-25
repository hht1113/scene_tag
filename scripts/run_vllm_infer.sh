#!/bin/bash
###############################################################################
# 多标签 v3 测试集推理脚本
#
# 模型:   30B multilabel_v3 — checkpoint-282 (3 epoch 完整训练)
# 数据集: qwen3_sft_test_dataset_segment_multilabel_v3 (测试集)
# 单卡推理
###############################################################################

mkdir -p /root/workspace/LLaMA-Factory/infer_results

cd /root/workspace/LLaMA-Factory

export DISABLE_VERSION_CHECK=1
export CUDA_VISIBLE_DEVICES=${GPU_ID:-3}

CHECKPOINT="/mnt/pfs/houhaotian/saves/Qwen3-VL-30B-A3B-Instruct/full/sft_segment_multilabel_v3_8gpu/checkpoint-282"
DATASET="qwen3_sft_test_dataset_segment_multilabel_v3"
SAVE_NAME="/root/workspace/LLaMA-Factory/infer_results/test_multilabel_v3_30B_checkpoint282.jsonl"

echo "============================================================"
echo "  多标签 v3 测试集推理"
echo "  模型:   $CHECKPOINT"
echo "  数据集: $DATASET"
echo "  GPU:    $CUDA_VISIBLE_DEVICES"
echo "  开始:   $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

python scripts/vllm_infer.py \
    --model_name_or_path "$CHECKPOINT" \
    --dataset "$DATASET" \
    --template "qwen3_vl_nothink" \
    --save_name "$SAVE_NAME" \
    --cutoff_len 20000 \
    --max_new_tokens 512 \
    --batch_size 1 \
    --video_fps 2.0 \
    --video_maxlen 40 \
    --image_max_pixels 65536 \
    --temperature 0.01 \
    --top_p 0.95 \
    --enable_thinking false

echo "============================================================"
echo "  完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果: $SAVE_NAME"
echo "============================================================"
