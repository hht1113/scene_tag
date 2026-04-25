#!/bin/bash
###############################################################################
# Qwen3-VL-4B-Instruct LoRA 推理脚本
#
# 基座模型:   Qwen3-VL-4B-Instruct
# LoRA 权重:  checkpoint-600
# 测试集:     qwen3_sft_test_dataset_segment_upsample
# 框架:       LLaMA-Factory + vLLM
#
# 用法:
#   bash run_vllm_infer_4B_lora.sh
###############################################################################

# ========================= Conda 环境初始化 =========================
CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen3}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -n "$CONDA_EXE" ]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
else
    echo "[ERROR] 找不到 conda，请检查 conda 安装路径"
    exit 1
fi

conda activate "$CONDA_ENV_NAME"
echo "[INFO] 已激活 conda 环境: $CONDA_ENV_NAME (python: $(which python))"

# ========================= 配置 =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

BASE_MODEL="/mnt/pfs/qwen3/Qwen3-VL-4B-Instruct"
ADAPTER_PATH="/mnt/pfs/houhaotian/saves/Qwen3-VL-4B-Instruct/lora/checkpoint-600"
DATASET="qwen3_sft_test_dataset_segment_upsample"
TEMPLATE="qwen3_vl_nothink"
SAVE_NAME="${PROJECT_DIR}/infer_results/12tags_Qwen3-VL-4B_lora_sft_segment_upsample.jsonl"

mkdir -p "$(dirname "$SAVE_NAME")"

# ========================= 环境检查 =========================
echo "============================================================"
echo "  Qwen3-VL-4B LoRA 推理 (vLLM)"
echo "============================================================"
echo ""
echo "[INFO] 基座模型:     $BASE_MODEL"
echo "[INFO] LoRA 权重:    $ADAPTER_PATH"
echo "[INFO] 测试集:       $DATASET"
echo "[INFO] 输出文件:     $SAVE_NAME"
echo "[INFO] 当前时间:     $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -d "$BASE_MODEL" ]; then
    echo "[ERROR] 基座模型不存在: $BASE_MODEL"
    exit 1
fi

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "[ERROR] LoRA 权重不存在: $ADAPTER_PATH"
    exit 1
fi

export DISABLE_VERSION_CHECK=1

# ========================= 启动推理 =========================
echo "============================================================"
echo "  开始推理 @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

python scripts/vllm_infer.py \
    --model_name_or_path "$BASE_MODEL" \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --dataset "$DATASET" \
    --template "$TEMPLATE" \
    --save_name "$SAVE_NAME" \
    --cutoff_len 15000 \
    --max_new_tokens 512 \
    --batch_size 1 \
    --video_fps 2.0 \
    --video_maxlen 40 \
    --image_max_pixels 65536

INFER_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $INFER_EXIT_CODE -eq 0 ]; then
    echo "  推理完成 @ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  结果保存: $SAVE_NAME"
else
    echo "  推理失败 (exit code: $INFER_EXIT_CODE) @ $(date '+%Y-%m-%d %H:%M:%S')"
fi
echo "============================================================"

exit $INFER_EXIT_CODE
