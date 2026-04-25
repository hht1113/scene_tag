#!/bin/bash
###############################################################################
# 批量推理：在新平衡测试集上评估所有无泄露模型
#
# 新测试集: qwen3_sft_test_dataset_segment_balanced_v2 (360条, 每类30条)
# 所有模型均基于 train_segment_upsample 训练，与新测试集零泄露
###############################################################################

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
    echo "[ERROR] 找不到 conda"
    exit 1
fi

conda activate "$CONDA_ENV_NAME"
echo "[INFO] conda 环境: $CONDA_ENV_NAME (python: $(which python))"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

export DISABLE_VERSION_CHECK=1

DATASET="qwen3_sft_test_dataset_segment_balanced_v2"
TEMPLATE="qwen3_vl_nothink"
OUTDIR="${PROJECT_DIR}/infer_results/balanced_v2"
mkdir -p "$OUTDIR"

BASE_30B="/mnt/pfs/qwen3/Qwen3-VL-30B-A3B-Instruct"
BASE_8B="/mnt/pfs/qwen3/Qwen3-VL-8B-Instruct"
BASE_4B="/mnt/pfs/qwen3/Qwen3-VL-4B-Instruct"
SAVES="/mnt/pfs/houhaotian/saves"

run_infer() {
    local NAME="$1"
    local MODEL="$2"
    local ADAPTER="$3"
    local EXTRA_ARGS="$4"
    local SAVE_FILE="${OUTDIR}/${NAME}.jsonl"

    echo ""
    echo "============================================================"
    echo "  推理: ${NAME}"
    echo "  模型: ${MODEL}"
    [ -n "$ADAPTER" ] && echo "  适配器: ${ADAPTER}"
    echo "  输出: ${SAVE_FILE}"
    echo "  开始: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    if [ -f "$SAVE_FILE" ]; then
        echo "  [SKIP] 输出已存在，跳过"
        return 0
    fi

    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID:-0} python scripts/vllm_infer.py \
        --model_name_or_path \"${MODEL}\" \
        --dataset \"${DATASET}\" \
        --template \"${TEMPLATE}\" \
        --save_name \"${SAVE_FILE}\" \
        --cutoff_len 20000 \
        --max_new_tokens 512 \
        --batch_size 1 \
        --video_fps 2.0 \
        --video_maxlen 40 \
        --enable_thinking false"

    if [ -n "$ADAPTER" ]; then
        CMD="${CMD} --adapter_name_or_path \"${ADAPTER}\""
    fi

    if [ -n "$EXTRA_ARGS" ]; then
        CMD="${CMD} ${EXTRA_ARGS}"
    fi

    eval $CMD
    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  [OK] 完成 @ $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "  [FAIL] exit code: $EXIT_CODE @ $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    return $EXIT_CODE
}

echo ""
echo "============================================================"
echo "  批量推理 - 新平衡测试集 (360条, 每类30条)"
echo "  共 6 个模型 (排除动态采样模型)"
echo "============================================================"
echo ""

# ── 1. 30B 12w pixels (3ep) → checkpoint-339 ──
run_infer \
    "30B_12w_pixels_3ep" \
    "${SAVES}/Qwen3-VL-30B-A3B-Instruct/full-12w-pixels/sft_segment_upsample_8gpu_5epoch/checkpoint-339" \
    "" \
    "--image_max_pixels 120000"

# ── 2. 30B 旧版 6.5w (2ep) → checkpoint-226 ──
run_infer \
    "30B_65k_pixels_2ep" \
    "${SAVES}/Qwen3-VL-30B-A3B-Instruct/full/sft_segment_upsample_8gpu_0317/checkpoint-226" \
    "" \
    "--image_max_pixels 65536"

# ── 3. 30B 20w pixels (~3.5ep) → checkpoint-400 ──
run_infer \
    "30B_20w_pixels_3.5ep" \
    "${SAVES}/Qwen3-VL-30B-A3B-Instruct/full-20w-pixels/sft_segment_upsample_8gpu_5epoch/checkpoint-400" \
    "" \
    "--image_max_pixels 200000"

# ── 4. 30B 20w pixels (~4.4ep) → checkpoint-500 ──
run_infer \
    "30B_20w_pixels_4.4ep" \
    "${SAVES}/Qwen3-VL-30B-A3B-Instruct/full-20w-pixels/sft_segment_upsample_8gpu_5epoch/checkpoint-500" \
    "" \
    "--image_max_pixels 200000"

# ── 5. 8B 全参 → checkpoint-58 ──
run_infer \
    "8B_full_2ep" \
    "${SAVES}/Qwen3-VL-8B-Instruct/full/sft_segment_upsample_8gpu/checkpoint-58" \
    "" \
    "--image_max_pixels 65536"

# ── 6. 4B LoRA → checkpoint-600 ──
run_infer \
    "4B_lora" \
    "${BASE_4B}" \
    "${SAVES}/Qwen3-VL-4B-Instruct/lora/checkpoint-600" \
    "--image_max_pixels 65536"

echo ""
echo "============================================================"
echo "  全部完成 @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果保存在: ${OUTDIR}/"
echo "============================================================"
ls -la "${OUTDIR}/"
