#!/bin/bash
# 在 batch2 的同池 1k 公平子集上启动 dual-agent A/B 实验

set -e

export no_proxy="${no_proxy},10.10.64.144"
export NO_PROXY="${NO_PROXY},10.10.64.144"

API_BASE="${1:-http://10.10.64.144:2754/v1}"
MODEL_NAME="${2:-Qwen3.5-397B-A17B}"
SUBSET_DIR="${3:-scene_tag/ab_eval/batch2_samepool_1k}"
VIDEO_LIST="${SUBSET_DIR}/subset_video_list.txt"
RESULT_DIR="${SUBSET_DIR}/dual_agent"
LOG_DIR="${SUBSET_DIR}/logs"
SCRIPT="scene_tag/dual_agent_distillation.py"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "============================================"
echo "  Dual-Agent A/B Eval"
echo "  API: $API_BASE"
echo "  模型: $MODEL_NAME"
echo "  子集: $VIDEO_LIST ($(wc -l < "$VIDEO_LIST") videos)"
echo "============================================"

PROMPTS=(
    "04_Intersection"
    "05_LaneCruising"
    "02_TrafficLight"
)

for PROMPT_NAME in "${PROMPTS[@]}"; do
    PROMPT_FILE="scene_tag/prompt_txt/${PROMPT_NAME}.txt"
    OUTPUT_FILE="${RESULT_DIR}/mining_${PROMPT_NAME}_dual_agent.json"
    LOG_FILE="${LOG_DIR}/mining_${PROMPT_NAME}_dual_agent.log"

    echo "启动 dual-agent: ${PROMPT_NAME}"
    nohup python3 "$SCRIPT" \
        --api_base "$API_BASE" \
        --model_name "$MODEL_NAME" \
        --prompt_file "$PROMPT_FILE" \
        --video_list "$VIDEO_LIST" \
        --output "$OUTPUT_FILE" \
        --concurrency 4 \
        --annotator_temperature 0.0 \
        --judge_temperature 0.0 \
        > "$LOG_FILE" 2>&1 &
    echo "  PID: $!"
    sleep 2
done

echo ""
echo "已启动 3 个核心组 dual-agent 实验。"
