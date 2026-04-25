#!/bin/bash
# Round-B label inference v3 — iterate all v2 group prompts sequentially

set -e

MODEL="/mnt/pfs/qwen3/Qwen3-8B"
INPUT="/mnt/pfs/chenruize/dataset/round_a_results_v2_final.jsonl"
OUTPUT_DIR="scripts/distillation"
PROMPT_DIR="scripts/distillation/group_prompts"

TP_SIZE=8
MAX_MODEL_LEN=20000
TEMPERATURE=0.2
BATCH_SIZE=32
GPU_MEM=0.90

PROMPTS=(
    "DynamicInteraction_v2.txt"
    "TrafficLight_v2.txt"
    "StartStop_v2.txt"
    "Intersection_v2.txt"
    "LaneChange_v2.txt"
    "IntersectionInteraction_v2.txt"
    "LaneCruising_v2.txt"
)

TOTAL=${#PROMPTS[@]}
IDX=0

for PROMPT_FILE in "${PROMPTS[@]}"; do
    IDX=$((IDX + 1))
    GROUP_NAME="${PROMPT_FILE%_v2.txt}"
    GROUP_LOWER=$(echo "$GROUP_NAME" | tr '[:upper:]' '[:lower:]')
    OUTPUT_FILE="${OUTPUT_DIR}/round_b_label_results_${GROUP_LOWER}_v8.json"

    echo "=============================================="
    echo "[${IDX}/${TOTAL}] Group: ${GROUP_NAME}"
    echo "  Prompt: ${PROMPT_DIR}/${PROMPT_FILE}"
    echo "  Output: ${OUTPUT_FILE}"
    echo "=============================================="

    python scripts/distillation/round_b_label_infer_v3.py \
        --model "${MODEL}" \
        --input "${INPUT}" \
        --output "${OUTPUT_FILE}" \
        --prompt "${PROMPT_DIR}/${PROMPT_FILE}" \
        --tensor-parallel-size "${TP_SIZE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --temperature "${TEMPERATURE}" \
        --batch-size "${BATCH_SIZE}" \
        --gpu-memory-utilization "${GPU_MEM}"

    echo "[${IDX}/${TOTAL}] ${GROUP_NAME} done."
    echo ""
done

echo "All ${TOTAL} groups completed."
