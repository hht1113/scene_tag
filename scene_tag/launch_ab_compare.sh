#!/bin/bash
# ============================================================
# 三模型 A/B/C 对比实验
#
# 维度一：精调验证（6 个交通标志/标线识别 prompt × 5000 张图片）
# 维度二：泛化验证
#   - 困难场景挖掘（3 个 prompt × 5000 张图片）
#   - 视频驾驶行为挖掘（2 个 prompt × 1000 条视频）
#
# 三个模型：
#   A: doubao     — 豆包精调模型（需填入API地址和模型名）
#   B: qwen235b   — Qwen3-VL-235B-FP8（本地部署 http://127.0.0.1:8000/v1）
#   C: qwen35     — Qwen3.5-397B-A17B（百度云 http://10.10.64.144:2754/v1）
#
# 用法：
#   bash scene_tag/launch_ab_compare.sh all              # 三模型全部任务
#   bash scene_tag/launch_ab_compare.sh doubao            # doubao 全部任务
#   bash scene_tag/launch_ab_compare.sh doubao finetune   # doubao 精调验证
#   bash scene_tag/launch_ab_compare.sh doubao generalize # doubao 泛化验证
#   bash scene_tag/launch_ab_compare.sh doubao image      # doubao 全部图片任务
#   bash scene_tag/launch_ab_compare.sh doubao video      # doubao 视频任务
# ============================================================

set -e

MODE="${1:-all}"
TASK_TYPE="${2:-both}"  # finetune / generalize / image / video / both

# ========== 配置区 ==========

DOUBAO_API_BASE="http://YOUR_DOUBAO_API/v1"
DOUBAO_MODEL="YOUR_DOUBAO_MODEL_NAME"

QWEN235B_API_BASE="http://127.0.0.1:8000/v1"
QWEN235B_MODEL="qwen3-vl-235b"

QWEN35_API_BASE="http://10.10.64.144:2754/v1"
QWEN35_MODEL="Qwen3.5-397B-A17B"

# ========== 公共配置 ==========

VIDEO_SCRIPT="scene_tag/12_distillation.py"
IMAGE_SCRIPT="scene_tag/14_image_distillation.py"
RESULT_DIR="scene_tag/results_ab_compare"
LOG_DIR="scene_tag/logs"

VIDEO_SAMPLE="$RESULT_DIR/sample_1000.txt"
IMAGE_SAMPLE="$RESULT_DIR/sample_images.txt"

# 精调验证：6 个交通标志/标线
FINETUNE_PROMPTS=(
    "img_uturn_arrow"
    "img_left_uturn_arrow"
    "img_construction_sign"
    "img_children_sign"
    "img_no_uturn_sign"
    "img_speed_limit_sign"
)

# 泛化验证-图片：3 个困难场景
GENERALIZE_IMG_PROMPTS=(
    "img_road_surface_water"
    "img_dual_countdown"
    "img_portable_tld"
)

# 泛化验证-视频：2 组驾驶行为
VIDEO_PROMPTS=("04_Intersection" "05_LaneCruising")

export no_proxy="${no_proxy},10.10.64.144,127.0.0.1"
export NO_PROXY="${NO_PROXY},10.10.64.144,127.0.0.1"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

# ========== 通用图片任务函数 ==========

run_image_batch() {
    local MODEL_TAG="$1"
    local API_BASE="$2"
    local MODEL_NAME="$3"
    local LABEL="$4"
    shift 4
    local PROMPTS=("$@")

    echo ""
    echo "============================================"
    echo "  [$LABEL] 图片任务 - $MODEL_TAG"
    echo "  API: $API_BASE"
    echo "  模型: $MODEL_NAME"
    echo "  数据: $IMAGE_SAMPLE ($(wc -l < "$IMAGE_SAMPLE") 张)"
    echo "  Prompt 数: ${#PROMPTS[@]}"
    echo "============================================"

    if ! curl -s --connect-timeout 10 "${API_BASE}/models" 2>/dev/null | grep -q "id\|model\|data"; then
        echo "  ⚠ API不可达: $API_BASE，跳过"
        return 1
    fi
    echo "  API 连接成功"

    for P in "${PROMPTS[@]}"; do
        OUTPUT="$RESULT_DIR/${MODEL_TAG}_${P}.json"
        LOG="$LOG_DIR/ab_${MODEL_TAG}_${P}.log"

        if [ -f "$OUTPUT" ]; then
            local DONE=$(python3 -c "import json; print(len(json.load(open('$OUTPUT'))))" 2>/dev/null)
            echo "  $P: 已有 $DONE 条结果（断点续传）"
        fi

        echo "  启动 $P → $OUTPUT"
        nohup python3 "$IMAGE_SCRIPT" \
            --api_base "$API_BASE" \
            --model_name "$MODEL_NAME" \
            --image_list "$IMAGE_SAMPLE" \
            --prompt_file "scene_tag/prompt_txt/${P}.txt" \
            --output "$OUTPUT" \
            --concurrency 4 \
            --request_timeout 120 \
            > "$LOG" 2>&1 &

        echo "    PID: $!"
        sleep 1
    done
}

# ========== 视频任务 ==========

run_video_tasks() {
    local MODEL_TAG="$1"
    local API_BASE="$2"
    local MODEL_NAME="$3"

    echo ""
    echo "============================================"
    echo "  [泛化-视频] 视频任务 - $MODEL_TAG"
    echo "  API: $API_BASE"
    echo "  模型: $MODEL_NAME"
    echo "  数据: $VIDEO_SAMPLE ($(wc -l < "$VIDEO_SAMPLE") 条)"
    echo "============================================"

    if ! curl -s --connect-timeout 10 "${API_BASE}/models" 2>/dev/null | grep -q "id\|model\|data"; then
        echo "  ⚠ API不可达: $API_BASE，跳过"
        return 1
    fi
    echo "  API 连接成功"

    for P in "${VIDEO_PROMPTS[@]}"; do
        OUTPUT="$RESULT_DIR/${MODEL_TAG}_${P}.json"
        LOG="$LOG_DIR/ab_${MODEL_TAG}_${P}.log"

        if [ -f "$OUTPUT" ]; then
            local DONE=$(python3 -c "import json; print(len(json.load(open('$OUTPUT'))))" 2>/dev/null)
            echo "  $P: 已有 $DONE 条结果（断点续传）"
        fi

        echo "  启动 $P → $OUTPUT"
        nohup python3 "$VIDEO_SCRIPT" \
            --api_base "$API_BASE" \
            --model_name "$MODEL_NAME" \
            --video_list "$VIDEO_SAMPLE" \
            --prompt_file "scene_tag/prompt_txt/${P}.txt" \
            --output "$OUTPUT" \
            --resolution 256 \
            --concurrency 2 \
            --min_confidence 50 \
            --request_timeout 300 \
            > "$LOG" 2>&1 &

        echo "    PID: $!"
        sleep 1
    done
}

# ========== 模型调度 ==========

run_model() {
    local MODEL_TAG="$1"
    local API_BASE="$2"
    local MODEL_NAME="$3"

    case "$TASK_TYPE" in
        finetune)
            run_image_batch "$MODEL_TAG" "$API_BASE" "$MODEL_NAME" "精调验证" "${FINETUNE_PROMPTS[@]}"
            ;;
        generalize)
            run_image_batch "$MODEL_TAG" "$API_BASE" "$MODEL_NAME" "泛化-图片" "${GENERALIZE_IMG_PROMPTS[@]}"
            run_video_tasks "$MODEL_TAG" "$API_BASE" "$MODEL_NAME"
            ;;
        image)
            run_image_batch "$MODEL_TAG" "$API_BASE" "$MODEL_NAME" "精调验证" "${FINETUNE_PROMPTS[@]}"
            run_image_batch "$MODEL_TAG" "$API_BASE" "$MODEL_NAME" "泛化-图片" "${GENERALIZE_IMG_PROMPTS[@]}"
            ;;
        video)
            run_video_tasks "$MODEL_TAG" "$API_BASE" "$MODEL_NAME"
            ;;
        both)
            run_image_batch "$MODEL_TAG" "$API_BASE" "$MODEL_NAME" "精调验证" "${FINETUNE_PROMPTS[@]}"
            run_image_batch "$MODEL_TAG" "$API_BASE" "$MODEL_NAME" "泛化-图片" "${GENERALIZE_IMG_PROMPTS[@]}"
            run_video_tasks "$MODEL_TAG" "$API_BASE" "$MODEL_NAME"
            ;;
    esac
}

case "$MODE" in
    doubao)
        run_model "doubao" "$DOUBAO_API_BASE" "$DOUBAO_MODEL"
        ;;
    qwen235b)
        run_model "qwen235b" "$QWEN235B_API_BASE" "$QWEN235B_MODEL"
        ;;
    qwen35)
        run_model "qwen35" "$QWEN35_API_BASE" "$QWEN35_MODEL"
        ;;
    all)
        run_model "doubao" "$DOUBAO_API_BASE" "$DOUBAO_MODEL" || true
        run_model "qwen235b" "$QWEN235B_API_BASE" "$QWEN235B_MODEL" || true
        run_model "qwen35" "$QWEN35_API_BASE" "$QWEN35_MODEL" || true
        ;;
    *)
        echo "用法: bash $0 {all|doubao|qwen235b|qwen35} [finetune|generalize|image|video|both]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  对比实验已启动"
echo "  精调验证: 5000 张 × 6 prompts"
echo "  泛化-图片: 5000 张 × 3 prompts"
echo "  泛化-视频: 1000 条 × 2 prompts"
echo "  结果: $RESULT_DIR/"
echo "  监控: tail -f $LOG_DIR/ab_*.log"
echo "============================================"
