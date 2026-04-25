#!/bin/bash
# ============================================================
# Doubao (ADVC) 一键启动脚本
#
# 用法：
#   # 图片任务（精调验证 + 泛化验证，共 4 个任务）
#   bash scene_tag/launch_doubao.sh image YOUR_ADVC_API_KEY
#
#   # 视频任务（路口通行 + 车道巡航，共 2 个任务）
#   bash scene_tag/launch_doubao.sh video YOUR_ADVC_API_KEY
#
#   # 全部任务
#   bash scene_tag/launch_doubao.sh all YOUR_ADVC_API_KEY
#
#   # 指定模型名（默认 ADVC-Data-Reasoning）
#   bash scene_tag/launch_doubao.sh image YOUR_KEY ADVC-Data-Reasoning
# ============================================================

set -e

TASK_TYPE="${1:-all}"
API_KEY="${2}"
MODEL_NAME="${3:-ADVC-Data-Reasoning}"

if [ -z "$API_KEY" ]; then
    echo "用法: bash $0 {image|video|all} <ADVC_API_KEY> [模型名]"
    echo ""
    echo "示例:"
    echo "  bash scene_tag/launch_doubao.sh image abc123def456"
    echo "  bash scene_tag/launch_doubao.sh video abc123def456"
    echo "  bash scene_tag/launch_doubao.sh all abc123def456 doubao-seed-1-6-vision-250815"
    exit 1
fi

# ========== 配置 ==========
API_BASE="https://ai-beijing.volcadvc.com/api/v1"
IMAGE_SCRIPT="scene_tag/14_image_distillation.py"
VIDEO_SCRIPT="scene_tag/12_distillation.py"
RESULT_DIR="scene_tag/results_ab_compare"
LOG_DIR="scene_tag/logs"
IMAGE_SAMPLE="$RESULT_DIR/sample_images.txt"
VIDEO_SAMPLE="$RESULT_DIR/sample_1000.txt"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "============================================"
echo "  Doubao (ADVC) 对比实验"
echo "  API:   $API_BASE"
echo "  模型:  $MODEL_NAME"
echo "  任务:  $TASK_TYPE"
echo "============================================"

# ========== 连通性测试 ==========
echo ""
echo "[测试] 验证 API 连通性..."
TEST_RESULT=$(curl -s --connect-timeout 15 \
    "$API_BASE/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":5}" 2>&1)

if echo "$TEST_RESULT" | grep -q '"choices"'; then
    echo "  ✅ API 连通，模型可用！"
elif echo "$TEST_RESULT" | grep -q 'ServerOverloaded\|Too Many Requests\|429'; then
    echo "  ⚠ API 可达但服务暂时过载 (429)，任务启动后会自动重试"
elif echo "$TEST_RESULT" | grep -q 'Forbidden\|Unauthorized'; then
    echo "  ❌ API Key 无效"
    exit 1
elif echo "$TEST_RESULT" | grep -q 'NotFound'; then
    echo "  ❌ 模型不存在: $MODEL_NAME"
    exit 1
else
    echo "  ⚠ API 返回: $(echo $TEST_RESULT | head -c 200)"
    echo "  继续启动任务..."
fi

# ========== 图片任务 ==========
run_image_tasks() {
    echo ""
    echo "============================================"
    echo "  启动图片任务（4 个）"
    echo "  数据: $IMAGE_SAMPLE ($(wc -l < "$IMAGE_SAMPLE") 张)"
    echo "============================================"

    # 精调验证（6标签合并）
    echo "  [1/4] 精调验证（6标签）..."
    nohup python3 "$IMAGE_SCRIPT" \
        --api_base "$API_BASE" \
        --model_name "$MODEL_NAME" \
        --api_key "$API_KEY" \
        --image_list "$IMAGE_SAMPLE" \
        --prompt_file "scene_tag/prompt_txt/img_finetune_6labels.txt" \
        --output "$RESULT_DIR/doubao_img_finetune_6labels.json" \
        --concurrency 4 \
        --request_timeout 120 \
        > "$LOG_DIR/ab_doubao_img_finetune_6labels.log" 2>&1 &
    echo "    PID: $!"

    sleep 1

    # 泛化-雨天积水
    echo "  [2/4] 泛化-雨天积水..."
    nohup python3 "$IMAGE_SCRIPT" \
        --api_base "$API_BASE" \
        --model_name "$MODEL_NAME" \
        --api_key "$API_KEY" \
        --image_list "$IMAGE_SAMPLE" \
        --prompt_file "scene_tag/prompt_txt/img_road_surface_water.txt" \
        --output "$RESULT_DIR/doubao_img_road_surface_water.json" \
        --concurrency 4 \
        --request_timeout 120 \
        > "$LOG_DIR/ab_doubao_img_road_surface_water.log" 2>&1 &
    echo "    PID: $!"

    sleep 1

    # 泛化-双灯倒计时
    echo "  [3/4] 泛化-双灯倒计时..."
    nohup python3 "$IMAGE_SCRIPT" \
        --api_base "$API_BASE" \
        --model_name "$MODEL_NAME" \
        --api_key "$API_KEY" \
        --image_list "$IMAGE_SAMPLE" \
        --prompt_file "scene_tag/prompt_txt/img_dual_countdown.txt" \
        --output "$RESULT_DIR/doubao_img_dual_countdown.json" \
        --concurrency 4 \
        --request_timeout 120 \
        > "$LOG_DIR/ab_doubao_img_dual_countdown.log" 2>&1 &
    echo "    PID: $!"

    sleep 1

    # 泛化-便携式信号灯
    echo "  [4/4] 泛化-便携式信号灯..."
    nohup python3 "$IMAGE_SCRIPT" \
        --api_base "$API_BASE" \
        --model_name "$MODEL_NAME" \
        --api_key "$API_KEY" \
        --image_list "$IMAGE_SAMPLE" \
        --prompt_file "scene_tag/prompt_txt/img_portable_tld.txt" \
        --output "$RESULT_DIR/doubao_img_portable_tld.json" \
        --concurrency 4 \
        --request_timeout 120 \
        > "$LOG_DIR/ab_doubao_img_portable_tld.log" 2>&1 &
    echo "    PID: $!"
}

# ========== 视频任务 ==========
run_video_tasks() {
    echo ""
    echo "============================================"
    echo "  启动视频任务（2 个）"
    echo "  数据: $VIDEO_SAMPLE ($(wc -l < "$VIDEO_SAMPLE") 条)"
    echo "============================================"

    # 路口通行
    echo "  [1/2] 路口通行（22标签）..."
    nohup python3 "$VIDEO_SCRIPT" \
        --api_base "$API_BASE" \
        --model_name "$MODEL_NAME" \
        --video_list "$VIDEO_SAMPLE" \
        --prompt_file "scene_tag/prompt_txt/04_Intersection.txt" \
        --output "$RESULT_DIR/doubao_04_Intersection.json" \
        --resolution 256 \
        --concurrency 2 \
        --min_confidence 50 \
        --request_timeout 300 \
        > "$LOG_DIR/ab_doubao_04_Intersection.log" 2>&1 &
    echo "    PID: $!"

    sleep 1

    # 车道巡航
    echo "  [2/2] 车道巡航（18标签）..."
    nohup python3 "$VIDEO_SCRIPT" \
        --api_base "$API_BASE" \
        --model_name "$MODEL_NAME" \
        --video_list "$VIDEO_SAMPLE" \
        --prompt_file "scene_tag/prompt_txt/05_LaneCruising.txt" \
        --output "$RESULT_DIR/doubao_05_LaneCruising.json" \
        --resolution 256 \
        --concurrency 2 \
        --min_confidence 50 \
        --request_timeout 300 \
        > "$LOG_DIR/ab_doubao_05_LaneCruising.log" 2>&1 &
    echo "    PID: $!"
}

# ========== 执行 ==========
case "$TASK_TYPE" in
    image)
        run_image_tasks
        ;;
    video)
        run_video_tasks
        ;;
    all)
        run_image_tasks
        run_video_tasks
        ;;
    *)
        echo "未知任务类型: $TASK_TYPE"
        echo "可选: image / video / all"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "  Doubao 任务已启动"
echo "  结果: $RESULT_DIR/doubao_*.json"
echo "  日志: tail -f $LOG_DIR/ab_doubao_*.log"
echo "============================================"
