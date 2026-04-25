#!/bin/bash
# 30B 微调模型挖掘脚本
# 用 Qwen3-VL-30B checkpoint-282 部署 + 在 rawdata 扩大池上挖掘
#
# Step 1: 部署30B模型 (4xA100, TP=4)
# Step 2: 等待服务就绪
# Step 3: 启动挖掘
#
# 用法: bash scene_tag/launch_30b_mining.sh

set -e

MODEL_PATH="/mnt/pfs/houhaotian/saves/Qwen3-VL-30B-A3B-Instruct/full/sft_segment_multilabel_v3_8gpu/checkpoint-282"
MODEL_NAME="qwen3-vl-30b-sft"
PORT=8000
TP=4
MINING_POOL="/root/workspace/LLaMA-Factory/data/mining_pool_all.json"
RESULT_DIR="/root/workspace/LLaMA-Factory/scene_tag/results_30b"
LOG_DIR="/root/workspace/LLaMA-Factory/scene_tag/logs"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

# 检查挖掘池是否已构建
if [ ! -f "$MINING_POOL" ]; then
    echo "错误: 挖掘池尚未构建完成: $MINING_POOL"
    echo "请等待 build_mining_pool.py 完成后再运行"
    exit 1
fi

POOL_SIZE=$(python3 -c "import json; print(len(json.load(open('$MINING_POOL'))))")
echo "============================================"
echo "  30B 微调模型挖掘"
echo "  模型: $MODEL_PATH"
echo "  挖掘池: $MINING_POOL ($POOL_SIZE 条)"
echo "============================================"

# Step 1: 部署30B模型
echo ""
echo "[Step 1] 部署 30B 模型 (TP=$TP, port=$PORT)..."

# 先检查是否已经在运行
if curl -s --noproxy '*' --connect-timeout 3 http://127.0.0.1:$PORT/health | grep -q ""; then
    echo "  端口 $PORT 已有服务，跳过部署"
else
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size $TP \
        --port $PORT \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code \
        --max-num-seqs 8 \
        --limit-mm-per-prompt '{"image": 40}' \
        --allowed-local-media-path / \
        > "$LOG_DIR/vllm_30b.log" 2>&1 &

    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"
    echo "  日志: $LOG_DIR/vllm_30b.log"

    # Step 2: 等待服务就绪
    echo ""
    echo "[Step 2] 等待服务就绪..."
    for i in $(seq 1 120); do
        if curl -s --noproxy '*' --connect-timeout 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ""; then
            echo "  服务就绪! (等待了 ${i}0 秒)"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "  超时: 服务未能在 20 分钟内启动"
            echo "  查看日志: tail -50 $LOG_DIR/vllm_30b.log"
            exit 1
        fi
        sleep 10
    done
fi

# Step 3: 提取视频列表并启动挖掘
echo ""
echo "[Step 3] 启动挖掘..."

# 从mining_pool_all.json提取视频路径列表
VIDEO_LIST="$RESULT_DIR/video_list_all.txt"
python3 -c "
import json
with open('$MINING_POOL') as f:
    data = json.load(f)
with open('$VIDEO_LIST', 'w') as f:
    for item in data:
        for v in item.get('videos', []):
            f.write(v + '\n')
print(f'视频列表: {len(data)} 条 → $VIDEO_LIST')
"

# 用训练集的prompt格式（P00的12标签prompt）挖掘
# 30B模型是用这个prompt微调的，所以用原始prompt效果最好
PROMPT_FILE="$RESULT_DIR/system_prompt_30b.txt"
python3 -c "
import json
with open('/root/workspace/LLaMA-Factory/data/qwen3_sft_train_segment_multilabel_v3.json') as f:
    data = json.load(f)
sys_prompt = data[0]['system']
if sys_prompt.startswith('system\n'):
    sys_prompt = sys_prompt[7:]
with open('$PROMPT_FILE', 'w') as f:
    f.write(sys_prompt)
print(f'System prompt 已提取 ({len(sys_prompt)} 字符)')
"

# 启动挖掘（单任务，30B模型可以开更高并发）
NO_PROXY="*" no_proxy="*" nohup python3 scene_tag/12_distillation.py \
    --api_base http://127.0.0.1:$PORT/v1 \
    --model_name "$MODEL_NAME" \
    --video_list "$VIDEO_LIST" \
    --prompt_file "$PROMPT_FILE" \
    --output "$RESULT_DIR/mining_30b_all.json" \
    --resolution 256 \
    --concurrency 4 \
    --min_confidence 50 \
    --request_timeout 120 \
    > "$LOG_DIR/mining_30b.log" 2>&1 &

MINING_PID=$!
echo "  挖掘已启动 PID: $MINING_PID"
echo "  结果: $RESULT_DIR/mining_30b_all.json"
echo "  日志: $LOG_DIR/mining_30b.log"
echo ""
echo "============================================"
echo "  30B 模型比 235B 快得多:"
echo "  - 30B MoE 只激活 3B 参数 (vs 235B 激活 22B)"
echo "  - max_num_seqs=8 (vs 4)"
echo "  - concurrency=4"
echo "  - 预计 ~1-2s/segment"
echo "  - ~27K segments 预计 ~8-15h"
echo "============================================"
