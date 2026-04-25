#!/bin/bash
# ============================================================
#  场景标签挖掘：用 12 标签 SFT 模型推理挖掘池
# ============================================================
#
#  模型:  checkpoint-226 (Qwen3-VL-30B-A3B, 12 类场景标签)
#  挖掘池: mining_pool_expanded.json (30000+ 条 20s 视频, output 为空)
#  输出:  mining_results.jsonl (JSONL, 每行含 video/output/labels)
#
#  用法:  bash launch_mining.sh
# ============================================================

set -e

MODEL_PATH="/mnt/pfs/houhaotian/saves/Qwen3-VL-30B-A3B-Instruct/full/sft_segment_upsample_8gpu_0317/checkpoint-226"
MODEL_NAME="scene-tagger-30b"
PORT=8000
GPU=0          # 30B MoE (A3B) 单卡 A100 即可

POOL="/root/workspace/LLaMA-Factory/data/mining_pool_expanded.json"
OUTPUT="/root/workspace/LLaMA-Factory/data/mining_results.jsonl"

# ---------- Step 1: 启动 vLLM ----------
echo "[1/2] 启动 vLLM 服务 (GPU $GPU, port $PORT)..."
CUDA_VISIBLE_DEVICES=$GPU nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port $PORT \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"video":1}' \
  > /tmp/vllm_mining.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID, 等待启动..."

# 等 vLLM 就绪
for i in $(seq 1 60); do
  if curl -s http://127.0.0.1:$PORT/v1/models | grep -q "$MODEL_NAME" 2>/dev/null; then
    echo "  vLLM 就绪 (${i}s)"
    break
  fi
  sleep 5
done

# ---------- Step 2: 推理 ----------
echo "[2/2] 开始推理..."
cd /root/workspace/LLaMA-Factory/scene_tag
python run_mining_inference.py \
  --pool "$POOL" \
  --output "$OUTPUT" \
  --api-base "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL_NAME" \
  --workers 8 \
  --resume

echo "推理完成, 结果: $OUTPUT"
echo "关闭 vLLM: kill $VLLM_PID"
kill $VLLM_PID 2>/dev/null
