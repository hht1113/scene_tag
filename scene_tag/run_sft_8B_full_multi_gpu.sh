#!/bin/bash
###############################################################################
# Qwen3-VL-4B-Instruct 全参 SFT 多卡分布式训练脚本
#
# 数据集:   qwen3_sft_train_segment_upsample.json
# 微调方式: 全参微调 + DeepSpeed ZeRO-3
# 框架:     LLaMA-Factory + torchrun 分布式训练
#
# 用法:
#   8卡训练:  bash run_sft_multi_gpu.sh
#   指定GPU:  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_sft_multi_gpu.sh
#   多机训练: NNODES=2 NODE_RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 bash run_sft_multi_gpu.sh
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

# ========================= 基础配置 =========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${SCRIPT_DIR}/train_qwen3vl_sft_multi_gpu.yaml"

cd "$PROJECT_DIR"

# ========================= 环境检查 =========================
echo "============================================================"
echo "  Qwen3-VL-4B 全参 SFT 多卡训练 (DeepSpeed ZeRO-3)"
echo "============================================================"
echo ""
echo "[INFO] 项目目录:   $PROJECT_DIR"
echo "[INFO] 配置文件:   $CONFIG_FILE"
echo "[INFO] 当前时间:   $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查数据集文件
DATASET_FILE="${PROJECT_DIR}/data/qwen3_sft_train_segment_upsample.json"
if [ ! -f "$DATASET_FILE" ]; then
    echo "[ERROR] 数据集文件不存在: $DATASET_FILE"
    exit 1
fi
echo "[INFO] 数据集文件: $DATASET_FILE"

# 检查 DeepSpeed 配置文件
DS_CONFIG="${PROJECT_DIR}/examples/deepspeed/ds_z3_config.json"
if [ ! -f "$DS_CONFIG" ]; then
    echo "[ERROR] DeepSpeed 配置文件不存在: $DS_CONFIG"
    exit 1
fi
echo "[INFO] DeepSpeed:  $DS_CONFIG"

# 检查 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] 未找到 nvidia-smi，请确认 GPU 环境"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "[INFO] 检测到 GPU 数量: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

if [ "$GPU_COUNT" -lt 1 ]; then
    echo "[ERROR] 未检测到可用 GPU"
    exit 1
fi

# ========================= 分布式训练配置 =========================
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$GPU_COUNT}"

echo "[INFO] 分布式训练配置:"
echo "  NNODES:          $NNODES"
echo "  NODE_RANK:       $NODE_RANK"
echo "  NPROC_PER_NODE:  $NPROC_PER_NODE"
echo "  MASTER_ADDR:     $MASTER_ADDR"
echo "  MASTER_PORT:     $MASTER_PORT"
echo ""

# 打印有效 batch size
PER_DEVICE_BS=4
GRAD_ACCUM=4
EFFECTIVE_BS=$((PER_DEVICE_BS * GRAD_ACCUM * NPROC_PER_NODE))
echo "[INFO] 有效 batch size: $PER_DEVICE_BS × $GRAD_ACCUM × $NPROC_PER_NODE = $EFFECTIVE_BS"
echo ""

# ========================= 启动训练 =========================
echo "============================================================"
echo "  开始训练 @ $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""

FORCE_TORCHRUN=1 \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
NPROC_PER_NODE=$NPROC_PER_NODE \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
  llamafactory-cli train "$CONFIG_FILE"

TRAIN_EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "  训练完成 @ $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo "  训练失败 (exit code: $TRAIN_EXIT_CODE) @ $(date '+%Y-%m-%d %H:%M:%S')"
fi
echo "============================================================"

exit $TRAIN_EXIT_CODE
