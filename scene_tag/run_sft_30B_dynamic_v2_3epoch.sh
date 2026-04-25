#!/bin/bash
###############################################################################
# Qwen3-VL-30B-A3B 续训: 动态采样 v2 + cutoff_len=10000 + 3 epoch
#
# 基座:   add_tags_dynamic 2epoch 检查点
# 数据:   动态采样 v2 (6897 样本, 已验证有效的配比, 新随机种子)
# DS:     ZeRO-3 (overlap_comm=true, 无 CPU offload)
# 关键变更: cutoff_len 20000 → 10000
#
# 用法:
#   bash run_sft_30B_dynamic_v2_3epoch.sh
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
CONFIG_FILE="${SCRIPT_DIR}/train_qwen3vl_30B_full_multi_gpu_dynamic_v2_3epoch.yaml"

cd "$PROJECT_DIR"

export DISABLE_VERSION_CHECK=1

# ========================= 环境检查 =========================
echo "============================================================"
echo "  Qwen3-VL-30B-A3B 续训 (动态采样 v2 + cutoff=10000)"
echo "  基座: add_tags_dynamic 2epoch checkpoint"
echo "  DeepSpeed: ZeRO-3 (overlap_comm=true, 无offload)"
echo "============================================================"
echo ""
echo "[INFO] 项目目录:   $PROJECT_DIR"
echo "[INFO] 配置文件:   $CONFIG_FILE"
echo "[INFO] 当前时间:   $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

DATASET_FILE="${PROJECT_DIR}/data/qwen3_sft_train_segment_add_tags_dynamic_v2.json"
if [ ! -f "$DATASET_FILE" ]; then
    echo "[ERROR] 数据集文件不存在: $DATASET_FILE"
    echo "[INFO] 请先运行: python3 scene_tag/6_resample_dataset_dynamic_v2.py"
    exit 1
fi
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$DATASET_FILE'))))")
echo "[INFO] 数据集文件: $DATASET_FILE ($SAMPLE_COUNT 样本)"

DS_CONFIG="${PROJECT_DIR}/examples/deepspeed/ds_z3.json"
if [ ! -f "$DS_CONFIG" ]; then
    echo "[ERROR] DeepSpeed 配置文件不存在: $DS_CONFIG"
    exit 1
fi
echo "[INFO] DeepSpeed:  $DS_CONFIG (ZeRO-3, overlap_comm=true, 无offload)"

MODEL_DIR="/mnt/pfs/houhaotian/saves/Qwen3-VL-30B-A3B-Instruct/full/sft_segment_add_tags_dynamic_8gpu"
if [ ! -d "$MODEL_DIR" ]; then
    echo "[ERROR] 基座模型目录不存在: $MODEL_DIR"
    exit 1
fi
echo "[INFO] 基座模型:   $MODEL_DIR"
echo "[INFO] cutoff_len: 10000"

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

PER_DEVICE_BS=1
GRAD_ACCUM=4
EFFECTIVE_BS=$((PER_DEVICE_BS * GRAD_ACCUM * NPROC_PER_NODE))
STEPS_PER_EPOCH=$(( (SAMPLE_COUNT + EFFECTIVE_BS - 1) / EFFECTIVE_BS ))
TOTAL_STEPS=$((STEPS_PER_EPOCH * 3))
echo "[INFO] 有效 batch size: $PER_DEVICE_BS × $GRAD_ACCUM × $NPROC_PER_NODE = $EFFECTIVE_BS"
echo "[INFO] 预计 ~${STEPS_PER_EPOCH} 步/epoch, 共 ~${TOTAL_STEPS} 步 (3 epoch)"
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
