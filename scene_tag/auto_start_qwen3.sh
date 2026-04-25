#!/bin/bash
cd /root/workspace/LLaMA-Factory
export no_proxy="${no_proxy},10.10.64.144,127.0.0.1"
export NO_PROXY="${NO_PROXY},10.10.64.144,127.0.0.1"

echo "[$(date)] 等待 Qwen3-VL-235B 部署完成..."

while true; do
    if curl -s --connect-timeout 5 --noproxy '*' http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "qwen3-vl-235b"; then
        echo "[$(date)] Qwen3-VL-235B 部署完成！开始启动推理任务..."
        break
    fi
    sleep 30
done

sleep 10

echo "[$(date)] 启动精调验证任务..."
nohup python3 scene_tag/14_image_distillation.py \
    --api_base "http://127.0.0.1:8000/v1" \
    --model_name "qwen3-vl-235b" \
    --image_list "scene_tag/results_ab_compare/sample_images.txt" \
    --prompt_file "scene_tag/prompt_txt/img_finetune_6labels.txt" \
    --output "scene_tag/results_ab_compare/qwen235b_img_finetune_6labels.json" \
    --concurrency 2 \
    --request_timeout 180 \
    > scene_tag/logs/ab_qwen235b_img_finetune_6labels.log 2>&1 &
echo "[$(date)] 精调验证 PID: $!"

sleep 2

echo "[$(date)] 启动泛化-雨天积水..."
nohup python3 scene_tag/14_image_distillation.py \
    --api_base "http://127.0.0.1:8000/v1" \
    --model_name "qwen3-vl-235b" \
    --image_list "scene_tag/results_ab_compare/sample_images.txt" \
    --prompt_file "scene_tag/prompt_txt/img_road_surface_water.txt" \
    --output "scene_tag/results_ab_compare/qwen235b_img_road_surface_water.json" \
    --concurrency 2 \
    --request_timeout 180 \
    > scene_tag/logs/ab_qwen235b_img_road_surface_water.log 2>&1 &
echo "[$(date)] 雨天积水 PID: $!"

sleep 2

echo "[$(date)] 启动泛化-双灯倒计时..."
nohup python3 scene_tag/14_image_distillation.py \
    --api_base "http://127.0.0.1:8000/v1" \
    --model_name "qwen3-vl-235b" \
    --image_list "scene_tag/results_ab_compare/sample_images.txt" \
    --prompt_file "scene_tag/prompt_txt/img_dual_countdown.txt" \
    --output "scene_tag/results_ab_compare/qwen235b_img_dual_countdown.json" \
    --concurrency 2 \
    --request_timeout 180 \
    > scene_tag/logs/ab_qwen235b_img_dual_countdown.log 2>&1 &
echo "[$(date)] 双灯倒计时 PID: $!"

sleep 2

echo "[$(date)] 启动泛化-便携式信号灯..."
nohup python3 scene_tag/14_image_distillation.py \
    --api_base "http://127.0.0.1:8000/v1" \
    --model_name "qwen3-vl-235b" \
    --image_list "scene_tag/results_ab_compare/sample_images.txt" \
    --prompt_file "scene_tag/prompt_txt/img_portable_tld.txt" \
    --output "scene_tag/results_ab_compare/qwen235b_img_portable_tld.json" \
    --concurrency 2 \
    --request_timeout 180 \
    > scene_tag/logs/ab_qwen235b_img_portable_tld.log 2>&1 &
echo "[$(date)] 便携式信号灯 PID: $!"

echo ""
echo "[$(date)] === Qwen3-VL-235B 全部 4 个推理任务已启动 ==="
echo "监控: tail -f scene_tag/logs/ab_qwen235b_*.log"
