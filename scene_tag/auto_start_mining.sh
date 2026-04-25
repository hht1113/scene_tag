#!/bin/bash
cd /root/workspace/LLaMA-Factory

echo "[$(date)] 等待30B服务就绪..."
while true; do
    RESULT=$(curl -s --noproxy '*' --connect-timeout 2 http://127.0.0.1:8000/v1/models 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null)
    if [ -n "$RESULT" ]; then
        echo "[$(date)] 服务就绪: $RESULT"
        break
    fi
    sleep 10
done

echo "[$(date)] 启动挖掘..."
NO_PROXY="*" no_proxy="*" python3 scene_tag/12_distillation.py \
    --api_base http://127.0.0.1:8000/v1 \
    --model_name "qwen3-vl-30b-sft" \
    --video_list scene_tag/results_30b/video_list_all.txt \
    --prompt_file scene_tag/results_30b/system_prompt_30b.txt \
    --output scene_tag/results_30b/mining_30b_all.json \
    --resolution 256 \
    --concurrency 4 \
    --min_confidence 50 \
    --request_timeout 120

echo "[$(date)] 挖掘完成!"
