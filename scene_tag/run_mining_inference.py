"""
挖掘池推理脚本：用 12 标签 SFT 模型对挖掘池视频做场景标注。

前置条件：
  1. 挖掘池 JSON 已构建：mining_pool_expanded.json（30000+ 条，output 为空）
  2. 12 标签 SFT 模型已部署为 vLLM 服务

Usage:
  # 1) 启动 vLLM 服务（checkpoint-226 为例，30B 需双卡）
  CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model /mnt/pfs/houhaotian/saves/Qwen3-VL-30B-A3B-Instruct/full/sft_segment_upsample_8gpu_0317/checkpoint-226 \
    --served-model-name scene-tagger-30b \
    --host 127.0.0.1 --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 2 \
    --limit-mm-per-prompt '{"video":1}'

  # 2) 运行推理（断点续跑，Ctrl+C 安全中断）
  python run_mining_inference.py \
    --pool /root/workspace/LLaMA-Factory/data/mining_pool_expanded.json \
    --output /root/workspace/LLaMA-Factory/data/mining_results.jsonl \
    --api-base http://127.0.0.1:8000/v1 \
    --model scene-tagger-30b \
    --workers 8

  # 3) 切片完成后重建挖掘池再跑
  python build_mining_pool_v3.py --step build
  python run_mining_inference.py --pool ... --output ... --resume
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

VALID_LABELS = {
    "TrafficLight_StraightStopOrGo",
    "TrafficLight_LeftTurnStopOrGo",
    "LaneChange_NavForIntersection",
    "LaneChange_AvoidSlowVRU",
    "LaneChange_AvoidStaticVehicle",
    "DynamicInteraction_VRUInLaneCrossing",
    "DynamicInteraction_VehicleInLaneCrossing",
    "DynamicInteraction_StandardVehicleCutIn",
    "StartStop_StartFromMainRoad",
    "StartStop_ParkRoadside",
    "Intersection_StandardUTurn",
    "LaneCruising_Straight",
    "else",
}


def parse_labels(text):
    """从模型输出中解析 <driving_maneuver> 标签。"""
    pattern = re.compile(
        r"<driving_maneuver>(.*?)</driving_maneuver>\s*"
        r"from\s*<start_time>([\d.]+)</start_time>\s*"
        r"to\s*<end_time>([\d.]+)</end_time>"
    )
    results = []
    for m in pattern.finditer(text):
        label = m.group(1).strip()
        start = float(m.group(2))
        end = float(m.group(3))
        results.append({"label": label, "start": start, "end": end,
                        "valid": label in VALID_LABELS})
    return results


def encode_video_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def infer_one(item, api_base, model, session, max_retries=3):
    """对单条挖掘池样本做推理。"""
    video_path = item["videos"][0]
    if not os.path.exists(video_path):
        return {"video": video_path, "error": "file_not_found", "output": ""}

    b64 = encode_video_b64(video_path)
    messages = []
    if item.get("system"):
        messages.append({"role": "system", "content": item["system"]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
            {"type": "text", "text": item["instruction"]},
        ],
    })

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            r = session.post(
                f"{api_base}/chat/completions",
                json=payload, timeout=120,
            )
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                labels = parse_labels(content)
                return {
                    "video": video_path,
                    "output": content,
                    "labels": labels,
                    "num_valid": sum(1 for lb in labels if lb["valid"]),
                    "num_else": sum(1 for lb in labels if lb["label"] == "else"),
                }
            else:
                time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)

    return {"video": video_path, "error": "api_failed", "output": ""}


def load_done(output_path):
    """加载已完成的视频路径，用于断点续跑。"""
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    r = json.loads(line.strip())
                    done.add(r.get("video", ""))
                except Exception:
                    pass
    return done


def main():
    parser = argparse.ArgumentParser(description="挖掘池推理（12 标签模型）")
    parser.add_argument("--pool", required=True, help="挖掘池 JSON 路径")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="scene-tagger-30b")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（调试用）")
    args = parser.parse_args()

    with open(args.pool) as f:
        pool = json.load(f)
    print(f"挖掘池总条数: {len(pool)}")

    done = set()
    if args.resume:
        done = load_done(args.output)
        print(f"已完成: {len(done)}, 待推理: {len(pool) - len(done)}")

    todo = [item for item in pool if item["videos"][0] not in done]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"本次推理: {len(todo)}")

    session = requests.Session()
    session.trust_env = False

    outf = open(args.output, "a")
    ok = err = 0
    label_counter = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(infer_one, item, args.api_base, args.model, session): item
            for item in todo
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="推理中"):
            result = future.result()
            outf.write(json.dumps(result, ensure_ascii=False) + "\n")
            outf.flush()
            if "error" in result:
                err += 1
            else:
                ok += 1
                for lb in result.get("labels", []):
                    label_counter[lb["label"]] = label_counter.get(lb["label"], 0) + 1

    outf.close()

    print(f"\n{'='*60}")
    print(f"  推理完成: 成功 {ok}, 失败 {err}")
    print(f"  输出文件: {args.output}")
    print(f"{'='*60}")
    print(f"  标签分布:")
    for lb, cnt in sorted(label_counter.items(), key=lambda x: -x[1]):
        marker = " ✓" if lb in VALID_LABELS else " ✗ UNKNOWN"
        print(f"    {lb:<55} {cnt:>6}{marker}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
