#!/usr/bin/env python3
"""
严禁靠边场景视频 — 重试脚本

1. 从已保存的 JSON 元数据重新下载视频（无需重新生成）
2. 对完全缺失的场景重新提交生成任务
3. 支持 --ref_image 模式
"""

import argparse
import json
import os
import time
import requests
import glob as _glob

ALL_PP_TAGS = [
    "PP_NoStopSignZone", "PP_BusLaneAndStation", "PP_FireLaneEntrance",
    "PP_CrosswalkZone", "PP_IntersectionSpecialRoad", "PP_ExpresswayMainRoad",
    "PP_InsideTunnel", "PP_LongDownhill", "PP_PoorVisibilityZone",
    "PP_NonMotorLaneSidewalk", "PP_FireHydrantZone", "PP_HospitalEntrance",
    "PP_SchoolEntrance", "PP_EventVenueEntrance", "PP_MetroTransitHub",
    "PP_ResidentialGateCrowded", "PP_MilitaryGovZone", "PP_GPSUnstableZone",
    "PP_SingleLaneMainRoad", "PP_HighTrafficIntersection",
    "PP_TaxiRideshareWaiting", "PP_LoadingUnloadingZone",
    "PP_IndustrialParkEntrance",
]

API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seedance-2-0-260128"


def download_video(video_url: str, output_path: str, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            resp = requests.get(video_url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"    下载成功: {output_path} ({size_mb:.1f} MB)")
            return True
        except Exception as e:
            print(f"    下载失败 (尝试 {attempt+1}/{retries}): {e}")
            time.sleep(3)
    return False


def poll_task(api_key: str, task_id: str, max_wait: int = 600) -> dict | None:
    url = f"{API_BASE}/contents/generations/tasks/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            status = result.get("status", "")
            if status in ("SUCCESS", "succeeded", "complete"):
                return result
            elif status in ("FAILED", "failed"):
                print(f"    任务失败: {result.get('fail_reason', '未知')}")
                return None
            elapsed = int(time.time() - start)
            print(f"    [{elapsed}s] {status}")
            time.sleep(15)
        except Exception as e:
            print(f"    轮询错误: {e}")
            time.sleep(10)
    return None


def submit_and_wait(api_key: str, prompt: str, ref_image_path: str | None = None) -> dict | None:
    content = []
    if ref_image_path:
        import base64
        from PIL import Image
        import io
        img = Image.open(ref_image_path).resize((640, 360))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        content.append({"type": "image_url", "image_url": {"url": img_b64}, "role": "reference_image"})
    content.append({"type": "text", "text": prompt})

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {"model": MODEL, "content": content, "resolution": "720p", "ratio": "16:9", "duration": 10}

    for attempt in range(3):
        try:
            resp = requests.post(f"{API_BASE}/contents/generations/tasks", json=body, headers=headers, timeout=60)
            resp.raise_for_status()
            task_id = resp.json().get("id")
            if task_id:
                print(f"    提交成功: {task_id}")
                return poll_task(api_key, task_id)
        except Exception as e:
            print(f"    提交失败 (尝试 {attempt+1}/3): {e}")
            time.sleep(5)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--output_dir", default="/mnt/pfs/houhaotian/prohibited_parking_videos")
    parser.add_argument("--ref_image", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = "_ref" if args.ref_image else ""
    ref_label = " (带参考图)" if args.ref_image else ""

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gen_video", os.path.join(os.path.dirname(__file__), "16_generate_video.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TAG_PROMPTS = mod.TAG_PROMPTS

    print(f"\n{'='*60}")
    print(f"  严禁靠边场景 — 重试/补全{ref_label}")
    print(f"{'='*60}\n")

    success = 0
    fail = 0
    skip = 0

    for tag in ALL_PP_TAGS:
        mp4_name = f"{tag}_seedance2{suffix}.mp4"
        json_name = f"{tag}_seedance2{suffix}.json"
        mp4_path = os.path.join(args.output_dir, mp4_name)
        json_path = os.path.join(args.output_dir, json_name)

        if os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 100_000:
            size_mb = os.path.getsize(mp4_path) / 1024 / 1024
            print(f"[{tag}] 已存在 ({size_mb:.1f} MB)，跳过")
            skip += 1
            continue

        print(f"\n[{tag}]{ref_label}")

        if os.path.isfile(json_path):
            with open(json_path) as f:
                meta = json.load(f)
            video_url = meta.get("result", {}).get("content", {}).get("video_url", "")
            if video_url:
                print(f"  JSON 存在，尝试重新下载...")
                if download_video(video_url, mp4_path):
                    success += 1
                    continue
                print(f"  URL 可能过期，重新轮询任务...")
                task_id = meta.get("task_id")
                if task_id:
                    result = poll_task(args.api_key, task_id)
                    if result:
                        new_url = result.get("content", {}).get("video_url", "")
                        if new_url and download_video(new_url, mp4_path):
                            meta["result"] = result
                            with open(json_path, "w") as f:
                                json.dump(meta, f, ensure_ascii=False, indent=2)
                            success += 1
                            continue

        print(f"  需要重新生成...")
        if tag not in TAG_PROMPTS:
            print(f"  错误: 未找到 prompt 配置")
            fail += 1
            continue

        prompt = TAG_PROMPTS[tag]["prompt"]
        result = submit_and_wait(args.api_key, prompt, args.ref_image)
        if result:
            video_url = result.get("content", {}).get("video_url", "")
            if video_url and download_video(video_url, mp4_path):
                with open(json_path, "w") as f:
                    json.dump({
                        "task_id": result.get("id", ""),
                        "tag": tag,
                        "model": MODEL,
                        "prompt": prompt,
                        "duration": 10,
                        "resolution": "720p",
                        "ref_image": args.ref_image,
                        "result": result,
                    }, f, ensure_ascii=False, indent=2)
                success += 1
                continue
        fail += 1

    print(f"\n{'='*60}")
    print(f"  完成{ref_label}! 成功: {success}, 跳过: {skip}, 失败: {fail}")
    print(f"  总计: {success + skip}/23 可用")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
