#!/usr/bin/env python3
"""
严禁靠边场景 — 健壮视频生成（自动重试网络错误）

每个场景：提交 → 轮询（带重试） → 下载（带重试）
网络错误自动等待后重试，不会丢失已提交的任务。
"""

import argparse
import base64
import io
import json
import os
import time
import requests

API_BASE = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seedance-2-0-260128"

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


def _request_with_retry(method, url, max_retries=5, backoff=10, **kwargs):
    kwargs.setdefault("timeout", 90)
    for attempt in range(max_retries):
        try:
            resp = method(url, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as e:
            wait = backoff * (attempt + 1)
            print(f"      网络错误 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"      等待 {wait}s 后重试...")
                time.sleep(wait)
    raise RuntimeError(f"请求失败 {max_retries} 次: {url}")


def load_ref_image(path):
    from PIL import Image
    img = Image.open(path).resize((640, 360))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def submit_task(api_key, prompt, ref_b64=None):
    content = []
    if ref_b64:
        content.append({"type": "image_url", "image_url": {"url": ref_b64}, "role": "reference_image"})
    content.append({"type": "text", "text": prompt})
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {"model": MODEL, "content": content, "resolution": "720p", "ratio": "16:9", "duration": 10}

    resp = _request_with_retry(requests.post, f"{API_BASE}/contents/generations/tasks",
                               json=body, headers=headers)
    task_id = resp.json().get("id")
    print(f"    提交成功: {task_id}")
    return task_id


def poll_until_done(api_key, task_id, max_wait=900):
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{API_BASE}/contents/generations/tasks/{task_id}"
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = _request_with_retry(requests.get, url, headers=headers, max_retries=3, backoff=15)
            result = resp.json()
            status = result.get("status", "")
            if status in ("SUCCESS", "succeeded", "complete"):
                return result
            if status in ("FAILED", "failed"):
                print(f"    生成失败: {result.get('fail_reason', '未知')}")
                return None
            inner = result.get("data", {})
            if isinstance(inner, dict):
                if inner.get("status") == "succeeded":
                    return result
                if inner.get("status") == "failed":
                    return None
            elapsed = int(time.time() - start)
            print(f"    [{elapsed}s] {status or 'running'}")
            time.sleep(15)
        except Exception as e:
            print(f"    轮询异常: {e}，等待 30s...")
            time.sleep(30)
    print(f"    超时 ({max_wait}s)")
    return None


def download_video(url, path):
    for attempt in range(5):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(path) / 1024 / 1024
            if size_mb > 0.5:
                print(f"    下载完成: {os.path.basename(path)} ({size_mb:.1f} MB)")
                return True
            else:
                print(f"    文件太小 ({size_mb:.1f} MB)，重试...")
        except Exception as e:
            print(f"    下载失败 ({attempt+1}/5): {e}")
        time.sleep(10)
    return False


def extract_video_url(result):
    url = result.get("content", {}).get("video_url", "")
    if not url:
        inner = result.get("data", {})
        if isinstance(inner, dict):
            url = inner.get("content", {}).get("video_url", "")
    return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--output_dir", default="/mnt/pfs/houhaotian/prohibited_parking_videos")
    parser.add_argument("--ref_image", default=None)
    parser.add_argument("--tags", default=None, help="逗号分隔的标签（默认全部PP标签）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_gv", os.path.join(os.path.dirname(os.path.abspath(__file__)), "16_generate_video.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    TAG_PROMPTS = mod.TAG_PROMPTS

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else ALL_PP_TAGS
    suffix = "_ref" if args.ref_image else ""
    ref_label = " [带参考图]" if args.ref_image else ""
    ref_b64 = load_ref_image(args.ref_image) if args.ref_image else None

    print(f"\n{'='*60}")
    print(f"  严禁靠边场景 — 健壮生成{ref_label}")
    print(f"  标签数: {len(tags)}")
    print(f"{'='*60}\n")

    ok, fail, skip = 0, 0, 0

    for i, tag in enumerate(tags, 1):
        mp4_path = os.path.join(args.output_dir, f"{tag}_seedance2{suffix}.mp4")
        json_path = os.path.join(args.output_dir, f"{tag}_seedance2{suffix}.json")

        if os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 500_000:
            sz = os.path.getsize(mp4_path) / 1024 / 1024
            print(f"[{i}/{len(tags)}] {tag} — 已存在 ({sz:.1f}MB)，跳过")
            skip += 1
            continue

        print(f"\n[{i}/{len(tags)}] {tag}{ref_label}")

        if tag not in TAG_PROMPTS:
            print(f"  跳过: 无 prompt 配置")
            fail += 1
            continue

        prompt = TAG_PROMPTS[tag]["prompt"]

        try:
            task_id = submit_task(args.api_key, prompt, ref_b64)
            result = poll_until_done(args.api_key, task_id)
            if not result:
                fail += 1
                continue

            video_url = extract_video_url(result)
            if not video_url:
                print(f"  无 video_url")
                fail += 1
                continue

            if download_video(video_url, mp4_path):
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "task_id": task_id, "tag": tag, "model": MODEL,
                        "prompt": prompt, "ref_image": args.ref_image,
                        "duration": 10, "resolution": "720p", "result": result,
                    }, f, ensure_ascii=False, indent=2)
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  异常: {e}")
            fail += 1

    print(f"\n{'='*60}")
    print(f"  完成{ref_label}! 成功: {ok}, 跳过: {skip}, 失败: {fail}")
    print(f"  可用总数: {ok + skip}/{len(tags)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
