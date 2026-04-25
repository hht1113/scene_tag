#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-A 结构化感知提取脚本（v1.5 Schema）

通过 Qwen3.5-397B API 对驾驶视频进行 6 模块结构化感知提取，
禁止输出任何场景分类标签（01-07），仅做感知层面的客观描述。

用法:
    # 50 条 pilot
    python 17_round_a_distillation.py \
        --api_base http://10.10.64.144:2754/v1 \
        --video_dir /mnt/pfs/sampled_videos_5k/slices_20s/ \
        --output scene_tag/results/round_a_results.jsonl \
        --max_videos 50 --concurrency 4

    # 全量
    python 17_round_a_distillation.py \
        --api_base http://10.10.64.144:2754/v1 \
        --pool_json /root/workspace/LLaMA-Factory/data/mining_pool_all.json \
        --output scene_tag/results/round_a_results.jsonl \
        --concurrency 8
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import requests

# ---------------------------------------------------------------------------
# Proxy bypass
# ---------------------------------------------------------------------------
def ensure_no_proxy(api_base: str) -> None:
    host = urlparse(api_base).hostname
    if not host:
        return
    for key in ("no_proxy", "NO_PROXY"):
        cur = os.environ.get(key, "")
        entries = [item.strip() for item in cur.split(",") if item.strip()]
        if host not in entries:
            entries.append(host)
            os.environ[key] = ",".join(entries)


NO_PROXY = {"http": None, "https": None}

# ---------------------------------------------------------------------------
# Round-A System Prompt (v1.5)
# ---------------------------------------------------------------------------

def load_round_a_prompt(prompt_path: str = None) -> Tuple[str, str]:
    """Load Round-A v2 prompt from external file, or use fallback defaults.
    Returns (system_prompt, user_template)."""
    if prompt_path and os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        sys_match = re.search(
            r"={10,}\nSYSTEM\n={10,}\n(.*?)(?=\n={10,}\nUSER\n={10,})",
            content, re.DOTALL
        )
        user_match = re.search(
            r"={10,}\nUSER\n={10,}\n(.*?)(?=\n={10,}\n(?:（可选）|结束))",
            content, re.DOTALL
        )

        system_prompt = sys_match.group(1).strip() if sys_match else ""
        user_template = user_match.group(1).strip() if user_match else ""

        if system_prompt and user_template:
            return system_prompt, user_template
        print(f"WARNING: Failed to parse prompt file {prompt_path}, using fallback", flush=True)

    system_prompt = (
        "你是一名自动驾驶前视/车载相机视频理解专家，任务是对给定时长的一段视频 clip 做客观、可校验的结构化感知描述。\n\n"
        "硬性规则：\n"
        "1. 只依据画面中真实可见的信息；clip 之外的路况、他车意图、未出现在画面中的信号灯变化一律不得编造。\n"
        "2. 所有判断需合理给出 confidence（0–1 浮点数）；看不清的字段用枚举中的 unknown / not_visible / false。\n"
        "3. 时间一律使用相对本 clip 起点的秒数，范围必须在 [0, T]；segments、time_span 不得越界。\n"
        "4. 禁止输出驾驶场景 taxonomy 大类（如 DynamicInteraction、TrafficLight 等 01–07 标签）。\n"
        "5. 最终回复仅包含一个 JSON 对象，不要 Markdown 代码围栏、不要前后解释文字、不要重复 JSON。"
    )
    user_template = "请仅观看该视频，并输出一个 JSON 对象。"
    return system_prompt, user_template


ROUND_A_PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_vl_round_a_prompt_v2.txt")

# ---------------------------------------------------------------------------
# Frame extraction with timestamps
# ---------------------------------------------------------------------------

def extract_frames_with_timestamps(
    video_path: str,
    sample_fps: float = 2.0,
    max_frames: int = 40,
    resolution: Tuple[int, int] = (640, 640),
) -> Tuple[List[str], List[float]]:
    """Extract frames as base64 JPEG with their timestamps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 10.0

    duration = total_frames / fps
    frame_interval = max(1, int(round(fps / sample_fps)))

    frames_b64: List[str] = []
    timestamps: List[float] = []
    frame_idx = 0

    while len(frames_b64) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                frame_idx += 1
                continue
            frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
            timestamps.append(round(frame_idx / fps, 2))

        frame_idx += 1

    cap.release()
    return frames_b64, timestamps


# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------

def call_api(
    api_base: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    request_timeout: int = 600,
) -> Tuple[str, Optional[str]]:
    """Single API call. Returns (text, error_or_None)."""
    ensure_no_proxy(api_base)
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            timeout=(10, request_timeout),
            proxies=NO_PROXY,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return "", "empty choices"
        msg = choices[0].get("message") or {}
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
        return content.strip(), None
    except requests.exceptions.Timeout:
        return "", f"timeout ({request_timeout}s)"
    except requests.exceptions.HTTPError as e:
        return "", f"HTTP {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return "", str(e)


def call_api_with_retry(
    api_base: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    request_timeout: int = 600,
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    last_err = None
    for attempt in range(max_retries):
        text, err = call_api(api_base, model_name, messages, temperature, max_tokens, request_timeout)
        if err:
            last_err = err
            wait = min(2 ** attempt * 5, 60)
            print(f"  [retry {attempt+1}/{max_retries}] {err}, waiting {wait}s")
            time.sleep(wait)
            continue
        if text.strip():
            return text, None
        last_err = "empty response"
    return "", last_err


# ---------------------------------------------------------------------------
# JSON validation
# ---------------------------------------------------------------------------

REQUIRED_TOP_KEYS = {"ego_motion", "lane_and_markings", "road_layout", "traffic_control", "traffic_agents", "scene_context"}


def try_parse_json(raw: str) -> Tuple[Optional[Dict], str]:
    """Attempt to parse JSON from raw text, stripping thinking blocks and fences."""
    text = raw.strip()

    # Strip Qwen3.5 thinking blocks: <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block (greedy)
    start = text.find("{")
    if start == -1:
        return None, f"No JSON object found in response ({len(raw)} chars)"

    # Use bracket counting to find the matching closing brace
    depth = 0
    in_string = False
    escape_next = False
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        # Fallback: try rfind
        end = text.rfind("}")

    if end > start:
        try:
            return json.loads(text[start : end + 1]), ""
        except json.JSONDecodeError as e:
            return None, f"JSON parse error after extraction: {e}"

    return None, f"No valid JSON object found ({len(raw)} chars)"


def validate_round_a(data: Dict[str, Any], clip_duration: float = 20.0) -> List[str]:
    """Light validation. Returns list of issues (empty = passed)."""
    issues = []

    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        issues.append(f"missing top-level keys: {missing}")

    ego = data.get("ego_motion")
    if isinstance(ego, dict):
        summary = ego.get("summary") or {}
        lon = summary.get("longitudinal", "")
        stop_urg = summary.get("stop_urgency")
        if lon not in ("stop", "decelerate", "emergency_stop") and stop_urg is not None:
            issues.append("stop_urgency should be null when longitudinal is not stop/decelerate/emergency_stop")

        segments = ego.get("segments") or []
        for seg in segments:
            if isinstance(seg, dict):
                s, e = seg.get("start", 0), seg.get("end", 0)
                if s < -0.1 or e > clip_duration + 0.5:
                    issues.append(f"ego_motion segment time out of range: [{s}, {e}]")

    tc = data.get("traffic_control")
    if isinstance(tc, dict):
        tl = tc.get("traffic_light") or {}
        for head in tl.get("heads") or []:
            if isinstance(head, dict):
                if head.get("is_flashing_guess") is True and head.get("flash_color_guess") is None:
                    issues.append(f"head {head.get('id')}: is_flashing=true but flash_color_guess is null")
                if head.get("is_flashing_guess") is False and head.get("flash_color_guess") is not None:
                    issues.append(f"head {head.get('id')}: is_flashing=false but flash_color_guess is not null")

    ta = data.get("traffic_agents")
    if isinstance(ta, dict):
        agent_ids = set()
        for agent in ta.get("agents") or []:
            if isinstance(agent, dict):
                aid = agent.get("id", "")
                if aid in agent_ids:
                    issues.append(f"duplicate agent id: {aid}")
                agent_ids.add(aid)

                interactions = agent.get("interaction_with_ego") or []
                has_cut_in = "cut_in" in interactions
                has_lead = "lead_vehicle" in interactions
                if has_cut_in and agent.get("cut_in_detail") is None:
                    issues.append(f"agent {aid}: has cut_in but missing cut_in_detail")
                if not has_cut_in and agent.get("cut_in_detail") is not None:
                    issues.append(f"agent {aid}: no cut_in but cut_in_detail present")
                if has_lead and agent.get("lead_vehicle_braking_detail") is None:
                    issues.append(f"agent {aid}: has lead_vehicle but missing lead_vehicle_braking_detail")

    return issues


# ---------------------------------------------------------------------------
# Single video processing
# ---------------------------------------------------------------------------

def process_one_video(
    video_path: str,
    api_base: str,
    model_name: str,
    sample_fps: float,
    max_frames: int,
    resolution: Tuple[int, int],
    temperature: float,
    max_tokens: int,
    request_timeout: int,
    max_retries: int,
    clip_duration: float = 20.0,
    system_prompt: str = "",
    user_template: str = "",
    camera_view: str = "front_wide",
) -> Dict[str, Any]:
    """Process a single video through Round-A."""
    t0 = time.time()

    frames_b64, timestamps = extract_frames_with_timestamps(
        video_path, sample_fps, max_frames, resolution
    )
    if not frames_b64:
        raise ValueError(f"No frames extracted: {video_path}")

    image_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]

    sys_text = system_prompt.replace("{{CLIP_DURATION_SEC}}", str(clip_duration))
    usr_text = user_template.replace(
        "{{CLIP_DURATION_SEC}}", str(clip_duration)
    ).replace(
        "{{CAMERA_VIEW}}", camera_view
    ).replace(
        "{{FRAME_TIMESTAMPS_HINT}}",
        f"均匀 {sample_fps}FPS 采样，覆盖 0–{clip_duration}s"
    )

    messages = [
        {"role": "system", "content": sys_text},
        {"role": "user", "content": image_content + [{"type": "text", "text": usr_text}]},
    ]

    raw_text, api_err = call_api_with_retry(
        api_base, model_name, messages, temperature, max_tokens, request_timeout, max_retries
    )

    raw_preview = (raw_text[:500] + "...") if raw_text and len(raw_text) > 500 else (raw_text or "")

    result: Dict[str, Any] = {
        "clip_id": _make_clip_id(video_path),
        "video_path": video_path,
        "teacher_model": model_name,
        "teacher_sampling": {
            "policy": f"uniform_{sample_fps}fps",
            "resolution": f"{resolution[0]}x{resolution[1]}",
            "num_frames": len(frames_b64),
        },
        "frame_timestamps_sec": timestamps,
        "round_a": None,
        "raw_response_length": len(raw_text) if raw_text else 0,
        "raw_response_preview": raw_preview,
        "validation": {"passed": False, "issues": []},
        "error": api_err,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if api_err:
        result["validation"]["issues"] = [f"API error: {api_err}"]
        return result

    parsed, parse_err = try_parse_json(raw_text)
    if parsed is None:
        result["error"] = parse_err
        result["validation"]["issues"] = [parse_err]
        debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "round_a_debug_raw.txt")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w", encoding="utf-8") as df:
            df.write(raw_text)
        print(f"  DEBUG: raw response saved to {debug_path}", flush=True)
        return result

    issues = validate_round_a(parsed, clip_duration)
    result["round_a"] = parsed
    result["validation"]["passed"] = len(issues) == 0
    result["validation"]["issues"] = issues
    result["error"] = None

    return result


def _make_clip_id(video_path: str) -> str:
    p = Path(video_path)
    return f"{p.parent.name}/{p.stem}"


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def load_processed_set(output_path: str) -> set:
    """Load already processed video paths from JSONL output."""
    processed = set()
    if not os.path.exists(output_path):
        return processed
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    vp = obj.get("video_path", "")
                    if vp and obj.get("round_a") is not None:
                        processed.add(vp)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return processed


def load_video_paths(args: argparse.Namespace) -> List[str]:
    if args.pool_json:
        with open(args.pool_json, "r", encoding="utf-8") as f:
            pool = json.load(f)
        paths = []
        for item in pool:
            videos = item.get("videos") or []
            if videos:
                paths.append(videos[0])
        return paths
    if args.video_dir:
        return sorted(str(p) for p in Path(args.video_dir).rglob("*.mp4") if p.is_file())
    if args.video_list:
        with open(args.video_list, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    raise ValueError("Must specify --pool_json, --video_dir, or --video_list")


def run_batch(args: argparse.Namespace) -> None:
    video_paths = load_video_paths(args)
    print(f"Total videos found: {len(video_paths)}", flush=True)

    if args.max_videos:
        video_paths = video_paths[: args.max_videos]
        print(f"Limited to first {args.max_videos} videos", flush=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    processed = load_processed_set(args.output)
    pending = [v for v in video_paths if v not in processed]
    print(f"Already processed: {len(processed)}, pending: {len(pending)}", flush=True)

    if not pending:
        print("All videos already processed.")
        return

    resolution = (args.resolution, args.resolution)
    success = 0
    failed = 0
    lock = __import__("threading").Lock()

    prompt_file = getattr(args, "prompt_file", None) or ROUND_A_PROMPT_FILE
    system_prompt, user_template = load_round_a_prompt(prompt_file)
    print(f"Prompt loaded: {len(system_prompt)} chars system, {len(user_template)} chars user", flush=True)

    camera_view = getattr(args, "camera_view", "front_wide")

    def _process(video_path: str) -> Optional[Dict]:
        try:
            return process_one_video(
                video_path=video_path,
                api_base=args.api_base,
                model_name=args.model_name,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                resolution=resolution,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                system_prompt=system_prompt,
                user_template=user_template,
                camera_view=camera_view,
            )
        except Exception as e:
            traceback.print_exc()
            return {
                "clip_id": _make_clip_id(video_path),
                "video_path": video_path,
                "round_a": None,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _append_result(result: Dict) -> None:
        nonlocal success, failed
        with lock:
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            if result.get("round_a") is not None:
                success += 1
            else:
                failed += 1
            total_done = success + failed
            if total_done % 10 == 0 or total_done <= 5:
                print(f"  [{total_done}/{len(pending)}] success={success} failed={failed}", flush=True)

    print(f"\nStarting Round-A extraction (concurrency={args.concurrency})...\n", flush=True)
    start_time = time.time()

    if args.concurrency <= 1:
        for i, vp in enumerate(pending):
            result = _process(vp)
            if result:
                _append_result(result)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_process, vp): vp for vp in pending}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    _append_result(result)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Round-A extraction complete.")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Time:    {elapsed:.0f}s ({elapsed/max(success+failed,1):.1f}s/video)")
    print(f"  Output:  {args.output}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-A structured perception extraction (v1.5 schema)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--api_base", type=str, default="http://10.10.64.144:2754/v1")
    parser.add_argument("--model_name", type=str, default="Qwen3.5-397B-A17B")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pool_json", type=str, help="mining pool JSON file")
    input_group.add_argument("--video_dir", type=str, help="directory of mp4 files")
    input_group.add_argument("--video_list", type=str, help="text file with one video path per line")

    parser.add_argument("--output", type=str, default="scene_tag/results/round_a_results.jsonl")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)

    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--request_timeout", type=int, default=600)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Round-A prompt txt file (default: scene_tag/qwen3_vl_round_a_prompt_v2.txt)")
    parser.add_argument("--camera_view", type=str, default="front_wide",
                        help="Camera view name for prompt (default: front_wide)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Round-A Distillation v1.5", flush=True)
    print(f"  API:        {args.api_base}", flush=True)
    print(f"  Model:      {args.model_name}", flush=True)
    print(f"  Resolution: {args.resolution}x{args.resolution}", flush=True)
    print(f"  FPS:        {args.sample_fps}", flush=True)
    print(f"  MaxFrames:  {args.max_frames}", flush=True)
    print(f"  Temp:       {args.temperature}", flush=True)
    print(f"  Output:     {args.output}", flush=True)
    print(flush=True)

    run_batch(args)


if __name__ == "__main__":
    main()
