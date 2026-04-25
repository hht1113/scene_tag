#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-B 场景分类脚本：基于 Round-A 结构化感知 + 视频，进行 7 组标签分类。

核心策略（最大化准确率）：
1. 读取 Round-A JSON，用规则预过滤出候选类别（通常 2-4 个），跳过无关类别
2. 对每个候选类别，用现有 prompt_txt/01-07 的详细分类 prompt + Round-A JSON 作为上下文 + 视频帧
3. 合并多标签结果，输出 JSONL

用法:
    python 19_round_b_classification.py \
        --api_base http://NEW_IP:PORT/v1 \
        --round_a_file scene_tag/results/round_a_results_v2.jsonl \
        --output scene_tag/results/round_b_results.jsonl \
        --concurrency 4 --max_videos 50
"""

import argparse
import base64
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import cv2
import requests

# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------
NO_PROXY = {"http": None, "https": None}

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

# ---------------------------------------------------------------------------
# Category prompt loading
# ---------------------------------------------------------------------------
PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_txt")

CATEGORY_FILES = {
    "01_DynamicInteraction": "01_DynamicInteraction.txt",
    "02_TrafficLight": "02_TrafficLight.txt",
    "03_StartStop": "03_StartStop.txt",
    "04_Intersection": "04_Intersection.txt",
    "05_LaneCruising": "05_LaneCruising.txt",
    "06_LaneChange": "06_LaneChange.txt",
    "07_IntersectionInteraction": "07_IntersectionInteraction.txt",
}

def load_category_prompts() -> Dict[str, str]:
    prompts = {}
    for cat_key, fname in CATEGORY_FILES.items():
        fpath = os.path.join(PROMPT_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                prompts[cat_key] = f.read()
    return prompts

# ---------------------------------------------------------------------------
# Round-A based pre-filtering
# ---------------------------------------------------------------------------

def prefilter_categories(round_a: Dict[str, Any]) -> Set[str]:
    """Use Round-A JSON to determine which categories are worth running.
    Returns a set of category keys to run."""
    candidates = set()

    ego = round_a.get("ego_motion", {})
    segs = ego.get("segments", [])
    avoidance = ego.get("avoidance_maneuver", {})
    compound = ego.get("compound_maneuver_guess", {})
    following = ego.get("following_behavior", {})

    lane = round_a.get("lane_and_markings", {})
    lce = lane.get("lane_change_evidence", {})

    road = round_a.get("road_layout", {})
    intersection = road.get("intersection_topology_guess", "none")
    geo_cues = road.get("road_geometry_cues", [])

    tc = round_a.get("traffic_control", {})
    tl = tc.get("traffic_light", {})
    any_light = tl.get("any_traffic_light_visible", False)

    ta = round_a.get("traffic_agents", {})
    agents = ta.get("agents", [])

    sc = round_a.get("scene_context", {})
    flow = sc.get("traffic_flow_state", {}).get("overall", "unknown")
    queue = sc.get("queue_ahead", {})

    has_interaction = False
    has_cut_in = False
    has_cross_path = False
    has_lead_brake = False
    agent_categories = set()

    for a in agents:
        interactions = a.get("interaction_with_ego", [])
        cat = a.get("category", "")
        agent_categories.add(cat)
        if any(i not in ("none_apparent",) for i in interactions):
            has_interaction = True
        if "cut_in" in interactions:
            has_cut_in = True
        if "cross_path" in interactions or "pedestrian_near_crosswalk" in interactions:
            has_cross_path = True
        if "lead_vehicle" in interactions:
            braking = a.get("lead_vehicle_braking_detail", {})
            if braking.get("braking_intensity") in ("sudden_hard", "gradual"):
                has_lead_brake = True

    has_stop_start = any(
        s.get("longitudinal") in ("stop", "start_from_stop", "emergency_stop", "decelerate")
        for s in segs
    )
    has_lane_change = any(
        s.get("lateral") in (
            "lane_change_left", "lane_change_right",
            "borrow_oncoming_lane_left", "borrow_oncoming_lane_right",
            "cross_line_bypass_left", "cross_line_bypass_right",
            "nudge_left", "nudge_right",
        )
        for s in segs
    ) or lce.get("lane_change_in_progress_guess", False)

    is_intersection = intersection not in ("none", "unclear")
    is_intersection_approach = "intersection_approach" in geo_cues or "intersection_interior" in geo_cues

    # --- Pre-filter rules ---

    # 01_DynamicInteraction: any cut-in, cross-path, lead brake, or SOD
    if has_cut_in or has_cross_path or has_lead_brake:
        candidates.add("01_DynamicInteraction")
    if "static_small_object" in agent_categories:
        candidates.add("01_DynamicInteraction")

    # 02_TrafficLight: traffic light visible
    if any_light:
        candidates.add("02_TrafficLight")

    # 03_StartStop: stop/start transitions
    if has_stop_start:
        candidates.add("03_StartStop")

    # 04_Intersection: intersection geometry
    if is_intersection or is_intersection_approach:
        candidates.add("04_Intersection")

    # 05_LaneCruising: always a candidate (most common)
    candidates.add("05_LaneCruising")

    # 06_LaneChange: any lateral movement
    if has_lane_change:
        candidates.add("06_LaneChange")
    if avoidance.get("strategy") not in ("none", "unknown", None):
        candidates.add("06_LaneChange")
    if compound.get("type") in ("overtake",):
        candidates.add("06_LaneChange")

    # 07_IntersectionInteraction: intersection + agent interaction
    if (is_intersection or is_intersection_approach) and has_interaction:
        candidates.add("07_IntersectionInteraction")
    if (is_intersection or is_intersection_approach) and has_cross_path:
        candidates.add("07_IntersectionInteraction")

    return candidates

# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    sample_fps: float = 2.0,
    max_frames: int = 40,
    resolution: Tuple[int, int] = (640, 640),
) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0

    frame_interval = max(1, int(round(fps / sample_fps)))
    frames_b64: List[str] = []
    frame_idx = 0

    while len(frames_b64) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
        frame_idx += 1

    cap.release()
    return frames_b64

# ---------------------------------------------------------------------------
# API calling
# ---------------------------------------------------------------------------

SEGMENT_PATTERN = re.compile(
    r"<driving_maneuver>([^<]+)</driving_maneuver>\s+"
    r"from\s+<start_time>([\d.]+)</start_time>\s+"
    r"to\s+<end_time>([\d.]+)</end_time>\s+seconds",
    re.IGNORECASE,
)

def call_api(
    api_base: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    request_timeout: int = 600,
) -> Tuple[str, Optional[str]]:
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
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return content.strip(), None
    except requests.exceptions.Timeout:
        return "", f"timeout ({request_timeout}s)"
    except requests.exceptions.HTTPError as e:
        return "", f"HTTP {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return "", str(e)


def call_with_retry(api_base, model_name, messages, temperature=0.3,
                    max_tokens=2048, request_timeout=600, max_retries=3):
    last_err = None
    for attempt in range(max_retries):
        text, err = call_api(api_base, model_name, messages, temperature, max_tokens, request_timeout)
        if err:
            last_err = err
            time.sleep(min(2 ** attempt * 3, 30))
            continue
        if text.strip():
            return text, None
        last_err = "empty response"
    return "", last_err

# ---------------------------------------------------------------------------
# Parse classification output
# ---------------------------------------------------------------------------

def parse_segments(raw_text: str) -> List[Dict[str, Any]]:
    segments = []
    for match in SEGMENT_PATTERN.finditer(raw_text):
        label = match.group(1).strip()
        start = float(match.group(2))
        end = float(match.group(3))
        if label == "not_applicable":
            continue
        if start >= end:
            continue
        segments.append({
            "label": label,
            "start": round(max(0.0, min(start, 20.0)), 1),
            "end": round(max(0.0, min(end, 20.0)), 1),
        })
    return segments

# ---------------------------------------------------------------------------
# Build Round-B messages
# ---------------------------------------------------------------------------

def build_round_b_messages(
    category_prompt: str,
    round_a_json: Dict[str, Any],
    frames_b64: List[str],
) -> List[Dict[str, Any]]:
    """Build messages for Round-B: category prompt + Round-A context + video."""

    round_a_summary = json.dumps(round_a_json, ensure_ascii=False, separators=(",", ":"))
    if len(round_a_summary) > 6000:
        compact = {
            "ego_motion": round_a_json.get("ego_motion", {}),
            "road_layout": {
                k: round_a_json.get("road_layout", {}).get(k)
                for k in ["road_type_hints", "intersection_topology_guess",
                           "ego_maneuver_slot_guess", "road_geometry_cues",
                           "waiting_zone", "roadside_facilities", "road_width_impression"]
            },
            "traffic_control": round_a_json.get("traffic_control", {}),
            "traffic_agents": round_a_json.get("traffic_agents", {}),
            "scene_context": {
                k: round_a_json.get("scene_context", {}).get(k)
                for k in ["traffic_flow_state", "queue_ahead", "time_of_day_guess",
                           "weather_visibility"]
            },
        }
        round_a_summary = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))

    system_text = (
        "You are a strict labeling agent.\n"
        "You MUST follow the task specification below, but you must output ONLY the final label segments.\n"
        "Do NOT output any reasoning, analysis, observations, bullet points, markdown, or chain-of-thought.\n"
        "Do NOT explain your answer.\n"
        "Do NOT output placeholders such as LABEL, XXX, or YYY.\n"
        "If no valid label applies, output only the full-duration not_applicable segment in the exact required format.\n"
        "Return plain text only.\n\n"
        "=== TASK SPECIFICATION ===\n"
        f"{category_prompt}\n\n"
        "=== ROUND-A STRUCTURED PERCEPTION (for reference, do NOT blindly trust — verify against the video) ===\n"
        f"{round_a_summary}"
    )

    image_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in frames_b64
    ]

    user_text = (
        "Below are frames extracted at 2 fps from a 20-second ego-vehicle driving video, "
        "shown in chronological order. "
        "A structured perception analysis (Round-A) is provided in the system prompt for reference. "
        "Use it as context but VERIFY against the actual video frames. "
        "Output ONLY the final segment lines. Do NOT include analysis or reasoning."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": image_content + [{"type": "text", "text": user_text}]},
    ]

    return messages

# ---------------------------------------------------------------------------
# Process one video
# ---------------------------------------------------------------------------

def process_one_video(
    round_a_entry: Dict[str, Any],
    category_prompts: Dict[str, str],
    api_base: str,
    model_name: str,
    resolution: Tuple[int, int],
    temperature: float,
    max_retries: int,
    request_timeout: int,
) -> Dict[str, Any]:
    t0 = time.time()
    video_path = round_a_entry["video_path"]
    clip_id = round_a_entry["clip_id"]
    round_a = round_a_entry.get("round_a")

    if not round_a:
        return {
            "clip_id": clip_id,
            "video_path": video_path,
            "labels": [],
            "categories_checked": [],
            "error": "no round_a data",
            "elapsed_sec": round(time.time() - t0, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    candidates = prefilter_categories(round_a)

    frames_b64 = extract_frames(video_path, resolution=resolution)
    if not frames_b64:
        return {
            "clip_id": clip_id,
            "video_path": video_path,
            "labels": [],
            "categories_checked": list(candidates),
            "error": f"no frames extracted from {video_path}",
            "elapsed_sec": round(time.time() - t0, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    all_labels = []
    categories_checked = []
    errors = []

    for cat_key in sorted(candidates):
        if cat_key not in category_prompts:
            continue
        categories_checked.append(cat_key)

        messages = build_round_b_messages(
            category_prompts[cat_key], round_a, frames_b64
        )

        raw_text, err = call_with_retry(
            api_base, model_name, messages,
            temperature=temperature,
            max_retries=max_retries,
            request_timeout=request_timeout,
        )

        if err:
            errors.append(f"{cat_key}: {err}")
            continue

        segments = parse_segments(raw_text)
        for seg in segments:
            seg["source_category"] = cat_key
        all_labels.extend(segments)

    deduped = _dedup_labels(all_labels)

    return {
        "clip_id": clip_id,
        "video_path": video_path,
        "labels": deduped,
        "num_labels": len(deduped),
        "categories_checked": categories_checked,
        "num_categories_checked": len(categories_checked),
        "prefilter_skipped": sorted(set(CATEGORY_FILES.keys()) - candidates),
        "error": "; ".join(errors) if errors else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _dedup_labels(labels: List[Dict]) -> List[Dict]:
    """Remove exact duplicate labels (same label + overlapping time)."""
    seen = set()
    deduped = []
    for seg in labels:
        key = (seg["label"], seg["start"], seg["end"])
        if key not in seen:
            seen.add(key)
            deduped.append(seg)
    return deduped

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def load_round_a_entries(round_a_file: str, max_videos: Optional[int] = None) -> List[Dict]:
    entries = []
    with open(round_a_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("round_a") is not None:
                    entries.append(d)
            except json.JSONDecodeError:
                continue
    if max_videos:
        entries = entries[:max_videos]
    return entries


def load_processed_set(output_path: str) -> set:
    processed = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                vp = obj.get("video_path", "")
                if vp:
                    processed.add(vp)
            except json.JSONDecodeError:
                continue
    return processed


def run_batch(args: argparse.Namespace) -> None:
    category_prompts = load_category_prompts()
    print(f"Loaded {len(category_prompts)} category prompts", flush=True)

    entries = load_round_a_entries(args.round_a_file, args.max_videos)
    print(f"Round-A entries loaded: {len(entries)}", flush=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    processed = load_processed_set(args.output)
    pending = [e for e in entries if e["video_path"] not in processed]
    print(f"Already processed: {len(processed)}, pending: {len(pending)}", flush=True)

    if not pending:
        print("All entries already processed.", flush=True)
        return

    resolution = (args.resolution, args.resolution)
    success = 0
    failed = 0
    total_labels = 0
    lock = __import__("threading").Lock()

    def _process(entry: Dict) -> Optional[Dict]:
        try:
            return process_one_video(
                entry, category_prompts, args.api_base, args.model_name,
                resolution, args.temperature, args.max_retries, args.request_timeout,
            )
        except Exception as e:
            traceback.print_exc()
            return {
                "clip_id": entry.get("clip_id", ""),
                "video_path": entry.get("video_path", ""),
                "labels": [],
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def _append_result(result: Dict) -> None:
        nonlocal success, failed, total_labels
        with lock:
            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            n_labels = len(result.get("labels", []))
            total_labels += n_labels
            if result.get("error"):
                failed += 1
            else:
                success += 1
            total_done = success + failed
            if total_done % 10 == 0 or total_done <= 5:
                print(
                    f"  [{total_done}/{len(pending)}] success={success} failed={failed} "
                    f"labels_found={total_labels} avg_cats={sum(len(r.get('categories_checked',[])) for r in [result])/1:.0f}",
                    flush=True,
                )

    print(f"\nStarting Round-B classification (concurrency={args.concurrency})...\n", flush=True)
    start_time = time.time()

    if args.concurrency <= 1:
        for entry in pending:
            result = _process(entry)
            if result:
                _append_result(result)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_process, e): e for e in pending}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    _append_result(result)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}", flush=True)
    print(f"Round-B classification complete.", flush=True)
    print(f"  Success: {success}", flush=True)
    print(f"  Failed:  {failed}", flush=True)
    print(f"  Total labels found: {total_labels}", flush=True)
    print(f"  Time:    {elapsed:.0f}s ({elapsed/max(success+failed,1):.1f}s/video)", flush=True)
    print(f"  Output:  {args.output}", flush=True)
    print(f"{'=' * 60}", flush=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round-B classification based on Round-A perception + video",
    )
    parser.add_argument("--api_base", type=str, default="http://10.10.64.144:2754/v1")
    parser.add_argument("--model_name", type=str, default="Qwen3.5-397B-A17B")
    parser.add_argument("--round_a_file", type=str, required=True,
                        help="Round-A results JSONL file")
    parser.add_argument("--output", type=str, default="scene_tag/results/round_b_results.jsonl")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--request_timeout", type=int, default=600)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Round-B Classification", flush=True)
    print(f"  API:         {args.api_base}", flush=True)
    print(f"  Model:       {args.model_name}", flush=True)
    print(f"  Round-A:     {args.round_a_file}", flush=True)
    print(f"  Resolution:  {args.resolution}x{args.resolution}", flush=True)
    print(f"  Output:      {args.output}", flush=True)
    print(flush=True)
    run_batch(args)


if __name__ == "__main__":
    main()
