#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双 Agent 视频打标脚本（OpenAI 兼容 API）。

流程:
1. Annotator Agent: 根据分类 prompt 对 20 秒视频打标签
2. Judge Agent: 重新看同一视频，对初稿做 accepted / corrected / rejected 判定

适用场景:
- 使用 Qwen3.5 OpenAI 兼容服务进行多轮多模态审核
- 产出可直接用于后续人工抽查或冷启动数据筛选的结构化结果
"""

import argparse
import base64
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import requests


NO_PROXY = {"http": None, "https": None}

DEFAULT_ANNOTATOR_USER_PROMPT = (
    "Below are frames extracted at 2 fps from a 20-second ego-vehicle driving video, "
    "shown in chronological order. Carefully analyze these frames and output the "
    "driving behavior segments strictly following the system prompt. "
    "Output ONLY the final segment lines. Do NOT include analysis, reasoning, chain-of-thought, "
    "or any explanatory text."
)

DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_RETRIES = 3
DEFAULT_JUDGE_MAX_TOKENS = 768

SEGMENT_PATTERN = re.compile(
    r"<driving_maneuver>([^<]+)</driving_maneuver>\s+"
    r"from\s+<start_time>([\d.]+)</start_time>\s+"
    r"to\s+<end_time>([\d.]+)</end_time>\s+seconds",
    re.IGNORECASE,
)


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


def extract_frames_from_video(
    video_path: str,
    sample_fps: float = 2.0,
    max_frames: int = 40,
    resolution: Tuple[int, int] = (256, 256),
) -> List[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

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
            if not ok:
                raise ValueError(f"JPEG 编码失败: {video_path}")
            frames_b64.append(base64.b64encode(buffer).decode("utf-8"))

        frame_idx += 1

    cap.release()
    return frames_b64


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_annotator_system_prompt(base_prompt: str) -> str:
    return (
        "You are a strict labeling agent.\n"
        "You MUST follow the task specification below, but you must output ONLY the final label segments.\n"
        "Do NOT output any reasoning, analysis, observations, bullet points, markdown, or chain-of-thought.\n"
        "Do NOT explain your answer.\n"
        "Do NOT output placeholders such as LABEL, XXX, or YYY.\n"
        "If no valid label applies, output only the full-duration not_applicable segment in the exact required format.\n"
        "Return plain text only.\n\n"
        "=== TASK SPECIFICATION ===\n"
        f"{base_prompt}"
    )


def extract_target_labels(prompt_text: str) -> List[str]:
    labels: List[str] = []
    patterns = [
        re.compile(r"^\s{2,}([A-Za-z][A-Za-z0-9_]+)\s+\(", re.MULTILINE),
        re.compile(r"^\s*\d+\.\s+([A-Za-z][A-Za-z0-9_]+)\s+\(", re.MULTILINE),
    ]

    for pattern in patterns:
        for match in pattern.findall(prompt_text):
            if match not in labels:
                labels.append(match)

    if "not_applicable" not in labels:
        labels.append("not_applicable")
    return labels


def build_image_content(frames_b64: List[str]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        }
        for b64 in frames_b64
    ]


def extract_text(response_json: Optional[Dict[str, Any]]) -> str:
    if not response_json:
        return ""
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts).strip()
    return ""


def call_chat_completion(
    *,
    api_base: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    request_timeout: int,
    enable_thinking: bool = False,
    response_format: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    ensure_no_proxy(api_base)
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    try:
        response = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            timeout=(10, request_timeout),
            proxies=NO_PROXY,
        )
        response.raise_for_status()
        return extract_text(response.json()), None
    except Exception as exc:
        return "", str(exc)


def call_text_with_retries(
    *,
    api_base: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    request_timeout: int,
    max_retries: int,
    enable_thinking: bool = False,
    response_format: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    last_error: Optional[str] = None
    last_text = ""
    for _ in range(max_retries):
        text, error = call_chat_completion(
            api_base=api_base,
            model_name=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            request_timeout=request_timeout,
            enable_thinking=enable_thinking,
            response_format=response_format,
        )
        if error:
            last_error = error
            continue
        if text.strip():
            return text, None
        last_text = text
        last_error = "empty_response"

    return last_text, last_error


def parse_segments(raw_text: str, allowed_labels: List[str]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for match in SEGMENT_PATTERN.finditer(raw_text):
        label = match.group(1).strip()
        start = float(match.group(2))
        end = float(match.group(3))

        if label not in allowed_labels:
            continue
        if start >= end:
            continue

        segments.append(
            {
                "label": label,
                "start": round(max(0.0, min(start, 20.0)), 1),
                "end": round(max(0.0, min(end, 20.0)), 1),
            }
        )
    return segments


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def build_judge_prompt(
    draft_output: str,
    allowed_labels: List[str],
    draft_segments: Optional[List[Dict[str, Any]]] = None,
    custom_template: Optional[str] = None,
) -> str:
    allowed_block = "\n".join(f"- {label}" for label in allowed_labels)
    if draft_segments:
        draft_block = "\n".join(
            f"<driving_maneuver>{seg['label']}</driving_maneuver> "
            f"from <start_time>{seg['start']:.1f}</start_time> "
            f"to <end_time>{seg['end']:.1f}</end_time> seconds."
            for seg in draft_segments
        )
    else:
        draft_block = (
            "[NO_VALID_DRAFT_SEGMENTS]\n"
            "The annotator did not return any valid final label segments. "
            "Treat the draft as invalid and independently review the video."
        )

    if custom_template:
        return (
            custom_template.replace("{{allowed_labels}}", allowed_block)
            .replace("{{draft_output}}", draft_block)
        )

    return f"""
You are a strict senior reviewer for autonomous-driving scene tags.

You must independently re-check the same video and review the annotator draft.
Do NOT trust the draft unless it is clearly supported by the video.
You must output ONLY valid JSON. Do NOT output any analysis before or after the JSON.
Do NOT output markdown fences. Do NOT output placeholder labels such as LABEL, XXX, or YYY.

Allowed labels:
{allowed_block}

Annotator draft:
{draft_block}

Your task:
1. Check whether the annotator draft is fully supported by the video.
2. If correct, return accepted.
3. If incorrect but fixable, return corrected and provide the corrected final output.
4. If the draft is unreliable / unsupported / malformed, return rejected.

Important rules:
- accepted: final_output should be the final segment text to keep.
- corrected: final_output must be the corrected segment text using ONLY the allowed labels.
- rejected: final_output should be an empty string.
- If the correct answer is "not_applicable", use corrected with a full-duration not_applicable segment.
- final_output MUST use a REAL label from the allowed labels list. Never use placeholder text like LABEL.
- Keep time format exactly like:
  <driving_maneuver>Real_Label_Name</driving_maneuver> from <start_time>0.0</start_time> to <end_time>20.0</end_time> seconds.

Output STRICT JSON only:
{{
  "verdict": "accepted | corrected | rejected",
  "final_output": "string",
  "reason": ["short reason 1", "short reason 2"]
}}
""".strip()


def parse_judge_output(raw_text: str) -> Dict[str, Any]:
    cleaned = strip_code_fences(raw_text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)

    try:
        data = json.loads(cleaned)
        verdict = data.get("verdict", "").strip().lower()
        if verdict not in {"accepted", "corrected", "rejected"}:
            raise ValueError(f"非法 judge verdict: {verdict}")

        reason = data.get("reason", [])
        if isinstance(reason, str):
            reason = [reason]
        elif not isinstance(reason, list):
            reason = [str(reason)]

        return {
            "verdict": verdict,
            "final_output": data.get("final_output", "") or "",
            "reason": reason,
        }
    except Exception:
        lowered = cleaned.lower()
        verdict = None
        for candidate in ("accepted", "corrected", "rejected"):
            if candidate in lowered:
                verdict = candidate
                break
        if verdict is None:
            raise

        seg_match = re.search(
            r"(<driving_maneuver>.*?</driving_maneuver>.*?seconds\.)",
            cleaned,
            re.DOTALL | re.IGNORECASE,
        )
        final_output = seg_match.group(1).strip() if seg_match else ""
        reason = [cleaned[:300].replace("\n", " ")]
        return {
            "verdict": verdict,
            "final_output": final_output,
            "reason": reason,
        }


def load_video_paths(args: argparse.Namespace) -> List[str]:
    if args.video_path:
        return [args.video_path]
    if args.video_dir:
        return sorted(str(p) for p in Path(args.video_dir).glob("**/*.mp4") if p.is_file())
    if args.video_list:
        with open(args.video_list, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise ValueError("必须指定 --video_path / --video_dir / --video_list")


def process_single_video(
    video_path: str,
    annotator_prompt: str,
    judge_template: Optional[str],
    allowed_labels: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    frames_b64 = extract_frames_from_video(
        video_path,
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
        resolution=(args.resolution, args.resolution),
    )
    if not frames_b64:
        raise ValueError(f"未提取到任何帧: {video_path}")

    image_content = build_image_content(frames_b64)

    annotator_messages = [
        {"role": "system", "content": build_annotator_system_prompt(annotator_prompt)},
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": DEFAULT_ANNOTATOR_USER_PROMPT}],
        },
    ]
    annotator_text, annotator_error = call_chat_completion(
        api_base=args.api_base,
        model_name=args.model_name,
        messages=annotator_messages,
        max_tokens=args.max_tokens,
        temperature=args.annotator_temperature,
        request_timeout=args.request_timeout,
        enable_thinking=args.enable_thinking,
        response_format=None,
    )
    if annotator_error or not annotator_text.strip():
        annotator_text, annotator_error = call_text_with_retries(
            api_base=args.api_base,
            model_name=args.model_name,
            messages=annotator_messages,
            max_tokens=args.max_tokens,
            temperature=args.annotator_temperature,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            enable_thinking=args.enable_thinking,
            response_format=None,
        )
    if annotator_error or not annotator_text.strip():
        raise RuntimeError(f"annotator 调用失败: {annotator_error}")

    annotator_segments = parse_segments(annotator_text, allowed_labels)

    judge_prompt = build_judge_prompt(
        draft_output=annotator_text,
        draft_segments=annotator_segments,
        allowed_labels=allowed_labels,
        custom_template=judge_template,
    )
    judge_messages = [
        {"role": "system", "content": "You are a strict multimodal judge."},
        {
            "role": "user",
            "content": image_content + [{"type": "text", "text": judge_prompt}],
        },
    ]
    judge_result = None
    judge_text = ""
    judge_error = None
    last_parse_error = None
    for _ in range(args.max_retries):
        judge_text, judge_error = call_text_with_retries(
            api_base=args.api_base,
            model_name=args.model_name,
            messages=judge_messages,
            max_tokens=args.judge_max_tokens,
            temperature=args.judge_temperature,
            request_timeout=args.request_timeout,
            max_retries=1,
            enable_thinking=args.enable_thinking,
            response_format=None if args.disable_judge_json_mode else {"type": "json_object"},
        )
        if judge_error or not judge_text.strip():
            last_parse_error = judge_error or "empty_judge_response"
            continue
        try:
            judge_result = parse_judge_output(judge_text)
            last_parse_error = None
            break
        except Exception as exc:
            last_parse_error = str(exc)
            continue

    if judge_result is None:
        raise RuntimeError(f"judge 调用失败: {last_parse_error}")

    verdict = judge_result["verdict"]

    if verdict == "accepted":
        final_output = judge_result["final_output"] or annotator_text
    elif verdict == "corrected":
        final_output = judge_result["final_output"]
    else:
        final_output = ""

    final_segments = parse_segments(final_output, allowed_labels) if final_output else []

    return {
        "video_path": video_path,
        "frame_count": len(frames_b64),
        "raw_output": final_output,
        "segments": final_segments,
        "annotator_raw_output": annotator_text,
        "annotator_segments": annotator_segments,
        "judge_raw_output": judge_text,
        "judge_verdict": verdict,
        "judge_reason": judge_result["reason"],
        "final_output": final_output,
        "final_segments": final_segments,
        "accepted_for_bootstrap": verdict in {"accepted", "corrected"} and len(final_segments) > 0,
    }


def batch_run(
    video_paths: List[str],
    annotator_prompt: str,
    judge_template: Optional[str],
    allowed_labels: List[str],
    args: argparse.Namespace,
) -> None:
    results: List[Dict[str, Any]] = []
    processed = set()

    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed = {item["video_path"] for item in results}
            print(f"加载已有结果: {len(results)} 条，支持断点续跑")
        except Exception:
            results = []
            processed = set()

    if args.max_videos:
        video_paths = video_paths[: args.max_videos]

    pending = [path for path in video_paths if path not in processed]
    print(f"总视频数: {len(video_paths)} | 已处理: {len(processed)} | 待处理: {len(pending)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    def _save() -> None:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _run_one(path: str) -> Dict[str, Any]:
        try:
            return process_single_video(path, annotator_prompt, judge_template, allowed_labels, args)
        except Exception as exc:
            traceback.print_exc()
            return {
                "video_path": path,
                "error": str(exc),
                "raw_output": "",
                "segments": [],
                "annotator_raw_output": "",
                "annotator_segments": [],
                "judge_raw_output": "",
                "judge_verdict": "rejected",
                "judge_reason": [str(exc)],
                "final_output": "",
                "final_segments": [],
                "accepted_for_bootstrap": False,
            }

    if args.concurrency <= 1:
        for idx, path in enumerate(pending, start=1):
            print(f"[{idx}/{len(pending)}] {path}")
            results.append(_run_one(path))
            _save()
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_run_one, path): path for path in pending}
            for idx, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                _save()
                print(
                    f"[{idx}/{len(pending)}] "
                    f"{Path(result['video_path']).name} "
                    f"-> {result.get('judge_verdict', 'unknown')}"
                )

    accepted = sum(1 for item in results if item.get("accepted_for_bootstrap"))
    corrected = sum(1 for item in results if item.get("judge_verdict") == "corrected")
    rejected = sum(1 for item in results if item.get("judge_verdict") == "rejected")
    print(
        f"完成: {len(results)} 条 | accepted_for_bootstrap={accepted} | "
        f"corrected={corrected} | rejected={rejected}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双 Agent 视频打标（Annotator + Judge）")
    parser.add_argument("--api_base", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen3.5-397B-A17B")
    parser.add_argument("--prompt_file", type=str, required=True, help="Annotator prompt")
    parser.add_argument("--judge_prompt_file", type=str, default="", help="可选，自定义 Judge 模板")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str)
    input_group.add_argument("--video_dir", type=str)
    input_group.add_argument("--video_list", type=str)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--sample_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--judge_max_tokens", type=int, default=DEFAULT_JUDGE_MAX_TOKENS)
    parser.add_argument("--request_timeout", type=int, default=300)
    parser.add_argument("--max_retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--enable_thinking", action="store_true", help="启用模型显式 thinking；默认关闭以稳定结构化输出")
    parser.add_argument("--disable_judge_json_mode", action="store_true", help="禁用 judge 的 json_object 模式")
    parser.add_argument("--annotator_temperature", type=float, default=0.0)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotator_prompt = load_prompt(args.prompt_file)
    judge_template = load_prompt(args.judge_prompt_file) if args.judge_prompt_file else None
    allowed_labels = extract_target_labels(annotator_prompt)
    video_paths = load_video_paths(args)

    print(f"模型: {args.model_name}")
    print(f"视频数: {len(video_paths)}")
    print(f"标签数: {len(allowed_labels)}")
    batch_run(video_paths, annotator_prompt, judge_template, allowed_labels, args)


if __name__ == "__main__":
    main()
