#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将旧版 mining_pool (P00 分类 prompt) 转换为 Round-A v2 结构化感知 prompt 格式。

输入：mining_pool_all.json（旧版，含 12 类分类 prompt）
输出：mining_pool_round_a_v2.json（新版，含 v2 中文 Round-A prompt）

用法:
    python 18_convert_pool_to_round_a.py \
        --input /root/workspace/LLaMA-Factory/data/mining_pool_all.json \
        --prompt_file scene_tag/qwen3_vl_round_a_prompt_v2.txt \
        --output /root/workspace/LLaMA-Factory/data/mining_pool_round_a_v2.json \
        --clip_duration 20.0 \
        --camera_view front_wide
"""

import argparse
import json
import os
import re
import sys


def load_prompt_sections(prompt_path: str) -> tuple:
    """从 v2 prompt 文件中提取 SYSTEM 和 USER 部分。"""
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

    if not sys_match or not user_match:
        print("ERROR: 无法从 prompt 文件中提取 SYSTEM/USER 部分", file=sys.stderr)
        sys.exit(1)

    return sys_match.group(1).strip(), user_match.group(1).strip()


def main():
    parser = argparse.ArgumentParser(description="Convert mining pool to Round-A v2 prompt format")
    parser.add_argument("--input", type=str, required=True, help="Input mining pool JSON")
    parser.add_argument("--prompt_file", type=str, required=True, help="Round-A v2 prompt txt file")
    parser.add_argument("--output", type=str, required=True, help="Output mining pool JSON")
    parser.add_argument("--clip_duration", type=float, default=20.0)
    parser.add_argument("--camera_view", type=str, default="front_wide")
    parser.add_argument("--sample_fps", type=float, default=2.0)
    args = parser.parse_args()

    print(f"Loading prompt: {args.prompt_file}")
    system_prompt, user_template = load_prompt_sections(args.prompt_file)

    system_text = system_prompt.replace("{{CLIP_DURATION_SEC}}", str(args.clip_duration))
    user_text = user_template.replace(
        "{{CLIP_DURATION_SEC}}", str(args.clip_duration)
    ).replace(
        "{{CAMERA_VIEW}}", args.camera_view
    ).replace(
        "{{FRAME_TIMESTAMPS_HINT}}",
        f"均匀 {args.sample_fps}FPS 采样，覆盖 0–{args.clip_duration}s"
    )

    instruction = f"<video>\n{user_text}"

    print(f"System prompt: {len(system_text)} chars")
    print(f"User instruction: {len(instruction)} chars")

    print(f"\nLoading input pool: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        pool = json.load(f)
    print(f"  Total entries: {len(pool)}")

    new_pool = []
    for item in pool:
        videos = item.get("videos", [])
        if not videos:
            continue

        new_item = {
            "instruction": instruction,
            "input": "",
            "output": "",
            "videos": videos,
            "system": system_text,
            "prompt_variant": "teacher_round_a_qwen3vl_v2",
            "round_a_clip_duration_sec": args.clip_duration,
            "round_a_camera_id": args.camera_view,
        }
        new_pool.append(new_item)

    print(f"\nConverted: {len(new_pool)} entries")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(new_pool, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {args.output}")
    print(f"File size: {os.path.getsize(args.output) / 1e6:.1f} MB")

    info_path = os.path.join(os.path.dirname(args.output), "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        dataset_name = os.path.splitext(os.path.basename(args.output))[0]
        info[dataset_name] = {
            "file_name": os.path.basename(args.output),
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "videos": "videos",
                "system": "system",
            },
        }
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"Updated dataset_info.json with key: {dataset_name}")


if __name__ == "__main__":
    main()
