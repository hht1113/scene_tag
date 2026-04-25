#!/usr/bin/env python3
"""
Round-B 标签分类推理 v4 — 多模态版本（图片+文本），支持 thinking 模式。

在 v3 纯文本基础上，支持在 user prompt 中插入示意图，
借助 VLM 的 OCR + 空间理解能力增强场景分类。

Usage:

python round_b_label_infer_v4_multimodal.py \
    --model /mnt/pfs/qwen3.5/Qwen3.5-9B \
    --input /mnt/pfs/chenruize/dataset/round_a_results_v2_final.jsonl \
    --output round_b_results_intersection_v4.json \
    --prompt group_prompts/IntersectionInteraction_v2.txt \
    --images group_prompts/intersection_interaction_vehicle.png \
             group_prompts/intersection_interaction_vru.png \
    --enable-thinking \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --max-tokens 4096 \
    --temperature 0.6 \
    --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

_SECTION_SEP = "## USER ##"


def load_prompt(prompt_path: str) -> tuple[str, str]:
    raw = Path(prompt_path).read_text(encoding="utf-8")
    if _SECTION_SEP not in raw:
        raise ValueError(f"Prompt file must contain '{_SECTION_SEP}' separator: {prompt_path}")
    sys_part, user_part = raw.split(_SECTION_SEP, 1)
    sys_part = sys_part.replace("## SYSTEM ##", "").strip()
    user_part = user_part.strip()
    return sys_part, user_part


def build_user_content(user_template: str, response: dict, image_paths: list[str]) -> list[dict]:
    """Build multimodal user content with images + text (vLLM image_url format)."""
    content_parts: list[dict] = []

    if image_paths:
        content_parts.append({
            "type": "text",
            "text": "以下是该大类各子标签的场景示意图，展示了自车和他车/VRU在路口中的位置关系和行驶轨迹。请参考这些示意图理解各标签的空间含义：\n",
        })
        for img_path in image_paths:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"file://{img_path}"},
            })
        content_parts.append({
            "type": "text",
            "text": "\n---\n\n",
        })

    text = user_template.replace(
        "{response_json}",
        json.dumps(response, ensure_ascii=False, indent=2),
    )
    content_parts.append({"type": "text", "text": text})

    return content_parts


def run_vllm_batch(
    model_path: str,
    records: list[dict[str, Any]],
    system_prompt: str,
    user_template: str,
    image_paths: list[str],
    *,
    batch_size: int = 16,
    max_model_len: int = 16384,
    tensor_parallel_size: int = 1,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    gpu_memory_utilization: float = 0.90,
    enable_thinking: bool = True,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": len(image_paths)} if image_paths else None,
    )
    if image_paths:
        llm_kwargs["allowed_local_media_path"] = "/"
    llm = LLM(**llm_kwargs)

    chat_template_kwargs = {}
    if enable_thinking:
        chat_template_kwargs["enable_thinking"] = True

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    abs_image_paths = [str(Path(p).resolve()) for p in image_paths]

    results: list[dict[str, Any]] = []
    n_batches = math.ceil(len(records) / batch_size)

    skipped: list[int] = []
    for bi in range(n_batches):
        batch = records[bi * batch_size : (bi + 1) * batch_size]
        conversations = []
        valid_rows: list[dict[str, Any]] = []
        for row in batch:
            raw = row.get("round_a")
            if isinstance(raw, dict):
                resp = raw
            elif isinstance(raw, str):
                try:
                    resp = json.loads(raw)
                except json.JSONDecodeError as exc:
                    idx = row.get("sample_index", "?")
                    print(f"WARNING: skipping sample_index={idx} — invalid JSON: {exc}")
                    skipped.append(idx)
                    results.append({
                        "sample_index": idx,
                        "video_path": row.get("video_path"),
                        "structured_label": None,
                        "raw_generation": None,
                        "skip_reason": f"invalid response JSON: {exc}",
                    })
                    continue
            else:
                idx = row.get("sample_index", "?")
                print(f"WARNING: skipping sample_index={idx} — unexpected type {type(raw).__name__}")
                skipped.append(idx)
                results.append({
                    "sample_index": idx,
                    "video_path": row.get("video_path"),
                    "structured_label": None,
                    "raw_generation": None,
                    "skip_reason": f"unexpected response type: {type(raw).__name__}",
                })
                continue

            user_content = build_user_content(user_template, resp, abs_image_paths)

            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ])
            valid_rows.append(row)

        if not conversations:
            print(f"[batch {bi + 1}/{n_batches}] all rows skipped")
            continue

        outputs = llm.chat(
            conversations,
            sampling_params=sampling,
            chat_template_kwargs=chat_template_kwargs,
        )

        for row, out in zip(valid_rows, outputs):
            generated = out.outputs[0].text.strip()
            label = _try_parse_json(generated)
            results.append({
                "sample_index": row.get("sample_index"),
                "video_path": row.get("video_path"),
                "label_en": row.get("label_en"),
                "all_labels": row.get("all_labels"),
                "structured_label": label,
                "raw_generation": generated,
            })

        print(f"[batch {bi + 1}/{n_batches}] done — {len(results)}/{len(records)}")

    if skipped:
        print(f"Total skipped: {len(skipped)} — indices: {skipped}")

    return results


def _try_parse_json(text: str) -> dict | str:
    clean = text.strip()
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        clean = "\n".join(lines).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return clean


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True, type=str)
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--prompt", required=True, type=str)
    p.add_argument("--images", nargs="*", default=[], help="示意图路径列表，插入 user prompt")
    p.add_argument("--enable-thinking", action="store_true", help="启用 thinking 模式")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-records", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    system_prompt, user_template = load_prompt(args.prompt)
    print(f"Loaded prompt from {args.prompt}")
    print(f"  System: {len(system_prompt)} chars")
    print(f"  User template: {len(user_template)} chars")
    print(f"  Images: {args.images}")
    print(f"  Thinking: {args.enable_thinking}")

    raw_text = Path(args.input).read_text(encoding="utf-8")
    try:
        records = json.loads(raw_text)
        if not isinstance(records, list):
            print("ERROR: input JSON must be a list", file=sys.stderr)
            return 1
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    if args.max_records is not None:
        records = records[:args.max_records]
    print(f"Input records: {len(records)}")

    results = run_vllm_batch(
        model_path=args.model,
        records=records,
        system_prompt=system_prompt,
        user_template=user_template,
        image_paths=args.images,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_thinking=args.enable_thinking,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results written to {args.output} ({len(results)} records)")

    parsed_ok = sum(1 for r in results if isinstance(r["structured_label"], dict))
    print(f"JSON parse success: {parsed_ok}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
