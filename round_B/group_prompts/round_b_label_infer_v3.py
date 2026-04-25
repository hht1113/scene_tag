#!/usr/bin/env python3
"""
Round-B 标签分类推理 v3 — 基于 vLLM 离线批量推理。

与 v2 的区别：prompt 文件已包含完整的标签定义和约束（如 *_v2.txt），
无需额外的 --tag-tree 参数进行合并，直接加载即可。

Usage:

python scripts/distillation/round_b_label_infer_v3.py \
    --model /root/workspace/model_zoo/Qwen3-8B \
    --input scripts/distillation/round_a_results_v2.jsonl \
    --output scripts/distillation/round_b_label_results_dynamic_v3.json \
    --prompt scripts/distillation/group_prompts/DynamicInteraction_v2.txt

python scripts/distillation/round_b_label_infer_v3.py \
    --model /mnt/pfs/chenruize/model_zoo/Qwen3-235B-A22B-FP8 \
    --input /mnt/pfs/chenruize/dataset/round_a_results_v2_final.jsonl \
    --output scripts/distillation/round_b_label_results_trafficlight_v3.json \
    --prompt scripts/distillation/group_prompts/TrafficLight_v2.txt \
    --tensor-parallel-size 8 \
    --max-model-len 20000 \
    --temperature 0.2 \
    --batch-size 64 \
    --gpu-memory-utilization 0.95
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────
# 1. Prompt loading (self-contained prompt file)
# ──────────────────────────────────────────────

_SECTION_SEP = "## USER ##"


def load_prompt(prompt_path: str) -> tuple[str, str]:
    """Load system and user prompts from a self-contained prompt file.

    The file must contain ``## SYSTEM ##`` and ``## USER ##`` sections.
    All tag definitions, constraints, and interpretation guidance are
    already embedded — no external tag-tree merging required.
    """
    raw = Path(prompt_path).read_text(encoding="utf-8")
    if _SECTION_SEP not in raw:
        raise ValueError(f"Prompt file must contain '{_SECTION_SEP}' separator: {prompt_path}")
    sys_part, user_part = raw.split(_SECTION_SEP, 1)
    sys_part = sys_part.replace("## SYSTEM ##", "").strip()
    user_part = user_part.strip()
    return sys_part, user_part


def build_user_prompt(user_template: str, response: dict) -> str:
    return user_template.replace(
        "{response_json}",
        json.dumps(response, ensure_ascii=False, indent=2),
    )


# ──────────────────────────────────────────────
# 2. vLLM offline batch inference
# ──────────────────────────────────────────────

def run_vllm_batch(
    model_path: str,
    records: list[dict[str, Any]],
    system_prompt: str,
    user_template: str,
    *,
    batch_size: int = 32,
    max_model_len: int = 8192,
    tensor_parallel_size: int = 1,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    gpu_memory_utilization: float = 0.90,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        chat_template = llm.llm_engine.tokenizer.tokenizer.chat_template
        if chat_template and "thinking" in chat_template:
            sampling = SamplingParams(
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_tokens=max_tokens,
            )
            llm.llm_engine.tokenizer.tokenizer.chat_template = chat_template.replace(
                "enable_thinking=true", "enable_thinking=true"
            )
            print("[thinking mode] enabled — temperature=0.6, top_p=0.95, top_k=20")
    except Exception:
        pass

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
                    print(f"WARNING: skipping sample_index={idx} — response is truncated/invalid JSON: {exc}")
                    skipped.append(idx)
                    results.append({
                        "sample_index": idx,
                        "video_path": row.get("video_path"),
                        "label_en": row.get("label_en"),
                        "all_labels": row.get("all_labels"),
                        "structured_label": None,
                        "raw_generation": None,
                        "skip_reason": f"invalid response JSON: {exc}",
                    })
                    continue
            else:
                idx = row.get("sample_index", "?")
                print(f"WARNING: skipping sample_index={idx} — response is {type(raw).__name__}, expected dict/str")
                skipped.append(idx)
                results.append({
                    "sample_index": idx,
                    "video_path": row.get("video_path"),
                    "label_en": row.get("label_en"),
                    "all_labels": row.get("all_labels"),
                    "structured_label": None,
                    "raw_generation": None,
                    "skip_reason": f"unexpected response type: {type(raw).__name__}",
                })
                continue
            conversations.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(user_template, resp)},
            ])
            valid_rows.append(row)

        if not conversations:
            print(f"[batch {bi + 1}/{n_batches}] all rows skipped")
            continue

        outputs = llm.chat(conversations, sampling_params=sampling)

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
        print(f"Total skipped due to invalid response: {len(skipped)} — indices: {skipped}")

    return results


def _try_parse_json(text: str) -> dict | str:
    clean = text.strip()
    import re
    think_match = re.search(r"</think>\s*(.*)", clean, re.DOTALL)
    if think_match:
        clean = think_match.group(1).strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        clean = "\n".join(lines).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return clean


# ──────────────────────────────────────────────
# 3. CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Local model path or HF repo id")
    p.add_argument("--input", required=True, type=str, help="Round-A results JSON/JSONL")
    p.add_argument("--output", required=True, type=str, help="Output JSON with structured_label")
    p.add_argument(
        "--prompt", required=True, type=str,
        help="Self-contained prompt file (e.g. group_prompts/DynamicInteraction_v2.txt). "
             "Must contain ## SYSTEM ## and ## USER ## sections with all tag definitions embedded.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-records", type=int, default=None, help="Limit number of input records (for debugging)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    system_prompt, user_template = load_prompt(args.prompt)
    print(f"Loaded self-contained prompt from {args.prompt}")
    print(f"  System prompt length: {len(system_prompt)} chars")
    print(f"  User template length: {len(user_template)} chars")

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
    total_records = len(records)
    print(f"Input records: {total_records}")

    results = run_vllm_batch(
        model_path=args.model,
        records=records,
        system_prompt=system_prompt,
        user_template=user_template,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
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
