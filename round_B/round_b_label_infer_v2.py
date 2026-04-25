#!/usr/bin/env python3
"""
Round-B 标签分类推理 — 基于 vLLM 离线批量推理。

python scripts/distillation/round_b_label_infer_v2.py \
    --model /root/workspace/model_zoo/Qwen3-8B \
    --input scripts/distillation/round_a_results_v2.jsonl \
    --output scripts/distillation/round_b_label_results_v4.json \
    --prompt scripts/distillation/round_b_prompt.txt \
    --tag-tree scripts/distillation/tag_structure_tree.txt

python scripts/distillation/round_b_label_infer_v2.py \
    --model /root/workspace/model_zoo/Qwen3-8B \
    --input scripts/distillation/round_a_results_v2.jsonl \
    --output scripts/distillation/round_b_label_results_v4.json \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --max-tokens 4096 \
    --temperature 0.1 \
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
# 1. Tag taxonomy parser
# ──────────────────────────────────────────────

TAG_TREE_DEFAULT = str(Path(__file__).resolve().parent / "tag_structure_tree.txt")


def parse_tag_tree(path: str) -> dict[str, dict]:
    """Parse tag_structure_tree.txt into taxonomy dict.

    Returns:
        {
            "01_DynamicInteraction": {
                "cn": "车道动态交互",
                "subs": [
                    {"cn": "邻车道车辆紧急切入", "en": "DynamicInteraction_EmergencyVehicleCutIn"},
                    ...
                ]
            },
            ...
        }
    """
    taxonomy: dict[str, dict] = {}
    current_major: str | None = None
    lines = Path(path).read_text(encoding="utf-8").splitlines()

    import re
    major_re = re.compile(r"^组\d+[：:]")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if major_re.match(line):
            after_colon = line.split("：", 1)[-1].strip() if "：" in line else line.split(":", 1)[-1].strip()
            parts = after_colon.split(None, 1)
            major_code = parts[0]
            major_cn = parts[1] if len(parts) > 1 else ""
            current_major = major_code
            taxonomy[major_code] = {"cn": major_cn, "subs": []}
        elif current_major:
            tokens = line.rsplit(None, 1)
            if len(tokens) == 2:
                sub_cn, sub_en = tokens[0].strip(), tokens[1].strip()
            else:
                sub_cn, sub_en = line, ""
            taxonomy[current_major].setdefault("subs", []).append(
                {"cn": sub_cn, "en": sub_en}
            )
    return taxonomy


def taxonomy_to_prompt_block(taxonomy: dict[str, dict]) -> str:
    """Format taxonomy as a readable enumeration for the prompt."""
    parts: list[str] = []
    for code, info in taxonomy.items():
        header = f"{code}（{info['cn']}）"
        sub_lines = [f"  - {s['cn']}  [{s['en']}]" for s in info["subs"]]
        parts.append(header + "\n" + "\n".join(sub_lines))
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# 2. Prompt construction (loaded from external txt)
# ──────────────────────────────────────────────

PROMPT_DEFAULT = str(Path(__file__).resolve().parent / "round_b_prompt.txt")

_SECTION_SEP = "## USER ##"


def load_prompt_templates(prompt_path: str) -> tuple[str, str]:
    """Load system and user templates from a prompt txt file.

    The file must contain a ``## SYSTEM ##`` section and a ``## USER ##`` section.
    """
    raw = Path(prompt_path).read_text(encoding="utf-8")
    if _SECTION_SEP not in raw:
        raise ValueError(f"Prompt file must contain '{_SECTION_SEP}' separator: {prompt_path}")
    sys_part, user_part = raw.split(_SECTION_SEP, 1)
    sys_part = sys_part.replace("## SYSTEM ##", "").strip()
    user_part = user_part.strip()
    return sys_part, user_part


def build_system_prompt(system_template: str, taxonomy: dict[str, dict]) -> str:
    block = taxonomy_to_prompt_block(taxonomy)
    return system_template.replace("{taxonomy_block}", block)


def build_user_prompt(user_template: str, response: dict) -> str:
    return user_template.replace(
        "{response_json}",
        json.dumps(response, ensure_ascii=False, indent=2),
    )


# ──────────────────────────────────────────────
# 3. vLLM offline batch inference
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

    results: list[dict[str, Any]] = []
    n_batches = math.ceil(len(records) / batch_size)

    skipped: list[int] = []
    for bi in range(n_batches):
        batch = records[bi * batch_size : (bi + 1) * batch_size]
        conversations = []
        valid_rows: list[dict[str, Any]] = []
        for row in batch:
            #raw = row.get("response")
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

        outputs = llm.chat(
            conversations,
            sampling_params=sampling,
            chat_template_kwargs={"enable_thinking": False},
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
        print(f"Total skipped due to invalid response: {len(skipped)} — indices: {skipped}")

    return results


def _try_parse_json(text: str) -> dict | str:
    import re
    clean = text.strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        clean = "\n".join(lines).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(clean[start:end+1])
            except json.JSONDecodeError:
                pass
        return clean


# ──────────────────────────────────────────────
# 4. CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Local model path or HF repo id")
    p.add_argument("--input", required=True, type=str, help="Round-A results JSON")
    p.add_argument("--output", required=True, type=str, help="Output JSON with structured_label")
    p.add_argument("--tag-tree", type=str, default=TAG_TREE_DEFAULT, help="tag_structure_tree.txt path")
    p.add_argument("--prompt", type=str, default=PROMPT_DEFAULT, help="round_b_prompt.txt path")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    taxonomy = parse_tag_tree(args.tag_tree)
    print(f"Loaded taxonomy: {len(taxonomy)} major categories, "
          f"{sum(len(v['subs']) for v in taxonomy.values())} sub categories")

    system_template, user_template = load_prompt_templates(args.prompt)
    print(f"Loaded prompt templates from {args.prompt}")
    system_prompt = build_system_prompt(system_template, taxonomy)
    raw_text = Path(args.input).read_text(encoding="utf-8")
    try:
        records = json.loads(raw_text)
        if not isinstance(records, list):
            print("ERROR: input JSON must be a list", file=sys.stderr)
            return 1
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    # records = records[:300]  # 去掉条数限制
    print(f"Input records: {len(records)}")

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
