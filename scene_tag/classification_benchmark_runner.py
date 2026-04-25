#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests


DEFAULT_TASK_INDEX = Path(
    "/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare/reviewed_benchmark/classification_compatible_tasks.json"
)
DEFAULT_MODEL_CONFIG = Path(
    "/root/workspace/LLaMA-Factory/scene_tag/scene_classification_benchmark_models.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare/classification_benchmark_runs"
)


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def detect_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    if ext == ".gif":
        return "image/gif"
    return "image/jpeg"


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_truthy_text(text: str) -> Optional[bool]:
    normalized = text.strip().lower()
    if normalized in {"true", "yes", "1", "正确", "符合"}:
        return True
    if normalized in {"false", "no", "0", "错误", "不符合"}:
        return False
    return None


def parse_boolean_result(raw_text: str) -> tuple[Optional[bool], str]:
    text = raw_text.strip()

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            value = parsed.get("result")
            if isinstance(value, bool):
                return value, "json_bool"
            if isinstance(value, str):
                normalized = normalize_truthy_text(value)
                if normalized is not None:
                    return normalized, "json_str"
            if isinstance(value, list) and value:
                normalized = normalize_truthy_text(str(value[0]))
                if normalized is not None:
                    return normalized, "json_list"
        except json.JSONDecodeError:
            pass

    patterns = [
        r"RESULT\s*[:：]\s*(true|false|yes|no)",
        r"(?:结论|结果|判定)\s*[:：]\s*(true|false|yes|no|符合|不符合)",
        r'"result"\s*:\s*(true|false)',
        r'"result"\s*:\s*"(true|false|yes|no|符合|不符合)"',
        r"\b(true|false|yes|no)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            normalized = normalize_truthy_text(match.group(1))
            if normalized is not None:
                return normalized, "regex"

    return None, "parse_error"


TASK_NAME_EN = {
    "区域限速标志": "zone speed limit sign",
    "左转掉头组合箭头": "left-turn and U-turn combined arrow marking",
    "掉头箭头": "U-turn arrow marking",
    "注意儿童标志": "children warning sign",
    "禁止掉头标志": "no U-turn sign",
    "道路施工标志": "road construction sign",
    "雨天路面积水": "wet road surface with standing water (rainy day)",
    "双灯+中间倒计时": "dual traffic light heads with countdown timer in between",
    "便携式临时信号灯": "portable temporary traffic signal light",
    "锥桶": "traffic cone",
    "水马": "water-filled barrier (water horse)",
    "夜间行车场景": "nighttime driving scene",
    "交通灯组": "traffic light cluster / signal head group",
    "红色左转箭头交通灯": "red left-turn arrow traffic light",
    "红色左转箭头灯": "red left-turn arrow signal",
    "白天道路护栏": "daytime road guardrail / barrier",
    "施工区域": "construction zone",
    "前方车辆": "vehicle ahead",
    "左转待转区": "left-turn waiting zone marking",
    "场景图": "scene overview image",
}


def build_prompt(task_name: str, english: bool = False) -> str:
    if english:
        en_name = TASK_NAME_EN.get(task_name, task_name)
        return (
            "You are a strict road scene classifier. "
            f"Determine whether the image contains the target scene/object: `{en_name}`.\n"
            "Requirements:\n"
            "1. Only answer for the current task. Do not expand to similar concepts.\n"
            "2. If the image clearly contains the target, return true; otherwise return false.\n"
            "3. Output exactly two lines, nothing else.\n"
            "4. Line 1 must be: RESULT: true  OR  RESULT: false\n"
            "5. Line 2 must be: REASON: (brief reason in <=20 words)\n"
            "Example output:\n"
            "RESULT: true\n"
            "REASON: Target object clearly visible in image"
        )
    return (
        "你是一个严格的道路场景分类器。"
        f"请判断图片中是否存在目标场景/目标：`{task_name}`。\n"
        "要求：\n"
        "1. 只回答当前任务，不要扩展到相似概念。\n"
        "2. 如果图片明确包含该目标，返回 true；否则返回 false。\n"
        "3. 最终只输出两行，不要输出其他内容。\n"
        "4. 第一行必须是 RESULT: true 或 RESULT: false。\n"
        "5. 第二行必须是 REASON: 不超过20字。\n"
        "输出示例：\n"
        "RESULT: true\n"
        "REASON: 图片中可见目标"
    )


@dataclass
class TaskRecord:
    task_id: str
    task_name: str
    task_slug: str
    task_file: Path
    source_component: str
    positive_item_count: int


class OpenAICompatibleVisionClient:
    def __init__(
        self,
        api_base: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout_seconds: int = 120,
        extra_body: Optional[dict] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.extra_body = extra_body or {}
        self.session = requests.Session()
        self.session.trust_env = False

    def list_models(self):
        response = self.session.get(f"{self.api_base}/models", timeout=10)
        response.raise_for_status()
        return response.json()

    def classify_image(self, image_path: str, task_name: str, english: bool = False) -> dict:
        started_at = time.perf_counter()
        mime = detect_mime_type(image_path)
        payload = {
            "model": self.model_name,
            "temperature": 0.0,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{encode_image_to_base64(image_path)}"
                            },
                        },
                        {"type": "text", "text": build_prompt(task_name, english=english)},
                    ],
                }
            ],
        }
        payload.update(self.extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self.session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        raw_text = body["choices"][0]["message"]["content"].strip()
        parsed_bool, parse_mode = parse_boolean_result(raw_text)
        latency_seconds = time.perf_counter() - started_at

        return {
            "raw_output": raw_text,
            "predicted_positive": parsed_bool,
            "parse_mode": parse_mode,
            "latency_seconds": round(latency_seconds, 4),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="场景分类 benchmark runner")
    parser.add_argument("--task-index", type=Path, default=DEFAULT_TASK_INDEX)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-positives-per-task", type=int, default=None)
    parser.add_argument("--negatives-per-task", type=int, default=None, help="默认与正样本数相同")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-name-regex", type=str, default=None)
    parser.add_argument("--model-name-regex", type=str, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--health-check-only", action="store_true")
    parser.add_argument("--english-prompt", action="store_true", help="Use English prompts instead of Chinese")
    return parser.parse_args()


def load_task_records(task_index_path: Path, task_name_regex: Optional[str], max_tasks: Optional[int]) -> list[TaskRecord]:
    tasks = load_json(task_index_path)
    records = []
    for task in tasks:
        if task_name_regex and not re.search(task_name_regex, task["task_name"]):
            continue
        records.append(
            TaskRecord(
                task_id=task["task_id"],
                task_name=task["task_name"],
                task_slug=task["task_slug"],
                task_file=Path(task["task_file"]),
                source_component=task["source_component"],
                positive_item_count=task["positive_item_count"],
            )
        )
    if max_tasks is not None:
        records = records[:max_tasks]
    return records


def load_task_items(task_record: TaskRecord) -> list[str]:
    payload = load_json(task_record.task_file)
    return [item["image_path"] for item in payload.get("items", [])]


def build_eval_dataset(
    task_records: list[TaskRecord],
    max_positives_per_task: Optional[int],
    negatives_per_task: Optional[int],
    seed: int,
):
    rng = random.Random(seed)
    task_to_items = {task.task_slug: load_task_items(task) for task in task_records}
    all_images = {
        task.task_slug: set(items)
        for task, items in zip(task_records, task_to_items.values())
    }

    datasets = []
    all_unique_images = set()
    for task in task_records:
        positives = list(task_to_items[task.task_slug])
        rng.shuffle(positives)
        if max_positives_per_task is not None:
            positives = positives[:max_positives_per_task]

        positive_set = set(positives)
        negative_pool = []
        for other_task in task_records:
            if other_task.task_slug == task.task_slug:
                continue
            for image_path in task_to_items[other_task.task_slug]:
                if image_path not in positive_set:
                    negative_pool.append(image_path)

        negative_pool = list(dict.fromkeys(negative_pool))
        rng.shuffle(negative_pool)
        negative_count = negatives_per_task if negatives_per_task is not None else len(positives)
        negatives = negative_pool[:negative_count]

        samples = [{"image_path": path, "label": 1} for path in positives] + [
            {"image_path": path, "label": 0} for path in negatives
        ]
        rng.shuffle(samples)

        all_unique_images.update(positives)
        all_unique_images.update(negatives)
        datasets.append(
            {
                "task_id": task.task_id,
                "task_name": task.task_name,
                "task_slug": task.task_slug,
                "source_component": task.source_component,
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "samples": samples,
            }
        )

    return {
        "tasks": datasets,
        "unique_image_count": len(all_unique_images),
    }


def safe_divide(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def compute_binary_metrics(predictions: list[dict]) -> dict:
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for row in predictions:
        predicted_positive = row.get("predicted_positive")
        label = row["label"]
        if predicted_positive is None:
            if label == 1:
                fn += 1
            else:
                fp += 1
            continue
        if label == 1 and predicted_positive is True:
            tp += 1
        elif label == 0 and predicted_positive is False:
            tn += 1
        elif label == 0 and predicted_positive is True:
            fp += 1
        elif label == 1 and predicted_positive is False:
            fn += 1
    total = len(predictions)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = round((2 * precision * recall) / (precision + recall), 4) if (precision + recall) else 0.0
    accuracy = safe_divide(tp + tn, total)
    parse_errors = sum(1 for row in predictions if row["parse_mode"] == "parse_error")
    avg_latency = round(sum(row["latency_seconds"] for row in predictions) / total, 4) if total else 0.0
    return {
        "sample_count": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parse_error_count": parse_errors,
        "avg_latency_seconds": avg_latency,
    }


def run_single_model(model_cfg: dict, dataset_bundle: dict, timeout_seconds: int, english_prompt: bool = False) -> dict:
    client = OpenAICompatibleVisionClient(
        api_base=model_cfg["api_base"],
        model_name=model_cfg["model_name"],
        api_key=model_cfg.get("api_key"),
        timeout_seconds=timeout_seconds,
        extra_body=model_cfg.get("extra_body"),
    )

    model_started_at = time.perf_counter()
    task_results = []
    all_predictions = []

    for task in dataset_bundle["tasks"]:
        task_predictions = []
        for sample in task["samples"]:
            inference = client.classify_image(sample["image_path"], task["task_name"], english=english_prompt)
            predicted_label = None
            if inference["predicted_positive"] is True:
                predicted_label = 1
            elif inference["predicted_positive"] is False:
                predicted_label = 0
            row = {
                "image_path": sample["image_path"],
                "label": sample["label"],
                "predicted_label": predicted_label,
                "predicted_positive": inference["predicted_positive"],
                "parse_mode": inference["parse_mode"],
                "latency_seconds": inference["latency_seconds"],
                "raw_output": inference["raw_output"],
            }
            task_predictions.append(row)
            all_predictions.append(row)

        task_metrics = compute_binary_metrics(task_predictions)
        task_results.append(
            {
                "task_id": task["task_id"],
                "task_name": task["task_name"],
                "task_slug": task["task_slug"],
                "source_component": task["source_component"],
                "positive_count": task["positive_count"],
                "negative_count": task["negative_count"],
                "metrics": task_metrics,
                "predictions": task_predictions,
            }
        )

    total_seconds = time.perf_counter() - model_started_at
    overall_metrics = compute_binary_metrics(all_predictions)
    overall_metrics["total_wall_time_seconds"] = round(total_seconds, 4)
    overall_metrics["throughput_samples_per_second"] = round(
        len(all_predictions) / total_seconds, 4
    ) if total_seconds else 0.0

    return {
        "model": model_cfg["name"],
        "api_base": model_cfg["api_base"],
        "model_name": model_cfg["model_name"],
        "overall_metrics": overall_metrics,
        "task_results": task_results,
    }


def build_summary_markdown(run_payload: dict, summary_rows: list[dict], summary_path: Path) -> str:
    lines = [
        "# Classification Benchmark Report",
        "",
        f"- 生成时间: `{utc_now_iso()}`",
        f"- summary: `{summary_path}`",
        f"- task_index: `{run_payload['task_index']}`",
        f"- model_config: `{run_payload['model_config']}`",
        f"- benchmark task 数: `{run_payload['dataset']['task_count']}`",
        f"- benchmark unique image 数: `{run_payload['dataset']['unique_image_count']}`",
        "",
        "## 模型总表",
        "",
        "| 模型 | Accuracy | Precision | Recall | F1 | Avg Latency(s) | Throughput(samples/s) | Parse Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['avg_latency_seconds']:.4f} | {row['throughput_samples_per_second']:.4f} | {row['parse_error_count']} |"
        )

    lines.append("")
    return "\n".join(lines)


def health_check_models(model_cfgs: list[dict], timeout_seconds: int) -> list[dict]:
    results = []
    for cfg in model_cfgs:
        status = {
            "name": cfg["name"],
            "api_base": cfg["api_base"],
            "model_name": cfg["model_name"],
            "healthy": False,
            "error": None,
        }
        try:
            client = OpenAICompatibleVisionClient(
                api_base=cfg["api_base"],
                model_name=cfg["model_name"],
                api_key=cfg.get("api_key"),
                timeout_seconds=timeout_seconds,
                extra_body=cfg.get("extra_body"),
            )
            models_json = client.list_models()
            status["healthy"] = True
            status["models_response_excerpt"] = json.dumps(models_json, ensure_ascii=False)[:500]
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
        results.append(status)
    return results


def main():
    args = parse_args()
    task_records = load_task_records(args.task_index, args.task_name_regex, args.max_tasks)
    model_cfgs = load_json(args.model_config)

    if args.model_name_regex:
        model_cfgs = [cfg for cfg in model_cfgs if re.search(args.model_name_regex, cfg["name"])]

    model_cfgs = [cfg for cfg in model_cfgs if cfg.get("enabled", True)]

    output_dir = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    health_status = []
    should_run_health_check = args.health_check_only or (not args.skip_health_check)
    if should_run_health_check:
        health_status = health_check_models(model_cfgs, args.timeout_seconds)
        write_json(output_dir / "health_check.json", health_status)
        if args.health_check_only:
            print(json.dumps(health_status, ensure_ascii=False, indent=2))
            return
        model_cfgs = [cfg for cfg, status in zip(model_cfgs, health_status) if status["healthy"]]

    dataset_bundle = build_eval_dataset(
        task_records=task_records,
        max_positives_per_task=args.max_positives_per_task,
        negatives_per_task=args.negatives_per_task,
        seed=args.seed,
    )

    run_payload = {
        "generated_at": utc_now_iso(),
        "task_index": str(args.task_index),
        "model_config": str(args.model_config),
        "dataset": {
            "task_count": len(dataset_bundle["tasks"]),
            "unique_image_count": dataset_bundle["unique_image_count"],
            "max_positives_per_task": args.max_positives_per_task,
            "negatives_per_task": args.negatives_per_task,
            "seed": args.seed,
        },
        "health_status": health_status,
        "model_results": [],
    }

    if not model_cfgs:
        write_json(output_dir / "benchmark_results.json", run_payload)
        (output_dir / "benchmark_summary.md").write_text(
            "# Classification Benchmark Report\n\n- 没有可用的健康模型，未执行正式评测。\n",
            encoding="utf-8",
        )
        print("没有可用的健康模型，已写出健康检查结果。")
        print(f"输出目录: {output_dir}")
        return

    summary_rows = []
    for cfg in model_cfgs:
        result = run_single_model(cfg, dataset_bundle, args.timeout_seconds, english_prompt=args.english_prompt)
        run_payload["model_results"].append(result)
        summary_rows.append(
            {
                "model": result["model"],
                **result["overall_metrics"],
            }
        )

    write_json(output_dir / "benchmark_results.json", run_payload)
    write_json(output_dir / "benchmark_summary_rows.json", summary_rows)
    summary_md = build_summary_markdown(run_payload, summary_rows, output_dir / "benchmark_summary.md")
    (output_dir / "benchmark_summary.md").write_text(summary_md, encoding="utf-8")

    print("=" * 60)
    print("场景分类 benchmark 完成")
    print(f"输出目录: {output_dir}")
    print(f"模型数: {len(model_cfgs)}")
    print(f"任务数: {len(dataset_bundle['tasks'])}")
    print(f"图片数: {dataset_bundle['unique_image_count']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
