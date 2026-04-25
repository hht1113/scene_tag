#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片场景挖掘脚本（API 客户端）

通过 OpenAI 兼容 API 对单张图片进行场景判定（二分类），
用于三模型对比实验中的图片任务。

用法:
    python 14_image_distillation.py \
        --api_base http://10.10.64.144:2754/v1 \
        --model_name Qwen3.5-397B-A17B \
        --image_list scene_tag/results_ab_compare/sample_images.txt \
        --prompt_file scene_tag/prompt_txt/img_road_surface_water.txt \
        --output scene_tag/results_ab_compare/qwen35_img_road_surface_water.json \
        --concurrency 4

    python 14_image_distillation.py \
        --api_base http://10.10.64.144:2754/v1 \
        --model_name Qwen3.5-397B-A17B \
        --image_path /mnt/pfs/rawdata/X6S5609/some_image.webp \
        --prompt_file scene_tag/prompt_txt/img_dual_countdown.txt
"""

import os
import re
import sys
import json
import time
import base64
import argparse
import traceback
import requests
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",10.10.64.144"
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",10.10.64.144"
NO_PROXY = {"http": None, "https": None}


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".bmp": "image/bmp", ".gif": "image/gif",
    }
    return mime_map.get(ext, "image/jpeg")


def try_parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"parse_error": True, "raw_text": text}


KNOWN_LABELS = [
    "掉头箭头", "左转掉头组合箭头", "道路施工标志",
    "注意儿童标志", "禁止掉头标志", "区域限速标志",
]


def try_parse_multilabel_csv(text: str) -> dict:
    """Parse multi-label CSV output, with fallback for verbose reasoning."""
    labels = {}
    for line in text.strip().splitlines():
        line = line.strip().lstrip("*-·• ")
        if not line:
            continue
        parts = re.split(r'[，,]\s*', line, maxsplit=1)
        if len(parts) == 2:
            label_name = parts[0].strip().strip("*")
            value = parts[1].strip().strip("。.").lower()
            if value in ("true", "false"):
                labels[label_name] = value == "true"

    if len(labels) >= 3:
        return {"labels": labels}

    for label in KNOWN_LABELS:
        pattern = rf'{re.escape(label)}[\s\S]*?(?:结论|结果|判定)[：:\s]*(True|False|true|false)'
        match = re.search(pattern, text)
        if match:
            labels[label] = match.group(1).lower() == "true"
        else:
            pattern2 = rf'{re.escape(label)}[\s\S]{{0,300}}?(True|False|true|false)'
            match2 = re.search(pattern2, text)
            if match2:
                labels[label] = match2.group(1).lower() == "true"

    if labels:
        return {"labels": labels}
    return {"parse_error": True, "raw_text": text}


class ImageAnnotationClient:

    def __init__(
        self,
        api_base: str,
        model_name: str,
        prompt_text: str,
        request_timeout: int = 120,
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.prompt_text = prompt_text
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.api_key = api_key

        print(f"\n图片标注客户端配置:")
        print(f"  API 地址:     {self.api_base}")
        print(f"  模型名称:     {self.model_name}")
        print(f"  认证:         {'API Key' if self.api_key else '无'}")
        print(f"  请求超时:     {self.request_timeout}s")
        print(f"  最大重试:     {self.max_retries}")
        print(f"  Prompt 长度:  {len(self.prompt_text)} chars")
        print()

    def annotate_image(self, image_path: str) -> Dict:
        t0 = time.time()

        b64 = encode_image_to_base64(image_path)
        mime = detect_mime_type(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": self.prompt_text},
                ],
            }
        ]

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
        }

        raw_text = self._call_api_with_retry(request_body)

        parsed_csv = try_parse_multilabel_csv(raw_text)
        if "labels" in parsed_csv:
            elapsed = time.time() - t0
            return {
                "image_path": image_path,
                "result": {k: v for k, v in parsed_csv["labels"].items() if v},
                "labels": parsed_csv["labels"],
                "raw_output": raw_text,
                "elapsed_seconds": round(elapsed, 2),
            }

        parsed = try_parse_json(raw_text)
        result_value = self._extract_result(parsed)
        elapsed = time.time() - t0

        return {
            "image_path": image_path,
            "result": result_value,
            "parsed": parsed,
            "raw_output": raw_text,
            "elapsed_seconds": round(elapsed, 2),
        }

    def _extract_result(self, parsed: dict) -> str:
        if "parse_error" in parsed:
            return "parse_error"
        result_list = parsed.get("result", [])
        if isinstance(result_list, list) and result_list:
            val = str(result_list[0]).strip().lower()
            return val
        result_str = parsed.get("result", "")
        if isinstance(result_str, str):
            return result_str.strip().lower()
        return "unknown"

    def _call_api_with_retry(self, request_body: Dict) -> str:
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=request_body,
                    headers=headers,
                    timeout=self.request_timeout,
                    proxies=NO_PROXY,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            except requests.exceptions.Timeout:
                last_error = f"API 请求超时 ({self.request_timeout}s)"
                print(f"  [重试 {attempt}/{self.max_retries}] {last_error}")
            except requests.exceptions.ConnectionError as e:
                last_error = f"连接失败: {e}"
                print(f"  [重试 {attempt}/{self.max_retries}] {last_error}")
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP 错误: {e.response.status_code} - {e.response.text[:200]}"
                print(f"  [重试 {attempt}/{self.max_retries}] {last_error}")
                if 400 <= e.response.status_code < 500:
                    break
            except (KeyError, IndexError) as e:
                last_error = f"API 响应格式异常: {e}"
                print(f"  [重试 {attempt}/{self.max_retries}] {last_error}")
                break
            except Exception as e:
                last_error = f"未知错误: {e}"
                print(f"  [重试 {attempt}/{self.max_retries}] {last_error}")

            if attempt < self.max_retries:
                wait_time = min(2 ** attempt * 5, 60)
                print(f"  等待 {wait_time}s 后重试...")
                time.sleep(wait_time)

        raise RuntimeError(f"API 调用失败（已重试 {self.max_retries} 次）: {last_error}")


def batch_annotate_images(
    client: ImageAnnotationClient,
    image_paths: List[str],
    output_json: str,
    max_images: Optional[int] = None,
    concurrency: int = 1,
):
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

    existing_results = []
    processed_paths = set()
    if os.path.exists(output_json):
        with open(output_json, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        processed_paths = {r["image_path"] for r in existing_results}

    if max_images:
        image_paths = image_paths[:max_images]

    pending = [p for p in image_paths if p not in processed_paths]
    skip_count = len(image_paths) - len(pending)

    print(f"\n{'=' * 60}")
    print(f"  批量图片标注")
    print(f"  总数:         {len(image_paths)}")
    print(f"  已处理(跳过): {skip_count}")
    print(f"  待处理:       {len(pending)}")
    print(f"  并发数:       {concurrency}")
    print(f"  输出文件:     {output_json}")
    print(f"{'=' * 60}\n")

    if not pending:
        print("所有图片已处理完毕")
        return

    results = list(existing_results)
    success_count = skip_count
    error_count = 0
    positive_count = sum(1 for r in existing_results if r.get("result") not in ("false", "none", "not_applicable", "parse_error", "unknown", "error"))

    def _process_single(args):
        idx, img_path = args
        try:
            result = client.annotate_image(img_path)
            return result, None
        except Exception as e:
            traceback.print_exc()
            return {
                "image_path": img_path,
                "result": "error",
                "parsed": {},
                "raw_output": "",
                "error": str(e),
            }, str(e)

    def _save_results():
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _handle_result(result, error):
        nonlocal success_count, error_count, positive_count
        if error:
            error_count += 1
        else:
            success_count += 1
        results.append(result)

        r = result.get("result")
        if isinstance(r, dict):
            is_positive = any(r.values())
        else:
            is_positive = r not in (
                "false", "none", "not_applicable", "parse_error", "unknown", "error",
                "['none']", "['false']",
            )
        if is_positive:
            positive_count += 1

        total_done = success_count + error_count
        elapsed = result.get("elapsed_seconds", 0)
        status = "+" if is_positive else "-"
        r = result.get("result", "?")
        result_str = str(r) if isinstance(r, dict) else str(r)
        print(
            f"  [{total_done}/{len(image_paths)}] {status} "
            f"result={result_str[:40]:<40s} "
            f"{elapsed:.1f}s  "
            f"(positive: {positive_count})"
        )

        if total_done % 10 == 0:
            _save_results()

    if concurrency <= 1:
        for idx, img_path in enumerate(pending):
            result, error = _process_single((idx, img_path))
            _handle_result(result, error)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_process_single, (idx, p)): (idx, p)
                for idx, p in enumerate(pending)
            }
            for future in as_completed(futures):
                result, error = future.result()
                _handle_result(result, error)

    _save_results()

    print(f"\n{'=' * 60}")
    print(f"  完成: {success_count} 成功, {error_count} 失败")
    print(f"  Positive: {positive_count}")
    print(f"  结果: {output_json}")
    print(f"{'=' * 60}")


def check_api_health(api_base: str) -> bool:
    base_url = api_base.rstrip("/")
    if base_url.endswith("/v1"):
        health_url = base_url[:-3] + "/health"
    else:
        health_url = base_url + "/health"
    try:
        resp = requests.get(health_url, timeout=10, proxies=NO_PROXY, verify=False)
        return resp.status_code == 200
    except Exception:
        try:
            resp = requests.get(f"{base_url}/models", timeout=10, proxies=NO_PROXY, verify=False)
            return resp.status_code == 200
        except Exception:
            return False


def parse_args():
    parser = argparse.ArgumentParser(description="图片场景挖掘（API 客户端）")

    parser.add_argument("--api_base", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None,
                        help="API Key (Bearer token). Can also set ARK_API_KEY env var.")

    parser.add_argument("--image_path", type=str, help="单张图片路径")
    parser.add_argument("--image_list", type=str, help="图片列表文件（每行一个路径）")
    parser.add_argument("--output", type=str, default="results/image_annotations.json")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request_timeout", type=int, default=120)

    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    print(f"  已加载 Prompt: {args.prompt_file} ({len(prompt_text)} chars)")

    api_key = args.api_key or os.environ.get("ARK_API_KEY")

    client = ImageAnnotationClient(
        api_base=args.api_base,
        model_name=args.model_name,
        prompt_text=prompt_text,
        request_timeout=args.request_timeout,
        api_key=api_key,
    )

    if args.image_path:
        print(f"\n单张图片模式: {args.image_path}")
        result = client.annotate_image(args.image_path)
        print(f"\n结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.image_list:
        with open(args.image_list, "r") as f:
            image_paths = [line.strip() for line in f if line.strip()]
        print(f"  已加载图片列表: {args.image_list} ({len(image_paths)} 张)")

        batch_annotate_images(
            client=client,
            image_paths=image_paths,
            output_json=args.output,
            max_images=args.max_images,
            concurrency=args.concurrency,
        )
        return

    print("请指定 --image_path（单张）或 --image_list（批量）")
    sys.exit(1)


if __name__ == "__main__":
    main()
