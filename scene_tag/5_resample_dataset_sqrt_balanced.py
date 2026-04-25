"""
sqrt 温度采样重平衡脚本

策略：
  - 对每个类别使用 sqrt(count) 作为采样权重（相当于 T=2 温度采样）
  - 确保不下采样任何类别（所有原始数据保留）
  - 对少数类进行上采样，但限制最大上采样倍数（默认 5x）防止过拟合
  - 最终分布：从极端不均衡 → 适度均衡

输入：基础数据集 qwen3_sft_train_segment_add_tags.json（无上采样的原始数据）
输出：新数据集 qwen3_sft_train_segment_add_tags_sqrt_balanced.json
"""

import json
import copy
import math
import random
import logging
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "qwen3_sft_train_segment_add_tags.json"
OUTPUT_FILE = DATA_DIR / "qwen3_sft_train_segment_add_tags_sqrt_balanced.json"

TEMPERATURE = 2.0      # T=2 即 sqrt 采样；T 越大越均匀
MAX_UPSAMPLE_RATIO = 5  # 单个类别最多上采样到原来的 5 倍
RANDOM_SEED = 42


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"加载数据集: {path}  样本数: {len(data)}")
    return data


def extract_label(item: dict) -> str:
    label = item.get("label_en", "")
    if not label:
        import re
        m = re.search(r"<driving_maneuver>(.*?)</driving_maneuver>", item.get("output", ""))
        if m:
            label = m.group(1)
    return label or "unknown"


def print_distribution(counter: Counter, title: str):
    total = sum(counter.values())
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'=' * 75}")
    print(f"  {title}  (共 {total} 样本, {len(counter)} 类)")
    print(f"{'=' * 75}")
    print(f"  {'Category':<55} {'Count':>6}  {'Ratio':>6}")
    print(f"  {'-' * 70}")
    for label, count in sorted_items:
        print(f"  {label:<55} {count:>6}  {count/total*100:>5.1f}%")
    print(f"  {'-' * 70}")
    max_c = sorted_items[0][1]
    min_c = sorted_items[-1][1]
    print(f"  Max/Min 比: {max_c/min_c:.1f}x")
    print(f"{'=' * 75}\n")


def sqrt_balanced_resample(data: list, temperature: float, max_ratio: int, seed: int) -> list:
    """
    基于 sqrt 温度的重采样:
      target_i = original_count_i * (max_count / original_count_i) ^ (1/T)
    T=2 时即 target_i = original_count_i * sqrt(max_count / original_count_i)
                       = sqrt(original_count_i * max_count)
    """
    rng = random.Random(seed)

    category_map: dict[str, list] = defaultdict(list)
    for item in data:
        category_map[extract_label(item)].append(item)

    counts = {k: len(v) for k, v in category_map.items()}
    max_count = max(counts.values())

    print_distribution(Counter(counts), "原始分布")

    resampled = []
    target_counts = {}

    for label, items in category_map.items():
        c = len(items)
        raw_target = c * (max_count / c) ** (1.0 / temperature)
        target = int(round(raw_target))

        target = max(target, c)
        target = min(target, c * max_ratio)

        target_counts[label] = target

        resampled.extend(items)

        needed = target - c
        if needed > 0:
            extras = rng.choices(items, k=needed)
            resampled.extend([copy.deepcopy(x) for x in extras])

    rng.shuffle(resampled)

    print_distribution(Counter(target_counts), f"重采样后分布 (T={temperature}, max_ratio={max_ratio}x)")

    return resampled


def main():
    if not INPUT_FILE.exists():
        logger.error(f"输入文件不存在: {INPUT_FILE}")
        return

    data = load_dataset(INPUT_FILE)

    resampled = sqrt_balanced_resample(
        data,
        temperature=TEMPERATURE,
        max_ratio=MAX_UPSAMPLE_RATIO,
        seed=RANDOM_SEED,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resampled, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存重采样数据集: {OUTPUT_FILE}  样本数: {len(resampled)}")

    print(f"\n输出文件: {OUTPUT_FILE}")
    print(f"样本数:   {len(data)} → {len(resampled)}  (+{len(resampled)-len(data)})")


if __name__ == "__main__":
    main()
