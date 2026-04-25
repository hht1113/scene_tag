"""
动态采样 v2: 基于已验证有效的配比策略重新生成数据集

策略（来自 add_tags_dynamic 实验的经验）:
  - 降采样头部主导类 (LaneCruising_Straight 0.3x)
  - 定向上采样尾部稀有类 (2-3x)
  - 中间段类别保持不变
  - 重点提升关键驾驶场景的比例

输入: qwen3_sft_train_segment_add_tags.json (5825 样本, 原始分布)
输出: qwen3_sft_train_segment_add_tags_dynamic_v2.json
"""

import json
import copy
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
OUTPUT_FILE = DATA_DIR / "qwen3_sft_train_segment_add_tags_dynamic_v2.json"

RANDOM_SEED = 2026

# 已验证有效的采样倍率 (来自 add_tags_dynamic 实验)
CATEGORY_MULTIPLIERS = {
    "LaneCruising_Straight":                        0.3,   # 头部主导类降采样
    "TrafficLight_LeftTurnStopOrGo":                1.0,
    "TrafficLight_StraightStopOrGo":                2.0,   # 红绿灯直行场景重要，上采样
    "Intersection_StandardUTurn":                   1.0,
    "DynamicInteraction_VehicleInLaneCrossing":     1.0,
    "LaneChange_NavForIntersection":                3.0,   # 路口变道是关键场景
    "LaneChange_AvoidStaticVehicle":                1.0,
    "DynamicInteraction_StandardVehicleCutIn":      1.0,
    "DynamicInteraction_VRUInLaneCrossing":         2.5,   # VRU 交互安全相关
    "StartStop_ParkRoadside":                       2.0,
    "LaneChange_AvoidSlowVRU":                      2.0,
    "StartStop_StartFromMainRoad":                  3.0,   # 极稀有但重要
}

DEFAULT_MULTIPLIER = 1.0


def load_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("加载数据集: %s  样本数: %d", path, len(data))
    return data


def extract_label(item: dict) -> str:
    import re
    label = item.get("label_en", "")
    if not label:
        m = re.search(r"<driving_maneuver>(.*?)</driving_maneuver>", item.get("output", ""))
        if m:
            label = m.group(1)
    return label or "unknown"


def print_distribution(counter: Counter, title: str):
    total = sum(counter.values())
    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print("\n" + "=" * 78)
    print("  {}  (共 {} 样本, {} 类)".format(title, total, len(counter)))
    print("=" * 78)
    print("  {:<55} {:>6}  {:>6}".format("Category", "Count", "Ratio"))
    print("  " + "-" * 72)
    for label, count in sorted_items:
        print("  {:<55} {:>6}  {:>5.1f}%".format(label, count, count / total * 100))
    print("  " + "-" * 72)
    max_c = sorted_items[0][1]
    min_c = sorted_items[-1][1]
    print("  Max/Min 比: {:.1f}x".format(max_c / min_c))
    print("=" * 78)


def dynamic_resample(data: list, multipliers: dict, seed: int) -> list:
    rng = random.Random(seed)

    category_map: dict[str, list] = defaultdict(list)
    for item in data:
        category_map[extract_label(item)].append(item)

    orig_counter = Counter({k: len(v) for k, v in category_map.items()})
    print_distribution(orig_counter, "原始分布")

    resampled = []
    target_counter = Counter()

    for label, items in category_map.items():
        mult = multipliers.get(label, DEFAULT_MULTIPLIER)
        c = len(items)
        target = max(1, int(round(c * mult)))

        target_counter[label] = target

        if target <= c:
            selected = rng.sample(items, target)
            resampled.extend(selected)
        else:
            resampled.extend(items)
            needed = target - c
            extras = rng.choices(items, k=needed)
            resampled.extend([copy.deepcopy(x) for x in extras])

    rng.shuffle(resampled)

    print_distribution(target_counter, "动态采样后分布")

    print("\n  采样倍率明细:")
    print("  " + "-" * 72)
    for label in sorted(multipliers.keys(), key=lambda x: orig_counter.get(x, 0), reverse=True):
        orig = orig_counter.get(label, 0)
        tgt = target_counter.get(label, 0)
        mult = multipliers.get(label, DEFAULT_MULTIPLIER)
        action = "downsample" if mult < 1 else ("upsample" if mult > 1 else "keep")
        print("  {:<50} {:>4} -> {:>4}  ({:.1f}x {})".format(label, orig, tgt, mult, action))
    print("  " + "-" * 72)

    return resampled


def main():
    if not INPUT_FILE.exists():
        logger.error("输入文件不存在: %s", INPUT_FILE)
        return

    data = load_dataset(INPUT_FILE)

    resampled = dynamic_resample(data, CATEGORY_MULTIPLIERS, RANDOM_SEED)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(resampled, f, ensure_ascii=False, indent=2)

    logger.info("已保存: %s  样本数: %d", OUTPUT_FILE, len(resampled))
    print("\n输出文件: {}".format(OUTPUT_FILE))
    print("样本数:   {} -> {}  ({:+d})".format(len(data), len(resampled), len(resampled) - len(data)))


if __name__ == "__main__":
    main()
