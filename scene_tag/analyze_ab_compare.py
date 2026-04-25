"""
三模型 A/B/C 对比分析脚本

读取 results_ab_compare/ 下的结果，对比三个模型在同一批数据上的：
- 各标签命中数
- 命中率
- 标签覆盖度
- 同一视频上的标签一致性

用法:
    python scene_tag/analyze_ab_compare.py
"""

import json
import os
from collections import defaultdict

RESULT_DIR = "/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare"
MODELS = ["doubao", "qwen235b", "qwen35"]
PROMPTS = ["04_Intersection", "05_LaneCruising"]

P00_LABELS = {
    "DynamicInteraction_VRUInLaneCrossing", "DynamicInteraction_VehicleInLaneCrossing",
    "DynamicInteraction_StandardVehicleCutIn", "TrafficLight_StraightStopOrGo",
    "TrafficLight_LeftTurnStopOrGo", "LaneChange_NavForIntersection",
    "LaneChange_AvoidSlowVRU", "LaneChange_AvoidStaticVehicle",
    "StartStop_StartFromMainRoad", "StartStop_ParkRoadside",
    "Intersection_StandardUTurn", "LaneCruising_Straight",
}


def load_results(model, prompt):
    fp = os.path.join(RESULT_DIR, f"{model}_{prompt}.json")
    if not os.path.exists(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def extract_hits(data):
    hits = defaultdict(int)
    video_labels = {}
    for item in data:
        vp = item["video_path"]
        labels = []
        for seg in item.get("segments", []):
            label = seg.get("label", "")
            if label in P00_LABELS or label == "not_applicable":
                continue
            hits[label] += 1
            labels.append(label)
        if labels:
            video_labels[vp] = set(labels)
    return dict(hits), video_labels


def main():
    for prompt in PROMPTS:
        print(f"\n{'='*70}")
        print(f"  {prompt}")
        print(f"{'='*70}")

        model_data = {}
        for model in MODELS:
            data = load_results(model, prompt)
            if data is None:
                print(f"  {model}: 无数据")
                continue
            hits, video_labels = extract_hits(data)
            model_data[model] = {
                "total": len(data),
                "hits": hits,
                "video_labels": video_labels,
                "hit_videos": len(video_labels),
                "total_hits": sum(hits.values()),
            }
            print(f"  {model}: {len(data)} videos, {sum(hits.values())} 命中段, {len(video_labels)} 命中视频")

        if len(model_data) < 2:
            print("  不足2个模型有数据，跳过对比")
            continue

        all_labels = set()
        for md in model_data.values():
            all_labels.update(md["hits"].keys())

        print(f"\n  {'标签':<55s}", end="")
        for model in MODELS:
            if model in model_data:
                print(f" {model:>10s}", end="")
        print()
        print(f"  {'-'*55}", end="")
        for model in MODELS:
            if model in model_data:
                print(f" {'-'*10}", end="")
        print()

        for label in sorted(all_labels):
            print(f"  {label:<55s}", end="")
            for model in MODELS:
                if model in model_data:
                    cnt = model_data[model]["hits"].get(label, 0)
                    print(f" {cnt:>10d}", end="")
            print()

        print(f"  {'合计':<55s}", end="")
        for model in MODELS:
            if model in model_data:
                print(f" {model_data[model]['total_hits']:>10d}", end="")
        print()

        # 一致性分析
        active_models = [m for m in MODELS if m in model_data]
        if len(active_models) >= 2:
            all_videos = set()
            for md in model_data.values():
                all_videos.update(md["video_labels"].keys())

            agree = 0
            disagree = 0
            for vp in all_videos:
                labels_per_model = [model_data[m]["video_labels"].get(vp, set()) for m in active_models if m in model_data]
                if all(l == labels_per_model[0] for l in labels_per_model):
                    agree += 1
                else:
                    disagree += 1

            total_compared = agree + disagree
            if total_compared > 0:
                print(f"\n  标签一致率: {agree}/{total_compared} = {agree/total_compared*100:.1f}%")

    # 保存对比报告
    report_path = os.path.join(RESULT_DIR, "ab_compare_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 三模型对比报告\n\n")
        f.write("自动生成，详见终端输出。\n")
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
