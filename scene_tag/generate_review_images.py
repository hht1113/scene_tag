#!/usr/bin/env python3
"""
生成 Bad Case 审查图片板

每个混淆对生成一张图片，每行一个bad case，显示6帧+标签信息。
可以在 Cursor 中直接查看 PNG 文件。
"""

import json
import re
import os
import cv2
import numpy as np
from collections import defaultdict

PRED_FILE = "/root/workspace/LLaMA-Factory/VQA/json/12tags_Qwen3-VL-30B_full_add_tags_dynamic.jsonl"
TEST_FILE = "/root/workspace/LLaMA-Factory/data/qwen3_sft_test_segment.json"
OUTPUT_DIR = "/root/workspace/LLaMA-Factory/scene_tag/bad_case_review/sheets"

FRAME_W, FRAME_H = 320, 180
N_FRAMES = 6
LABEL_H = 60
GAP = 4

MULTI_LABEL_PAIRS = {
    ("LaneChange_AvoidStaticVehicle", "LaneChange_NavForIntersection"),
    ("LaneChange_AvoidStaticVehicle", "LaneChange_AvoidSlowVRU"),
    ("LaneChange_NavForIntersection", "LaneChange_AvoidSlowVRU"),
    ("DynamicInteraction_VehicleInLaneCrossing", "DynamicInteraction_StandardVehicleCutIn"),
    ("TrafficLight_StraightStopOrGo", "TrafficLight_LeftTurnStopOrGo"),
    ("TrafficLight_StraightStopOrGo", "StartStop_StartFromMainRoad"),
    ("TrafficLight_LeftTurnStopOrGo", "Intersection_StandardUTurn"),
    ("DynamicInteraction_VehicleInLaneCrossing", "LaneChange_NavForIntersection"),
    ("LaneChange_NavForIntersection", "LaneCruising_Straight"),
    ("DynamicInteraction_StandardVehicleCutIn", "LaneCruising_Straight"),
    ("LaneChange_AvoidStaticVehicle", "DynamicInteraction_VRUInLaneCrossing"),
    ("LaneChange_AvoidStaticVehicle", "DynamicInteraction_StandardVehicleCutIn"),
    ("DynamicInteraction_VRUInLaneCrossing", "DynamicInteraction_StandardVehicleCutIn"),
    ("TrafficLight_StraightStopOrGo", "DynamicInteraction_VehicleInLaneCrossing"),
    ("StartStop_StartFromMainRoad", "StartStop_ParkRoadside"),
    ("LaneChange_AvoidSlowVRU", "DynamicInteraction_VRUInLaneCrossing"),
    ("DynamicInteraction_VehicleInLaneCrossing", "LaneChange_AvoidStaticVehicle"),
    ("LaneChange_AvoidStaticVehicle", "LaneCruising_Straight"),
    ("DynamicInteraction_StandardVehicleCutIn", "DynamicInteraction_VRUInLaneCrossing"),
    ("TrafficLight_LeftTurnStopOrGo", "LaneChange_NavForIntersection"),
}


def normalize_pair(a, b):
    return tuple(sorted([a, b]))


def parse_maneuvers(text):
    return re.findall(r'<driving_maneuver>(.*?)</driving_maneuver>', text)


def extract_frames(video_path, n_frames=6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if total <= 0:
        return []
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (FRAME_W, FRAME_H))
            t = idx / fps
            cv2.putText(frame_resized, f"{t:.1f}s", (5, FRAME_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame_resized, f"{t:.1f}s", (4, FRAME_H - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_resized, f"{t:.1f}s", (5, FRAME_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            frames.append(frame_resized)
    cap.release()
    return frames


def put_text_cn(img, text, pos, font_scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def create_case_row(case, row_w):
    """Create a single row image for one bad case: label area + 6 frames"""
    row_h = LABEL_H + FRAME_H
    row = np.ones((row_h, row_w, 3), dtype=np.uint8) * 40

    is_multi = case['is_multi']
    bg_color = (30, 80, 50) if is_multi else (50, 30, 30)
    row[:LABEL_H, :] = bg_color

    idx_text = f"#{case['index']}"
    put_text_cn(row, idx_text, (10, 20), 0.6, (200, 200, 200), 2)

    gt_text = f"GT: {case['gt_label']}"
    put_text_cn(row, gt_text, (80, 20), 0.5, (180, 220, 255), 1)

    pred_text = f"Pred: {case['pred_label']}"
    put_text_cn(row, pred_text, (80, 42), 0.5, (180, 255, 180) if is_multi else (100, 100, 255), 1)

    if is_multi:
        put_text_cn(row, "[Possible Multi-Label]", (650, 20), 0.5, (0, 255, 255), 1)

    video_name = os.path.basename(case['video_path'])
    put_text_cn(row, f"Video: {video_name}", (650, 42), 0.4, (160, 160, 160), 1)

    frames = case.get('frames', [])
    x_offset = 0
    for i, frame in enumerate(frames):
        y_start = LABEL_H
        x_start = x_offset
        if x_start + FRAME_W <= row_w:
            row[y_start:y_start + FRAME_H, x_start:x_start + FRAME_W] = frame
        x_offset += FRAME_W + GAP

    return row


def create_sheet(pair_name, cases, is_multi, output_path):
    """Create a sheet image for a confusion pair group"""
    row_w = N_FRAMES * (FRAME_W + GAP) - GAP
    row_w = max(row_w, 1000)

    title_h = 70
    row_h = LABEL_H + FRAME_H + GAP
    total_h = title_h + len(cases) * row_h + 10

    sheet = np.ones((total_h, row_w, 3), dtype=np.uint8) * 30

    if is_multi:
        sheet[:title_h, :] = (30, 80, 50)
        tag = "[POSSIBLE MULTI-LABEL]"
        tag_color = (0, 255, 255)
    else:
        sheet[:title_h, :] = (80, 30, 30)
        tag = "[LIKELY REAL ERROR]"
        tag_color = (100, 100, 255)

    put_text_cn(sheet, pair_name, (15, 30), 0.7, (255, 255, 255), 2)
    put_text_cn(sheet, f"{tag}  |  {len(cases)} cases", (15, 55), 0.55, tag_color, 1)

    for i, case in enumerate(cases):
        row = create_case_row(case, row_w)
        y_start = title_h + i * row_h
        sheet[y_start:y_start + row_h - GAP, :] = row

    cv2.imwrite(output_path, sheet, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PRED_FILE, 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]
    with open(TEST_FILE, 'r') as f:
        test_data = json.load(f)

    n = min(len(predictions), len(test_data))
    bad_cases = []

    for i in range(n):
        pred_obj = predictions[i]
        test_obj = test_data[i]
        gt_list = parse_maneuvers(pred_obj['label'])
        pred_list = parse_maneuvers(pred_obj['predict'])
        gt = gt_list[0] if gt_list else "Unrecognized"
        pred = pred_list[0] if pred_list else "Unrecognized"
        if gt == pred:
            continue

        video_path = test_obj.get('videos', [''])[0]
        is_multi = normalize_pair(gt, pred) in MULTI_LABEL_PAIRS

        print(f"[{i:3d}] Extracting: {os.path.basename(video_path)}")
        frames = extract_frames(video_path, N_FRAMES) if video_path and os.path.exists(video_path) else []

        bad_cases.append({
            'index': i,
            'gt_label': gt,
            'pred_label': pred,
            'video_path': video_path,
            'is_multi': is_multi,
            'frames': frames,
        })

    pair_groups = defaultdict(list)
    for case in bad_cases:
        pair = normalize_pair(case['gt_label'], case['pred_label'])
        pair_groups[pair].append(case)

    sorted_pairs = sorted(pair_groups.items(), key=lambda x: (-len(x[1])))

    print(f"\n生成审查图片...")
    file_list = []

    for idx, (pair, cases) in enumerate(sorted_pairs):
        is_multi = pair in MULTI_LABEL_PAIRS
        pair_name = f"{pair[0]}  <->  {pair[1]}"
        prefix = "MULTI" if is_multi else "ERROR"
        safe_name = f"{idx+1:02d}_{prefix}_{pair[0]}_vs_{pair[1]}"
        if len(safe_name) > 80:
            safe_name = safe_name[:80]
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
        create_sheet(pair_name, cases, is_multi, output_path)
        file_list.append((output_path, pair_name, len(cases), is_multi))
        print(f"  [{prefix}] {pair_name}: {len(cases)} cases -> {os.path.basename(output_path)}")

    print(f"\n{'='*60}")
    print(f"共生成 {len(file_list)} 张审查图片")
    print(f"保存目录: {OUTPUT_DIR}")
    print(f"\n建议查看顺序:")
    for path, name, cnt, is_multi in file_list:
        tag = "可能多标签" if is_multi else "可能真错误"
        print(f"  {os.path.basename(path):<75s} ({cnt}个, {tag})")


if __name__ == "__main__":
    main()
