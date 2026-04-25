#!/usr/bin/env python3
"""
生成 Bad Case 可视化审查 HTML 页面

从每个 bad case 视频中抽取 6 帧关键帧，生成 HTML 页面。
审查人员可以快速浏览帧画面，判断模型预测是否合理，
无需逐个播放视频。

用法: python3 generate_review_html.py
输出: /root/workspace/LLaMA-Factory/scene_tag/bad_case_review/ 目录
"""

import json
import re
import os
import base64
import cv2
import numpy as np
from collections import defaultdict
from io import BytesIO

PRED_FILE = "/root/workspace/LLaMA-Factory/VQA/json/12tags_Qwen3-VL-30B_full_add_tags_dynamic.jsonl"
TEST_FILE = "/root/workspace/LLaMA-Factory/data/qwen3_sft_test_segment.json"
OUTPUT_DIR = "/root/workspace/LLaMA-Factory/scene_tag/bad_case_review"
N_FRAMES = 6


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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    if total_frames <= 0:
        return []

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            time_sec = idx / fps
            frame_small = cv2.resize(frame, (384, 216))
            _, buf = cv2.imencode('.jpg', frame_small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64 = base64.b64encode(buf).decode('utf-8')
            frames.append({
                'b64': b64,
                'time': f"{time_sec:.1f}s",
                'frame_idx': int(idx),
            })

    cap.release()
    return frames


def generate_html(bad_cases, output_path):
    multi_cases = [c for c in bad_cases if c['is_multi']]
    non_multi_cases = [c for c in bad_cases if not c['is_multi']]

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>Bad Case 可视化审查 - Dynamic Model</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5; padding: 20px; color: #333; }}
h1 {{ text-align: center; margin-bottom: 10px; color: #1a1a2e; }}
.summary {{ background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 900px; margin-left: auto; margin-right: auto; }}
.summary h2 {{ color: #16213e; margin-bottom: 10px; }}
.summary table {{ width: 100%; border-collapse: collapse; }}
.summary td, .summary th {{ padding: 6px 12px; border-bottom: 1px solid #eee; text-align: left; }}
.summary th {{ background: #f8f9fa; font-weight: 600; }}
.section-title {{ font-size: 1.4em; margin: 30px 0 15px 0; padding: 10px 15px; background: #16213e;
                  color: #fff; border-radius: 6px; max-width: 1200px; margin-left: auto; margin-right: auto; }}
.filter-bar {{ background: #fff; border-radius: 8px; padding: 15px 20px; margin-bottom: 20px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 1200px; margin-left: auto; margin-right: auto; }}
.filter-bar label {{ font-weight: 600; margin-right: 10px; }}
.filter-bar select {{ padding: 6px 12px; border-radius: 4px; border: 1px solid #ccc; font-size: 14px; }}
.filter-bar button {{ padding: 6px 16px; border-radius: 4px; border: none; background: #e94560;
                     color: white; cursor: pointer; font-size: 14px; margin-left: 10px; }}
.filter-bar button:hover {{ background: #c23152; }}
.case-card {{ background: #fff; border-radius: 8px; margin-bottom: 16px; padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); max-width: 1200px; margin-left: auto; margin-right: auto;
            border-left: 5px solid #e94560; }}
.case-card.multi-label {{ border-left-color: #f59e0b; }}
.case-card.verdict-correct {{ border-left-color: #10b981; opacity: 0.7; }}
.case-card.verdict-wrong {{ border-left-color: #ef4444; }}
.case-header {{ display: flex; justify-content: space-between; align-items: flex-start;
               margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }}
.case-header .idx {{ font-weight: 700; color: #1a1a2e; font-size: 1.1em; }}
.label-box {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.85em; font-weight: 600; }}
.label-gt {{ background: #dbeafe; color: #1e40af; }}
.label-pred {{ background: #fef3c7; color: #92400e; }}
.label-multi {{ background: #fef3c7; color: #d97706; border: 1px dashed #d97706; font-size: 0.8em; padding: 2px 8px; }}
.frames {{ display: flex; gap: 6px; overflow-x: auto; padding: 8px 0; }}
.frame-wrap {{ flex-shrink: 0; text-align: center; }}
.frame-wrap img {{ border-radius: 4px; border: 1px solid #e5e7eb; display: block; cursor: pointer; }}
.frame-wrap img:hover {{ border-color: #3b82f6; box-shadow: 0 0 8px rgba(59,130,246,0.3); }}
.frame-time {{ font-size: 0.75em; color: #6b7280; margin-top: 2px; }}
.video-path {{ font-size: 0.75em; color: #9ca3af; margin-top: 4px; word-break: break-all; }}
.verdict-buttons {{ margin-top: 8px; display: flex; gap: 8px; }}
.verdict-buttons button {{ padding: 4px 14px; border-radius: 4px; border: 1px solid #ccc;
                          cursor: pointer; font-size: 13px; font-weight: 500; }}
.btn-correct {{ background: #d1fae5; color: #065f46; border-color: #6ee7b7; }}
.btn-correct:hover {{ background: #a7f3d0; }}
.btn-wrong {{ background: #fee2e2; color: #991b1b; border-color: #fca5a5; }}
.btn-wrong:hover {{ background: #fecaca; }}
.btn-unsure {{ background: #f3f4f6; color: #4b5563; border-color: #d1d5db; }}
.btn-unsure:hover {{ background: #e5e7eb; }}
.stats-bar {{ position: fixed; bottom: 0; left: 0; right: 0; background: #1a1a2e; color: #fff;
             padding: 10px 20px; display: flex; gap: 20px; font-size: 14px; z-index: 100;
             justify-content: center; }}
.stats-bar span {{ font-weight: 600; }}
.modal-overlay {{ display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
                  background: rgba(0,0,0,0.85); z-index: 200; justify-content: center; align-items: center; cursor: pointer; }}
.modal-overlay.active {{ display: flex; }}
.modal-overlay img {{ max-width: 95vw; max-height: 95vh; border-radius: 8px; }}
#export-btn {{ position: fixed; top: 20px; right: 20px; padding: 8px 20px; background: #3b82f6;
              color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; z-index: 100; }}
#export-btn:hover {{ background: #2563eb; }}
</style>
</head>
<body>
<h1>Bad Case 可视化审查</h1>
<p style="text-align:center;color:#6b7280;margin-bottom:20px;">模型: 12tags_Qwen3-VL-30B_full_add_tags_dynamic &nbsp;|&nbsp; 总 Bad Case: {len(bad_cases)}</p>

<button id="export-btn" onclick="exportResults()">导出审查结果</button>

<div class="summary">
<h2>总览</h2>
<table>
<tr><th>指标</th><th>数值</th></tr>
<tr><td>Bad Case 总数</td><td>{len(bad_cases)}</td></tr>
<tr><td>疑似多标签问题</td><td style="color:#d97706;font-weight:600;">{len(multi_cases)} ({len(multi_cases)/len(bad_cases)*100:.0f}%)</td></tr>
<tr><td>可能是真错误</td><td style="color:#ef4444;font-weight:600;">{len(non_multi_cases)} ({len(non_multi_cases)/len(bad_cases)*100:.0f}%)</td></tr>
</table>
<p style="margin-top:12px;font-size:0.9em;color:#6b7280;">
点击每个 case 下方的按钮来标记你的判断：<br>
<strong style="color:#065f46;">预测正确</strong> = 模型预测的标签确实合理（标注不完整）<br>
<strong style="color:#991b1b;">预测错误</strong> = 模型预测确实不对<br>
<strong style="color:#4b5563;">不确定</strong> = 需要看视频才能判断
</p>
</div>

<div class="filter-bar">
<label>按混淆对筛选:</label>
<select id="filter-pair" onchange="filterCases()">
<option value="all">全部</option>
<option value="multi">仅疑似多标签</option>
<option value="non-multi">仅非多标签</option>
""")

    pair_counts = defaultdict(int)
    for c in bad_cases:
        pair = normalize_pair(c['gt_label'], c['pred_label'])
        pair_counts[f"{pair[0]} ↔ {pair[1]}"] += 1
    for pair_name, cnt in sorted(pair_counts.items(), key=lambda x: -x[1]):
        html_parts.append(f'<option value="{pair_name}">{pair_name} ({cnt})</option>\n')

    html_parts.append("""</select>
<button onclick="filterCases()">筛选</button>
</div>

<div class="stats-bar">
<span>已标记: <span id="stat-total">0</span> / """ + str(len(bad_cases)) + """</span>
<span style="color:#6ee7b7;">预测正确: <span id="stat-correct">0</span></span>
<span style="color:#fca5a5;">预测错误: <span id="stat-wrong">0</span></span>
<span style="color:#d1d5db;">不确定: <span id="stat-unsure">0</span></span>
</div>

<div id="modal" class="modal-overlay" onclick="this.classList.remove('active')">
<img id="modal-img" src="">
</div>

<div id="cases-container">
""")

    for case in bad_cases:
        is_multi = case['is_multi']
        card_class = "case-card multi-label" if is_multi else "case-card"
        pair_name = f"{normalize_pair(case['gt_label'], case['pred_label'])[0]} ↔ {normalize_pair(case['gt_label'], case['pred_label'])[1]}"

        html_parts.append(f"""
<div class="{card_class}" id="case-{case['index']}" data-pair="{pair_name}" data-multi="{str(is_multi).lower()}" data-verdict="">
<div class="case-header">
<span class="idx">#{case['index']}</span>
<div>
<span class="label-box label-gt">GT: {case['gt_label']}</span>
<span class="label-box label-pred">Pred: {case['pred_label']}</span>
""")
        if is_multi:
            html_parts.append('<span class="label-multi">疑似多标签</span>')
        html_parts.append("""</div></div>
<div class="frames">
""")

        for frame in case.get('frames', []):
            html_parts.append(f"""<div class="frame-wrap">
<img src="data:image/jpeg;base64,{frame['b64']}" width="384" height="216"
     onclick="showModal(this.src)" loading="lazy">
<div class="frame-time">{frame['time']}</div>
</div>
""")

        if not case.get('frames'):
            html_parts.append('<p style="color:#ef4444;padding:20px;">视频文件不存在或无法读取</p>')

        html_parts.append(f"""</div>
<div class="video-path">{case['video_path']}</div>
<div class="verdict-buttons">
<button class="btn-correct" onclick="setVerdict({case['index']}, 'correct')">预测正确 (标注不完整)</button>
<button class="btn-wrong" onclick="setVerdict({case['index']}, 'wrong')">预测错误</button>
<button class="btn-unsure" onclick="setVerdict({case['index']}, 'unsure')">不确定 (需看视频)</button>
</div>
</div>
""")

    html_parts.append("""</div>

<script>
const verdicts = {};

function setVerdict(idx, verdict) {
    verdicts[idx] = verdict;
    const card = document.getElementById('case-' + idx);
    card.classList.remove('verdict-correct', 'verdict-wrong');
    if (verdict === 'correct') card.classList.add('verdict-correct');
    else if (verdict === 'wrong') card.classList.add('verdict-wrong');
    updateStats();
}

function updateStats() {
    let correct = 0, wrong = 0, unsure = 0;
    for (const v of Object.values(verdicts)) {
        if (v === 'correct') correct++;
        else if (v === 'wrong') wrong++;
        else if (v === 'unsure') unsure++;
    }
    document.getElementById('stat-total').textContent = Object.keys(verdicts).length;
    document.getElementById('stat-correct').textContent = correct;
    document.getElementById('stat-wrong').textContent = wrong;
    document.getElementById('stat-unsure').textContent = unsure;
}

function filterCases() {
    const filterVal = document.getElementById('filter-pair').value;
    const cards = document.querySelectorAll('.case-card');
    cards.forEach(card => {
        const pair = card.dataset.pair;
        const isMulti = card.dataset.multi;
        let show = true;
        if (filterVal === 'multi') show = isMulti === 'true';
        else if (filterVal === 'non-multi') show = isMulti === 'false';
        else if (filterVal !== 'all') show = pair === filterVal;
        card.style.display = show ? 'block' : 'none';
    });
}

function showModal(src) {
    document.getElementById('modal-img').src = src;
    document.getElementById('modal').classList.add('active');
}

function exportResults() {
    const results = [];
    document.querySelectorAll('.case-card').forEach(card => {
        const idx = card.id.replace('case-', '');
        const gt = card.querySelector('.label-gt').textContent.replace('GT: ', '');
        const pred = card.querySelector('.label-pred').textContent.replace('Pred: ', '');
        const verdict = verdicts[idx] || 'unreviewed';
        results.push({ index: idx, gt: gt, pred: pred, verdict: verdict });
    });
    const blob = new Blob([JSON.stringify(results, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'bad_case_review_results.json'; a.click();
}
</script>
</body>
</html>""")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))


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

        print(f"[{i:3d}/{n}] 抽帧: {os.path.basename(video_path)}  GT={gt}  Pred={pred}")
        frames = extract_frames(video_path, N_FRAMES) if video_path and os.path.exists(video_path) else []

        bad_cases.append({
            'index': i,
            'gt_label': gt,
            'pred_label': pred,
            'video_path': video_path,
            'is_multi': is_multi,
            'frames': frames,
        })

    bad_cases.sort(key=lambda c: (not c['is_multi'], c['gt_label'], c['pred_label']))

    output_path = os.path.join(OUTPUT_DIR, "bad_case_review.html")
    generate_html(bad_cases, output_path)
    print(f"\n审查页面已生成: {output_path}")
    print(f"总 Bad Case: {len(bad_cases)}")
    print(f"疑似多标签: {sum(1 for c in bad_cases if c['is_multi'])}")
    print(f"\n用浏览器打开 HTML 文件即可开始审查")


if __name__ == "__main__":
    main()
