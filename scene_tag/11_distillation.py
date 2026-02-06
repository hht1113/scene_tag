#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-235B 视频驾驶行为标注脚本
支持20秒视频切片，输出结构化驾驶行为标签（15类 + else）
支持 vLLM tensor parallel 分布式推理
支持 confidence 阈值过滤

用法:
    # 单视频标注（自动检测GPU数量）
    python 11_distillation.py \
        --video_path /data/video_001.mp4 \
        --model_path Qwen/Qwen3-VL-235B-A22B-Instruct

    # 8卡分布式推理 + 批量标注
    python 11_distillation.py \
        --video_dir /data/videos_20s/ \
        --model_path Qwen/Qwen3-VL-235B-A22B-Instruct \
        --tp 8 \
        --output results/annotations.json \
        --min_confidence 75

    # FP8量化版（4卡即可）
    python 11_distillation.py \
        --video_dir /data/videos_20s/ \
        --model_path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
        --tp 4
"""

import os
import re
import json
import argparse
import torch
import cv2
import tempfile
import shutil
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from vllm import LLM, SamplingParams


# ==================== 1. 类别定义（15类 + else） ====================
# 每个类别包含：定义、视觉线索、消歧规则
DRIVING_MANEUVER_CATEGORIES = {
    "TrafficLight_StraightStopOrGo": {
        "definition": "Ego vehicle stops at or starts from a traffic light for straight-line movement.",
        "visual_cues": (
            "Traffic light clearly visible ahead in ego's lane direction; "
            "ego vehicle decelerates to a full stop OR accelerates from standstill; "
            "steering wheel remains centered (no turning); "
            "lane markings and road geometry indicate straight-ahead path."
        ),
        "distinguish": (
            "vs LaneCruising_Straight: THIS label requires a traffic light causing the stop/go; "
            "vs StartStop_StartFromMainRoad: if a traffic light is the reason for stopping, use THIS label."
        ),
    },
    "TrafficLight_LeftTurnStopOrGo": {
        "definition": "Ego vehicle stops at or starts from a traffic light for left-turn movement.",
        "visual_cues": (
            "Traffic light with left-turn arrow/signal visible; "
            "ego vehicle positioned in a left-turn lane (often leftmost lane); "
            "left turn signal may be blinking on dashboard; "
            "ego decelerates to stop or begins moving when light turns green for left turn."
        ),
        "distinguish": (
            "vs Intersection_LeftTurn: THIS label is the stop/go phase at the light BEFORE the turn; "
            "Intersection_LeftTurn is the actual turning maneuver. They can appear sequentially."
        ),
    },
    "LaneChange_NavForIntersection": {
        "definition": "Ego vehicle changes lane for navigation purposes when approaching an intersection.",
        "visual_cues": (
            "Ego vehicle performs lateral movement crossing lane markings; "
            "intersection or junction visible ahead; "
            "NO obstacle or slow object in current lane (the motivation is navigation, not avoidance); "
            "turn signal activated before lane change."
        ),
        "distinguish": (
            "vs LaneChange_AvoidSlowVRU / LaneChange_AvoidStaticVehicle: "
            "THIS is navigation-motivated (getting into the correct lane for turning), "
            "NOT obstacle avoidance. Check if there is an obstacle in the original lane."
        ),
    },
    "LaneChange_AvoidSlowVRU": {
        "definition": (
            "Ego vehicle changes lane to avoid slow-moving vulnerable road users "
            "(pedestrians walking in lane, cyclists, scooter/e-bike riders)."
        ),
        "visual_cues": (
            "VRU visible ahead in ego's current lane, moving slowly in the same direction; "
            "ego vehicle performs a complete lane change (crosses lane markings) to overtake/avoid; "
            "VRU is NOT crossing the road — they are traveling along the lane."
        ),
        "distinguish": (
            "vs DynamicInteraction_VRUInLaneCrossing: HERE ego changes lane to go around a slow VRU; "
            "THERE ego slows/stops because VRU is crossing perpendicular to the lane. "
            "Key difference: lane change vs slow-down/stop."
        ),
    },
    "LaneChange_AvoidStaticVehicle": {
        "definition": (
            "Ego vehicle changes lane to avoid a stationary or parked vehicle "
            "that is blocking or partially blocking the current lane."
        ),
        "visual_cues": (
            "Stationary vehicle visible ahead in ego's lane (parked car, delivery truck, "
            "broken-down vehicle with hazard lights); "
            "ego performs lane change to go around it; "
            "the obstacle vehicle has ZERO velocity."
        ),
        "distinguish": (
            "vs LaneChange_AvoidSlowVRU: obstacle HERE is a vehicle (car/truck), not a person/cyclist. "
            "vs LaneChange_NavForIntersection: HERE there IS an obstacle motivating the lane change."
        ),
    },
    "DynamicInteraction_VRUInLaneCrossing": {
        "definition": (
            "Ego vehicle interacts with a vulnerable road user (pedestrian, cyclist) "
            "who is crossing the ego vehicle's lane path (perpendicular or diagonal crossing)."
        ),
        "visual_cues": (
            "VRU enters ego's lane from the side (crosswalk, jaywalking, cycling across); "
            "ego vehicle decelerates, stops, or swerves slightly to yield; "
            "VRU's movement direction is roughly perpendicular to ego's travel direction."
        ),
        "distinguish": (
            "vs LaneChange_AvoidSlowVRU: HERE ego typically slows/stops and does NOT change lane; "
            "THERE ego changes lane. Also, HERE the VRU is CROSSING, THERE the VRU is moving ALONG the lane."
        ),
    },
    "DynamicInteraction_VehicleInLaneCrossing": {
        "definition": (
            "Ego vehicle interacts with another vehicle crossing the ego's lane path "
            "(e.g., vehicle turning from a side street, vehicle crossing at an unsignalized intersection)."
        ),
        "visual_cues": (
            "Another vehicle enters or crosses ego's lane from a perpendicular or diagonal direction; "
            "ego adjusts speed (typically decelerates) to avoid collision; "
            "the other vehicle's trajectory intersects ego's lane."
        ),
        "distinguish": (
            "vs DynamicInteraction_StandardVehicleCutIn: CROSSING is roughly perpendicular; "
            "CUT-IN is a lateral merge into the same lane from an adjacent lane. "
            "Key: crossing = different direction; cut-in = same direction."
        ),
    },
    "DynamicInteraction_StandardVehicleCutIn": {
        "definition": (
            "Another vehicle from an adjacent lane merges/cuts in front of the ego vehicle "
            "into ego's lane, typically requiring ego to decelerate."
        ),
        "visual_cues": (
            "Vehicle in adjacent lane moves laterally into ego's lane ahead of ego; "
            "the merging vehicle and ego are traveling in the SAME general direction; "
            "ego may need to brake; following distance suddenly decreases."
        ),
        "distinguish": (
            "vs VehicleInLaneCrossing: CUT-IN vehicles travel in the same direction and merge laterally; "
            "CROSSING vehicles travel in a different direction and cross the lane."
        ),
    },
    "DynamicInteraction_LeadVehicleEmergencyBrake": {
        "definition": (
            "Lead vehicle ahead suddenly brakes very hard (deceleration >= 0.3g or brake lights flash rapidly), "
            "forcing the ego vehicle to react with immediate hard braking to avoid rear-end collision."
        ),
        "visual_cues": (
            "Lead vehicle's brake lights activate suddenly and intensely; "
            "the gap between ego and lead vehicle closes rapidly; "
            "ego vehicle performs hard braking (visible deceleration, nose dip); "
            "reaction happens within 1-2 seconds of lead vehicle braking. "
            "This is a SAFETY-CRITICAL event — be especially precise about timing."
        ),
        "distinguish": (
            "vs normal following: Emergency braking is SUDDEN and HARD, not gradual speed adjustment. "
            "Only label this when the deceleration is clearly emergency-level."
        ),
    },
    "StartStop_StartFromMainRoad": {
        "definition": (
            "Ego vehicle starts moving from a fully stopped position on a main road, "
            "where the stop was NOT caused by a traffic light."
        ),
        "visual_cues": (
            "Ego vehicle was stationary (0 km/h) on the main road; "
            "begins accelerating forward; "
            "no traffic light visible as the cause of the stop; "
            "could be after yielding, waiting for traffic gap, or brief stop."
        ),
        "distinguish": (
            "vs TrafficLight_StraightStopOrGo: if a traffic light caused the stop, "
            "use the TrafficLight label instead. "
            "vs StartStop_ParkRoadside: THIS is starting to move; THAT is stopping to park."
        ),
    },
    "StartStop_ParkRoadside": {
        "definition": (
            "Ego vehicle intentionally decelerates and stops/parks at the roadside "
            "(not at a traffic light or intersection)."
        ),
        "visual_cues": (
            "Ego vehicle moves toward the road edge/curb; "
            "gradually decelerates to a complete stop; "
            "may activate hazard lights or right turn signal; "
            "the stop is intentional (parking), not forced by traffic."
        ),
        "distinguish": (
            "vs TrafficLight stops: parking is intentional at roadside, not at traffic control. "
            "vs any other stop: this is a deliberate pull-over to the side of the road."
        ),
    },
    "Intersection_LeftTurn": {
        "definition": (
            "Ego vehicle executes a left turn at an intersection "
            "(including protected/unprotected left turns)."
        ),
        "visual_cues": (
            "Intersection clearly visible; "
            "ego vehicle's steering wheel rotates significantly leftward (>30 degrees); "
            "vehicle trajectory curves to the left; "
            "turn signal may be active; "
            "ego transitions from one road direction to a roughly perpendicular left direction."
        ),
        "distinguish": (
            "vs TrafficLight_LeftTurnStopOrGo: THAT is the stop/go at the light; "
            "THIS is the actual turning maneuver. They are sequential phases. "
            "vs Intersection_StandardUTurn: left turn is ~90 degrees; U-turn is ~180 degrees."
        ),
    },
    "Intersection_RightTurn": {
        "definition": (
            "Ego vehicle executes a right turn at an intersection "
            "(including protected/unprotected right turns)."
        ),
        "visual_cues": (
            "Intersection clearly visible; "
            "ego vehicle's steering wheel rotates significantly rightward (>30 degrees); "
            "vehicle trajectory curves to the right; "
            "turn signal may be active; "
            "ego transitions from one road direction to a roughly perpendicular right direction."
        ),
        "distinguish": (
            "Similar to Intersection_LeftTurn but in the opposite direction."
        ),
    },
    "Intersection_StandardUTurn": {
        "definition": (
            "Ego vehicle makes a U-turn (approximately 180-degree turn) "
            "at an intersection or designated U-turn area."
        ),
        "visual_cues": (
            "Ego vehicle makes a very wide left turn approaching 180 degrees; "
            "after the maneuver, ego is traveling in the opposite direction; "
            "typically occurs at intersection with U-turn permitted sign or wide median opening."
        ),
        "distinguish": (
            "vs Intersection_LeftTurn: U-turn is ~180 degrees (reverses direction); "
            "left turn is ~90 degrees (turns onto perpendicular road)."
        ),
    },
    "LaneCruising_Straight": {
        "definition": (
            "Ego vehicle cruises straight in its lane at a relatively steady speed "
            "without any notable events, maneuvers, or interactions."
        ),
        "visual_cues": (
            "No significant steering input; "
            "relatively constant speed; "
            "staying within lane markings; "
            "no notable interactions with other road users; "
            "no traffic lights, intersections, or obstacles requiring response."
        ),
        "distinguish": (
            "Use this ONLY when genuinely nothing notable is happening. "
            "If there is ANY identifiable maneuver from the other 14 categories, use that instead."
        ),
    },
}

CATEGORY_LABELS = list(DRIVING_MANEUVER_CATEGORIES.keys())

# 构建详细类别定义文本（用于 prompt）
CATEGORY_LIST_STR = "\n".join(
    [f"  {i+1}. {label}" for i, label in enumerate(CATEGORY_LABELS)]
)

CATEGORY_DEFINITIONS_DETAILED = ""
for i, (label, info) in enumerate(DRIVING_MANEUVER_CATEGORIES.items(), 1):
    CATEGORY_DEFINITIONS_DETAILED += (
        f"{i}. {label}\n"
        f"   Definition: {info['definition']}\n"
        f"   Visual cues: {info['visual_cues']}\n"
        f"   Disambiguation: {info['distinguish']}\n\n"
    )
CATEGORY_DEFINITIONS_DETAILED += (
    "16. else\n"
    "   Definition: Any driving behavior not matching the above 15 categories.\n"
    "   Use sparingly. If unsure between a specific label and 'else', prefer 'else' to avoid mislabeling.\n"
)


# ==================== 2. 系统 Prompt ====================
SYSTEM_PROMPT = f"""You are an expert autonomous driving scene annotator. Your task is to analyze a 20-second ego-vehicle driving video and identify ALL driving maneuvers with precise timing and high accuracy.

You MUST label every maneuver using ONLY the predefined categories below.

=== AVAILABLE LABELS (15 categories + else) ===
{CATEGORY_LIST_STR}
  16. else  (ONLY when no label above matches)

=== DETAILED CATEGORY DEFINITIONS ===
{CATEGORY_DEFINITIONS_DETAILED}
=== LABELING RULES (STRICTLY FOLLOW) ===
1. Assign a label ONLY if the action CLEARLY matches the category definition.
2. Report your confidence (0-100%) for each segment honestly.
3. Minimum segment duration: 1 second.
4. Start and end times MUST be whole integers (e.g., 0, 3, 8, 15, 20). Do NOT use decimals.
5. Time range must be within [0, 20] seconds.
6. Segments must be in chronological order and should cover the entire 20 seconds.
7. Adjacent segments with the same label should be merged into one.
8. If multiple DIFFERENT maneuvers happen simultaneously, list them as separate lines (overlapping times are OK for different labels).
9. For safety-critical events (emergency brake, VRU crossing), be especially precise about start/end timing.

=== PRIORITY RULES (when a scene could match multiple categories) ===
- Emergency braking (LeadVehicleEmergencyBrake) > all other labels
- Specific interaction labels > generic labels (e.g., VRUInLaneCrossing > else)
- Intersection turn labels > traffic light labels (if turning phase, use turn label)
- Lane change for obstacle avoidance > lane change for navigation
- If confidence < 60% for any specific label, use "else" instead

=== OUTPUT FORMAT (ONE LINE PER SEGMENT, NO OTHER TEXT) ===
<driving_maneuver>LABEL</driving_maneuver> from <start_time>START</start_time> to <end_time>END</end_time> seconds (confidence: XX%)

=== EXAMPLE OUTPUT ===
<driving_maneuver>LaneCruising_Straight</driving_maneuver> from <start_time>0</start_time> to <end_time>5</end_time> seconds (confidence: 95%)
<driving_maneuver>TrafficLight_StraightStopOrGo</driving_maneuver> from <start_time>5</start_time> to <end_time>9</end_time> seconds (confidence: 88%)
<driving_maneuver>Intersection_LeftTurn</driving_maneuver> from <start_time>9</start_time> to <end_time>15</end_time> seconds (confidence: 82%)
<driving_maneuver>LaneCruising_Straight</driving_maneuver> from <start_time>15</start_time> to <end_time>20</end_time> seconds (confidence: 91%)

IMPORTANT: Output ONLY the structured annotation lines. Do NOT add any reasoning, explanation, or commentary."""

USER_PROMPT = (
    "Carefully analyze this 20-second driving video frame by frame. "
    "Identify ALL ego vehicle maneuvers with precise integer time boundaries "
    "and confidence scores. Cover the entire 20-second duration."
)


# ==================== 3. 视频预处理 ====================
def preprocess_video(
    video_path: str,
    target_resolution: Tuple[int, int] = (256, 256),
    max_frames: int = 40,
) -> Tuple[str, str, int]:
    """
    预处理视频：读取2FPS输入视频，将每帧下采样到 target_resolution，
    保存为临时视频文件。

    Args:
        video_path: 输入视频路径（已经是2FPS）
        target_resolution: 目标分辨率 (width, height)
        max_frames: 最大帧数限制

    Returns:
        (preprocessed_video_path, temp_dir, frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  原始视频: {orig_w}x{orig_h}, {fps:.1f}fps, {total_frames}帧")

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="vl_anno_")
    output_path = os.path.join(temp_dir, "preprocessed.mp4")

    # 写入预处理后的视频（保持原始fps，resize到目标分辨率）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # 下采样到目标分辨率（INTER_AREA 适合缩小）
        resized = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_AREA)
        writer.write(resized)
        frame_count += 1

    cap.release()
    writer.release()

    print(
        f"  预处理后: {target_resolution[0]}x{target_resolution[1]}, "
        f"{fps:.1f}fps, {frame_count}帧"
    )
    return output_path, temp_dir, frame_count


# ==================== 4. vLLM 推理核心类 ====================
class Qwen3VLVideoAnnotator:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.95,
        target_resolution: Tuple[int, int] = (256, 256),
        max_frames: int = 40,
        min_confidence: int = 70,
        **kwargs,
    ):
        """
        初始化 Qwen3-VL-235B 视频标注器

        Args:
            model_path: HuggingFace模型ID或本地路径
            tensor_parallel_size: 张量并行GPU数量
                - 235B FP16/BF16 需 8 卡
                - 235B FP8 量化版可 4 卡
            max_model_len: 最大序列长度
            gpu_memory_utilization: GPU显存利用率
            target_resolution: 帧下采样目标分辨率 (width, height)
            max_frames: 最大输入帧数
            min_confidence: 最低置信度阈值（低于此值的 segment 将被过滤）
        """
        print("=" * 60)
        print("初始化 Qwen3-VL-235B 视频标注器")
        print("=" * 60)
        print(f"  模型路径:     {model_path}")
        print(f"  张量并行:     {tensor_parallel_size} GPUs")
        print(f"  最大序列长度: {max_model_len}")
        print(f"  帧分辨率:     {target_resolution}")
        print(f"  最大帧数:     {max_frames}")
        print(f"  置信度阈值:   {min_confidence}%")

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enable_chunked_prefill=False,  # 视频任务建议关闭
            max_num_seqs=1,  # 单视频推理，避免OOM
            limit_mm_per_prompt={"video": 1},
            allowed_local_media_path="/",  # 允许加载本地视频文件
            **kwargs,
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,  # 确定性输出
            max_tokens=2048,
            stop=["\n\n\n"],  # 防止过长输出
        )

        self.target_resolution = target_resolution
        self.max_frames = max_frames
        self.min_confidence = min_confidence

        print("模型加载完成!\n")

    def annotate_video(self, video_path: str) -> Dict:
        """
        对单个 20 秒视频切片进行标注

        Returns:
            {
                "video_path": str,
                "segments": [...],           # 通过置信度过滤的有效 segments
                "segments_dropped": [...],   # 被过滤掉的低置信度 segments
                "raw_output": str,
                "frame_count": int,
                "min_confidence": int
            }
        """
        # 1. 预处理视频（下采样到 256x256）
        preprocessed_path, temp_dir, frame_count = preprocess_video(
            video_path,
            target_resolution=self.target_resolution,
            max_frames=self.max_frames,
        )

        try:
            # 2. 构造多模态输入
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": f"file://{preprocessed_path}"
                            },
                        },
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ]

            # 3. 推理
            outputs = self.llm.chat(
                messages=[messages],
                sampling_params=self.sampling_params,
            )
            raw_text = outputs[0].outputs[0].text.strip()
            print(f"  模型原始输出:\n{raw_text}")

            # 4. 解析输出
            all_segments = self._parse_output(raw_text)

            # 5. 置信度过滤
            filtered_segments = [
                s for s in all_segments if s["confidence"] >= self.min_confidence
            ]
            dropped_segments = [
                s for s in all_segments if s["confidence"] < self.min_confidence
            ]
            if dropped_segments:
                print(
                    f"  过滤掉 {len(dropped_segments)} 个低置信度段"
                    f"（阈值 {self.min_confidence}%）"
                )

            return {
                "video_path": video_path,
                "segments": filtered_segments,
                "segments_dropped": dropped_segments,
                "raw_output": raw_text,
                "frame_count": frame_count,
                "min_confidence": self.min_confidence,
            }

        finally:
            # 清理临时文件
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _parse_output(self, raw_text: str) -> List[Dict]:
        """
        解析模型输出为结构化标签

        示例输入:
        <driving_maneuver>Intersection_LeftTurn</driving_maneuver> from <start_time>3</start_time> to <end_time>8</end_time> seconds (confidence: 95%)

        时间戳解析为整数（如果模型输出了小数，四舍五入为整数）。
        """
        segments = []

        pattern = (
            r"<driving_maneuver>([^<]+)</driving_maneuver>\s+"
            r"from\s+<start_time>([\d.]+)</start_time>\s+"
            r"to\s+<end_time>([\d.]+)</end_time>\s+seconds\s+"
            r"\(confidence:\s*(\d+)%\)"
        )

        for match in re.finditer(pattern, raw_text):
            label = match.group(1).strip()
            start = int(round(float(match.group(2))))  # 四舍五入为整数
            end = int(round(float(match.group(3))))  # 四舍五入为整数
            conf = int(match.group(4))

            # 验证标签合法性
            if label not in CATEGORY_LABELS + ["else"]:
                print(f"  警告: 无效标签 '{label}'，跳过")
                continue

            # 验证时间范围
            start = max(0, min(start, 20))
            end = max(0, min(end, 20))
            if start >= end:
                print(f"  警告: 无效时间范围 [{start}-{end}]，跳过")
                continue

            segments.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "confidence": conf,
                }
            )

        if not segments:
            print("  警告: 未解析到有效标签，返回空结果")

        return segments


# ==================== 5. 批量处理 ====================
def batch_annotate_videos(
    video_dir: str,
    output_json: str,
    model_path: str,
    tensor_parallel_size: int = 8,
    min_confidence: int = 70,
    max_videos: Optional[int] = None,
    target_resolution: Tuple[int, int] = (256, 256),
    max_frames: int = 40,
):
    """批量处理视频目录下所有 .mp4 文件"""
    annotator = Qwen3VLVideoAnnotator(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        target_resolution=target_resolution,
        max_frames=max_frames,
        min_confidence=min_confidence,
    )

    # 确保输出目录存在
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    video_files = sorted(
        [f for f in Path(video_dir).glob("*.mp4") if f.is_file()]
    )

    if max_videos:
        video_files = video_files[:max_videos]

    total = len(video_files)
    print(f"\n共发现 {total} 个视频文件\n")

    for i, video_path in enumerate(video_files):
        print(f"{'=' * 60}")
        print(f"[{i + 1}/{total}] 处理: {video_path.name}")
        print(f"{'=' * 60}")

        try:
            result = annotator.annotate_video(str(video_path))
            results.append(result)

            # 实时保存（防中断丢失）
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            # 打印结果摘要
            print(f"\n  有效标注段 ({len(result['segments'])} 个):")
            for seg in result["segments"]:
                print(
                    f"    {seg['label']:45s} "
                    f"[{seg['start']:2d}s - {seg['end']:2d}s] "
                    f"conf={seg['confidence']}%"
                )

            if result.get("segments_dropped"):
                print(
                    f"  被过滤的低置信度段 ({len(result['segments_dropped'])} 个):"
                )
                for seg in result["segments_dropped"]:
                    print(
                        f"    {seg['label']:45s} "
                        f"[{seg['start']:2d}s - {seg['end']:2d}s] "
                        f"conf={seg['confidence']}% [DROPPED]"
                    )

        except Exception as e:
            print(f"  处理失败: {e}")
            traceback.print_exc()
            results.append(
                {
                    "video_path": str(video_path),
                    "segments": [],
                    "error": str(e),
                }
            )
            continue

    print(f"\n{'=' * 60}")
    success_count = sum(1 for r in results if "error" not in r)
    print(f"全部完成! 成功标注 {success_count}/{total} 个视频")
    print(f"结果保存至: {output_json}")
    print(f"{'=' * 60}")


# ==================== 6. 命令行入口 ====================
def get_tensor_parallel_size(tp_arg: str) -> int:
    """解析 tensor parallel 参数，支持 'auto' 自动检测"""
    if tp_arg.lower() == "auto":
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            raise RuntimeError(
                "未检测到可用 GPU! Qwen3-VL-235B 需要至少 4 块 GPU (FP8) 或 8 块 GPU (FP16)。"
            )
        print(f"自动检测到 {gpu_count} 块 GPU，将全部用于张量并行推理")
        return gpu_count
    else:
        tp = int(tp_arg)
        available = torch.cuda.device_count()
        if tp > available:
            raise RuntimeError(
                f"请求 {tp} 块 GPU 但仅检测到 {available} 块可用 GPU"
            )
        return tp


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-235B 视频驾驶行为标注工具（支持分布式推理）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 单视频标注（自动检测GPU数量）
  python 11_distillation.py \\
      --video_path /data/video_001.mp4 \\
      --model_path Qwen/Qwen3-VL-235B-A22B-Instruct

  # 8卡分布式推理 + 批量标注
  python 11_distillation.py \\
      --video_dir /data/videos_20s/ \\
      --model_path Qwen/Qwen3-VL-235B-A22B-Instruct \\
      --tp 8 \\
      --output results/annotations.json \\
      --min_confidence 75

  # FP8量化版（可用4卡运行）
  python 11_distillation.py \\
      --video_dir /data/videos_20s/ \\
      --model_path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \\
      --tp 4

  # 只处理前20个视频，置信度阈值80%
  python 11_distillation.py \\
      --video_dir /data/videos_20s/ \\
      --model_path /local/models/qwen3-vl-235b \\
      --tp auto \\
      --max_videos 20 \\
      --min_confidence 80
        """,
    )

    # 输入（二选一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video_path", type=str, help="单个视频文件路径"
    )
    input_group.add_argument(
        "--video_dir", type=str, help="视频目录路径（批量处理所有 .mp4 文件）"
    )

    # 模型配置
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型路径（HuggingFace ID 或本地路径）",
    )
    parser.add_argument(
        "--tp",
        type=str,
        default="auto",
        help=(
            "张量并行 GPU 数量（默认 auto 自动检测）。"
            "235B FP16 需 8 卡，FP8 可 4 卡"
        ),
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=32768,
        help="最大序列长度（默认 32768）",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="GPU 显存利用率（默认 0.95）",
    )

    # 视频处理
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="帧下采样分辨率（默认 256，即 256x256 正方形）",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=40,
        help="最大输入帧数（默认 40，即 20s x 2fps）",
    )

    # 过滤
    parser.add_argument(
        "--min_confidence",
        type=int,
        default=70,
        help="最低置信度阈值 (0-100)，低于此值的标注段将被过滤（默认 70）",
    )

    # 输出
    parser.add_argument(
        "--output",
        type=str,
        default="annotations.json",
        help="输出 JSON 文件路径（默认 annotations.json）",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="最大处理视频数量（默认全部处理）",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 解析 GPU 数量
    tp_size = get_tensor_parallel_size(args.tp)
    target_res = (args.resolution, args.resolution)

    if args.video_path:
        # ===== 单视频模式 =====
        annotator = Qwen3VLVideoAnnotator(
            model_path=args.model_path,
            tensor_parallel_size=tp_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            target_resolution=target_res,
            max_frames=args.max_frames,
            min_confidence=args.min_confidence,
        )

        result = annotator.annotate_video(args.video_path)

        # 打印结果
        print(f"\n{'=' * 60}")
        print(f"标注结果 ({len(result['segments'])} 个有效段):")
        print(f"{'=' * 60}")
        for seg in result["segments"]:
            print(
                f"  {seg['label']:45s} "
                f"[{seg['start']:2d}s - {seg['end']:2d}s] "
                f"conf={seg['confidence']}%"
            )

        if result.get("segments_dropped"):
            print(
                f"\n被过滤的低置信度段 ({len(result['segments_dropped'])} 个):"
            )
            for seg in result["segments_dropped"]:
                print(
                    f"  {seg['label']:45s} "
                    f"[{seg['start']:2d}s - {seg['end']:2d}s] "
                    f"conf={seg['confidence']}% [DROPPED]"
                )

        # 保存结果
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {args.output}")

    else:
        # ===== 批量模式 =====
        batch_annotate_videos(
            video_dir=args.video_dir,
            output_json=args.output,
            model_path=args.model_path,
            tensor_parallel_size=tp_size,
            min_confidence=args.min_confidence,
            max_videos=args.max_videos,
            target_resolution=target_res,
            max_frames=args.max_frames,
        )
