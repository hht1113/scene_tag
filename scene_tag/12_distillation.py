#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL-235B 视频驾驶行为标注脚本（API 客户端）

连接已部署的 vLLM 服务（由 11_deploy.py 启动），
从视频列表读取视频，切片抽帧，通过 API 进行驾驶行为标注。

前置条件:
    先在另一个终端运行: python 11_deploy.py

用法:
    # 批量标注 1000 个视频
    python 12_distillation.py \
        --api_base http://localhost:8000/v1 \
        --video_list /mnt/pfs/houhaotian/sampled_1000_videos.txt \
        --output results/annotations.json

    # 单视频标注
    python 12_distillation.py \
        --api_base http://localhost:8000/v1 \
        --video_path /data/video_001.mp4

    # 自定义参数
    python 12_distillation.py \
        --api_base http://localhost:8000/v1 \
        --video_list /mnt/pfs/houhaotian/sampled_1000_videos.txt \
        --concurrency 2 \
        --min_confidence 75 \
        --output results/annotations.json

    # 处理目录下所有视频
    python 12_distillation.py \
        --api_base http://localhost:8000/v1 \
        --video_dir /data/videos_20s/ \
        --max_videos 100
"""

import os
import re
import sys
import json
import time
import base64
import argparse
import traceback
import cv2
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# 绕过系统 HTTP 代理，直接连接本地 vLLM 服务
NO_PROXY = {"http": None, "https": None}


# ==================== 1. 类别定义（17类 + else） ====================
# 每个类别包含：定义、视觉线索、消歧规则
DRIVING_MANEUVER_CATEGORIES = {
    "TrafficLight_StraightStopOrGo": {
        "definition": (
            "Ego vehicle stops at or starts from a traffic light for STRAIGHT-LINE movement ONLY. "
            "This label requires the ego vehicle to be in a STRAIGHT-GOING lane AND "
            "the traffic light controlling straight movement to be visible. "
            "The ego must show a clear TRANSITION: either decelerating to stop OR accelerating from stop."
        ),
        "visual_cues": (
            "Traffic light for straight direction clearly visible; "
            "ego is in a straight-going lane (NOT a left-turn lane or right-turn lane); "
            "ego decelerates to a full stop OR accelerates from standstill; "
            "steering wheel remains centered; lane markings indicate straight-ahead path."
        ),
        "distinguish": (
            "CRITICAL DISTINCTIONS: "
            "1) If ego is in a LEFT-TURN LANE or the leftmost lane at an intersection → use TrafficLight_LeftTurnStopOrGo instead. "
            "2) If ego makes a RIGHT TURN at the intersection → use Intersection_RightTurn instead, NOT this label. "
            "3) If ego is completely stationary the ENTIRE video (never moves at all) → use 'else', NOT this label. "
            "This label REQUIRES a visible stop→go or go→stop TRANSITION during the video. "
            "vs StartStop_StartFromMainRoad: if a traffic light caused the stop, use THIS label."
        ),
    },
    "TrafficLight_LeftTurnStopOrGo": {
        "definition": (
            "Ego vehicle stops at or starts from a traffic light for LEFT-TURN movement. "
            "This includes: ego in a left-turn lane waiting for a left-turn signal, "
            "OR ego in the leftmost lane at an intersection preparing to turn left. "
            "A dedicated left-turn arrow signal does NOT need to be clearly visible — "
            "if ego is in a left-turn lane at a traffic light, this label applies."
        ),
        "visual_cues": (
            "Ego is positioned in a left-turn lane (often the leftmost lane); "
            "traffic light visible at intersection; "
            "ego decelerates to stop or begins moving for a left turn; "
            "left-turn arrow signal may or may not be clearly visible; "
            "lane markings or road geometry suggest left-turn intent."
        ),
        "distinguish": (
            "vs TrafficLight_StraightStopOrGo: If ego is in a left-turn lane, ALWAYS use THIS label, "
            "even if you cannot clearly see a left-turn arrow signal. "
            "vs Intersection_LeftTurn: THIS label is the stop/go phase at the light BEFORE the turn; "
            "Intersection_LeftTurn is the actual turning maneuver. They can appear sequentially."
        ),
    },
    "TrafficLight_DarkIntersectionPass": {
        "definition": (
            "Ego vehicle encounters an intersection where the traffic lights are OFF, dark, "
            "or malfunctioning (not illuminated). The ego vehicle is typically STUCK or CONFUSED — "
            "it may stop for a long time, proceed very cautiously, or creep forward inch by inch "
            "trying to find a safe gap to pass through. "
            "This is a trapped/stuck scenario: no signals guide traffic, "
            "so the ego must figure out when it is safe to go."
        ),
        "visual_cues": (
            "Traffic light structure visible but all lights are dark/off/unlit; "
            "ego stops and waits for an extended period, or creeps forward very slowly; "
            "other vehicles may also be hesitant or behave unpredictably; "
            "ego may inch forward, stop, inch forward again — typical stuck/trapped behavior; "
            "no signal indication visible (no red, green, or yellow lit)."
        ),
        "distinguish": (
            "vs TrafficLight_StraightStopOrGo/LeftTurnStopOrGo: THOSE require WORKING traffic lights; "
            "THIS label is for dark/off/malfunctioning traffic lights. "
            "vs AbnormalStop_StuckWaiting: If the traffic lights are dark/off, use THIS label. "
            "If lights are working but ego is stuck for other reasons, use AbnormalStop_StuckWaiting."
        ),
    },
    "AbnormalStop_StuckWaiting": {
        "definition": (
            "Ego vehicle is STUCK, TRAPPED, or WAITING for an abnormally long time in a complex situation, "
            "unable to proceed normally. The ego is essentially frozen in place or creeping very slowly, "
            "waiting for a safe opportunity to move. Common scenarios include:\n"
            "  - Unprotected left turn: ego is inside the intersection waiting for oncoming traffic to clear\n"
            "  - Right turn yielding: ego waits at a right turn for pedestrians/cyclists to cross\n"
            "  - Narrow road meeting: ego encounters oncoming traffic on a narrow road (会车) and must wait\n"
            "  - General trapped situation: ego cannot move forward due to complex traffic around it\n"
            "The key feature is that the ego vehicle is STATIONARY or barely moving for a PROLONGED period "
            "(typically >3 seconds) due to yielding, confusion, or being blocked."
        ),
        "visual_cues": (
            "Ego vehicle is completely stopped or moving extremely slowly (<5 km/h); "
            "ego is NOT at a normal red light — the situation is more complex; "
            "oncoming vehicles, pedestrians, or other obstacles prevent ego from proceeding; "
            "ego may be in the middle of an intersection, at a turn, or on a narrow road; "
            "other vehicles or VRUs are moving around/past the ego while ego waits; "
            "the waiting duration is noticeably long (several seconds to entire video)."
        ),
        "distinguish": (
            "vs TrafficLight stops: If ego is stopped at a WORKING red light, use TrafficLight label, NOT this. "
            "vs TrafficLight_DarkIntersectionPass: If traffic lights are dark/off, use THAT label instead. "
            "THIS label is for: working traffic lights BUT ego is stuck due to yielding/conflict "
            "(e.g., green light but cannot go because of oncoming traffic during unprotected left turn), "
            "OR no traffic light at all but ego is stuck in a conflict situation. "
            "vs normal driving: This is NOT normal waiting at a red light — it is an ABNORMAL prolonged stop."
        ),
    },
    "LaneChange_NavForIntersection": {
        "definition": (
            "Ego vehicle changes lane when approaching an intersection, for the purpose of "
            "getting into the correct lane for an upcoming turn or direction. "
            "Since navigation intent cannot be directly observed from video, "
            "ANY lane change near an intersection is treated as navigation-motivated. "
            "Examples: changing from a straight lane to a left-turn lane, "
            "from a straight lane to a right-turn lane, or repositioning between lanes near an intersection."
        ),
        "visual_cues": (
            "Ego performs lateral movement crossing lane markings; "
            "intersection or junction visible ahead (within ~200m); "
            "ego moves into a different-purpose lane (e.g., straight lane → turn lane); "
            "turn signal may be activated."
        ),
        "distinguish": (
            "vs LaneChange_AvoidSlowVRU / LaneChange_AvoidStaticVehicle: "
            "If the lane change happens near an intersection AND there is no obvious obstacle in the current lane, "
            "prefer THIS label. Only use avoidance labels when there is a CLEAR obstacle being avoided. "
            "This label should be used BROADLY — any lane change in the approach zone of an intersection."
        ),
    },
    "LaneChange_AvoidSlowVRU": {
        "definition": (
            "Ego vehicle performs a LANE CHANGE to avoid slow-moving vulnerable road users "
            "(pedestrians walking in lane, cyclists, scooter/e-bike riders). "
            "The LANE CHANGE is the critical element — ego must cross lane markings to go around the VRU."
        ),
        "visual_cues": (
            "VRU visible ahead in ego's current lane, moving slowly in the same direction; "
            "ego vehicle performs a COMPLETE lane change (fully crosses lane markings) to overtake/avoid; "
            "VRU is NOT crossing the road — they are traveling along the lane; "
            "ego's lateral displacement is significant (full lane change, not just a slight swerve)."
        ),
        "distinguish": (
            "MUST HAVE A LANE CHANGE. If ego only slows down without changing lane → NOT this label. "
            "vs DynamicInteraction_VRUInLaneCrossing: HERE ego changes lane to go around a slow VRU; "
            "THERE ego slows/stops because VRU is crossing perpendicular to the lane."
        ),
    },
    "LaneChange_AvoidStaticVehicle": {
        "definition": (
            "Ego vehicle performs a LANE CHANGE to avoid a stationary or parked vehicle "
            "that is blocking or partially blocking the current lane. "
            "The LANE CHANGE is the critical element — ego must cross lane markings to go around the obstacle."
        ),
        "visual_cues": (
            "Stationary vehicle visible ahead in ego's lane (parked car, delivery truck, "
            "broken-down vehicle with hazard lights); "
            "ego performs a COMPLETE lane change to go around it; "
            "the obstacle vehicle has ZERO velocity; "
            "ego's lateral displacement is significant (full lane change)."
        ),
        "distinguish": (
            "MUST HAVE A LANE CHANGE. If ego only slows down behind the static vehicle → NOT this label. "
            "vs LaneChange_AvoidSlowVRU: obstacle HERE is a vehicle (car/truck), not a person/cyclist. "
            "vs LaneChange_NavForIntersection: HERE there IS a blocking obstacle motivating the lane change."
        ),
    },
    "DynamicInteraction_VRUInLaneCrossing": {
        "definition": (
            "Ego vehicle interacts with a vulnerable road user (pedestrian, cyclist) "
            "who is crossing the ego vehicle's lane path (perpendicular or diagonal crossing). "
            "IMPORTANT: The ego vehicle must be MOVING or about to move — this is an active interaction. "
            "Do NOT use this label when ego is simply waiting at a red light and VRUs are crossing "
            "at the intersection as part of normal traffic flow."
        ),
        "visual_cues": (
            "VRU enters ego's lane from the side (crosswalk, jaywalking, cycling across); "
            "ego vehicle decelerates, stops, or swerves slightly to yield; "
            "VRU's movement direction is roughly perpendicular to ego's travel direction; "
            "ego was MOVING before the interaction OR is about to start moving."
        ),
        "distinguish": (
            "CRITICAL: If ego is stationary at a red light and VRUs are simply walking across "
            "the intersection crosswalk as part of normal signal-controlled traffic → NOT this label. "
            "This label requires ego to be ACTIVELY driving and needing to react to the VRU. "
            "vs LaneChange_AvoidSlowVRU: HERE ego slows/stops; THERE ego changes lane."
        ),
    },
    "DynamicInteraction_VehicleInLaneCrossing": {
        "definition": (
            "Ego vehicle interacts with another vehicle crossing the ego's lane path "
            "(e.g., vehicle turning from a side street, vehicle crossing at an unsignalized intersection). "
            "IMPORTANT: The ego vehicle must be MOVING or about to move — this is an active interaction. "
            "Do NOT use this label when ego is stopped at a red light and cross-traffic vehicles "
            "are simply flowing through the intersection as part of normal signal-controlled traffic."
        ),
        "visual_cues": (
            "Another vehicle enters or crosses ego's lane from a perpendicular or diagonal direction; "
            "ego adjusts speed (typically decelerates) to avoid collision; "
            "the other vehicle's trajectory intersects ego's lane; "
            "ego was MOVING before the interaction OR is about to start moving."
        ),
        "distinguish": (
            "CRITICAL: If ego is stationary at a red light and vehicles are simply driving through "
            "the perpendicular green-light direction as normal traffic → NOT this label. "
            "This label requires the crossing vehicle to create an actual conflict/interaction with ego. "
            "vs DynamicInteraction_StandardVehicleCutIn: CROSSING is roughly perpendicular; "
            "CUT-IN is a lateral merge from an adjacent lane in the same direction."
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
            "The lead vehicle ahead suddenly brakes very hard, AND the ego vehicle reacts "
            "with immediate hard braking. This label describes the PROCESS of emergency braking — "
            "both the lead vehicle and ego must be DECELERATING from a moving state. "
            "Do NOT use this label if the lead vehicle is already stationary/stopped the entire time."
        ),
        "visual_cues": (
            "Lead vehicle's brake lights activate suddenly and intensely; "
            "the gap between ego and lead vehicle closes rapidly; "
            "ego vehicle performs hard braking (visible rapid deceleration, nose dip); "
            "BOTH vehicles were MOVING before the event and come to a stop or near-stop; "
            "reaction happens within 1-2 seconds. "
            "This is a SAFETY-CRITICAL event — be especially precise about timing."
        ),
        "distinguish": (
            "CRITICAL: If the lead vehicle is ALREADY stopped and never moves → NOT this label. "
            "This label requires a DECELERATION PROCESS from moving to stopped/slow. "
            "vs normal car-following: Emergency braking is SUDDEN and HARD (visible jerk/nose-dip), "
            "not gradual speed adjustment. "
            "vs TrafficLight stop: if braking is due to a red light, use TrafficLight label instead."
        ),
    },
    "StartStop_StartFromMainRoad": {
        "definition": (
            "Ego vehicle starts moving from a fully stopped position at the ROADSIDE, "
            "where the stop was NOT caused by a traffic light. "
            "This is typically ego pulling out from a parked/stopped position on the side of the road. "
            "NOT a traffic light start — if a traffic light is visible and caused the stop, "
            "use a TrafficLight label instead."
        ),
        "visual_cues": (
            "Ego vehicle was stationary at the roadside (possibly parked or pulled over); "
            "begins accelerating forward and merging into traffic; "
            "NO traffic light visible as the cause of the stop; "
            "ego may be at the road edge/curb before starting; "
            "may activate turn signal to merge into traffic flow."
        ),
        "distinguish": (
            "vs TrafficLight_StraightStopOrGo: if a traffic light caused the stop, "
            "use the TrafficLight label instead. THIS label is for roadside/curbside starts only. "
            "vs StartStop_ParkRoadside: THIS is starting to move from roadside; THAT is stopping to park."
        ),
    },
    "StartStop_ParkRoadside": {
        "definition": (
            "Ego vehicle intentionally decelerates and pulls over to park at the ROADSIDE, "
            "typically moving toward the RIGHT side of the road to stop. "
            "This is NOT a traffic light stop or intersection stop."
        ),
        "visual_cues": (
            "Ego vehicle moves toward the RIGHT road edge/curb; "
            "gradually decelerates to a complete stop at the roadside; "
            "may activate hazard lights or right turn signal; "
            "the stop is intentional (parking/pulling over), not forced by traffic signals; "
            "ego ends up stopped at the road edge, not in a traffic lane."
        ),
        "distinguish": (
            "vs TrafficLight stops: parking is intentional at roadside, not at traffic control. "
            "vs any other stop: this is a deliberate pull-over to the side of the road (usually right side). "
            "vs StartStop_StartFromMainRoad: THIS is stopping to park; THAT is starting from parked."
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
            "(including protected/unprotected right turns, including right on red)."
        ),
        "visual_cues": (
            "Intersection clearly visible; "
            "ego vehicle's steering wheel rotates significantly rightward (>30 degrees); "
            "vehicle trajectory curves to the right; "
            "turn signal may be active; "
            "ego transitions from one road direction to a roughly perpendicular right direction."
        ),
        "distinguish": (
            "vs TrafficLight_StraightStopOrGo: If ego turns right at a traffic light, "
            "use THIS label (Intersection_RightTurn), NOT StraightStopOrGo. "
            "Straight stop/go is ONLY for straight-line movement."
        ),
    },
    "Intersection_StandardUTurn": {
        "definition": (
            "Ego vehicle makes a U-turn (approximately 180-degree turn) "
            "at an intersection or designated U-turn area. After the maneuver, "
            "the ego vehicle is traveling in the OPPOSITE direction on the same road."
        ),
        "visual_cues": (
            "Ego vehicle makes a very wide left turn approaching 180 degrees; "
            "after the maneuver, ego is traveling in the opposite direction; "
            "typically occurs at intersection with U-turn permitted sign, wide median opening, "
            "or at a traffic light intersection; "
            "the vehicle may briefly face oncoming traffic during the turn."
        ),
        "distinguish": (
            "vs Intersection_LeftTurn: U-turn is ~180 degrees (reverses direction completely); "
            "left turn is ~90 degrees (turns onto perpendicular road). "
            "If the ego ends up going in the OPPOSITE direction, it is a U-turn."
        ),
    },
    "LaneCruising_Straight": {
        "definition": (
            "Ego vehicle cruises straight in its lane at a CONSTANT speed (cruise control-like), "
            "with NO acceleration, NO deceleration, NO car-following adjustments, "
            "NO turning, NO lane changing, and NO interactions with other road users. "
            "This is a very strict label — the vehicle must be in a truly steady, uneventful state."
        ),
        "visual_cues": (
            "Absolutely no steering input; "
            "speed is visibly constant (no brake lights, no acceleration); "
            "no vehicles ahead requiring speed adjustment (NOT car-following); "
            "staying perfectly within lane markings; "
            "no traffic lights, intersections, pedestrians, or any objects requiring attention; "
            "the road ahead is clear and open."
        ),
        "distinguish": (
            "This label is VERY RARE. Most driving involves some speed variation or interaction. "
            "Do NOT use this label if: the ego vehicle is following another vehicle (even at steady distance), "
            "approaching any intersection, adjusting speed for any reason, or if ANY other label applies. "
            "If unsure between LaneCruising_Straight and any other label, ALWAYS prefer the other label."
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
    "18. else\n"
    "   Definition: Any driving behavior not matching the above 17 categories.\n"
    "   Use sparingly. If unsure between a specific label and 'else', prefer 'else' to avoid mislabeling.\n"
)


# ==================== 2. 系统 Prompt ====================
SYSTEM_PROMPT = f"""You are an expert autonomous driving scene annotator. Your task is to analyze a 20-second ego-vehicle driving video and identify ALL driving maneuvers with precise timing and high accuracy.

You MUST label every maneuver using ONLY the predefined categories below.

=== AVAILABLE LABELS (17 categories + else) ===
{CATEGORY_LIST_STR}
  18. else  (ONLY when no label above matches)

=== DETAILED CATEGORY DEFINITIONS ===
{CATEGORY_DEFINITIONS_DETAILED}
=== LABELING RULES (STRICTLY FOLLOW) ===
1. Assign a label ONLY if the action CLEARLY matches the category definition.
2. Report your confidence (0-100%) for each segment honestly.
3. Minimum segment duration: 1 second.
4. Start and end times MUST be whole integers (e.g., 0, 3, 8, 15, 20). Do NOT use decimals.
5. Time range must be within [0, 20] seconds.
6. TIME OVERLAP IS ALLOWED AND ENCOURAGED: Different maneuvers CAN and SHOULD overlap in time. For example, if a cut-in happens during seconds 8-12, the background driving state (e.g., TrafficLight_StraightStopOrGo) should still span the full period without being split. Output BOTH labels with overlapping time ranges.
7. Adjacent segments with the same label should be merged into one.
8. COMPLETE ACTION WINDOWS: Each maneuver's time range must cover the ENTIRE action lifecycle — from the initial preparation/approach phase, through the main action, to the completion/recovery phase. Do NOT start labeling mid-action. For example, a lane change should include the moment the vehicle begins steering, not just when it crosses the lane marking.
9. For safety-critical events (emergency brake, VRU crossing), be especially precise and generous about timing — include the full reaction window.
10. LaneCruising_Straight is an extremely strict label. It requires absolutely constant speed with zero interactions. If ANY other label could apply to ANY part of the time window, do NOT use LaneCruising_Straight for that period.
11. RED LIGHT WAITING RULE: When ego is STOPPED at a red light and does NOT move at all during the video, cross-traffic vehicles and pedestrians flowing through the intersection are NORMAL TRAFFIC, NOT interactions. Do NOT label them as VehicleInLaneCrossing or VRUInLaneCrossing. Only label interactions when ego is MOVING or about to move.
12. LEFT-TURN LANE RULE: If the ego vehicle is in a left-turn lane (typically the leftmost lane) at a traffic light, ALWAYS use TrafficLight_LeftTurnStopOrGo, NOT TrafficLight_StraightStopOrGo, regardless of whether a left-turn arrow is clearly visible.
13. LANE CHANGE LABELS require ego to ACTUALLY CHANGE LANE (cross lane markings). Simply slowing down or stopping is NOT a lane change.
14. LeadVehicleEmergencyBrake requires a DECELERATION PROCESS — both lead vehicle and ego must transition from moving to stopped. A lead vehicle that is already stationary does NOT trigger this label.
15. STUCK/TRAPPED DETECTION: If the ego vehicle is stuck or waiting for an abnormally long time (not a normal red light wait), look for AbnormalStop_StuckWaiting or TrafficLight_DarkIntersectionPass. Typical stuck scenarios: unprotected left turn waiting for oncoming traffic, right turn yielding to pedestrians, narrow road meeting oncoming vehicle, or stuck at a dark intersection.

=== PRIORITY RULES (when a scene could match multiple categories) ===
- Emergency braking (LeadVehicleEmergencyBrake) > all other labels
- Specific interaction labels > generic labels (e.g., VRUInLaneCrossing > else)
- Intersection turn labels > traffic light labels (if turning phase, use turn label)
- Right turn at traffic light → Intersection_RightTurn (NOT TrafficLight_StraightStopOrGo)
- Lane change for obstacle avoidance > lane change for navigation
- Any lane change near intersection with no obvious obstacle → NavForIntersection
- If confidence < 60% for any specific label, use "else" instead
- LaneCruising_Straight has the LOWEST priority — only use when absolutely nothing else applies

=== OUTPUT FORMAT (ONE LINE PER SEGMENT, NO OTHER TEXT) ===
<driving_maneuver>LABEL</driving_maneuver> from <start_time>START</start_time> to <end_time>END</end_time> seconds (confidence: XX%)

=== EXAMPLE OUTPUT (note: time overlaps between background state and events are correct) ===
<driving_maneuver>TrafficLight_StraightStopOrGo</driving_maneuver> from <start_time>0</start_time> to <end_time>20</end_time> seconds (confidence: 90%)
<driving_maneuver>DynamicInteraction_StandardVehicleCutIn</driving_maneuver> from <start_time>6</start_time> to <end_time>12</end_time> seconds (confidence: 85%)
<driving_maneuver>DynamicInteraction_VRUInLaneCrossing</driving_maneuver> from <start_time>15</start_time> to <end_time>19</end_time> seconds (confidence: 78%)

In this example, the ego vehicle is stopped/going at a traffic light for the full 20 seconds (background state), while a cut-in event occurs during seconds 6-12 and a VRU crossing during 15-19. The background label is NOT split.

IMPORTANT: Output ONLY the structured annotation lines. Do NOT add any reasoning, explanation, or commentary."""

USER_PROMPT = (
    "Below are frames extracted at 2 fps from a 20-second ego-vehicle driving video, "
    "shown in chronological order. "
    "Carefully analyze these frames to identify ALL ego vehicle maneuvers "
    "with precise integer time boundaries and confidence scores. "
    "Cover the entire 20-second duration."
)


# ==================== 3. 视频预处理（切片抽帧） ====================
def extract_frames_from_video(
    video_path: str,
    sample_fps: float = 2.0,
    max_frames: int = 40,
    resolution: Tuple[int, int] = (256, 256),
) -> List[str]:
    """
    从视频中按指定帧率抽帧，下采样后编码为 base64 JPEG 图片列表

    Args:
        video_path: 视频文件路径
        sample_fps: 抽帧帧率（默认 2fps，20s 视频 → 40 帧）
        max_frames: 最大帧数限制
        resolution: 目标分辨率 (width, height)

    Returns:
        List[str]: base64 编码的 JPEG 图片列表（按时间顺序）
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 帧率检查
    if fps <= 0:
        fps = 30.0  # 回退默认值

    duration = total_frames / fps
    frame_interval = max(1, int(round(fps / sample_fps)))

    print(f"  视频信息: {orig_w}x{orig_h}, {fps:.1f}fps, {total_frames}帧, {duration:.1f}s")
    print(f"  抽帧策略: 每{frame_interval}帧取1帧 (目标 {sample_fps}fps)")

    frames_b64 = []
    frame_idx = 0

    while len(frames_b64) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # 下采样到目标分辨率
            resized = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)
            # 编码为 JPEG → base64
            _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64_str = base64.b64encode(buffer).decode("utf-8")
            frames_b64.append(b64_str)

        frame_idx += 1

    cap.release()
    print(f"  抽取帧数: {len(frames_b64)} 帧, 分辨率: {resolution[0]}x{resolution[1]}")

    return frames_b64


# ==================== 4. API 标注客户端 ====================
class AnnotationClient:
    """通过 vLLM OpenAI 兼容 API 进行视频标注的客户端"""

    def __init__(
        self,
        api_base: str,
        model_name: str = "qwen3-vl-235b",
        min_confidence: int = 70,
        sample_fps: float = 2.0,
        max_frames: int = 40,
        resolution: Tuple[int, int] = (256, 256),
        request_timeout: int = 300,
        max_retries: int = 3,
    ):
        """
        Args:
            api_base: vLLM API 地址，如 http://localhost:8000/v1
            model_name: API 中的模型名称（需与 11_deploy.py 的 --model_name 一致）
            min_confidence: 最低置信度阈值
            sample_fps: 抽帧帧率
            max_frames: 最大帧数
            resolution: 目标分辨率
            request_timeout: API 请求超时（秒），视觉模型推理较慢
            max_retries: 失败重试次数
        """
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.resolution = resolution
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        print(f"\n标注客户端配置:")
        print(f"  API 地址:     {self.api_base}")
        print(f"  模型名称:     {self.model_name}")
        print(f"  抽帧帧率:     {self.sample_fps} fps")
        print(f"  最大帧数:     {self.max_frames}")
        print(f"  帧分辨率:     {self.resolution}")
        print(f"  置信度阈值:   {self.min_confidence}%")
        print(f"  请求超时:     {self.request_timeout}s")
        print(f"  最大重试:     {self.max_retries}")
        print()

    def annotate_video(self, video_path: str) -> Dict:
        """
        对单个视频进行标注

        流程: 切片抽帧 → 编码为 base64 → 发送 API 请求 → 解析输出 → 置信度过滤

        Returns:
            {
                "video_path": str,
                "segments": [...],
                "segments_dropped": [...],
                "raw_output": str,
                "frame_count": int,
                "min_confidence": int
            }
        """
        # 1. 切片抽帧
        frames_b64 = extract_frames_from_video(
            video_path,
            sample_fps=self.sample_fps,
            max_frames=self.max_frames,
            resolution=self.resolution,
        )

        if not frames_b64:
            raise ValueError(f"未能从视频提取到任何帧: {video_path}")

        # 2. 构造 API 请求（OpenAI Chat Completions 格式）
        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
            for b64 in frames_b64
        ]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": image_content + [
                    {"type": "text", "text": USER_PROMPT}
                ],
            },
        ]

        request_body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2048,
            "stop": ["\n\n\n"],
        }

        # 3. 发送 API 请求（带重试）
        raw_text = self._call_api_with_retry(request_body)
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
            "frame_count": len(frames_b64),
            "min_confidence": self.min_confidence,
        }

    def _call_api_with_retry(self, request_body: Dict) -> str:
        """带重试和指数退避的 API 调用"""
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.api_base}/chat/completions",
                    json=request_body,
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
                # 对于 4xx 错误不重试（请求本身有问题）
                if 400 <= e.response.status_code < 500:
                    break
            except (KeyError, IndexError) as e:
                last_error = f"API 响应格式异常: {e}"
                print(f"  [重试 {attempt}/{self.max_retries}] {last_error}")
                break  # 格式错误不重试
            except Exception as e:
                last_error = f"未知错误: {e}"
                print(f"  [重试 {attempt}/{self.max_retries}] {last_error}")

            # 指数退避
            if attempt < self.max_retries:
                wait_time = min(2 ** attempt * 5, 60)
                print(f"  等待 {wait_time}s 后重试...")
                time.sleep(wait_time)

        raise RuntimeError(f"API 调用失败（已重试 {self.max_retries} 次）: {last_error}")

    def _parse_output(self, raw_text: str) -> List[Dict]:
        """
        解析模型输出为结构化标签

        示例输入:
        <driving_maneuver>Intersection_LeftTurn</driving_maneuver> from <start_time>3</start_time> to <end_time>8</end_time> seconds (confidence: 95%)
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
            start = int(round(float(match.group(2))))
            end = int(round(float(match.group(3))))
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


# ==================== 5. 视频列表加载 ====================
def load_video_list(video_list_path: str) -> List[str]:
    """
    从文件列表加载视频路径

    支持:
      - 每行一个视频路径（绝对路径或相对于列表文件的相对路径）
      - 自动跳过空行和注释行（以 # 开头）
      - 支持常见视频格式
    """
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}
    base_dir = os.path.dirname(os.path.abspath(video_list_path))

    videos = []
    with open(video_list_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            path = line.strip()
            if not path or path.startswith("#"):
                continue

            # 处理相对路径
            if not os.path.isabs(path):
                path = os.path.join(base_dir, path)

            # 检查文件扩展名
            ext = os.path.splitext(path)[1].lower()
            if ext not in VIDEO_EXTENSIONS:
                print(f"  警告: 第{line_num}行 '{path}' 不是支持的视频格式，跳过")
                continue

            if not os.path.exists(path):
                print(f"  警告: 第{line_num}行 '{path}' 文件不存在，跳过")
                continue

            videos.append(path)

    return videos


# ==================== 6. 批量处理 ====================
def batch_annotate(
    client: AnnotationClient,
    video_paths: List[str],
    output_json: str,
    max_videos: Optional[int] = None,
    concurrency: int = 1,
):
    """
    批量处理视频标注

    支持:
      - 断点续传：已处理的视频自动跳过
      - 实时保存：每个视频完成后立即写入 JSON
      - 并发处理：可通过 concurrency 参数控制并发数
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 加载已有结果（断点续传）
    results = []
    processed_videos = set()
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed_videos = {r["video_path"] for r in results}
            print(f"加载已有结果: {len(results)} 条，将跳过已处理的视频")
        except (json.JSONDecodeError, KeyError):
            print("警告: 无法解析已有结果文件，将重新开始处理")
            results = []

    # 过滤已处理的视频
    if max_videos:
        video_paths = video_paths[:max_videos]

    pending_videos = [v for v in video_paths if v not in processed_videos]
    total = len(video_paths)
    skip_count = total - len(pending_videos)

    print(f"\n{'=' * 60}")
    print(f"批量标注任务")
    print(f"{'=' * 60}")
    print(f"  总视频数:     {total}")
    print(f"  已处理(跳过): {skip_count}")
    print(f"  待处理:       {len(pending_videos)}")
    print(f"  并发数:       {concurrency}")
    print(f"  输出文件:     {output_json}")
    print(f"{'=' * 60}\n")

    if not pending_videos:
        print("所有视频均已处理完成!")
        return

    def _process_single(idx_video):
        """处理单个视频"""
        idx, video_path = idx_video
        video_name = os.path.basename(video_path)
        print(f"\n{'=' * 60}")
        print(f"[{skip_count + idx + 1}/{total}] 处理: {video_name}")
        print(f"  路径: {video_path}")
        print(f"{'=' * 60}")

        try:
            result = client.annotate_video(video_path)
            return result, None
        except Exception as e:
            traceback.print_exc()
            error_result = {
                "video_path": video_path,
                "segments": [],
                "segments_dropped": [],
                "raw_output": "",
                "frame_count": 0,
                "error": str(e),
            }
            return error_result, str(e)

    # 顺序或并发处理
    if concurrency <= 1:
        for idx, video_path in enumerate(pending_videos):
            result, error = _process_single((idx, video_path))
            results.append(result)

            # 实时保存
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            _print_result_summary(result, error)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_process_single, (idx, vp)): (idx, vp)
                for idx, vp in enumerate(pending_videos)
            }

            for future in as_completed(futures):
                result, error = future.result()
                results.append(result)

                # 实时保存
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                _print_result_summary(result, error)

    # 最终统计
    print(f"\n{'=' * 60}")
    success_count = sum(1 for r in results if "error" not in r)
    print(f"全部完成! 成功标注 {success_count}/{len(results)} 个视频")
    print(f"结果保存至: {output_json}")
    print(f"{'=' * 60}")


def _print_result_summary(result: Dict, error: Optional[str]):
    """打印单个视频的标注结果摘要"""
    if error:
        print(f"  x 处理失败: {error}")
        return

    video_name = os.path.basename(result["video_path"])
    print(f"\n  -> {video_name} - 有效标注段 ({len(result['segments'])} 个):")
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


# ==================== 7. 服务连接检查 ====================
def check_server(api_base: str) -> bool:
    """检查 vLLM 服务是否可达"""
    try:
        # 从 api_base (http://host:port/v1) 提取 health URL
        base_url = api_base.rstrip("/")
        if base_url.endswith("/v1"):
            health_url = base_url[:-3] + "/health"
        else:
            health_url = base_url + "/health"

        resp = requests.get(health_url, timeout=10, proxies=NO_PROXY)
        return resp.status_code == 200
    except Exception:
        return False


# ==================== 8. 命令行入口 ====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-235B 视频驾驶行为标注（API 客户端，需先运行 11_deploy.py 部署服务）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
前置条件:
  先在另一个终端启动服务:
    python 11_deploy.py

使用示例:
  # 批量标注 1000 个视频
  python 12_distillation.py \\
      --api_base http://localhost:8000/v1 \\
      --video_list /mnt/pfs/houhaotian/sampled_1000_videos.txt \\
      --output results/annotations.json

  # 单视频标注
  python 12_distillation.py \\
      --api_base http://localhost:8000/v1 \\
      --video_path /data/video_001.mp4

  # 并发 + 置信度阈值
  python 12_distillation.py \\
      --api_base http://localhost:8000/v1 \\
      --video_list /mnt/pfs/houhaotian/sampled_1000_videos.txt \\
      --concurrency 2 \\
      --min_confidence 75

  # 处理视频目录（限制数量）
  python 12_distillation.py \\
      --api_base http://localhost:8000/v1 \\
      --video_dir /data/videos_20s/ \\
      --max_videos 100
        """,
    )

    # ===== 服务地址 =====
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API 地址（默认 http://localhost:8000/v1）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen3-vl-235b",
        help="API 中的模型名称，需与 11_deploy.py 的 --model_name 一致（默认 qwen3-vl-235b）",
    )

    # ===== 输入（三选一）=====
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video_path", type=str, help="单个视频文件路径"
    )
    input_group.add_argument(
        "--video_dir", type=str, help="视频目录路径（批量处理所有 .mp4 文件）"
    )
    input_group.add_argument(
        "--video_list",
        type=str,
        help="视频路径列表文件（每行一个视频路径）",
    )

    # ===== 视频处理参数 =====
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="帧下采样分辨率（默认 256，即 256x256）",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=40,
        help="每个视频最大帧数（默认 40，即 20s x 2fps）",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=2.0,
        help="抽帧帧率（默认 2.0 fps）",
    )

    # ===== 标注参数 =====
    parser.add_argument(
        "--min_confidence",
        type=int,
        default=70,
        help="最低置信度阈值 (0-100)（默认 70）",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=300,
        help="单次 API 请求超时时间（秒，默认 300）",
    )

    # ===== 批量处理参数 =====
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="并发请求数（默认 1）",
    )

    return parser.parse_args()


# ==================== 9. 主入口 ====================
def main():
    args = parse_args()
    target_res = (args.resolution, args.resolution)

    try:
        # ===== Step 1: 检查服务是否可用 =====
        print(f"检查 vLLM 服务: {args.api_base}")
        if not check_server(args.api_base):
            print(
                f"\n错误: 无法连接到 vLLM 服务 ({args.api_base})\n"
                f"请先在另一个终端运行:\n"
                f"  python 11_deploy.py\n"
                f"\n等待服务启动完成后再运行本脚本。"
            )
            sys.exit(1)
        print(f"-> 服务连接成功!\n")

        # ===== Step 2: 创建标注客户端 =====
        client = AnnotationClient(
            api_base=args.api_base,
            model_name=args.model_name,
            min_confidence=args.min_confidence,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            resolution=target_res,
            request_timeout=args.request_timeout,
        )

        # ===== Step 3: 执行标注 =====
        if args.video_path:
            # --- 单视频模式 ---
            result = client.annotate_video(args.video_path)

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
            # --- 批量模式 ---
            if args.video_list:
                video_paths = load_video_list(args.video_list)
                print(f"从列表文件加载: {args.video_list}")
            elif args.video_dir:
                video_paths = sorted(
                    [str(f) for f in Path(args.video_dir).glob("**/*.mp4") if f.is_file()]
                )
                print(f"从目录扫描: {args.video_dir}")
            else:
                raise ValueError("必须指定 --video_path, --video_dir 或 --video_list")

            print(f"共发现 {len(video_paths)} 个视频文件")

            if not video_paths:
                print("未找到任何视频文件，退出。")
                return

            batch_annotate(
                client=client,
                video_paths=video_paths,
                output_json=args.output,
                max_videos=args.max_videos,
                concurrency=args.concurrency,
            )

    except KeyboardInterrupt:
        print("\n\n用户中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
