#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-B 规则匹配脚本：基于 Round-A 结构化 JSON，用确定性规则匹配 122 个叶子标签。

不使用 LLM，纯 Python 规则引擎。每条 Round-A 可输出 0-N 个标签（多标签）。

用法:
    python 20_round_b_rule_matching.py \
        --round_a_file scene_tag/results/round_a_results_v2.jsonl \
        --output scene_tag/results/round_b_rule_results.jsonl

    # 限制条数
    python 20_round_b_rule_matching.py \
        --round_a_file scene_tag/results/round_a_results_v2.jsonl \
        --output scene_tag/results/round_b_rule_results.jsonl \
        --max_videos 100
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Helper accessors for Round-A JSON
# ---------------------------------------------------------------------------

def _seg_has(segs: list, field: str, values: set) -> bool:
    return any(s.get(field) in values for s in segs)

def _seg_any_lateral(segs: list, values: set) -> bool:
    return _seg_has(segs, "lateral", values)

def _seg_any_lon(segs: list, values: set) -> bool:
    return _seg_has(segs, "longitudinal", values)

def _agents(ra: dict) -> list:
    return (ra.get("traffic_agents") or {}).get("agents") or []

def _agents_with_interaction(ra: dict, interaction: str) -> list:
    return [a for a in _agents(ra)
            if interaction in (a.get("interaction_with_ego") or [])]

def _any_agent_category(ra: dict, cats: set) -> bool:
    return any(a.get("category") in cats for a in _agents(ra))

def _has_light(ra: dict) -> bool:
    tc = ra.get("traffic_control") or {}
    tl = tc.get("traffic_light") or {}
    return tl.get("any_traffic_light_visible", False)

def _heads(ra: dict) -> list:
    tc = ra.get("traffic_control") or {}
    tl = tc.get("traffic_light") or {}
    return tl.get("heads") or []

def _ego_maneuver(ra: dict) -> str:
    return (ra.get("road_layout") or {}).get("ego_maneuver_slot_guess", "unknown")

def _intersection(ra: dict) -> str:
    return (ra.get("road_layout") or {}).get("intersection_topology_guess", "none")

def _is_intersection(ra: dict) -> bool:
    return _intersection(ra) not in ("none", "unclear", "unknown")

def _geo_cues(ra: dict) -> list:
    return (ra.get("road_layout") or {}).get("road_geometry_cues") or []

def _in_intersection(ra: dict) -> bool:
    return _is_intersection(ra) or "intersection_interior" in _geo_cues(ra) or "intersection_approach" in _geo_cues(ra)

def _road_types(ra: dict) -> list:
    return (ra.get("road_layout") or {}).get("road_type_hints") or []

def _lane_funcs(ra: dict) -> list:
    return (ra.get("lane_and_markings") or {}).get("lane_function_hints") or []

def _segs(ra: dict) -> list:
    return (ra.get("ego_motion") or {}).get("segments") or []

def _avoidance(ra: dict) -> dict:
    return (ra.get("ego_motion") or {}).get("avoidance_maneuver") or {}

def _compound(ra: dict) -> dict:
    return (ra.get("ego_motion") or {}).get("compound_maneuver_guess") or {}

def _following(ra: dict) -> dict:
    return (ra.get("ego_motion") or {}).get("following_behavior") or {}

def _flow(ra: dict) -> str:
    sc = ra.get("scene_context") or {}
    return (sc.get("traffic_flow_state") or {}).get("overall", "unknown")

def _queue(ra: dict) -> dict:
    sc = ra.get("scene_context") or {}
    return sc.get("queue_ahead") or {}

def _waiting_zone(ra: dict) -> dict:
    return (ra.get("road_layout") or {}).get("waiting_zone") or {}

def _turning_layout(ra: dict) -> dict:
    return (ra.get("road_layout") or {}).get("turning_lane_layout") or {}

def _static_markings(ra: dict) -> dict:
    return (ra.get("road_layout") or {}).get("static_markings") or {}

def _roadside(ra: dict) -> list:
    return (ra.get("road_layout") or {}).get("roadside_facilities") or []

def _road_width(ra: dict) -> str:
    return (ra.get("road_layout") or {}).get("road_width_impression", "unknown")

def _warn_lights(ra: dict) -> dict:
    tc = ra.get("traffic_control") or {}
    return tc.get("warning_and_auxiliary_lights") or {}

def _signs(ra: dict) -> list:
    tc = ra.get("traffic_control") or {}
    ts = tc.get("traffic_signs") or {}
    return ts.get("signs") or []

def _lce(ra: dict) -> dict:
    return (ra.get("lane_and_markings") or {}).get("lane_change_evidence") or {}

def _static_infra(ra: dict) -> list:
    return (ra.get("road_layout") or {}).get("static_infrastructure") or []

# Flashing helpers
def _any_head_flashing(ra: dict, color: str = None) -> bool:
    for h in _heads(ra):
        if h.get("is_flashing_guess") is True:
            if color is None:
                return True
            if h.get("flash_color_guess") == color:
                return True
    return False

def _any_head_off(ra: dict) -> bool:
    return any(h.get("aspect_guess") == "off" for h in _heads(ra))

def _any_head_occluded(ra: dict) -> bool:
    return any(h.get("visibility") in ("occluded", "partial") for h in _heads(ra))

# ---------------------------------------------------------------------------
# Label rule functions — each returns List[dict] of matched labels
# ---------------------------------------------------------------------------

def rules_01_dynamic_interaction(ra: dict) -> List[dict]:
    results = []
    agents = _agents(ra)

    for a in agents:
        interactions = a.get("interaction_with_ego") or []
        cat = a.get("category", "")
        is_vru = cat in ("pedestrian", "cyclist", "two_wheeler")
        is_vehicle = cat in ("passenger_car", "truck_bus", "generic_vehicle")
        cd = a.get("cut_in_detail") or {}
        lvb = a.get("lead_vehicle_braking_detail") or {}
        ts = a.get("time_span", [[0, 20]])

        if "cut_in" in interactions:
            urgency = cd.get("urgency", "unknown")
            prior = cd.get("agent_prior_state_guess", "unknown")
            consec = cd.get("consecutive_lane_change_guess")

            if is_vehicle:
                if consec is True:
                    results.append({"label": "DynamicInteraction_ConsecutiveLaneChangeCutIn", "time_span": ts})
                elif prior == "stationary_then_started":
                    results.append({"label": "DynamicInteraction_StartupVehicleCutIn", "time_span": ts})
                elif urgency == "emergency":
                    results.append({"label": "DynamicInteraction_EmergencyVehicleCutIn", "time_span": ts})
                else:
                    results.append({"label": "DynamicInteraction_StandardVehicleCutIn", "time_span": ts})
            elif is_vru:
                if urgency == "emergency":
                    results.append({"label": "DynamicInteraction_EmergencyVRUCutIn", "time_span": ts})
                else:
                    results.append({"label": "DynamicInteraction_SlowVRUCutIn", "time_span": ts})

        if "cut_out" in interactions and is_vehicle:
            results.append({"label": "DynamicInteraction_LeadVehicleCutOut", "time_span": ts})

        if "gap_opening_then_cut_in" in interactions:
            results.append({"label": "DynamicInteraction_GapOpeningCutIn", "time_span": ts})

        if "lead_vehicle" in interactions:
            if lvb.get("braking_intensity") == "sudden_hard":
                results.append({"label": "DynamicInteraction_LeadVehicleSuddenBrake", "time_span": ts})

        if "cross_path" in interactions and not _in_intersection(ra):
            if is_vru:
                results.append({"label": "DynamicInteraction_VRUInLaneCrossing", "time_span": ts})
            elif is_vehicle:
                results.append({"label": "DynamicInteraction_VehicleInLaneCrossing", "time_span": ts})

    if _any_agent_category(ra, {"static_small_object"}):
        avd = _avoidance(ra)
        if avd.get("target_type_guess") == "static_obstacle" or avd.get("strategy") != "none":
            results.append({"label": "DynamicInteraction_StaticObjectReaction"})

    return results


def rules_02_traffic_light(ra: dict) -> List[dict]:
    results = []
    if not _has_light(ra) and not _warn_lights(ra).get("warning_light_visible") and not _warn_lights(ra).get("mobile_signal_visible"):
        return results

    maneuver = _ego_maneuver(ra)
    heads = _heads(ra)
    wl = _warn_lights(ra)

    if wl.get("mobile_signal_visible"):
        results.append({"label": "TrafficLight_MobileSignal"})
        return results

    if wl.get("warning_light_visible"):
        results.append({"label": "TrafficLight_WarningLight"})
        return results

    if _any_head_occluded(ra) and not any(h.get("visibility") == "clear" for h in heads):
        results.append({"label": "TrafficLight_OccludedSignal"})
        return results

    direction_map = {
        "go_straight": "Straight",
        "left_turn": "LeftTurn",
        "right_turn": "RightTurn",
        "uturn": "UTurn",
    }
    direction = direction_map.get(maneuver, "Straight")

    if _any_head_flashing(ra, "green"):
        results.append({"label": f"TrafficLight_{direction}GreenFlash"})
    elif _any_head_flashing(ra, "yellow"):
        results.append({"label": f"TrafficLight_{direction}YellowFlash"})
    elif _any_head_off(ra):
        results.append({"label": f"TrafficLight_{direction}DarkLight"})
    else:
        wz = _waiting_zone(ra)
        if wz.get("waiting_zone_visible") and wz.get("ego_relative_to_waiting_zone") in ("inside", "approaching"):
            results.append({"label": "TrafficLight_WaitingZoneStopOrGo"})
        elif maneuver == "uturn":
            results.append({"label": "TrafficLight_UTurnStopOrGo"})
        elif maneuver == "right_turn":
            results.append({"label": "TrafficLight_RightTurnStopOrGo"})
        elif maneuver == "left_turn":
            results.append({"label": "TrafficLight_LeftTurnStopOrGo"})
        else:
            results.append({"label": "TrafficLight_StraightStopOrGo"})

    return results


def rules_03_start_stop(ra: dict) -> List[dict]:
    results = []
    segs = _segs(ra)
    fol = _following(ra)

    has_stop = _seg_any_lon(segs, {"stop", "emergency_stop"})
    has_start = _seg_any_lon(segs, {"start_from_stop"})
    has_emergency = _seg_any_lon(segs, {"emergency_stop"}) or any(
        s.get("stop_urgency") == "emergency" for s in segs
    )

    if has_emergency and "main_road" in _road_types(ra):
        results.append({"label": "StartStop_EmergencyStopOnMainRoad"})

    if has_start and "non_motor_lane" in _road_types(ra):
        results.append({"label": "StartStop_StartFromNonMotorLane"})
    elif has_start and not _has_light(ra) and "main_road" in _road_types(ra):
        results.append({"label": "StartStop_StartFromMainRoad"})

    if "bus_station" in _roadside(ra) or "taxi_stand" in _roadside(ra):
        if has_stop:
            results.append({"label": "StartStop_StopAtStation"})

    if "structured_parking_spots" in _roadside(ra):
        if has_stop or _seg_any_lon(segs, {"decelerate"}):
            results.append({"label": "StartStop_ParkInStructuredSpot"})

    if (has_stop or _seg_any_lon(segs, {"decelerate"})) and not _has_light(ra):
        if "unstructured_roadside_parking" in _roadside(ra) or "parking_cars_line" in (
            (ra.get("lane_and_markings") or {}).get("curb_and_boundary") or []
        ):
            results.append({"label": "StartStop_ParkRoadside"})

    if fol.get("following_stop_guess") is True:
        results.append({"label": "StartStop_FollowingStop"})

    return results


def rules_04_intersection(ra: dict) -> List[dict]:
    results = []
    if not _in_intersection(ra):
        return results

    maneuver = _ego_maneuver(ra)
    has_light = _has_light(ra)
    topo = _intersection(ra)
    tl = _turning_layout(ra)
    wz = _waiting_zone(ra)
    flow = _flow(ra)
    comp = _compound(ra)
    roundabout = (ra.get("road_layout") or {}).get("roundabout_detail") or {}

    if topo == "roundabout":
        lc = roundabout.get("lane_count_guess", "unknown")
        if lc == "1":
            results.append({"label": "Intersection_SingleLaneRoundabout"})
        else:
            results.append({"label": "Intersection_MultiLaneRoundabout"})
        return results

    if topo == "T_junction" and not has_light:
        results.append({"label": "Intersection_TJunctionUnprotectedMerge"})
        return results

    if comp.get("type") == "three_point_uturn":
        results.append({"label": "Intersection_ThreePointUTurn"})
        return results

    if wz.get("waiting_zone_visible") and wz.get("ego_relative_to_waiting_zone") in ("inside", "approaching"):
        wzt = wz.get("waiting_zone_type_guess", "unknown")
        if comp.get("type") == "waiting_zone_uturn":
            results.append({"label": "Intersection_WaitingZoneUTurn"})
        elif wzt == "straight":
            results.append({"label": "Intersection_StraightWaitingZone"})
        elif wzt == "left_turn":
            results.append({"label": "Intersection_LeftTurnWaitingZone"})
        elif wzt == "text":
            results.append({"label": "Intersection_TextWaitingZone"})
        elif wzt == "combined_signal":
            results.append({"label": "Intersection_CombinedSignalWaitingZone"})
        elif wzt == "image":
            results.append({"label": "Intersection_ImageWaitingZone"})

    if maneuver == "uturn":
        results.append({"label": "Intersection_StandardUTurn"})
    elif maneuver == "left_turn":
        if tl.get("parallel_turning_visible") is True:
            results.append({"label": "Intersection_ParallelLeftTurn"})
        elif tl.get("dedicated_left_turn_lane_visible") is True:
            results.append({"label": "Intersection_DedicatedLeftTurnLane"})
        elif has_light:
            results.append({"label": "Intersection_ProtectedLeftTurn"})
        else:
            results.append({"label": "Intersection_UnprotectedLeftTurn"})
    elif maneuver == "right_turn":
        if tl.get("parallel_turning_visible") is True:
            results.append({"label": "Intersection_ParallelRightTurn"})
        elif tl.get("dedicated_right_turn_lane_visible") is True:
            results.append({"label": "Intersection_DedicatedRightTurnLane"})
        elif tl.get("right_turn_adjacent_non_motor_lane") is True:
            results.append({"label": "Intersection_RightTurnWithNonMotorLane"})
        elif has_light:
            results.append({"label": "Intersection_ProtectedRightTurn"})
        else:
            results.append({"label": "Intersection_ProtectedRightTurn"})
    elif maneuver == "go_straight":
        if topo == "misaligned_cross":
            results.append({"label": "Intersection_MisalignedStraight"})
        elif flow in ("congested", "gridlock"):
            results.append({"label": "Intersection_CongestedStraight"})
        elif has_light:
            results.append({"label": "Intersection_ProtectedStraight"})
        else:
            results.append({"label": "Intersection_UnprotectedStraight"})

    return results


def rules_05_lane_cruising(ra: dict) -> List[dict]:
    results = []
    segs = _segs(ra)
    fol = _following(ra)
    geo = _geo_cues(ra)
    flow = _flow(ra)
    queue = _queue(ra)
    sm = _static_markings(ra)
    lf = _lane_funcs(ra)

    if "sharp_curve" in geo:
        results.append({"label": "LaneCruising_SharpCurve"})

    if _road_width(ra) in ("very_narrow", "narrow_single_lane"):
        results.append({"label": "LaneCruising_NarrowSpace"})

    if "rural_road" in _road_types(ra):
        results.append({"label": "LaneCruising_RuralRoad"})

    if sm.get("speed_bump_visible"):
        results.append({"label": "LaneCruising_SpeedBump"})

    if "construction_zone" in _static_infra(ra):
        results.append({"label": "LaneCruising_ConstructionZone"})

    if sm.get("crosswalk_visible") and not _in_intersection(ra):
        results.append({"label": "LaneCruising_ZebraCrossing"})

    for s in _signs(ra):
        if s.get("type_guess") == "speed_limit":
            ctx = s.get("speed_limit_context", "unknown")
            if ctx == "road_permanent":
                results.append({"label": "LaneCruising_RoadSpeedLimit"})
            elif ctx in ("school_zone", "construction_zone"):
                results.append({"label": "LaneCruising_SceneSpeedLimit"})
            elif ctx == "intersection_approach":
                results.append({"label": "LaneCruising_IntersectionSpeedLimit"})

    if "variable_lane" in lf:
        results.append({"label": "LaneCruising_VariableLane"})
    if "bus_only" in lf:
        results.append({"label": "LaneCruising_BusLane"})
    if "tidal_lane" in lf:
        results.append({"label": "LaneCruising_TidalLane"})
    if "no_parking_zone" in lf or sm.get("no_parking_marking_visible"):
        results.append({"label": "LaneCruising_NoParkingZone"})

    if flow in ("congested", "gridlock"):
        if fol.get("is_following") and fol.get("following_target_type") == "vehicle":
            results.append({"label": "LaneCruising_CongestedFollowing"})
        elif queue.get("queue_visible") and queue.get("ego_in_queue") is False:
            results.append({"label": "LaneCruising_StaticVehicleQueueCongestion"})
        else:
            results.append({"label": "LaneCruising_OtherCongestion"})
    elif fol.get("is_following"):
        if fol.get("following_target_type") == "vru":
            results.append({"label": "LaneCruising_FollowingVRU"})
        elif fol.get("following_target_type") == "vehicle":
            results.append({"label": "LaneCruising_SteadyFollowing"})

    if not results:
        if _seg_any_lon(segs, {"maintain_speed"}) and _seg_any_lateral(segs, {"lane_keep"}):
            results.append({"label": "LaneCruising_Straight"})

    return results


def rules_06_lane_change(ra: dict) -> List[dict]:
    results = []
    segs = _segs(ra)
    avd = _avoidance(ra)
    comp = _compound(ra)
    lce = _lce(ra)

    has_lc = _seg_any_lateral(segs, {
        "lane_change_left", "lane_change_right",
        "borrow_oncoming_lane_left", "borrow_oncoming_lane_right",
        "cross_line_bypass_left", "cross_line_bypass_right",
    }) or lce.get("lane_change_in_progress_guess")

    if not has_lc:
        return results

    strategy = avd.get("strategy", "none")
    target = avd.get("target_type_guess", "none")
    borrow = lce.get("crossed_into_oncoming_lane_guess") is True or strategy == "borrow_oncoming_lane"
    cross_line = strategy == "cross_line_bypass"

    if comp.get("type") == "overtake":
        results.append({"label": "LaneChange_Overtake"})
        return results

    if cross_line:
        if target == "static_vehicle":
            results.append({"label": "LaneChange_CrossLineBypassStaticVehicles"})
        else:
            results.append({"label": "LaneChange_CrossLineBypassStaticObstacles"})
        return results

    if borrow:
        if target == "slow_vru":
            results.append({"label": "LaneChange_BorrowLaneAvoidSlowVRU"})
        elif target == "static_vehicle":
            results.append({"label": "LaneChange_BorrowLaneAvoidStaticVehicle"})
        elif target == "static_obstacle":
            results.append({"label": "LaneChange_BorrowLaneAvoidStaticObstacle"})
        else:
            oncoming = _agents_with_interaction(ra, "oncoming")
            if oncoming:
                results.append({"label": "LaneChange_BorrowOncomingLaneAvoidVehicle"})
            else:
                results.append({"label": "LaneChange_BorrowLaneAvoidStaticVehicle"})
        return results

    if strategy == "standard_lane_change" or strategy == "in_lane_nudge" or has_lc:
        if target == "static_vehicle":
            results.append({"label": "LaneChange_AvoidStaticVehicle"})
        elif target == "slow_vru":
            if _flow(ra) not in ("congested", "gridlock"):
                results.append({"label": "LaneChange_AvoidSlowVRU"})
            else:
                results.append({"label": "LaneChange_SlowVRUEfficiency"})
        elif target == "static_obstacle":
            results.append({"label": "LaneChange_AvoidStaticObstacle"})
        elif target == "slow_vehicle":
            results.append({"label": "LaneChange_SlowVehicleEfficiency"})
        else:
            results.append({"label": "LaneChange_NavForIntersection"})

    return results


def rules_07_intersection_interaction(ra: dict) -> List[dict]:
    results = []
    if not _in_intersection(ra):
        return results

    maneuver = _ego_maneuver(ra)
    agents = _agents(ra)

    for a in agents:
        interactions = a.get("interaction_with_ego") or []
        cat = a.get("category", "")
        is_vru = cat in ("pedestrian", "cyclist", "two_wheeler")
        is_vehicle = cat in ("passenger_car", "truck_bus", "generic_vehicle")
        heading = a.get("agent_own_heading_guess", "unknown")
        pos = a.get("ego_relative_position", "unknown")
        motion = a.get("motion_trend_relative_ego", "unknown")
        threat = a.get("approach_proximity_threat", "unknown")
        ts = a.get("time_span", [[0, 20]])

        if "cross_path" in interactions or "pedestrian_near_crosswalk" in interactions:
            if maneuver == "go_straight":
                if is_vru:
                    results.append({"label": "IntersectionInteraction_EgoStraight_VRUCrossing", "time_span": ts})
                elif is_vehicle and heading in ("perpendicular_from_left", "perpendicular_from_right"):
                    results.append({"label": "IntersectionInteraction_EgoStraight_VehicleLeftTurnCrossing", "time_span": ts})
            elif maneuver == "left_turn":
                if is_vru:
                    results.append({"label": "IntersectionInteraction_EgoLeftTurn_VRUCrossing", "time_span": ts})
                elif is_vehicle and heading in ("opposing", "same_direction"):
                    results.append({"label": "IntersectionInteraction_EgoLeftTurn_VehicleStraightCrossing", "time_span": ts})
            elif maneuver == "right_turn":
                if is_vru:
                    results.append({"label": "IntersectionInteraction_EgoRightTurn_VRUCrossing", "time_span": ts})
                elif is_vehicle:
                    results.append({"label": "IntersectionInteraction_EgoRightTurn_VehicleStraightCrossing", "time_span": ts})

        if "cut_in" in interactions and maneuver in ("go_straight", "right_turn"):
            if is_vru:
                results.append({"label": "IntersectionInteraction_EgoStraight_VRURightTurnCutIn", "time_span": ts})
            elif is_vehicle:
                results.append({"label": "IntersectionInteraction_EgoStraight_VehicleRightTurnCutIn", "time_span": ts})

        if threat in ("high", "moderate") and motion in ("parallel_same_dir", "parallel_opposing"):
            if is_vru:
                if motion == "parallel_opposing" or heading == "opposing":
                    results.append({"label": "IntersectionInteraction_OncomingVRUApproach", "time_span": ts})
                else:
                    results.append({"label": "IntersectionInteraction_ParallelVRUApproach", "time_span": ts})
            elif is_vehicle:
                if maneuver == "go_straight":
                    results.append({"label": "IntersectionInteraction_ParallelStraightVehicleApproach", "time_span": ts})
                elif maneuver == "left_turn":
                    results.append({"label": "IntersectionInteraction_ParallelLeftTurnVehicleApproach", "time_span": ts})
                elif maneuver == "right_turn":
                    results.append({"label": "IntersectionInteraction_ParallelRightTurnVehicleApproach", "time_span": ts})

    return results


# ---------------------------------------------------------------------------
# Master rule engine
# ---------------------------------------------------------------------------

ALL_RULE_GROUPS = [
    ("01_DynamicInteraction", rules_01_dynamic_interaction),
    ("02_TrafficLight", rules_02_traffic_light),
    ("03_StartStop", rules_03_start_stop),
    ("04_Intersection", rules_04_intersection),
    ("05_LaneCruising", rules_05_lane_cruising),
    ("06_LaneChange", rules_06_lane_change),
    ("07_IntersectionInteraction", rules_07_intersection_interaction),
]


def match_labels(ra: dict) -> List[dict]:
    all_labels = []
    for group_name, rule_fn in ALL_RULE_GROUPS:
        try:
            matched = rule_fn(ra)
            for m in matched:
                m["source_group"] = group_name
            all_labels.extend(matched)
        except Exception as e:
            print(f"  WARNING: rule {group_name} failed: {e}", flush=True)

    seen = set()
    deduped = []
    for lab in all_labels:
        key = lab["label"]
        if key not in seen:
            seen.add(key)
            deduped.append(lab)
    return deduped


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    print(f"Loading Round-A: {args.round_a_file}", flush=True)

    entries = []
    with open(args.round_a_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("round_a") is not None:
                    entries.append(d)
            except json.JSONDecodeError:
                continue

    if args.max_videos:
        entries = entries[:args.max_videos]

    print(f"Entries to process: {len(entries)}", flush=True)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    label_counter = Counter()
    total_labels = 0
    empty_count = 0

    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as fout:
        for i, entry in enumerate(entries):
            ra = entry["round_a"]
            labels = match_labels(ra)

            result = {
                "clip_id": entry.get("clip_id", ""),
                "video_path": entry.get("video_path", ""),
                "labels": labels,
                "num_labels": len(labels),
                "label_names": [l["label"] for l in labels],
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            total_labels += len(labels)
            if not labels:
                empty_count += 1
            for lab in labels:
                label_counter[lab["label"]] += 1

            if (i + 1) % 5000 == 0:
                print(f"  [{i+1}/{len(entries)}] labels_so_far={total_labels}", flush=True)

    elapsed = time.time() - t0

    print(f"\n{'=' * 60}", flush=True)
    print(f"Rule matching complete.", flush=True)
    print(f"  Processed:    {len(entries)}", flush=True)
    print(f"  Total labels: {total_labels} ({total_labels/max(len(entries),1):.2f}/video)", flush=True)
    print(f"  Empty (0 labels): {empty_count} ({empty_count/max(len(entries),1)*100:.1f}%)", flush=True)
    print(f"  Unique labels: {len(label_counter)}", flush=True)
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(len(entries),1)*1000:.1f}ms/video)", flush=True)
    print(f"  Output: {args.output}", flush=True)
    print(f"\nTop 30 labels:", flush=True)
    for label, count in label_counter.most_common(30):
        print(f"  {count:6d}  {label}", flush=True)
    print(f"{'=' * 60}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Round-B rule-based label matching from Round-A JSON")
    parser.add_argument("--round_a_file", type=str, required=True)
    parser.add_argument("--output", type=str, default="scene_tag/results/round_b_rule_results.jsonl")
    parser.add_argument("--max_videos", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
