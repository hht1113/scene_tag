#!/usr/bin/env python3
"""
Seedance 视频生成脚本

为驾驶场景标签生成合成训练视频。
异步任务：提交 → 轮询 → 下载视频。

用法:
    # Seedance 2.0（方舟）— 默认
    python scene_tag/16_generate_video.py \
        --api_key "YOUR_ARK_API_KEY" \
        --tag StartStop_StartFromNonMotorLane \
        --count 3

    # Seedance 1.5 Pro（ADVC）
    python scene_tag/16_generate_video.py \
        --api_key "YOUR_ADVC_API_KEY" \
        --tag StartStop_StartFromNonMotorLane \
        --model seedance15 \
        --count 1

    # 自定义 prompt
    python scene_tag/16_generate_video.py \
        --api_key "YOUR_ARK_API_KEY" \
        --prompt "你的提示词" \
        --count 1

    # 批量生成 10 个不同标签各 1 条
    python scene_tag/16_generate_video.py \
        --api_key "YOUR_ARK_API_KEY" \
        --batch_tags "Intersection_ThreePointUTurn,TrafficLight_MobileSignal,..." \
        --count 1
"""

import argparse
import json
import os
import time
import requests

MODEL_CONFIGS = {
    "seedance20": {
        "api_base": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seedance-2-0-260128",
        "submit_path": "/contents/generations/tasks",
        "poll_path": "/contents/generations/tasks",
    },
    "seedance15": {
        "api_base": "https://ai-beijing.volcadvc.com/api/v1",
        "model": "doubao-seedance-1-5-pro-251215",
        "submit_path": "/contents/generations/tasks",
        "poll_path": "/contents/generations/tasks",
    },
}

FW_CAMERA_STYLE = (
    "A realistic dashcam-style video from a CAR (not a bicycle, not a motorcycle). "
    "The ego vehicle is a regular passenger car. "
    "Camera style: exterior-mounted front-wide (FW) camera on the vehicle roof or front bumper, NOT inside the cabin. "
    "120-degree horizontal field of view with noticeable barrel distortion. "
    "The camera is mounted OUTSIDE the car — there is NO windshield, NO dashboard, NO rearview mirror, NO cabin interior visible. "
    "CRITICAL: there must be ABSOLUTELY NO vehicle parts visible — NO hood, NO bumper, NO windshield edges, "
    "NO wipers, NO handlebars, NO motorcycle handles, NO roof edges — NOTHING belonging to the ego vehicle. "
    "The bottom of the frame shows ONLY road surface. The entire frame is a completely clean, unobstructed forward view. "
    "The road converges toward a central vanishing point. "
    "Typical Chinese urban road: gray asphalt, white/yellow lane markings, green trees, concrete buildings. "
    "Looks like a real autonomous driving data recording camera — not cinematic, slightly washed-out colors. "
    "Vehicle speed feels moderate to slow — the road markings flow backward at a gentle, realistic pace, not fast."
)

TAG_PROMPTS = {
    "StartStop_StartFromNonMotorLane": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: Chinese city. The ego car is temporarily parked in a narrow bike lane on the far right of the road. "
            "The bike lane is narrow (about 2 meters wide), bordered by a solid white line on the left and the curb on the right. "
            "The main road to the left of the white line has MOVING traffic — cars, buses, and trucks are driving past normally. "
            "The right side has sidewalk, trees, and some parked bicycles along the curb. "
            "The road ahead of the ego car is clear and empty — absolutely NO cyclists, NO pedestrians in front. "
            "\n\n"
            "ACTION: "
            "During the first 3 seconds, ONLY the ego car is stopped. Other vehicles on the main road continue driving past. "
            "After 3 seconds, the ego car begins to move forward slowly from the bike lane. "
            "The road texture and white lane line begin flowing downward in the frame. "
            "Parked bicycles and curb objects slide backward. "
            "The acceleration is very gentle and smooth. No steering, no lane change. "
            "\n\n"
            "Core event: ego car starting from a non-motorized vehicle lane."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "LaneCruising_FollowingVRU": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "Chinese urban road scene with realistic traffic participants. "
            "The ego vehicle is moving slowly and steadily behind a vulnerable road user (VRU). "
            "The VRU is either a pedestrian walking ahead near the lane center, "
            "or a non-motorized vehicle such as a bicycle or electric bike riding ahead at low speed. "
            "\n\n"
            "Key behavior: "
            "The ego vehicle follows the VRU at a low and nearly constant speed (about 10-20 km/h). "
            "The following distance stays stable and safe throughout the entire video. "
            "No overtaking, no lane change, no sudden braking, no acceleration burst. "
            "The road surface markings and lane lines flow backward slowly and steadily under the camera. "
            "\n\n"
            "Scene details: "
            "Typical Chinese city road with lane markings, curb, roadside parked vehicles, "
            "traffic signs, other pedestrians on the sidewalk, and nearby moving cars. "
            "\n\n"
            "Core event: ego car steadily following a VRU at constant low speed."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    # ========== 10 个不常见场景标签（v2 修正） ==========
    "Intersection_ThreePointUTurn": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A NARROW Chinese urban side-street or alley, barely wider than two car lengths. "
            "The road is so narrow that a normal U-turn is IMPOSSIBLE — the car cannot turn around in one sweep. "
            "Walls, fences, or tightly parked vehicles line both sides, leaving very little room. "
            "No traffic signal. A few parked bicycles and utility poles on the sidewalk. "
            "\n\n"
            "ACTION: "
            "The ego car performs a three-point U-turn (K-turn) because the road is too narrow for a single U-turn: "
            "Step 1 — the car turns the steering wheel hard LEFT and creeps forward slowly; the entire scene rotates clockwise as the nose swings toward the opposite wall/curb, until the front almost touches it. "
            "Step 2 — the car stops, shifts to REVERSE, turns the wheel hard RIGHT, and backs up slowly; the scene shifts forward as the rear swings toward the near-side wall. "
            "Step 3 — the car stops again, shifts to DRIVE, turns the wheel hard LEFT, and creeps forward; the car is now facing the opposite direction and drives away slowly. "
            "The whole maneuver takes about 8 seconds, at very low speed (< 5 km/h). "
            "\n\n"
            "Core event: ego car performing a three-point U-turn in a narrow road where a single U-turn is impossible."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "TrafficLight_MobileSignal": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A Chinese urban road approaching a construction zone. "
            "There is NO permanent traffic light here — instead, a PORTABLE / TEMPORARY traffic signal stands on a metal tripod or wheeled base at the roadside. "
            "The portable signal is small, battery-powered, showing RED. "
            "Orange traffic cones and barriers partially block some lanes. Workers in reflective vests nearby. "
            "There are NO vehicles ahead of the ego car — the ego car is the FIRST in line. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the portable red signal and smoothly decelerates to a complete stop behind the cones. "
            "The car waits for several seconds with the scene completely still. "
            "Then the portable signal switches from RED to GREEN. "
            "The ego car begins to creep forward slowly, passing through the narrowed construction zone. "
            "Cones and barriers slide backward on one side as the car passes through. "
            "\n\n"
            "Core event: ego car stopping at and then proceeding through a portable/mobile traffic signal at a construction site."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "TrafficLight_StraightDarkLight": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A Chinese urban STRAIGHT road with ONE intersection ahead. "
            "The ego car's lane is a STRAIGHT-DRIVING lane — lane markings show straight-ahead arrows on the road. "
            "Above the intersection, there is EXACTLY ONE set of traffic lights. "
            "CRITICAL: that traffic light is completely DARK — all bulbs are OFF, no red, no yellow, no green. "
            "The light housings appear as black circles. This is a power failure. "
            "There must be NO other traffic lights visible anywhere in the scene — only this one dark signal. "
            "Other vehicles cautiously approach and inch through the intersection. "
            "\n\n"
            "ACTION: "
            "The ego car drives straight on the straight-only lane toward the intersection. "
            "As the ego car gets closer, the dark traffic light overhead becomes clearly visible — all lamps are off. "
            "The ego car slows down dramatically to nearly a stop. "
            "After carefully checking for cross traffic, the ego car creeps through the intersection. "
            "Road markings and the straight-ahead arrows flow backward slowly. "
            "\n\n"
            "Core event: ego car driving straight through an intersection where the only traffic light is completely dark."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "StartStop_EmergencyStopOnMainRoad": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A Chinese urban main road with multiple lanes. The ego car is the LEADING vehicle — "
            "there are NO cars ahead in its lane. The road ahead is clear and straight. "
            "An intersection with a traffic light is visible ahead in the distance. "
            "The ego car is driving at moderate city speed (30-40 km/h). "
            "\n\n"
            "ACTION: "
            "The ego car cruises forward for the first 4-5 seconds — road markings flow backward steadily. "
            "Then the traffic light ahead suddenly turns RED. "
            "The ego car performs an EMERGENCY BRAKE — sudden, hard deceleration. "
            "The camera may shake slightly from the abrupt stop. "
            "Road markings rapidly slow down and stop flowing. "
            "The ego car comes to a complete stop just before the intersection stop line. "
            "Vehicles in adjacent lanes also brake or continue through. "
            "The scene ends with the ego car stationary at the red light. "
            "\n\n"
            "Core event: ego car performing an emergency stop at a red light on a main road."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "DynamicInteraction_ConsecutiveLaneChangeCutIn": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A wide Chinese urban road with at least 3 lanes in the same direction. "
            "The ego car is driving straight in its lane at moderate speed. "
            "Traffic is moderate — vehicles visible in adjacent lanes. "
            "\n\n"
            "ACTION: "
            "ANOTHER vehicle (a sedan or SUV) starts in a lane that is TWO lanes away from the ego car (e.g., far-right lane). "
            "That other vehicle aggressively changes into the lane NEXT to the ego car first — this is visible as the other car appears from the far side and moves closer. "
            "Within just 2-3 seconds, that SAME other vehicle changes lanes AGAIN, this time directly into the ego car's lane, cutting in front of the ego car. "
            "The other vehicle squeezes into a tight gap ahead of the ego car. "
            "The ego car must brake to maintain safe distance. Road markings slow down as the ego car decelerates. "
            "The cut-in vehicle is now very close directly ahead. "
            "\n\n"
            "Core event: another vehicle performs rapid consecutive lane changes across 2 lanes and cuts into the ego car's lane."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "LaneChange_BorrowOncomingLaneAvoidVehicle": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A two-lane Chinese urban road (one lane each direction), separated by a yellow center line. "
            "In the ONCOMING lane (the lane for vehicles driving toward the ego car), there is a BROKEN-DOWN or PARKED vehicle blocking that lane. "
            "Because of this obstacle, an ONCOMING vehicle has crossed the yellow center line and is driving IN the ego car's lane, heading TOWARD the ego car. "
            "\n\n"
            "ACTION: "
            "The ego car is driving forward normally. "
            "Ahead, an oncoming vehicle appears — it is driving in the WRONG lane (the ego car's lane) to bypass the obstacle in the oncoming lane. "
            "The oncoming vehicle is heading straight toward the ego car in a near head-on situation. "
            "The ego car must slow down and steer RIGHT toward the road edge to yield space. "
            "The oncoming vehicle passes by on the ego car's LEFT side (very close), then returns to its own lane after clearing the obstacle. "
            "The ego car then steers back to the lane center and resumes normal driving. "
            "\n\n"
            "Core event: ego car avoiding an oncoming vehicle that has borrowed the ego car's lane due to an obstacle in the oncoming lane."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "Intersection_MisalignedStraight": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A Chinese urban intersection where the road on the FAR SIDE is significantly OFFSET from the road on the NEAR SIDE. "
            "The misalignment is LARGE and obvious — the far-side road is shifted about 3-5 meters to the LEFT or RIGHT. "
            "Looking through the intersection, the far-side lane markings are clearly NOT a straight continuation of the near-side lane markings. "
            "The road appears to 'jog' or 'zigzag' through the intersection. "
            "Traffic lights show green. A few other vehicles navigate the offset crossing. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the intersection on a straight lane. "
            "Upon entering the intersection, the ego car must make a noticeable lateral S-curve steering adjustment "
            "(first slightly left, then straighten) to align with the offset far-side lane. "
            "The road markings clearly show the stagger — lane lines on the far side are visibly shifted sideways. "
            "The ego car completes the crossing at slow speed and continues in the realigned lane. "
            "\n\n"
            "Core event: ego car navigating through a significantly misaligned / offset intersection where the far-side road is shifted several meters."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "LaneCruising_ConstructionZone": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A Chinese urban STRAIGHT road with an ACTIVE CONSTRUCTION ZONE on one side. "
            "The ego car's lane is a STRAIGHT-DRIVING lane — road markings show straight-ahead direction. "
            "Part of the road (typically the right side or an adjacent lane) is blocked by orange traffic cones, barriers, and temporary fencing. "
            "The remaining open lane is narrower than usual. "
            "Construction equipment (excavator, piles of dirt, concrete barriers) visible on the blocked side. "
            "Workers in reflective vests may be visible. Warning signs with arrows. "
            "\n\n"
            "ACTION: "
            "The ego car drives STRAIGHT ahead on the straight lane, approaching the construction zone. "
            "The car slows down and carefully passes through the narrowed section. "
            "Traffic cones and barriers slide backward on one side as the car passes. "
            "The road surface may appear rougher in the construction area. "
            "The car maintains steady low speed (15-25 km/h) through the zone, staying in the straight lane. "
            "After passing the zone, the road widens back to normal. "
            "\n\n"
            "Core event: ego car cruising straight through a construction zone on a straight road."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "IntersectionInteraction_LeftTurnVRUCrossing": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A Chinese urban signalized intersection. The ego car is in a LEFT-TURN lane. "
            "The left-turn signal is green. The ego car begins executing a LEFT TURN. "
            "\n\n"
            "ACTION: "
            "The ego car enters the intersection and begins turning LEFT. "
            "The entire scene visibly ROTATES as the car turns — buildings, poles, and road features sweep from the left side to the right side of the frame. "
            "The camera view continuously rotates clockwise during the turn. "
            "WHILE THE CAR IS MID-TURN (about halfway through the left turn), "
            "a VRU (pedestrian or cyclist) appears ahead, CROSSING the road/crosswalk that the ego car is turning into. "
            "The VRU walks or rides from right to left across the ego car's path. "
            "The ego car BRAKES and stops mid-turn to let the VRU pass. "
            "After the VRU clears, the ego car resumes and completes the left turn. "
            "\n\n"
            "Core event: ego car is actively turning left at an intersection when a VRU crosses its turning path, forcing a mid-turn stop."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "Intersection_SingleLaneRoundabout": {
        "prompt": (
            f"{FW_CAMERA_STYLE}"
            "\n\n"
            "SCENE: A SMALL single-lane roundabout in a Chinese urban residential area. "
            "The roundabout has a raised circular island in the center with some bushes or a small tree. "
            "CRITICAL: the circular road around the island is EXACTLY ONE lane wide — only enough for ONE car. "
            "There is NO second lane. The entry road is also a SINGLE lane leading into the roundabout. "
            "The roundabout is small — the center island is only about 5-8 meters in diameter. "
            "A yield sign or painted triangle marks the single-lane entry. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the roundabout on a SINGLE-LANE entry road. "
            "The car slows down and yields at the entry. "
            "Then the ego car enters the narrow circular road and follows the curve around the small center island. "
            "The center island with bushes rotates past on the LEFT side as the car curves around. "
            "The car drives about half the circle at very low speed (10-15 km/h), then exits onto an exit road. "
            "The entire path is single-lane — no lane markings inside the roundabout, just curbs on both sides. "
            "\n\n"
            "Core event: ego car navigating through a small single-lane roundabout with a single-lane entry."
        ),
        "duration": 10,
        "ratio": "16:9",
    },

    # ===== 严禁靠边场景（5大类 23子场景） ==========

    # ----- 类别1: 明确交通法规禁止区域 -----
    "PP_NoStopSignZone": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road with CLEARLY VISIBLE no-parking signs and road markings. "
            "The road has a solid YELLOW line along the curb (禁停黄线), and a yellow grid zone (网格线) is painted on the road surface. "
            "A red circular no-parking sign (禁止停车标志) is posted on a pole at the roadside. "
            "There are parked vehicles and shops along the sidewalk. Normal traffic flows on the road. "
            "\n\n"
            "ACTION: "
            "The ego car drives forward along the road. The yellow no-parking curb line is visible on the right side, sliding backward. "
            "As the car approaches, the yellow grid zone on the road surface becomes clearly visible. "
            "The red no-parking sign passes by on the right side of the frame. "
            "The ego car does NOT stop — it continues driving at moderate speed through the entire no-parking zone. "
            "\n\n"
            "Core event: ego car driving through a road segment with explicit no-parking signs and yellow curb markings."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_BusLaneAndStation": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban main road with a dedicated BUS LANE marked on the rightmost lane. "
            "The bus lane has '公交专用' (bus only) text painted in white on the road surface. "
            "Yellow bus lane boundary lines separate it from regular traffic lanes. "
            "A bus stop shelter with a bench and route signs is visible on the right side. "
            "A city bus is pulling into the bus stop. Passengers are waiting at the shelter. "
            "\n\n"
            "ACTION: "
            "The ego car drives in the regular traffic lane adjacent to the bus lane. "
            "The '公交专用' markings on the bus lane are clearly visible on the right side of the frame. "
            "As the ego car passes the bus stop, the shelter and waiting passengers slide backward in the frame. "
            "The city bus is stopped at the station. The ego car maintains speed and passes by. "
            "\n\n"
            "Core event: ego car passing a bus-only lane and bus station area on a Chinese urban road."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_FireLaneEntrance": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban area near a residential compound or commercial building. "
            "The road surface has large red/yellow text: '消防通道 禁止停车' (Fire Lane — No Parking). "
            "No-parking signs with fire lane markings are posted on both sides. "
            "The fire lane is wide and completely CLEAR — no vehicles parked in it. "
            "Building entrances and fire hydrants are visible along the lane. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the fire lane entrance slowly. "
            "The painted text '消防通道' on the ground becomes clearly visible and flows backward under the camera. "
            "The ego car drives past the fire lane entrance without stopping. "
            "Fire hydrants and no-parking signs slide backward on both sides. "
            "\n\n"
            "Core event: ego car passing a fire lane entrance with clear no-parking markings."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_CrosswalkZone": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road with a prominent WHITE CROSSWALK (斑马线) across the road. "
            "The zebra crossing has thick white stripes clearly visible on the gray asphalt. "
            "Several pedestrians are crossing or about to cross at the crosswalk. "
            "There is a pedestrian signal light showing green for pedestrians. "
            "Yellow yield-to-pedestrian warning signs are posted nearby. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the crosswalk and decelerates. "
            "Pedestrians are walking across the white zebra stripes in front of the car. "
            "The ego car comes to a complete stop before the crosswalk line. "
            "After pedestrians clear the crossing, the ego car slowly accelerates and crosses over the white stripes. "
            "The crosswalk markings flow backward under the camera. "
            "\n\n"
            "Core event: ego car yielding to pedestrians at a crosswalk zone."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_IntersectionSpecialRoad": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban intersection where multiple roads converge. "
            "Road surface has directional arrows and a white stop line. Traffic signals are visible overhead. "
            "There are lane markings guiding traffic flow. Vehicles from different directions pass through. "
            "The intersection is typical: wide, with clear markings, signal poles, and crosswalks at corners. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the intersection and slows down. "
            "The stop line and directional arrows on the road surface become clearly visible. "
            "The traffic light ahead shows red, and the ego car stops at the stop line. "
            "Other vehicles pass through the intersection from cross directions. "
            "When the light turns green, the ego car accelerates and drives through the intersection. "
            "\n\n"
            "Core event: ego car navigating through a busy urban intersection with traffic signals."
        ),
        "duration": 10,
        "ratio": "16:9",
    },

    # ----- 类别2: 高风险安全隐患区域 -----
    "PP_ExpresswayMainRoad": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese EXPRESSWAY or urban elevated highway with multiple lanes. "
            "The road is wide with clear white dashed lane dividers and a solid white line on the right marking the emergency lane. "
            "A green central median barrier separates opposing traffic. "
            "Green directional signs are overhead. Vehicles drive at high speed. "
            "The emergency lane on the far right is empty and marked with diagonal yellow stripes. "
            "\n\n"
            "ACTION: "
            "The ego car drives at highway speed in one of the main lanes. "
            "Road markings and lane lines flow backward rapidly. "
            "The green median and highway signs pass by overhead. "
            "Other vehicles (trucks, sedans) overtake or drive alongside in adjacent lanes. "
            "The emergency lane stays visible on the right, clearly unused. "
            "\n\n"
            "Core event: ego car driving at speed on a Chinese expressway main road."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_InsideTunnel": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: Inside a Chinese urban or highway TUNNEL. "
            "Concrete or tiled walls on both sides, with LED strip lighting on the ceiling. "
            "The tunnel is dimly lit compared to outside. Lane markings are visible on the road surface. "
            "Red tail lights of vehicles ahead glow in the dimness. "
            "Emergency exits and fire extinguisher cabinets are visible along the walls. "
            "\n\n"
            "ACTION: "
            "The ego car drives steadily through the tunnel at moderate speed. "
            "The overhead lights create rhythmic bright patches that flow backward. "
            "Wall-mounted fixtures and emergency signs slide past on both sides. "
            "Vehicles ahead maintain steady distance. The tunnel curves slightly. "
            "Eventually, a bright opening appears ahead — the tunnel exit. "
            "\n\n"
            "Core event: ego car driving through a tunnel interior."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_LongDownhill": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese mountain or overpass road with a LONG DOWNHILL slope. "
            "The road surface visibly tilts downward toward a distant point. "
            "Guardrails and reflective delineator posts line both sides. "
            "A yellow diamond warning sign showing a steep grade is visible ahead. "
            "The road may curve gently in the distance. Surrounding terrain is hilly. "
            "\n\n"
            "ACTION: "
            "The ego car descends the long downhill at controlled speed. "
            "The road surface ahead visibly slopes down. "
            "Guardrails and reflective posts slide backward steadily. "
            "The downhill grade warning sign passes on the right. "
            "The road markings flow backward at a moderate pace, indicating controlled speed. "
            "\n\n"
            "Core event: ego car descending a long downhill road segment."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_PoorVisibilityZone": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese road with POOR FORWARD VISIBILITY. "
            "The road goes around a SHARP CURVE, and the view ahead is blocked by buildings, vegetation, or a hillside on the inside of the curve. "
            "Curve warning signs (chevron arrows) are posted along the curve. "
            "Convex mirrors may be installed at the curve for visibility. "
            "Speed limit signs and rumble strips are visible before the curve. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the blind curve and decelerates significantly. "
            "Warning signs and chevron arrows slide past. "
            "The view ahead is blocked — the road disappears behind the obstruction. "
            "The ego car slowly navigates the curve, and as it rounds the bend, the road ahead gradually reveals itself. "
            "The car accelerates gently after clearing the blind section. "
            "\n\n"
            "Core event: ego car navigating a blind curve with obstructed forward visibility."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_NonMotorLaneSidewalk": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road with clearly separated zones: motor vehicle lanes on the left, "
            "a non-motorized vehicle lane (非机动车道) in the middle-right, and a pedestrian sidewalk on the far right. "
            "The non-motor lane is separated by a raised curb or green painted boundary. "
            "Bicycles and electric scooters ride in the non-motor lane. Pedestrians walk on the sidewalk. "
            "Trees line the boundary between the sidewalk and the road. "
            "\n\n"
            "ACTION: "
            "The ego car drives in the motor vehicle lane at moderate speed. "
            "On the right, bicycles and e-scooters are visible in the non-motor lane, moving in the same direction. "
            "Pedestrians walk along the tree-lined sidewalk further right. "
            "The boundary markings between lanes flow backward steadily. "
            "\n\n"
            "Core event: ego car driving past a non-motorized vehicle lane and pedestrian sidewalk."
        ),
        "duration": 10,
        "ratio": "16:9",
    },

    # ----- 类别3: 影响公共安全与秩序区域 -----
    "PP_FireHydrantZone": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road with a RED FIRE HYDRANT (消防栓) visible on the roadside curb. "
            "The area around the hydrant has yellow no-parking markings on the curb. "
            "The fire hydrant is a standard red cylindrical post about 0.7m tall. "
            "Nearby buildings and parked vehicles are visible, but the 5-meter zone around the hydrant is clear. "
            "\n\n"
            "ACTION: "
            "The ego car drives along the road at steady speed. "
            "The red fire hydrant appears on the right side and slides backward as the car passes. "
            "Yellow curb markings near the hydrant are clearly visible. "
            "The ego car does not slow down significantly — just passes normally. "
            "\n\n"
            "Core event: ego car passing a fire hydrant with a clear no-parking zone around it."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_HospitalEntrance": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road in front of a HOSPITAL entrance. "
            "The hospital building is visible with a red cross symbol and Chinese hospital signage. "
            "The main entrance gate has vehicle access lanes. An AMBULANCE with emergency markings is parked near the ER entrance. "
            "Road markings show an emergency lane (急救通道). People walk in and out of the hospital gate. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the hospital entrance area and slows down. "
            "The hospital sign and red cross symbol are clearly visible ahead. "
            "The ambulance parked at the ER entrance slides past on the right. "
            "Pedestrians (patients, visitors) cross near the hospital gate. "
            "The ego car drives past the hospital entrance at low speed. "
            "\n\n"
            "Core event: ego car driving past a hospital main entrance and emergency lane."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_SchoolEntrance": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road in front of a SCHOOL or KINDERGARTEN during drop-off/pick-up time. "
            "The school gate is visible with a sign. A large crowd of PARENTS and CHILDREN gathers near the entrance. "
            "Yellow 'SCHOOL ZONE' warning signs and speed bumps are on the road. "
            "A traffic coordinator in a reflective vest is directing traffic near the gate. "
            "Some vehicles are double-parked temporarily while dropping off children. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the school zone and slows down significantly due to the crowd. "
            "Children and parents are visible near the road edges and crossing the street. "
            "Speed bumps cause a slight camera bounce. The school gate and crowd pass by slowly. "
            "The ego car navigates carefully through the congested school area. "
            "\n\n"
            "Core event: ego car slowly driving past a school entrance during busy drop-off/pick-up period."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_EventVenueEntrance": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road near a LARGE EVENT VENUE (stadium or convention center) during an active event. "
            "The venue building is visible in the background. Temporary barriers and crowd-control fences line the road. "
            "Security personnel in uniforms stand at checkpoints. Large crowds of people walk toward the venue. "
            "Temporary traffic signs indicate detours or restricted areas. "
            "\n\n"
            "ACTION: "
            "The ego car drives slowly along the perimeter road outside the venue. "
            "Dense pedestrian crowds are visible, some crossing the road. "
            "Temporary barriers and security checkpoints slide past. "
            "The ego car navigates at very low speed through the event-affected area. "
            "\n\n"
            "Core event: ego car driving past a large venue entrance during an active event with crowd control."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_MetroTransitHub": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road near a METRO STATION exit and a major transit hub. "
            "The metro station entrance (with a large 'M' or station name sign) is visible on the sidewalk. "
            "Many commuters are streaming out of the metro exit. A taxi waiting area with queue lines is on the roadside. "
            "Several taxis and rideshare vehicles are lined up, waiting for passengers. A bus stop is also nearby. "
            "\n\n"
            "ACTION: "
            "The ego car drives past the metro station area at slow speed. "
            "Commuters pour out of the metro exit and walk near the road. "
            "Taxis and rideshare cars are stopped in the designated pickup zone. "
            "A bus pulls away from the nearby bus stop. "
            "The ego car navigates through the busy transit area. "
            "\n\n"
            "Core event: ego car passing a metro station exit and transit hub with dense pedestrian and vehicle activity."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_ResidentialGateCrowded": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban residential compound main gate. "
            "Multiple private cars are PARKED or WAITING on both sides of the road near the gate, narrowing the drivable space significantly. "
            "The gate has a boom barrier and a security booth. "
            "Pedestrians and electric scooters weave between the parked cars. "
            "The remaining passable width is barely enough for one vehicle. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the crowded residential gate at very low speed. "
            "Parked and idling vehicles on both sides create a narrow corridor. "
            "A pedestrian or e-scooter crosses in front. The boom barrier is visible ahead. "
            "The ego car carefully threads through the tight space past the gate area. "
            "\n\n"
            "Core event: ego car navigating through a residential compound gate crowded with parked vehicles."
        ),
        "duration": 10,
        "ratio": "16:9",
    },

    # ----- 类别4: 特定功能受限区域 -----
    "PP_MilitaryGovZone": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road near an IMPORTANT GOVERNMENT or institutional building. "
            "A tall perimeter wall with security cameras surrounds the compound. "
            "A guard booth with uniformed personnel is visible at the entrance. "
            "No-parking and no-stopping signs are prominently posted along the road. "
            "The road is clean, well-maintained, and has few other vehicles. "
            "\n\n"
            "ACTION: "
            "The ego car drives at a moderate, steady speed past the compound wall. "
            "The guard booth and security cameras slide past on the right side. "
            "No-parking signs are clearly visible. "
            "The ego car does not stop or slow down — it passes through the restricted zone normally. "
            "\n\n"
            "Core event: ego car driving past a sensitive government/institutional zone with no-parking restrictions."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_GPSUnstableZone": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban area with DENSE HIGH-RISE BUILDINGS or under a complex multi-level highway overpass. "
            "Tall buildings on both sides create an 'urban canyon' effect, blocking most of the sky. "
            "Highway ramps and overpasses stack above the road in multiple layers. "
            "Shadows from the structures create alternating dark and light patches on the road. "
            "\n\n"
            "ACTION: "
            "The ego car drives through the urban canyon at moderate speed. "
            "Tall buildings slide past on both sides, towering high above. "
            "The overhead structure creates shadow patterns that sweep across the road surface. "
            "The car navigates under the multi-level overpass structure. "
            "\n\n"
            "Core event: ego car driving through an urban canyon / under overpasses where GPS signal may be unstable."
        ),
        "duration": 10,
        "ratio": "16:9",
    },

    # ----- 类别5: 易引发拥堵或冲突区域 -----
    "PP_SingleLaneMainRoad": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban MAIN ROAD where most lanes are CLOSED due to construction or an incident. "
            "Only ONE lane remains open for traffic. Orange traffic cones, barriers, and warning signs funnel all traffic into the single lane. "
            "Vehicles queue up, bumper-to-bumper, crawling through the bottleneck. "
            "Workers or police may be directing traffic near the merge point. "
            "\n\n"
            "ACTION: "
            "The ego car is in the queue of vehicles approaching the single-lane bottleneck. "
            "Traffic cones and barriers narrow the road from the sides. "
            "The car ahead moves slowly; the ego car follows at walking speed. "
            "The single open lane is tight, with barriers close on both sides. "
            "The ego car crawls through the restricted section. "
            "\n\n"
            "Core event: ego car queuing and crawling through a main road reduced to a single lane."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_HighTrafficIntersection": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A very BUSY Chinese urban intersection with EXTREMELY HIGH traffic volume. "
            "All approach lanes are full of queued vehicles waiting at the signals. "
            "Multiple phases of traffic lights control the flow. "
            "Pedestrians, e-bikes, and buses all compete for space. "
            "The intersection is large, with wide crosswalks and multiple turning lanes. "
            "\n\n"
            "ACTION: "
            "The ego car is stuck in a long queue approaching the intersection. "
            "Vehicles ahead are bumper-to-bumper. The traffic light is red. "
            "Cross-traffic flows through the intersection while the ego car waits. "
            "E-bikes weave between waiting cars. "
            "When the light turns green, the queue moves forward slowly, and the ego car inches through. "
            "\n\n"
            "Core event: ego car in heavy traffic at a high-volume urban intersection."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_TaxiRideshareWaiting": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road near a COMMERCIAL DISTRICT or TRAIN STATION where many TAXIS and RIDESHARE vehicles are parked along the curb, waiting for passengers. "
            "Multiple vehicles have hazard lights flashing. Passengers are getting in and out of cars along the road. "
            "The parked taxis compress the drivable space to barely one lane. "
            "Some vehicles are double-parked. "
            "\n\n"
            "ACTION: "
            "The ego car drives slowly through the taxi/rideshare waiting area. "
            "Vehicles with hazard lights blink on both sides. "
            "A passenger steps into the road to approach a rideshare car, forcing the ego car to slow further. "
            "The ego car carefully navigates through the narrowed passage. "
            "\n\n"
            "Core event: ego car passing through a dense taxi/rideshare waiting zone with limited road space."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_LoadingUnloadingZone": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: A Chinese urban road in a COMMERCIAL area with designated LOADING/UNLOADING zones. "
            "A delivery TRUCK is parked in the loading zone, its rear doors open. "
            "Workers carry boxes between the truck and a building entrance. "
            "The loading zone is marked with special ground markings. "
            "Pallet jacks or hand carts are visible on the sidewalk. The truck partially extends into the driving lane. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the loading zone and encounters the partially blocking truck. "
            "Workers walk back and forth across the road carrying cargo. "
            "The ego car slows down and steers slightly left to pass the protruding truck. "
            "After clearing the loading zone, the road widens back to normal. "
            "\n\n"
            "Core event: ego car navigating past an active loading/unloading zone with a partially blocking truck."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
    "PP_IndustrialParkEntrance": {
        "prompt": (
            f"{FW_CAMERA_STYLE}\n\n"
            "SCENE: The ENTRANCE of a Chinese industrial or technology PARK. "
            "A gate with a boom barrier and a security guard booth is visible ahead. "
            "During morning rush hour, VEHICLES QUEUE in a long line waiting to enter the park. "
            "Speed bumps are on the approach road. The park name sign is displayed on the gate structure. "
            "Some vehicles are stopped temporarily on the roadside, dropping off passengers. "
            "\n\n"
            "ACTION: "
            "The ego car approaches the park entrance at low speed. "
            "The queue of vehicles ahead extends from the boom barrier. "
            "Speed bumps cause slight camera bounces. "
            "Workers walk along the roadside toward the gate. "
            "The ego car joins the queue and creeps forward toward the boom barrier. "
            "\n\n"
            "Core event: ego car approaching an industrial park entrance with queuing vehicles during rush hour."
        ),
        "duration": 10,
        "ratio": "16:9",
    },
}


def get_model_config(model_alias: str) -> dict:
    if model_alias not in MODEL_CONFIGS:
        raise ValueError(f"未知模型: {model_alias}，可选: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_alias]


def _load_ref_image_b64(ref_image_path: str) -> str:
    import base64
    from PIL import Image
    import io as _io
    img = Image.open(ref_image_path)
    img = img.resize((640, 360))
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def submit_generation(api_key: str, prompt: str, model_cfg: dict,
                       duration: int = 10, resolution: str = "720p",
                       ratio: str = "16:9",
                       ref_image_path: str | None = None) -> str:
    url = f"{model_cfg['api_base']}{model_cfg['submit_path']}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    content = []
    if ref_image_path:
        img_b64_url = _load_ref_image_b64(ref_image_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": img_b64_url},
            "role": "reference_image",
        })
    content.append({"type": "text", "text": prompt})

    body = {
        "model": model_cfg["model"],
        "content": content,
        "resolution": resolution,
        "ratio": ratio,
        "duration": duration,
    }

    resp = requests.post(url, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    task_id = result.get("id") or result.get("task_id")
    if not task_id:
        raise RuntimeError(f"未返回 task_id: {result}")
    print(f"  提交响应: {json.dumps(result, ensure_ascii=False)[:200]}")
    return task_id


def poll_task(api_key: str, task_id: str, model_cfg: dict,
              max_wait: int = 600) -> dict:
    url = f"{model_cfg['api_base']}{model_cfg['poll_path']}/{task_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    start = time.time()
    while time.time() - start < max_wait:
        resp = requests.get(url, headers=headers, timeout=60)
        resp.raise_for_status()
        result = resp.json()

        status = result.get("status", "")

        if status in ("SUCCESS", "succeeded", "complete"):
            return result
        elif status in ("FAILED", "failed"):
            reason = result.get("fail_reason", result.get("error", "未知原因"))
            raise RuntimeError(f"视频生成失败: {reason}")

        inner = result.get("data", {})
        if isinstance(inner, dict):
            inner_status = inner.get("status", "")
            if inner_status == "succeeded":
                return result
            elif inner_status == "failed":
                raise RuntimeError(f"视频生成失败: {inner.get('fail_reason', '未知')}")

        elapsed = int(time.time() - start)
        progress = result.get("progress", status or "waiting")
        print(f"  [{elapsed}s] 状态: {progress}")
        time.sleep(15)

    raise TimeoutError(f"等待超时 ({max_wait}s)")


def download_video(video_url: str, output_path: str):
    resp = requests.get(video_url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  已下载: {output_path} ({size_mb:.1f} MB)")


def extract_video_url(result: dict) -> str:
    video_url = result.get("content", {}).get("video_url", "")
    if not video_url:
        inner = result.get("data", {})
        if isinstance(inner, dict):
            video_url = inner.get("content", {}).get("video_url", "")
            if not video_url:
                inner2 = inner.get("data", {})
                if isinstance(inner2, dict):
                    video_url = inner2.get("content", {}).get("video_url", "")
    return video_url


def generate_one(api_key: str, model_cfg: dict, tag_name: str, prompt: str,
                 duration: int, resolution: str, ratio: str,
                 output_dir: str, suffix: str = "",
                 ref_image_path: str | None = None) -> str | None:
    try:
        task_id = submit_generation(
            api_key=api_key, prompt=prompt, model_cfg=model_cfg,
            duration=duration, resolution=resolution, ratio=ratio,
            ref_image_path=ref_image_path,
        )
        print(f"  Task ID: {task_id}")

        print(f"  等待生成...")
        result = poll_task(api_key, task_id, model_cfg)

        video_url = extract_video_url(result)
        if not video_url:
            print(f"  警告: 未找到 video_url，响应: {json.dumps(result, ensure_ascii=False)[:300]}")
            return None

        fname = f"{tag_name}_seedance2{suffix}.mp4"
        output_path = os.path.join(output_dir, fname)
        download_video(video_url, output_path)

        meta_path = output_path.replace(".mp4", ".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "tag": tag_name,
                "model": model_cfg["model"],
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
                "result": result,
            }, f, ensure_ascii=False, indent=2)
        return output_path

    except Exception as e:
        print(f"  错误: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Seedance 驾驶场景视频生成")
    parser.add_argument("--api_key", type=str, required=True, help="API Key")
    parser.add_argument("--model", type=str, default="seedance20",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="模型选择 (默认 seedance20)")
    parser.add_argument("--tag", type=str, default=None,
                        help="预设标签名")
    parser.add_argument("--batch_tags", type=str, default=None,
                        help="逗号分隔的标签名列表（批量生成）")
    parser.add_argument("--prompt", type=str, default=None, help="自定义 prompt（覆盖预设）")
    parser.add_argument("--count", type=int, default=1, help="每个标签的生成数量")
    parser.add_argument("--duration", type=int, default=10, help="视频时长 (2-12秒)")
    parser.add_argument("--resolution", type=str, default="720p", help="分辨率")
    parser.add_argument("--output_dir", type=str, default="scene_tag/generated_videos",
                        help="输出目录")
    parser.add_argument("--ref_image", type=str, default=None,
                        help="参考图路径（role=reference_image）")
    parser.add_argument("--skip_existing", action="store_true",
                        help="跳过已存在的视频文件")
    args = parser.parse_args()

    model_cfg = get_model_config(args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ref_image and not os.path.isfile(args.ref_image):
        print(f"参考图不存在: {args.ref_image}")
        return

    if args.batch_tags:
        tags = [t.strip() for t in args.batch_tags.split(",") if t.strip()]
    elif args.tag:
        tags = [args.tag]
    elif args.prompt:
        tags = ["custom"]
    else:
        print("请指定 --tag, --batch_tags, 或 --prompt")
        print(f"可用标签: {', '.join(TAG_PROMPTS.keys())}")
        return

    ref_label = "_ref" if args.ref_image else ""
    print(f"\n{'='*60}")
    print(f"  Seedance 视频生成")
    print(f"  模型: {model_cfg['model']}")
    print(f"  标签数: {len(tags)}, 每个生成: {args.count}")
    print(f"  时长: {args.duration}s, 分辨率: {args.resolution}")
    if args.ref_image:
        print(f"  参考图: {args.ref_image}")
    if args.skip_existing:
        print(f"  跳过已存在: 是")
    print(f"{'='*60}\n")

    results = []
    skipped = 0
    for tag_name in tags:
        if tag_name == "custom":
            prompt = args.prompt
            duration = args.duration
            ratio = "16:9"
        elif tag_name in TAG_PROMPTS:
            preset = TAG_PROMPTS[tag_name]
            prompt = args.prompt or preset["prompt"]
            duration = args.duration or preset.get("duration", 10)
            ratio = preset.get("ratio", "16:9")
        else:
            print(f"\n跳过未知标签: {tag_name}")
            continue

        for i in range(args.count):
            suffix = f"_v{i+1}" if args.count > 1 else ""
            fname = f"{tag_name}_seedance2{ref_label}{suffix}.mp4"
            output_path = os.path.join(args.output_dir, fname)

            if args.skip_existing and os.path.isfile(output_path):
                size_mb = os.path.getsize(output_path) / 1024 / 1024
                if size_mb > 0.1:
                    print(f"\n[{tag_name}] 已存在，跳过 ({size_mb:.1f} MB)")
                    results.append(output_path)
                    skipped += 1
                    continue

            print(f"\n[{tag_name}] ({i+1}/{args.count}) 提交生成任务...{' (带参考图)' if args.ref_image else ''}")
            out = generate_one(
                api_key=args.api_key, model_cfg=model_cfg,
                tag_name=tag_name, prompt=prompt,
                duration=duration, resolution=args.resolution, ratio=ratio,
                output_dir=args.output_dir,
                suffix=f"{ref_label}{suffix}",
                ref_image_path=args.ref_image,
            )
            if out:
                results.append(out)

    print(f"\n{'='*60}")
    print(f"  完成! 成功 {len(results)}/{len(tags) * args.count} (跳过 {skipped})")
    for r in results:
        print(f"    {r}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
