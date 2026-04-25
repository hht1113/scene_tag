"""
Generate per-category mining prompt files for 122 driving scene labels.
Each file is a self-contained system prompt for Qwen3-VL-235B cold-start mining.
Organization: 9 files, one per top-level category (一级类).
"""

import os
from collections import OrderedDict

OUTPUT_DIR = "/root/workspace/LLaMA-Factory/scene_tag/prompt_txt"

# P00 已有充足数据，不需要挖掘的标签
P00_EXCLUDE = {
    "DynamicInteraction_VRUInLaneCrossing",
    "DynamicInteraction_VehicleInLaneCrossing",
    "DynamicInteraction_StandardVehicleCutIn",
    "TrafficLight_StraightStopOrGo",
    "TrafficLight_LeftTurnStopOrGo",
    "LaneChange_NavForIntersection",
    "LaneChange_AvoidSlowVRU",
    "LaneChange_AvoidStaticVehicle",
    "StartStop_StartFromMainRoad",
    "StartStop_ParkRoadside",
    "Intersection_StandardUTurn",
    "LaneCruising_Straight",
}

COMMON_TERMINOLOGY = """TERMINOLOGY:
• VRU (vulnerable road users): pedestrians, bicycles, e-bikes, tricycles — excludes motor vehicles.
• Vehicle: motor vehicles only (cars, trucks, buses, motorcycles) — excludes bicycles, e-bikes, tricycles.
• Ego vehicle: the vehicle equipped with the camera recording the video; its behavior is the subject of annotation.
• FW (front-wide): the front-facing wide-angle camera view used for recording.
• Lane crossing: another road user moving approximately PERPENDICULAR to the ego's travel direction, traversing across the ego's lane.
• Cut-in: another road user moving approximately PARALLEL to the ego, merging INTO the ego's lane from an adjacent lane.
• Cut-out: the lead vehicle in the ego's lane changing OUT of the ego's lane to an adjacent lane.
• Nudge / in-lane avoidance: the ego makes a lateral offset WITHIN its lane (no lane change) to avoid an obstacle.
• Borrow-lane avoidance: the ego partially or fully enters the ONCOMING lane to bypass an obstacle, then returns.
• Cross-line bypass: the ego crosses lane markings to bypass a queue of stopped vehicles/obstacles.
• Overtake: the ego changes lanes, passes a slower vehicle, and returns to the original lane — a complete compound maneuver.
• SOD (Small Object Detection): small or distant objects such as traffic cones, debris, small animals, fallen cargo.
"""

COMMON_OUTPUT_FORMAT = """OUTPUT FORMAT:
<driving_maneuver>label_name</driving_maneuver> from <start_time>XX.X</start_time> to <end_time>YY.Y</end_time> seconds.
• Use the EXACT label names defined below.
• Multiple segments are allowed — separate with " and " in chronological order.
• If NO label in this category applies to any portion of the video, output:
  <driving_maneuver>not_applicable</driving_maneuver> from <start_time>0.0</start_time> to <end_time>20.0</end_time> seconds.
• Time precision: 1 decimal place (e.g., 5.0, 12.5).
• Minimum segment duration: 1.0 second.
"""

COMMON_RULES = """LABELING RULES:
1. Assign a label ONLY when the action clearly matches a predefined category (confidence ≥ 90%).
2. NEVER force-match ambiguous scenes; use not_applicable instead.
3. Multiple labels from this category may co-occur if multiple maneuvers are clearly present.
4. Time segments MUST be contiguous and cover the relevant portions of the 20-second video.
5. Base times on the video timeline (0.0 to 20.0 seconds).
6. Focus ONLY on the ego vehicle's behavior, not other vehicles' independent actions.
"""

COMMON_GUIDELINES = """IMPORTANT GUIDELINES:
1. Analyze the ENTIRE 20-second video thoroughly before labeling.
2. Match actions to the MOST SPECIFIC appropriate label.
3. When in doubt between two labels, check the DISAMBIGUATION RULES.
4. Ensure time segments accurately reflect when each maneuver occurs.
5. Maintain chronological order in output.
"""

# ─────────────────────────────────────────────
# CATEGORY DEFINITIONS
# ─────────────────────────────────────────────

CATEGORIES = OrderedDict()

# ============================================
# 1. DynamicInteraction (车道动态交互) — 12 labels
# ============================================
CATEGORIES["01_DynamicInteraction"] = {
    "title": "Lane Dynamic Interaction (车道动态交互)",
    "description": "Interactions between the ego vehicle and other road users within or adjacent to the ego's lane on a STRAIGHT road segment (NOT at intersections). Includes lane crossings, cut-ins, cut-outs, door openings, sudden braking, and small object reactions.",
    "labels": OrderedDict([
        ("DynamicInteraction_VRUInLaneCrossing", {
            "cn": "车道内VRU横穿",
            "def": [
                "A VRU (pedestrian, bicycle, e-bike, tricycle) actively CROSSES the ego's lane of travel.",
                "The VRU must be moving approximately PERPENDICULAR to the ego's direction — traversing across the lane, not walking/cycling along the same direction.",
                "Ego vehicle yields, slows, or stops because the VRU is entering or traversing the ego's lane (e.g., at a zebra crossing, jaywalking).",
                "Key visual cue: a VRU is visible moving LEFT-to-RIGHT or RIGHT-to-LEFT across the ego's path.",
                "Do NOT trigger when VRUs are merely nearby (sidewalks, adjacent lanes) without actively crossing.",
            ],
        }),
        ("DynamicInteraction_VehicleInLaneCrossing", {
            "cn": "车道内车辆横穿",
            "def": [
                "A motor vehicle crosses the ego's lane of travel — including reversing vehicles, cross-traffic vehicles entering/exiting driveways or side roads.",
                "Ego vehicle yields, slows, or stops for the crossing vehicle.",
                "Key visual cue: a vehicle moves perpendicular or at a sharp angle across the ego's forward path.",
                "Bicycles/e-bikes/tricycles crossing → use VRUInLaneCrossing instead.",
            ],
        }),
        ("DynamicInteraction_StandardVehicleCutIn", {
            "cn": "邻车道车辆常规切入",
            "def": [
                "A motor vehicle from an adjacent lane merges INTO the ego's lane ahead at normal speed, reducing the ego's following distance.",
                "The cut-in is smooth and predictable — the merging vehicle signals or gradually enters the lane.",
                "Key visual cue: a vehicle in an adjacent lane drifts into the ego's lane; the ego may decelerate in response.",
            ],
        }),
        ("DynamicInteraction_EmergencyVehicleCutIn", {
            "cn": "邻车道车辆紧急切入",
            "def": [
                "A motor vehicle abruptly and aggressively merges into the ego's lane with little warning, forcing the ego to brake hard.",
                "Distinct from standard cut-in by the URGENCY: short time-to-collision, sudden lateral movement, no or late signaling.",
                "Key visual cue: the merging vehicle appears suddenly in the ego's lane with minimal gap.",
            ],
        }),
        ("DynamicInteraction_StartupVehicleCutIn", {
            "cn": "邻车道车辆起步切入",
            "def": [
                "A motor vehicle that was previously STATIONARY (parked, stopped) in an adjacent lane starts moving and merges into the ego's lane.",
                "Distinct from standard cut-in: the cutting vehicle transitions from stopped → moving → into ego's lane.",
                "Key visual cue: a previously stopped vehicle pulls out from the roadside or adjacent lane into the ego's path.",
            ],
        }),
        ("DynamicInteraction_ConsecutiveLaneChangeCutIn", {
            "cn": "连续变道车辆切入",
            "def": [
                "A motor vehicle performs MULTIPLE consecutive lane changes and ends up cutting into the ego's lane.",
                "The vehicle crosses two or more lanes before entering the ego's lane.",
                "Key visual cue: a vehicle is seen rapidly crossing lane markings from a distant lane, eventually arriving in the ego's lane.",
            ],
        }),
        ("DynamicInteraction_SlowVRUCutIn", {
            "cn": "邻车道VRU缓速切入",
            "def": [
                "A VRU (bicycle, e-bike, tricycle) slowly drifts or merges into the ego's lane from an adjacent lane or road shoulder.",
                "The entry is gradual — the VRU may not be fully aware of the ego vehicle.",
                "Key visual cue: a slow-moving VRU near the lane boundary gradually encroaches into the ego's lane.",
            ],
        }),
        ("DynamicInteraction_EmergencyVRUCutIn", {
            "cn": "邻车道VRU紧急切入",
            "def": [
                "A VRU suddenly and unpredictably swerves or jumps into the ego's lane, requiring immediate reaction.",
                "Distinct from slow VRU cut-in by the ABRUPTNESS of the entry.",
                "Key visual cue: a VRU makes a sudden lateral movement into the ego's path.",
            ],
        }),
        ("DynamicInteraction_LeadVehicleCutOut", {
            "cn": "前车切出",
            "def": [
                "The vehicle directly ahead of the ego (lead vehicle) changes lanes OUT of the ego's lane to an adjacent lane.",
                "After the cut-out, the ego faces a new traffic situation: a new lead vehicle, an empty road, or a previously hidden obstacle.",
                "Key visual cue: the lead vehicle moves laterally out of the ego's lane; the scene ahead changes.",
                "Often co-occurs with SmallObjectReaction when the departing lead reveals a small obstacle.",
            ],
        }),
        ("DynamicInteraction_GapOpeningCutIn", {
            "cn": "豁口切入",
            "def": [
                "A vehicle or VRU enters the ego's lane through a GAP or OPENING (豁口) in a row of parked vehicles, barriers, fences, or other roadside structures.",
                "The intruding object emerges from a break/gap in an otherwise continuous barrier, making it hard to predict.",
                "Key visual cue: a gap or opening in roadside parked cars / barriers is visible; a vehicle or VRU suddenly appears through it into the ego's path.",
                "Distinct from standard cut-in: the intruder comes from BEHIND a physical barrier through a gap, not from an adjacent lane.",
            ],
        }),
        ("DynamicInteraction_LeadVehicleSuddenBrake", {
            "cn": "前车急刹",
            "def": [
                "The lead vehicle in the ego's lane brakes suddenly and unexpectedly, forcing the ego to perform emergency braking.",
                "Key visual cue: the lead vehicle's brake lights illuminate abruptly; the gap closes rapidly; the ego decelerates hard.",
                "Distinct from normal following-stop: the SUDDENNESS is the key feature — unexpected hard braking.",
            ],
        }),
        ("DynamicInteraction_StaticObjectReaction", {
            "cn": "静态障碍物反应(SOD)",
            "def": [
                "The ego vehicle reacts (brakes, swerves, or slows) to a STATIC obstacle in the lane — such as traffic cones, road debris, fallen cargo, rocks, potholes, or other SOD (Static Object Detection) targets.",
                "SOD targets are STATIONARY objects that are not detected by standard dynamic OD (Object Detection) — they are non-moving hazards on the road surface.",
                "The object may appear suddenly (e.g., after a lead vehicle cuts out and reveals it).",
                "Key visual cue: a stationary obstacle is visible on the road surface in the ego's path; the ego adjusts its trajectory.",
            ],
        }),
    ]),
    "disambiguation": [
        ("StandardVehicleCutIn vs EmergencyVehicleCutIn",
         "Standard: smooth, predictable merge with adequate gap. Emergency: abrupt, aggressive, forces hard braking. Judge by the URGENCY of the ego's reaction."),
        ("VRUInLaneCrossing vs SlowVRUCutIn",
         "Crossing: VRU moves PERPENDICULAR to traffic (left↔right). Cut-in: VRU moves roughly PARALLEL, drifting into the lane from the side. Check the VRU's movement DIRECTION relative to traffic flow."),
        ("LeadVehicleCutOut vs StandardVehicleCutIn",
         "Cut-out: the LEAD vehicle leaves the ego's lane. Cut-in: a SIDE vehicle enters the ego's lane. These are opposite events and may co-occur when a side vehicle cuts in while the lead cuts out."),
        ("GapOpeningCutIn vs StandardVehicleCutIn",
         "GapOpening: the intruder emerges from BEHIND a physical barrier through a gap (豁口). StandardCutIn: the intruder merges from a visible adjacent lane. Check whether the intruder comes from behind parked cars/barriers (Gap) or from a neighboring lane (Standard)."),
        ("LeadVehicleSuddenBrake vs SteadyFollowing/FollowingStop (LaneCruising/StartStop)",
         "SuddenBrake: UNEXPECTED hard braking requiring emergency response. FollowingStop: gradual, expected deceleration. SteadyFollowing: no braking event at all."),
    ],
}

# ============================================
# 2. TrafficLight (红绿灯通行) — 17 labels
# ============================================
CATEGORIES["02_TrafficLight"] = {
    "title": "Traffic Light Passage (红绿灯通行)",
    "description": "Ego vehicle behavior at traffic-light-controlled intersections, including stopping, starting, and passing through under various signal states (normal, flashing, dark/off, or special signals).",
    "labels": OrderedDict([
        ("TrafficLight_StraightStopOrGo", {
            "cn": "直行红绿灯起停",
            "def": [
                "Ego vehicle stops or starts at a traffic light for STRAIGHT-LINE movement through an intersection.",
                "Includes: waiting at red, departing on green, decelerating for yellow — all for straight-ahead travel.",
                "Key visual cue: after passing the intersection, the ego's heading remains approximately the SAME as before entering.",
            ],
        }),
        ("TrafficLight_LeftTurnStopOrGo", {
            "cn": "左转红绿灯起停",
            "def": [
                "Ego vehicle stops or starts at a traffic light for LEFT-TURN movement.",
                "REQUIRES clear visual evidence that the ego actually turns left: heading changes ~90° to the left after the intersection.",
                "Being in a left-turn lane or seeing a left-turn signal alone is NOT sufficient — the vehicle must visibly execute the turn.",
                "If the ego has not yet departed (still waiting), default to StraightStopOrGo unless turning motion is clearly beginning.",
            ],
        }),
        ("TrafficLight_RightTurnStopOrGo", {
            "cn": "右转红绿灯起停",
            "def": [
                "Ego vehicle stops or starts at a traffic light for RIGHT-TURN movement.",
                "The ego's heading changes ~90° to the right after the intersection.",
                "Key visual cue: ego turns right at a signalized intersection, controlled by a right-turn signal or a general signal.",
            ],
        }),
        ("TrafficLight_UTurnStopOrGo", {
            "cn": "掉头红绿灯起停",
            "def": [
                "Ego vehicle stops or starts at a traffic light specifically for a U-TURN maneuver.",
                "The ego's heading changes ~180° after the intersection — ending up facing the opposite direction.",
                "Distinct from LeftTurnStopOrGo: U-turn results in OPPOSITE heading, not perpendicular.",
            ],
        }),
        ("TrafficLight_WaitingZoneStopOrGo", {
            "cn": "待转区红绿灯起停",
            "def": [
                "Ego vehicle stops or starts within a designated WAITING ZONE (待转区) at a signalized intersection.",
                "The ego advances into the intersection's waiting zone during a preliminary signal phase, then proceeds when the final signal turns green.",
                "Key visual cue: ground markings or painted zones indicating a 待转区; ego stops INSIDE the intersection boundary, not at the stop line.",
            ],
        }),
        ("TrafficLight_StraightGreenFlash", {
            "cn": "直行绿闪通行",
            "def": [
                "Ego passes straight through an intersection while the traffic light is in GREEN FLASHING state (绿闪).",
                "Green flash indicates the green phase is about to end — the ego decides whether to proceed or stop.",
                "Key visual cue: the green light is visibly blinking/flashing as the ego approaches or enters the intersection.",
            ],
        }),
        ("TrafficLight_LeftTurnGreenFlash", {
            "cn": "左转绿闪通行",
            "def": [
                "Ego turns LEFT through an intersection while the left-turn signal is in GREEN FLASHING state (绿灯闪烁即将结束).",
                "The green light for left turn is visibly blinking, indicating the phase is about to end.",
                "The ego must decide to complete the left turn before the signal changes to yellow/red.",
                "Key visual cue: left-turn arrow or signal is flashing green; ego is executing or about to execute a left turn.",
            ],
        }),
        ("TrafficLight_RightTurnGreenFlash", {
            "cn": "右转绿闪通行",
            "def": [
                "Ego turns RIGHT through an intersection while the right-turn signal is in GREEN FLASHING state.",
                "The green light for right turn is visibly blinking, indicating the phase is about to end.",
                "Key visual cue: right-turn arrow signal is flashing green; ego is executing a right turn at the intersection.",
            ],
        }),
        ("TrafficLight_StraightYellowFlash", {
            "cn": "直行长黄闪通行",
            "def": [
                "Ego passes straight through an intersection where the signal is in persistent YELLOW FLASHING mode (长黄闪).",
                "This is NOT a brief yellow between green and red — it is a continuously flashing yellow indicating caution (common at low-traffic intersections or during off-peak hours).",
                "Key visual cue: the yellow light flashes repeatedly throughout the ego's approach and passage.",
            ],
        }),
        ("TrafficLight_LeftTurnYellowFlash", {
            "cn": "左转长黄闪通行",
            "def": [
                "Ego turns LEFT at an intersection where the signal is in persistent YELLOW FLASHING mode (长黄闪) — the yellow light continuously flashes as a caution signal.",
                "This is NOT a normal yellow phase between green and red — it is an abnormal/off-peak operating mode where the signal flashes yellow indefinitely.",
                "Key visual cue: yellow light flashes repeatedly throughout the approach; ego turns left through the intersection.",
            ],
        }),
        ("TrafficLight_RightTurnYellowFlash", {
            "cn": "右转长黄闪通行",
            "def": [
                "Ego turns RIGHT at an intersection where the signal is in persistent YELLOW FLASHING mode (长黄闪).",
                "The yellow light continuously flashes as a caution signal, not a normal phase transition.",
                "Key visual cue: yellow light flashes repeatedly; ego turns right through the intersection under caution.",
            ],
        }),
        ("TrafficLight_StraightDarkLight", {
            "cn": "直行黑灯通行",
            "def": [
                "Ego passes straight through an intersection where the traffic light is completely OFF / dark (黑灯).",
                "The signal is not functioning — all lights are unlit.",
                "Key visual cue: traffic light housing is visible but no light is illuminated.",
            ],
        }),
        ("TrafficLight_LeftTurnDarkLight", {
            "cn": "左转黑灯通行",
            "def": [
                "Ego turns LEFT at an intersection where the traffic light is completely OFF / dark (黑灯) — no lights are illuminated.",
                "The signal is malfunctioning or powered off; all lights are unlit.",
                "Key visual cue: traffic light housing visible but completely dark; ego turns left without signal guidance.",
            ],
        }),
        ("TrafficLight_RightTurnDarkLight", {
            "cn": "右转黑灯通行",
            "def": [
                "Ego turns RIGHT at an intersection where the traffic light is completely OFF / dark (黑灯).",
                "The signal is malfunctioning or powered off; all lights are unlit.",
                "Key visual cue: traffic light housing visible but completely dark; ego turns right without signal guidance.",
            ],
        }),
        ("TrafficLight_MobileSignal", {
            "cn": "移动红绿灯通行",
            "def": [
                "Ego passes through a location controlled by a MOBILE / temporary traffic signal (移动红绿灯).",
                "These are portable signals used at temporary construction zones or special events.",
                "Key visual cue: the traffic light is mounted on a portable stand or vehicle, not on a permanent pole.",
            ],
        }),
        ("TrafficLight_WarningLight", {
            "cn": "警示灯通行",
            "def": [
                "Ego passes through a location controlled by a WARNING/CAUTION light (警示灯) — a flashing beacon that warns of hazards but does not command stop/go.",
                "This is NOT a standard traffic signal — it is a single-light flashing beacon, often mounted on barriers, poles, or vehicles.",
                "Key visual cue: a single flashing yellow or red beacon light, distinct from a standard three-color traffic signal; often near construction zones, school zones, or temporary hazards.",
            ],
        }),
        ("TrafficLight_OccludedSignal", {
            "cn": "遮挡红绿灯通行",
            "def": [
                "Ego passes through an intersection where the traffic light is partially or fully OCCLUDED (遮挡) by trees, signs, other structures, or vehicles.",
                "The ego cannot clearly see the signal state and must proceed cautiously.",
                "Key visual cue: the traffic light is visibly blocked or obscured in the video.",
            ],
        }),
    ]),
    "disambiguation": [
        ("StraightStopOrGo vs LeftTurnStopOrGo vs UTurnStopOrGo",
         "Focus on the ego's ACTUAL trajectory AFTER the intersection. Straight: heading unchanged. LeftTurn: heading rotates ~90° left. UTurn: heading rotates ~180°. Do NOT infer from lane position or signal alone — require visible turning motion."),
        ("StopOrGo vs GreenFlash vs YellowFlash vs DarkLight",
         "StopOrGo: normal signal operation (steady red/green/yellow). GreenFlash: green light is BLINKING (end of green phase). YellowFlash: persistent yellow FLASHING (caution mode, not a phase transition). DarkLight: signal is completely OFF."),
        ("GreenFlash vs YellowFlash",
         "GreenFlash: the GREEN light blinks briefly before turning yellow/red — a normal phase-end warning. YellowFlash: the YELLOW light flashes continuously — an abnormal/caution operating mode, often lasting minutes or hours."),
    ],
}

# ============================================
# 3. StartStop (发车/停车) — 7 labels
# ============================================
CATEGORIES["03_StartStop"] = {
    "title": "Start and Stop (发车/停车)",
    "description": "Ego vehicle starting from a stationary position or coming to a stop on structured roads. Includes departures, roadside parking, station stops, emergency stops, and following-induced stops.",
    "labels": OrderedDict([
        ("StartStop_StartFromMainRoad", {
            "cn": "主路发车",
            "def": [
                "Ego vehicle accelerates from a STATIONARY position on a main road.",
                "This is the initial departure — the ego was stopped (e.g., after a traffic jam, after waiting) and now begins moving.",
                "Key visual cue: ego is initially stationary, then starts moving forward on a main road with lane markings.",
                "Distinct from TrafficLight_StopOrGo: no traffic light is the trigger; the departure is self-initiated or follows traffic flow resumption.",
            ],
        }),
        ("StartStop_StartFromNonMotorLane", {
            "cn": "非机动车道发车",
            "def": [
                "Ego vehicle starts from a stationary position on a NON-MOTOR VEHICLE LANE (非机动车道), road shoulder, or bike lane, then merges into the main traffic flow.",
                "The ego was stopped outside the main travel lanes — on a bike lane, service road, or shoulder area.",
                "Key visual cue: ego is initially positioned on a narrower lane or shoulder (often with bike lane markings); it then steers into the main road and accelerates to merge with traffic.",
                "Distinct from StartFromMainRoad: the starting position is NOT on the main road but on an auxiliary/shoulder lane.",
            ],
        }),
        ("StartStop_EmergencyStopOnMainRoad", {
            "cn": "主路紧急停车",
            "def": [
                "Ego vehicle performs an EMERGENCY STOP on the main road — a sudden, unplanned stop due to an unexpected hazard.",
                "Distinct from normal stops: the braking is hard and the stop is unplanned.",
                "Key visual cue: ego decelerates rapidly and stops in the travel lane (not at a designated stop point).",
            ],
        }),
        ("StartStop_StopAtStation", {
            "cn": "站点停车",
            "def": [
                "Ego vehicle stops at a designated STATION or stop point — such as a bus stop, shuttle stop, or designated pickup/dropoff location.",
                "The stop is at a MARKED/DESIGNATED location with infrastructure indicating it is a station.",
                "Key visual cue: bus stop shelter, station signage, platform markings, or designated bay area is visible near where the ego stops.",
                "Distinct from ParkRoadside: station stops have visible infrastructure (shelter, sign, platform); roadside stops do not.",
            ],
        }),
        ("StartStop_ParkInStructuredSpot", {
            "cn": "路侧结构化车位泊入",
            "def": [
                "Ego vehicle parks into a STRUCTURED PARKING SPOT on the roadside — a marked, designated parking space.",
                "Includes parallel parking, angled parking, or perpendicular parking into a marked space.",
                "Key visual cue: painted parking space lines are visible; ego maneuvers into the space.",
            ],
        }),
        ("StartStop_ParkRoadside", {
            "cn": "靠边停车",
            "def": [
                "Ego vehicle pulls over and stops at the ROADSIDE — a general stop at the road edge without a designated parking space.",
                "Includes temporary stops for drop-off, pickup, or brief waiting.",
                "Key visual cue: ego moves toward the road edge and stops; no structured parking space markings.",
                "Distinct from ParkInStructuredSpot: no marked parking space; just pulling over to the curb.",
            ],
        }),
        ("StartStop_FollowingStop", {
            "cn": "跟车停车",
            "def": [
                "Ego vehicle decelerates and stops because the LEAD VEHICLE ahead stops — NOT triggered by a traffic light or parking intent.",
                "The stop is reactive: the ego follows the lead vehicle's deceleration to a stop.",
                "Key visual cue: a lead vehicle is visible decelerating and stopping; ego follows suit; no traffic light is the cause.",
                "Distinct from TrafficLight stops: no signal involved. Distinct from EmergencyStop: the deceleration is gradual and expected.",
            ],
        }),
    ]),
    "disambiguation": [
        ("StartFromMainRoad vs TrafficLight_*StopOrGo",
         "If the start is triggered by a traffic light turning green, use TrafficLight_*StopOrGo. StartFromMainRoad is for non-signal-triggered departures."),
        ("ParkRoadside vs ParkInStructuredSpot",
         "StructuredSpot: there are visible PAINTED LINES for the parking space. Roadside: no painted space; ego just pulls to the curb."),
        ("FollowingStop vs EmergencyStopOnMainRoad vs LeadVehicleSuddenBrake (DynamicInteraction)",
         "FollowingStop: gradual, expected stop behind a decelerating lead. EmergencyStop: sudden, hard braking for unexpected hazard. SuddenBrake (DI category): specifically about the LEAD vehicle's sudden braking event, not the ego's stopping behavior."),
    ],
}

# ============================================
# 4. Intersection (路口通行) — 23 labels
# ============================================
CATEGORIES["04_Intersection"] = {
    "title": "Intersection Passage (路口通行)",
    "description": "Ego vehicle navigating through intersections — turning, going straight, making U-turns, using waiting zones, and traversing roundabouts. Focuses on the ego's trajectory and right-of-way context.",
    "labels": OrderedDict([
        ("Intersection_ProtectedLeftTurn", {
            "cn": "有保护左转",
            "def": [
                "Ego turns LEFT at an intersection controlled by a traffic signal — the intersection HAS traffic lights (any type: arrow, full signal, etc.).",
                "Protected means the intersection is SIGNALIZED (有红绿灯控制), regardless of whether there is a dedicated left-turn arrow.",
                "Key visual cue: traffic lights are visible at the intersection; ego executes a left turn.",
            ],
        }),
        ("Intersection_UnprotectedLeftTurn", {
            "cn": "无保护左转",
            "def": [
                "Ego turns LEFT at an intersection WITHOUT any traffic signal — an unsignalized intersection.",
                "No traffic lights exist at this intersection; the ego must assess right-of-way independently.",
                "Key visual cue: NO traffic lights visible at the intersection; ego turns left relying on gap judgment.",
            ],
        }),
        ("Intersection_ParallelLeftTurn", {
            "cn": "并行左转",
            "def": [
                "Ego turns LEFT alongside another vehicle that is ALSO turning left in a PARALLEL left-turn lane — both vehicles turn left simultaneously in adjacent lanes.",
                "There are at least TWO left-turn lanes, and the ego and another vehicle occupy them side by side during the turn.",
                "Key visual cue: another vehicle is visible turning left in the lane next to the ego; both vehicles curve left at the same time through the intersection.",
                "Distinct from ProtectedLeftTurn: this label emphasizes the PARALLEL vehicle, not just the signal protection.",
            ],
        }),
        ("Intersection_DedicatedLeftTurnLane", {
            "cn": "左转专用道",
            "def": [
                "Ego uses a physically SEPARATED or clearly DEDICATED left-turn lane (左转专用道) to turn left — the lane is channelized or has physical separation from through-traffic lanes.",
                "The dedicated lane guides the ego through the left turn with clear lane markings or physical barriers.",
                "Key visual cue: painted islands, curbs, or channelizing lines separate the left-turn lane from straight-through lanes; the ego follows this guided path.",
            ],
        }),
        ("Intersection_ProtectedStraight", {
            "cn": "有保护路口直行",
            "def": [
                "Ego goes STRAIGHT through an intersection that is controlled by traffic signals (有红绿灯控制).",
                "The intersection has working traffic lights; the ego proceeds straight without turning.",
                "Key visual cue: traffic lights visible at the intersection; ego's heading remains the same before and after passing through; no steering input.",
            ],
        }),
        ("Intersection_UnprotectedStraight", {
            "cn": "无保护路口直行",
            "def": [
                "Ego goes STRAIGHT through an intersection WITHOUT any traffic signal — an unsignalized intersection.",
                "Ego must assess right-of-way independently.",
                "Key visual cue: NO traffic lights at the intersection; ego proceeds straight with caution.",
            ],
        }),
        ("Intersection_MisalignedStraight", {
            "cn": "错位非对齐路口直行",
            "def": [
                "Ego goes straight through a MISALIGNED/OFFSET intersection — the entry and exit lanes are not directly aligned.",
                "The ego must adjust laterally while passing through.",
                "Key visual cue: the road segments on opposite sides of the intersection are offset/staggered.",
            ],
        }),
        ("Intersection_CongestedStraight", {
            "cn": "拥堵路口直行",
            "def": [
                "Ego goes straight through an intersection that is CONGESTED — traffic is backed up within or beyond the intersection.",
                "The ego moves in stop-and-go fashion through the intersection due to heavy traffic ahead.",
                "Key visual cue: vehicles are queued within the intersection or just beyond it; the ego creeps forward slowly; dense traffic visible in all directions.",
            ],
        }),
        ("Intersection_ProtectedRightTurn", {
            "cn": "有保护右转",
            "def": [
                "Ego turns RIGHT at an intersection controlled by traffic signals — specifically with a dedicated RIGHT-TURN signal (右转专用信号灯).",
                "Protected right turn requires a right-turn arrow signal controlling the right-turn phase.",
                "Key visual cue: a right-turn arrow signal is visible; ego turns right under signal control.",
            ],
        }),
        ("Intersection_DedicatedRightTurnLane", {
            "cn": "右转专用道",
            "def": [
                "Ego uses a physically SEPARATED or clearly DEDICATED right-turn lane (右转专用道) to turn right — the lane is channelized with physical separation from straight-through lanes.",
                "The ego follows a guided right-turn path that is separated by painted islands, curbs, or channelizing lines.",
                "Key visual cue: the right-turn lane is physically separated from the main intersection; ego follows this dedicated path to turn right, often bypassing the main signal.",
            ],
        }),
        ("Intersection_ParallelRightTurn", {
            "cn": "并行右转",
            "def": [
                "Ego turns RIGHT alongside another vehicle that is ALSO turning right in a PARALLEL right-turn lane — both vehicles turn right simultaneously in adjacent lanes.",
                "There are at least TWO right-turn lanes, and the ego and another vehicle occupy them side by side.",
                "Key visual cue: another vehicle is visible turning right in the lane next to the ego; both vehicles curve right at the same time.",
            ],
        }),
        ("Intersection_RightTurnWithNonMotorLane", {
            "cn": "右转右侧直行非机动车道",
            "def": [
                "Ego turns RIGHT while a non-motor-vehicle lane (bike lane) runs straight along the right side.",
                "Ego must watch for VRUs going straight in the adjacent bike lane during the right turn.",
                "Key visual cue: a painted bike lane is visible on the right side as the ego executes the right turn.",
            ],
        }),
        ("Intersection_StandardUTurn", {
            "cn": "普通U-turn",
            "def": [
                "Ego makes a standard U-TURN at an intersection, reversing direction (~180° heading change).",
                "Key visual cue: ego enters the intersection and exits heading the OPPOSITE direction.",
                "MUTUALLY EXCLUSIVE with LeftTurnStopOrGo — a U-turn must ONLY be labeled as UTurn.",
            ],
        }),
        ("Intersection_WaitingZoneUTurn", {
            "cn": "待转区U-turn",
            "def": [
                "Ego makes a U-turn using a designated WAITING ZONE (待转区) at the intersection.",
                "The ego first advances into the waiting zone, then completes the U-turn when signaled.",
                "Key visual cue: ground markings for a waiting zone; ego stops within it before completing the U-turn.",
            ],
        }),
        ("Intersection_ThreePointUTurn", {
            "cn": "三点U-turn",
            "def": [
                "Ego performs a THREE-POINT TURN (三点调头) to reverse direction — forward, reverse, forward.",
                "Used when the road is too narrow for a single-sweep U-turn.",
                "Key visual cue: ego drives forward, stops, reverses, stops, then drives forward in the new direction.",
            ],
        }),
        ("Intersection_StraightWaitingZone", {
            "cn": "直行待转区",
            "def": [
                "Ego enters a STRAIGHT-THROUGH waiting zone (直行待转区) at an intersection — a designated area past the stop line where vehicles wait during a preliminary signal phase.",
                "The ego advances beyond the normal stop line into the intersection's waiting zone, then proceeds when the final green signal activates.",
                "Key visual cue: ground markings (painted lines or text) define a waiting zone area past the stop line; ego stops inside this zone, ahead of the normal stop position.",
            ],
        }),
        ("Intersection_LeftTurnWaitingZone", {
            "cn": "左转待转区",
            "def": [
                "Ego enters a LEFT-TURN waiting zone (左转待转区) at an intersection — the ego advances into the intersection to a designated area and waits for the left-turn signal.",
                "During a preliminary phase (e.g., straight green), left-turn vehicles are allowed to enter the waiting zone inside the intersection.",
                "Key visual cue: painted waiting zone markings visible inside the intersection for left-turning vehicles; ego moves past the stop line into this zone and waits.",
            ],
        }),
        ("Intersection_TextWaitingZone", {
            "cn": "文字型待转区",
            "def": [
                "Ego uses a waiting zone that is marked with CHINESE TEXT characters painted on the road surface (e.g., '左转待转', '直行待转').",
                "The text on the ground explicitly indicates the waiting zone's purpose and direction.",
                "Key visual cue: large Chinese characters clearly visible painted on the road surface within the intersection area.",
            ],
        }),
        ("Intersection_CombinedSignalWaitingZone", {
            "cn": "组合灯控待转区",
            "def": [
                "Ego uses a waiting zone controlled by a COMBINED / multi-phase signal system — multiple signal heads or phases coordinate when vehicles may enter and leave the waiting zone.",
                "The signal setup is more complex than a simple green/red — it has dedicated signals for the waiting zone entry phase.",
                "Key visual cue: multiple traffic signal heads visible, some controlling the waiting zone entry separately from the main intersection signals.",
            ],
        }),
        ("Intersection_ImageWaitingZone", {
            "cn": "图像型待转区",
            "def": [
                "Ego uses a waiting zone marked with IMAGE/GRAPHIC symbols painted on the ground — directional arrows, diagrams, or pictographic indicators (not text).",
                "The ground markings use visual graphics rather than Chinese text to indicate the waiting zone.",
                "Key visual cue: large arrow symbols, directional diagrams, or pictographic markings on the road surface defining the waiting zone area.",
            ],
        }),
        ("Intersection_SingleLaneRoundabout", {
            "cn": "单车道小环岛",
            "def": [
                "Ego navigates through a SINGLE-LANE small roundabout — a compact circular intersection with one circulating lane around a central island.",
                "The ego enters the roundabout, follows the single lane around the circle, and exits at the desired road.",
                "Key visual cue: a small circular island in the center of the intersection; one lane circles around it; the ego follows the curve; the island and circulating lane are clearly visible.",
            ],
        }),
        ("Intersection_MultiLaneRoundabout", {
            "cn": "多车道环岛",
            "def": [
                "Ego navigates through a MULTI-LANE roundabout — a larger circular intersection with two or more circulating lanes.",
                "The ego may need to choose or change lanes within the roundabout to reach the desired exit.",
                "Key visual cue: a larger roundabout with multiple marked lanes circling the central island; other vehicles may be circulating alongside the ego in adjacent lanes.",
            ],
        }),
        ("Intersection_TJunctionUnprotectedMerge", {
            "cn": "T型无信号路口汇入",
            "def": [
                "Ego merges onto a main road from a side road at a T-JUNCTION without any traffic signal.",
                "Ego must yield to main road traffic and find a safe gap to merge.",
                "Key visual cue: a T-shaped intersection with no traffic lights; ego approaches from the stem of the T and turns onto the crossbar.",
            ],
        }),
    ]),
    "disambiguation": [
        ("Protected vs Unprotected (any direction)",
         "Protected (有保护) = the intersection HAS traffic lights controlling traffic. Unprotected (无保护) = the intersection has NO traffic lights at all (unsignalized). The distinction is simply: are there traffic signals at this intersection? YES → Protected, NO → Unprotected."),
        ("StandardUTurn vs LeftTurnStopOrGo (TrafficLight category)",
         "These are MUTUALLY EXCLUSIVE. UTurn: ~180° heading change. LeftTurn: ~90° heading change onto a perpendicular road. If the ego reverses direction, it is ALWAYS a UTurn, even at a traffic light."),
        ("WaitingZone types (Text vs Image vs Combined)",
         "Text: Chinese characters painted on road. Image: graphic arrows/symbols painted. Combined: controlled by multiple signal phases. Check the road surface markings."),
        ("TJunctionUnprotectedMerge vs UnprotectedStraight",
         "TJunction: ego is on the MINOR road merging onto the major road at a T-intersection. UnprotectedStraight: ego is on a through-road passing an unsignalized intersection. The distinction is the ego's position relative to the intersection geometry."),
    ],
}

# ============================================
# 5. LaneCruising (车道巡航) — 19 labels
# ============================================
CATEGORIES["05_LaneCruising"] = {
    "title": "Lane Cruising (车道巡航)",
    "description": "Ego vehicle maintaining its lane without notable lane-change, stop, or dynamic-interaction events. Includes normal cruising, following, congestion, speed-limited, and semantically-special-lane scenarios.",
    "labels": OrderedDict([
        ("LaneCruising_Straight", {
            "cn": "普通车道巡航",
            "def": [
                "Ego vehicle drives along its lane at steady speed with NO notable events — no lane changes, no significant interactions, no obstacles.",
                "This is the DEFAULT cruising state on a straight or gently curved road with free flow.",
                "Key visual cue: open road, consistent speed, no hazards or interactions.",
            ],
        }),
        ("LaneCruising_SharpCurve", {
            "cn": "大曲率弯道巡航",
            "def": [
                "Ego drives through a road segment that curves sharply — the vehicle is in the PROCESS OF TURNING along a curved road (not at an intersection).",
                "The ego's heading changes noticeably during the maneuver as it follows the road's curvature. This is a road-geometry turn, not an intersection turn.",
                "Key visual cue: the road ahead bends significantly (left or right); the scene rotates as the ego follows the curve; lane markings curve; the steering wheel is turned to follow the road.",
                "Distinct from Intersection turns: SharpCurve occurs on a continuous road segment (no intersection), the road itself is curved. Intersection turns occur at a junction between two roads.",
            ],
        }),
        ("LaneCruising_NarrowSpace", {
            "cn": "窄空间巡航",
            "def": [
                "Ego drives through a NARROW space where lateral clearance is very limited.",
                "Common scenarios: parked vehicles on BOTH sides squeezing the available width, a narrow road/alley, or vehicles in adjacent lanes closing in.",
                "Key visual cue: vehicles, walls, or obstacles visible close to BOTH left and right sides of the ego; the ego must drive carefully through the tight gap.",
            ],
        }),
        ("LaneCruising_RuralRoad", {
            "cn": "乡村道路巡航",
            "def": [
                "Ego cruises on a RURAL ROAD with distinct countryside characteristics.",
                "Key visual cue: farmland/fields alongside the road, village houses, unpaved or poorly-maintained road surface, no sidewalks or curbs, sparse or no lane markings, trees/vegetation encroaching the roadside, no urban buildings or infrastructure.",
                "The overall scene must clearly look like a rural/village environment, not a suburban or urban road.",
            ],
        }),
        ("LaneCruising_SpeedBump", {
            "cn": "减速带巡航",
            "def": [
                "Ego drives over one or more RAISED SPEED BUMPS (凸起减速带) on the road surface, decelerating to cross safely.",
                "Speed bumps are physical raised ridges across the road, causing the vehicle to bounce or jolt.",
                "Key visual cue: yellow/black striped raised bumps visible on the road; ego noticeably slows and the vehicle body pitches when crossing.",
            ],
        }),
        ("LaneCruising_ConstructionZone", {
            "cn": "施工区巡航",
            "def": [
                "Ego drives through a CONSTRUCTION ZONE — road work is active or indicated by barriers.",
                "Key visual cue: water-filled barriers (水马), traffic cones (锥桶), construction fencing, temporary guardrails, or other road-blocking obstacles that partially obstruct the road.",
                "Construction workers, machinery, or temporary lane markings may also be visible.",
            ],
        }),
        ("LaneCruising_ZebraCrossing", {
            "cn": "斑马线巡航",
            "def": [
                "Ego passes through a ZEBRA CROSSING (斑马线) area without stopping — no pedestrians are actively crossing.",
                "Key visual cue: white zebra stripes on the road; ego passes through without significant deceleration.",
                "If a VRU is actively crossing, use DynamicInteraction_VRUInLaneCrossing instead.",
            ],
        }),
        ("LaneCruising_CongestedFollowing", {
            "cn": "跟车拥堵巡航",
            "def": [
                "Ego follows the vehicle ahead in CONGESTED / stop-and-go traffic.",
                "The ego repeatedly accelerates and decelerates at low speed behind a lead vehicle in traffic jam conditions.",
                "Key visual cue: dense traffic, low speed, frequent speed changes, small following distance.",
            ],
        }),
        ("LaneCruising_StaticVehicleQueueCongestion", {
            "cn": "联排静止车拥堵巡航",
            "def": [
                "Ego drives past or alongside a QUEUE of stationary vehicles in the adjacent lane(s) — the ego's lane is flowing but neighboring lanes are completely jammed.",
                "A long row of stopped/barely-moving vehicles is visible to the side, while the ego continues at a higher speed in its own lane.",
                "Key visual cue: a continuous line of stopped vehicles in the neighboring lane(s); ego passes them; the contrast between ego's motion and the static queue is clear.",
            ],
        }),
        ("LaneCruising_OtherCongestion", {
            "cn": "其他拥堵巡航",
            "def": [
                "Ego drives in congested conditions not specifically covered by CongestedFollowing or StaticVehicleQueue — general slow-moving, dense traffic scenarios.",
                "The ego moves slowly but not in a strict stop-and-go pattern behind a single lead vehicle.",
                "Key visual cue: many vehicles visible in multiple lanes, all moving slowly; overall reduced speed; the congestion is general, not specific to following one vehicle.",
            ],
        }),
        ("LaneCruising_RoadSpeedLimit", {
            "cn": "道路限速巡航",
            "def": [
                "Ego cruises at a speed governed by a posted ROAD SPEED LIMIT sign — the speed limit is a general road regulation, not scene-specific.",
                "A speed limit sign (e.g., 40/60/80 km/h circular sign) is visible on the roadside or overhead.",
                "Key visual cue: circular speed limit sign with a number visible alongside the road; ego maintains a steady speed consistent with the posted limit.",
            ],
        }),
        ("LaneCruising_SceneSpeedLimit", {
            "cn": "场景限速巡航",
            "def": [
                "Ego cruises at reduced speed due to a SCENE-SPECIFIC speed restriction — the speed limit is tied to a particular environment, not just a general road limit.",
                "Examples: school zones (学校区域), hospital zones, residential area entrances, parking lot areas, or other special zones with their own speed limits.",
                "Key visual cue: scene-specific signage (e.g., '学校' + speed limit, residential gate, hospital entrance) is visible; the ego drives slower than normal road speed.",
            ],
        }),
        ("LaneCruising_IntersectionSpeedLimit", {
            "cn": "路口限速巡航",
            "def": [
                "Ego decelerates or cruises at reduced speed specifically because it is APPROACHING an intersection — a natural slowdown near a junction.",
                "The ego has not yet entered the intersection; it is in the approach phase, reducing speed as a precaution.",
                "Key visual cue: an intersection is visible ahead; the ego gradually reduces speed; no red light is the cause — the deceleration is precautionary.",
            ],
        }),
        ("LaneCruising_VariableLane", {
            "cn": "可变车道巡航",
            "def": [
                "Ego drives in a VARIABLE LANE (可变车道) — a lane whose permitted direction (straight/left/right) changes dynamically based on overhead LED signs.",
                "The lane assignment is not fixed; overhead electronic signs indicate the current allowed direction for this lane.",
                "Key visual cue: overhead LED directional arrow signs above the lane (pointing left, right, or straight); the lane may have dashed or special markings different from permanent lanes.",
            ],
        }),
        ("LaneCruising_BusLane", {
            "cn": "公交车道巡航",
            "def": [
                "Ego drives in or alongside a designated BUS LANE (公交车道) — a lane reserved for public transit buses.",
                "The bus lane is marked on the road surface with text or colored markings.",
                "Key visual cue: '公交专用' or 'BUS' text painted on the road surface; the lane may have colored (often red or yellow) surface markings; bus-only signage visible.",
            ],
        }),
        ("LaneCruising_TidalLane", {
            "cn": "潮汐车道巡航",
            "def": [
                "Ego drives in a TIDAL/REVERSIBLE LANE (潮汐车道) — a lane whose traffic direction reverses based on time of day and traffic demand.",
                "During peak hours, the lane may serve one direction; during off-peak, it serves the opposite direction.",
                "Key visual cue: movable barriers (often mechanical/automated), overhead direction indicators that can change, or special road markings indicating the lane is reversible.",
            ],
        }),
        ("LaneCruising_NoParkingZone", {
            "cn": "禁停区巡航",
            "def": [
                "Ego passes through a NO-PARKING / NO-STOPPING ZONE (禁停区) — an area where stopping or parking is prohibited.",
                "The zone is marked on the road surface to prevent vehicles from stopping in this area.",
                "Key visual cue: yellow zigzag lines (锯齿线), yellow cross-hatch markings, or '禁停' text painted on the road surface; often near bus stops, fire stations, or building entrances.",
            ],
        }),
        ("LaneCruising_FollowingVRU", {
            "cn": "跟VRU稳态行驶",
            "def": [
                "Ego follows a VRU (bicycle, e-bike, tricycle, pedestrian) at a STEADY LOW SPEED — unable to overtake, maintaining a consistent gap.",
                "Key visual cue: a VRU is directly ahead in the ego's lane; ego drives at VRU's speed (typically slow).",
                "Distinct from VRUInLaneCrossing: the VRU is moving in the SAME direction, not crossing.",
            ],
        }),
        ("LaneCruising_SteadyFollowing", {
            "cn": "稳态跟车",
            "def": [
                "Ego follows a lead VEHICLE at a STEADY speed with a consistent gap — normal non-congested car-following.",
                "Distinct from CongestedFollowing: traffic is flowing normally, no stop-and-go; the ego simply has a lead vehicle.",
                "Key visual cue: a vehicle ahead in the same lane; ego maintains consistent following distance at moderate speed.",
            ],
        }),
    ]),
    "disambiguation": [
        ("Straight vs SteadyFollowing",
         "Straight: NO lead vehicle or the lead is far enough to be irrelevant. SteadyFollowing: a lead vehicle is clearly present and constrains the ego's speed. If the ego is matching a lead vehicle's speed with a visible following gap, use SteadyFollowing."),
        ("SteadyFollowing vs CongestedFollowing",
         "SteadyFollowing: smooth, steady speed, no frequent braking. CongestedFollowing: stop-and-go, frequent speed changes, dense traffic. Check traffic density and speed variability."),
        ("FollowingVRU vs VRUInLaneCrossing (DynamicInteraction)",
         "FollowingVRU: VRU is in the SAME lane moving in the SAME direction; ego follows behind. VRUInLaneCrossing: VRU moves PERPENDICULAR to the ego, crossing the lane."),
        ("ZebraCrossing vs VRUInLaneCrossing (DynamicInteraction)",
         "ZebraCrossing: ego passes a zebra crossing WITHOUT stopping for pedestrians. If a VRU is actively crossing and the ego stops/yields, use VRUInLaneCrossing instead."),
    ],
}

# ============================================
# 6. LaneChange (变道/绕行) — 19 labels
# ============================================
CATEGORIES["06_LaneChange"] = {
    "title": "Lane Change and Bypass (变道/绕行)",
    "description": "Ego vehicle changing lanes or detouring — motivated by navigation, avoidance, efficiency, borrowing the oncoming lane, cross-line bypassing, overtaking, or suppressed lane changes.",
    "labels": OrderedDict([
        ("LaneChange_NavForIntersection", {
            "cn": "前方路口导航变道",
            "def": [
                "Ego changes lanes for NAVIGATION purposes as it approaches an intersection — e.g., getting into the correct turn lane.",
                "Driven by route/navigation requirements, not by obstacle avoidance.",
                "Key visual cue: ego switches lanes before an intersection; a turn or exit is imminent.",
                "NOTE: Requires navigation intent — FW video alone may not confirm this. Use with caution.",
            ],
        }),
        ("LaneChange_ShortConsecutiveNav", {
            "cn": "短距离连续导航变道",
            "def": [
                "Ego performs MULTIPLE lane changes in quick succession for navigation purposes — e.g., crossing several lanes to reach an exit.",
                "Key visual cue: ego crosses 2+ lanes in a short distance before a turn/exit.",
                "NOTE: Requires navigation intent — FW video alone may not confirm this.",
            ],
        }),
        ("LaneChange_CongestedNav", {
            "cn": "拥堵导航变道",
            "def": [
                "Ego changes lanes in CONGESTED traffic for navigation purposes — must merge through slow traffic to reach the correct lane.",
                "Key visual cue: ego changes lanes in dense, slow-moving traffic before a turn/exit.",
                "NOTE: Requires navigation intent — FW video alone may not confirm this.",
            ],
        }),
        ("LaneChange_AvoidSlowVRU", {
            "cn": "慢速VRU避让",
            "def": [
                "Ego performs a LATERAL AVOIDANCE (nudge or lane offset) WITHIN its lane to bypass a slow-moving VRU without fully changing lanes.",
                "The ego does not enter an adjacent lane — it adjusts position within its current lane.",
                "Key visual cue: a slow VRU in or near the ego's path; ego shifts laterally but stays in its lane.",
            ],
        }),
        ("LaneChange_AvoidStaticVehicle", {
            "cn": "静态障碍车避让",
            "def": [
                "Ego performs a LATERAL AVOIDANCE or lane change to bypass a STATIONARY VEHICLE (parked car, broken-down car, delivery truck) blocking or partially blocking the lane.",
                "Key visual cue: a stopped motor vehicle in the ego's lane; ego steers around it or changes lanes.",
            ],
        }),
        ("LaneChange_AvoidStaticObstacle", {
            "cn": "静态障碍物避让",
            "def": [
                "Ego avoids a STATIONARY NON-VEHICLE OBSTACLE in the lane — e.g., fallen cargo, construction barriers, large debris, trash bins.",
                "Key visual cue: a non-vehicle obstacle blocks part of the lane; ego adjusts path.",
                "Distinct from AvoidStaticVehicle: the obstacle is NOT a motor vehicle.",
            ],
        }),
        ("LaneChange_BorrowLaneAvoidSlowVRU", {
            "cn": "慢速VRU避让（借道）",
            "def": [
                "Ego enters the ONCOMING lane (借道) to bypass a slow-moving VRU, then returns to its original lane.",
                "Distinct from in-lane avoidance: the ego actually crosses the center line into the opposing lane.",
                "Key visual cue: ego crosses the center/dividing line to go around a VRU, then returns.",
            ],
        }),
        ("LaneChange_BorrowLaneAvoidStaticVehicle", {
            "cn": "静态障碍车避让（借道）",
            "def": [
                "Ego enters the ONCOMING lane (借道) to bypass a stationary motor vehicle blocking the lane, then returns to its original lane.",
                "The ego crosses the center/dividing line into the opposite-direction lane to go around a parked or broken-down vehicle.",
                "Key visual cue: ego crosses the center line, drives briefly in the oncoming lane past a stopped vehicle, then returns to its own lane.",
            ],
        }),
        ("LaneChange_BorrowLaneAvoidStaticObstacle", {
            "cn": "静态障碍物避让（借道）",
            "def": [
                "Ego enters the ONCOMING lane (借道) to bypass a stationary NON-VEHICLE obstacle (construction barriers, fallen debris, large objects), then returns.",
                "The obstacle is not a motor vehicle — it is a physical object blocking the lane.",
                "Key visual cue: ego crosses the center line to go around a non-vehicle obstacle on the road, then returns to its own lane.",
            ],
        }),
        ("LaneChange_BorrowOncomingLaneAvoidVehicle", {
            "cn": "对向车道障碍车借道避让",
            "def": [
                "Ego borrows the ONCOMING lane specifically to avoid an obstacle vehicle while navigating potential oncoming traffic.",
                "Emphasis on the interaction with ONCOMING vehicles during the borrow-lane maneuver.",
                "Key visual cue: ego is in the oncoming lane and must coordinate with approaching vehicles.",
            ],
        }),
        ("LaneChange_CrossLineBypassStaticVehicles", {
            "cn": "联排静止车跨线绕行",
            "def": [
                "Ego crosses lane markings to bypass a ROW/QUEUE of stationary vehicles (联排) — multiple parked or queued vehicles blocking the lane in a line.",
                "The ego must cross lane markings (not just nudge within the lane) to go around the entire queue.",
                "Key visual cue: a continuous line of stopped/parked cars occupies the ego's lane; ego crosses lane markings and drives around the entire queue before returning.",
                "Distinct from single-vehicle avoidance: this involves MULTIPLE vehicles in a row.",
            ],
        }),
        ("LaneChange_CrossLineBypassStaticObstacles", {
            "cn": "联排静态障碍物跨线绕行",
            "def": [
                "Ego crosses lane markings to bypass a ROW of stationary non-vehicle obstacles — such as a line of construction barriers, traffic cones, water-filled barriers, or other sequential obstacles.",
                "Multiple obstacles are arranged in a line, blocking the lane for an extended stretch.",
                "Key visual cue: a series of barriers/cones/obstacles arranged along the lane; ego crosses lane markings to bypass the entire row.",
            ],
        }),
        ("LaneChange_SlowVehicleEfficiency", {
            "cn": "慢速车效率变道",
            "def": [
                "Ego changes lanes to OVERTAKE a slow-moving vehicle for EFFICIENCY — the ego is not avoiding a hazard, but seeking faster travel.",
                "Distinct from avoidance: the motivation is speed, not safety.",
                "Key visual cue: a slower vehicle is ahead; ego changes lanes to pass; no immediate hazard.",
            ],
        }),
        ("LaneChange_SlowVRUEfficiency", {
            "cn": "慢速VRU效率变道",
            "def": [
                "Ego changes lanes to pass a slow-moving VRU (bicycle, e-bike, tricycle) for EFFICIENCY — not an emergency avoidance, but a proactive lane change for better traffic flow.",
                "The VRU is not an immediate hazard, but following it would significantly reduce the ego's speed.",
                "Key visual cue: a slow VRU ahead in the ego's lane; ego changes to an adjacent lane to maintain speed; no urgency or hard braking involved.",
                "Distinct from AvoidSlowVRU: efficiency change is to the ADJACENT lane; avoidance may be an in-lane nudge.",
            ],
        }),
        ("LaneChange_StaticObstacleEfficiency", {
            "cn": "静止障碍物效率变道",
            "def": [
                "Ego changes lanes proactively to avoid a static obstacle visible ahead — the obstacle is not an imminent threat but would require slowing down if the ego stayed in the lane.",
                "The lane change is a PROACTIVE efficiency decision, not a last-second emergency maneuver.",
                "Key visual cue: a static obstacle (not a vehicle) visible ahead in the lane; ego changes lanes early to maintain speed; the maneuver is smooth and planned.",
            ],
        }),
        ("LaneChange_CongestedQueueSuppressed", {
            "cn": "拥堵排队抑制变道",
            "def": [
                "Ego SUPPRESSES / does NOT change lanes despite wanting to — queuing in congestion prevents the lane change.",
                "This is a NON-ACTION label: the ego stays in its lane because changing is not possible.",
                "NOTE: Requires planning/decision information — FW video alone cannot confirm intent to change lanes.",
            ],
        }),
        ("LaneChange_NonMotorLaneSuppressed", {
            "cn": "非机动车道抑制变道",
            "def": [
                "Ego SUPPRESSES / does NOT change lanes because the adjacent lane is a NON-MOTOR-VEHICLE lane (非机动车道) — changing into it would be illegal.",
                "The ego stays in its lane despite potentially wanting to change, because the target lane is reserved for bicycles/e-bikes.",
                "Key visual cue: a bike lane or non-motor lane is visible adjacent to the ego; the ego remains in its lane.",
                "NOTE: Requires planning/decision information — FW video alone cannot confirm the intent to change lanes.",
            ],
        }),
        ("LaneChange_BusStopSuppressed", {
            "cn": "公交停泊港抑制变道",
            "def": [
                "Ego SUPPRESSES / does NOT change lanes near a BUS STOP BAY (公交停泊港) — changing into the bus bay area would be inappropriate.",
                "A bus stop bay or pull-in area is visible, and the ego avoids entering it.",
                "Key visual cue: a bus stop bay indentation in the road is visible; the ego stays in its lane rather than merging into the bus bay area.",
                "NOTE: Requires planning/decision information.",
            ],
        }),
        ("LaneChange_Overtake", {
            "cn": "超车",
            "def": [
                "Ego performs a complete OVERTAKE maneuver: changes to an adjacent lane → passes the slower vehicle → returns to the original lane.",
                "This is a COMPOUND action involving at least two lane changes and a passing phase.",
                "Key visual cue: ego moves to the left/right lane, accelerates past a vehicle, then moves back to the original lane.",
                "Distinct from efficiency lane change: overtake includes the RETURN to the original lane.",
            ],
        }),
    ]),
    "disambiguation": [
        ("AvoidSlowVRU (in-lane) vs BorrowLaneAvoidSlowVRU",
         "In-lane: ego stays within its lane, just shifts laterally. BorrowLane: ego crosses the center line into the oncoming lane. Check whether the ego crosses the center/dividing line."),
        ("AvoidStaticVehicle vs AvoidStaticObstacle",
         "Vehicle: the obstacle is a motor vehicle (car, truck, bus). Obstacle: the obstacle is NOT a motor vehicle (barriers, debris, cones). Identify the type of obstacle."),
        ("Efficiency lane change vs Overtake",
         "Efficiency: a single lane change to get past a slow object; the ego may stay in the new lane. Overtake: the ego changes lanes, passes, AND returns to the original lane — a complete three-phase maneuver."),
        ("Suppressed labels (CongestedQueue/NonMotor/BusStop)",
         "These are NON-ACTION labels requiring planning intent data. They cannot be reliably labeled from FW video alone. Use with caution and supplementary data sources."),
    ],
}

# ============================================
# 7. IntersectionInteraction (路口内动态交互) — 15 labels
# ============================================
CATEGORIES["07_IntersectionInteraction"] = {
    "title": "Intersection Dynamic Interaction (路口内动态交互)",
    "description": "Interactions between the ego and other road users WITHIN an intersection — crossings, cut-ins, and close approaches during turning or straight-through maneuvers at intersections.",
    "labels": OrderedDict([
        ("IntersectionInteraction_EgoStraight_VRUCrossing", {
            "cn": "路口自车直行VRU横穿",
            "def": [
                "While the ego goes STRAIGHT through an intersection, a VRU (pedestrian, cyclist, e-bike) CROSSES the ego's path perpendicularly.",
                "The ego must slow, stop, or yield to the crossing VRU within the intersection.",
                "Key visual cue: ego is going straight through the intersection; a VRU is walking/cycling across the ego's forward path (left-to-right or right-to-left).",
            ],
        }),
        ("IntersectionInteraction_EgoStraight_VehicleLeftTurnCrossing", {
            "cn": "路口自车直行他车左转横穿",
            "def": [
                "While the ego goes STRAIGHT through an intersection, a LEFT-TURNING vehicle from the CROSS STREET cuts across the ego's path.",
                "The conflicting vehicle is turning left from the perpendicular road and crosses in front of the ego.",
                "Key visual cue: ego is going straight; a vehicle from the side road turns left and crosses the ego's forward path; the ego brakes or yields.",
            ],
        }),
        ("IntersectionInteraction_EgoLeftTurn_VRUCrossing", {
            "cn": "路口自车左转VRU横穿",
            "def": [
                "While the ego is turning LEFT at an intersection, a VRU crosses the ego's turning path — typically on the crosswalk of the target road.",
                "The ego must yield to the VRU while executing or completing the left turn.",
                "Key visual cue: ego is mid-left-turn; a pedestrian/cyclist is crossing the road that the ego is turning INTO; the ego slows or stops to yield.",
            ],
        }),
        ("IntersectionInteraction_EgoLeftTurn_VehicleStraightCrossing", {
            "cn": "路口自车左转他车直行横穿",
            "def": [
                "While the ego turns LEFT, ONCOMING straight-through vehicles cross the ego's turning path — the classic unprotected-left-turn conflict.",
                "The ego must find a gap in oncoming traffic to complete its left turn.",
                "Key visual cue: ego is turning left; oncoming vehicles are going straight through the intersection, crossing the ego's intended path; the ego waits or proceeds carefully.",
            ],
        }),
        ("IntersectionInteraction_EgoRightTurn_VRUCrossing", {
            "cn": "路口自车右转VRU横穿",
            "def": [
                "While the ego turns RIGHT at an intersection, a VRU crosses the ego's turning path — typically on the crosswalk of the target road.",
                "The ego must yield to the VRU during or after the right turn.",
                "Key visual cue: ego is mid-right-turn; a pedestrian/cyclist is crossing the road the ego is turning INTO; the ego slows or stops.",
            ],
        }),
        ("IntersectionInteraction_EgoRightTurn_VehicleStraightCrossing", {
            "cn": "路口自车右转他车直行横穿",
            "def": [
                "While the ego turns RIGHT, a straight-through vehicle from the cross street crosses the ego's turning path.",
                "The conflicting vehicle is going straight on the perpendicular road while the ego is turning right.",
                "Key visual cue: ego is turning right; a vehicle from the cross street goes straight and crosses the ego's turning path.",
            ],
        }),
        ("IntersectionInteraction_EgoStraight_VRURightTurnCutIn", {
            "cn": "路口自车直行VRU右转切入",
            "def": [
                "While the ego goes STRAIGHT at an intersection, a right-turning VRU CUTS IN from the side — merging into the ego's path in roughly the SAME direction (not crossing perpendicular).",
                "The VRU enters from the roadside or an adjacent lane and merges ahead of the ego.",
                "Key visual cue: a VRU appears from the side of the intersection and enters the ego's lane/path, moving in the same general direction as the ego.",
                "Distinct from VRUCrossing: crossing is PERPENDICULAR; cut-in is roughly PARALLEL / same direction.",
            ],
        }),
        ("IntersectionInteraction_EgoStraight_VehicleRightTurnCutIn", {
            "cn": "路口自车直行他车右转切入",
            "def": [
                "While the ego goes STRAIGHT at an intersection, a right-turning motor vehicle CUTS IN from a side lane — merging into the ego's path in the same direction.",
                "The cutting vehicle enters from a side road, turn lane, or adjacent lane and merges ahead of the ego.",
                "Key visual cue: a vehicle appears from the side within the intersection and merges into the ego's forward path, reducing the ego's following distance.",
            ],
        }),
        ("IntersectionInteraction_RearSideVRUApproach", {
            "cn": "路口侧后VRU贴近",
            "def": [
                "A VRU approaches the ego from the REAR or SIDE at an intersection — the VRU is behind or beside the ego, creating a blind-spot hazard.",
                "The ego may not see the VRU in the front-wide camera; this is a proximity warning scenario.",
                "Key visual cue: a VRU may be barely visible at the extreme edge of the frame, or inferred from the traffic situation.",
                "NOTE: Requires side or rear camera for reliable detection — FW alone has limited visibility for rear/side approaches.",
            ],
        }),
        ("IntersectionInteraction_ParallelVRUApproach", {
            "cn": "路口并行VRU贴近",
            "def": [
                "A VRU moves PARALLEL to and alongside the ego vehicle at an intersection — both the VRU and ego are moving in the same general direction, close together.",
                "The VRU is in an adjacent lane, on the sidewalk edge, or in a bike lane, very close to the ego.",
                "Key visual cue: a VRU (cyclist, e-bike rider, pedestrian) is visible at the side of the frame, moving alongside the ego through the intersection at close range.",
            ],
        }),
        ("IntersectionInteraction_OncomingVRUApproach", {
            "cn": "路口逆行VRU贴近",
            "def": [
                "A VRU approaches the ego from the OPPOSITE direction at an intersection — head-on or near-head-on approach within the intersection.",
                "The VRU is coming toward the ego, creating a proximity concern during the ego's intersection maneuver.",
                "Key visual cue: a VRU is visible ahead, moving TOWARD the ego within the intersection; the gap between them is small.",
            ],
        }),
        ("IntersectionInteraction_RearSideVehicleApproach", {
            "cn": "路口侧后车辆贴近",
            "def": [
                "A motor vehicle approaches the ego from the REAR or SIDE at an intersection — the vehicle is in a blind spot or approaching from behind.",
                "This creates a collision risk, especially during turning maneuvers.",
                "NOTE: Requires side or rear camera for reliable detection — FW alone has limited visibility.",
            ],
        }),
        ("IntersectionInteraction_ParallelStraightVehicleApproach", {
            "cn": "路口并行直行车贴近",
            "def": [
                "A vehicle going STRAIGHT moves PARALLEL to and alongside the ego at an intersection — both vehicles are navigating the intersection side by side.",
                "The parallel vehicle is in an adjacent lane, going straight, while the ego is also traversing the intersection.",
                "Key visual cue: a vehicle in the adjacent lane is visible alongside the ego, both moving through the intersection; lateral clearance is small.",
            ],
        }),
        ("IntersectionInteraction_ParallelLeftTurnVehicleApproach", {
            "cn": "路口并行左转车贴近",
            "def": [
                "A LEFT-TURNING vehicle moves PARALLEL to and alongside the ego at an intersection — the other vehicle is turning left in an adjacent left-turn lane.",
                "Both the ego and the parallel vehicle are navigating the intersection simultaneously, creating close proximity.",
                "Key visual cue: a vehicle in the adjacent lane is visibly turning left alongside the ego; both vehicles curve through the intersection together.",
            ],
        }),
        ("IntersectionInteraction_ParallelRightTurnVehicleApproach", {
            "cn": "路口并行右转车贴近",
            "def": [
                "A RIGHT-TURNING vehicle moves PARALLEL to and alongside the ego at an intersection — the other vehicle is turning right in an adjacent lane.",
                "Both the ego and the parallel vehicle are navigating the intersection simultaneously at close range.",
                "Key visual cue: a vehicle in the adjacent lane is visibly turning right alongside the ego through the intersection.",
            ],
        }),
    ]),
    "disambiguation": [
        ("Crossing vs CutIn",
         "Crossing: the other road user moves PERPENDICULAR to the ego's direction. CutIn: the other road user moves roughly PARALLEL, merging into the ego's path."),
        ("IntersectionInteraction vs DynamicInteraction",
         "IntersectionInteraction: occurs WITHIN an intersection during turning/straight-through. DynamicInteraction: occurs on STRAIGHT road segments between intersections. Check the location — is the ego within an intersection?"),
        ("RearSide vs Parallel vs Oncoming approach",
         "RearSide: other road user approaches from BEHIND or SIDE (limited FW visibility). Parallel: other road user moves alongside in the SAME direction. Oncoming: other road user approaches HEAD-ON from the opposite direction."),
    ],
}

# ============================================
# 8. LaneApproach (车道贴近) — 4 labels
# ============================================
CATEGORIES["08_LaneApproach"] = {
    "title": "Lane Approach (车道贴近)",
    "description": "Another road user in an adjacent or oncoming lane approaches close to the ego vehicle — creating a proximity concern on STRAIGHT road segments (not at intersections).",
    "labels": OrderedDict([
        ("LaneApproach_AdjacentVRU", {
            "cn": "邻车道VRU贴近",
            "def": [
                "A VRU in an ADJACENT lane (same direction) moves close to the ego vehicle — their lateral distance is small enough to warrant caution.",
                "The VRU is NOT cutting in (not entering the ego's lane) but is riding/walking close to the lane boundary.",
                "Key visual cue: a VRU in the neighboring lane is visible near the lane line, close to the ego.",
            ],
        }),
        ("LaneApproach_AdjacentVehicle", {
            "cn": "邻车道车辆贴近",
            "def": [
                "A motor vehicle in an ADJACENT lane (same direction) drives close to the ego — small lateral clearance.",
                "The vehicle is NOT cutting in but is close to the lane boundary.",
                "Key visual cue: a vehicle in the neighboring lane drives uncomfortably close to the ego.",
            ],
        }),
        ("LaneApproach_OncomingVRU", {
            "cn": "逆向车道VRU贴近",
            "def": [
                "A VRU in the ONCOMING direction approaches close to the ego on a straight road — e.g., a cyclist riding against traffic, a pedestrian walking on the road toward the ego.",
                "The VRU is moving TOWARD the ego (head-on or near-head-on), creating a proximity hazard.",
                "Key visual cue: a VRU is visible ahead, getting closer, moving toward the ego; the lateral distance is small; the ego may need to adjust position.",
            ],
        }),
        ("LaneApproach_OncomingVehicle", {
            "cn": "逆向车道车辆贴近",
            "def": [
                "A motor vehicle in the ONCOMING direction approaches close to the ego on a straight road — the oncoming vehicle passes with very small lateral clearance.",
                "Common on narrow roads where two-way traffic has limited width, or when the oncoming vehicle drifts toward the center line.",
                "Key visual cue: an oncoming vehicle is visible ahead approaching the ego; it passes very close with minimal lateral gap; the ego may shift slightly to maintain distance.",
            ],
        }),
    ]),
    "disambiguation": [
        ("LaneApproach vs DynamicInteraction CutIn",
         "LaneApproach: the other road user stays in THEIR lane but is CLOSE. CutIn: the other road user actually enters the ego's lane. If the road user crosses the lane boundary, use a CutIn label."),
        ("LaneApproach vs IntersectionInteraction Approach",
         "LaneApproach: occurs on straight road segments BETWEEN intersections. IntersectionInteraction_*Approach: occurs WITHIN intersections. Check the location."),
    ],
}

# ============================================
# 9. MergeAndDiverge (主辅路/分合流) — 6 labels
# ============================================
CATEGORIES["09_MergeAndDiverge"] = {
    "title": "Merge, Diverge, and Service Road (主辅路/分合流)",
    "description": "Ego vehicle merging into, diverging from, or transitioning between main roads and service/frontage roads — ramp entries, ramp exits, lane-level and road-level splits and merges.",
    "labels": OrderedDict([
        ("MergeAndDiverge_LaneLevelDiverge", {
            "cn": "车道级分流",
            "def": [
                "Ego takes a LANE-LEVEL DIVERGE — the lane splits into two paths (e.g., a lane fork), and the ego follows one branch.",
                "Key visual cue: a single lane divides into two with painted gore markings; ego follows one path.",
                "Distinct from RoadLevelDiverge: the split is at the LANE level (one lane becomes two), not at the road level (e.g., a ramp exit).",
            ],
        }),
        ("MergeAndDiverge_RoadLevelDiverge", {
            "cn": "道路级分流",
            "def": [
                "Ego takes a ROAD-LEVEL DIVERGE — exiting to a ramp, service road, or a branching highway that physically separates from the main road.",
                "The split creates two SEPARATE road structures (main road and exit ramp/branch), not just two lanes on the same road.",
                "Key visual cue: an exit ramp or branch road physically separates from the main road; gore area markings (painted triangle) are visible; the ego follows the exit path.",
            ],
        }),
        ("MergeAndDiverge_StraightMerge", {
            "cn": "直行合流汇入",
            "def": [
                "Ego MERGES into the main road from a merging/acceleration lane while going roughly STRAIGHT — the ego's lane joins the main road at a shallow angle.",
                "The merge is a lane-joining maneuver, not a turn; the ego adjusts speed to find a gap in main road traffic.",
                "Key visual cue: the ego's lane narrows or its lane markings end as it merges into the main road; ego accelerates and adjusts position to join traffic flow.",
            ],
        }),
        ("MergeAndDiverge_RightTurnMerge", {
            "cn": "右转merge",
            "def": [
                "Ego MERGES into the main road via a RIGHT-TURN — the ego comes from a side road, ramp, or cross street and turns right to enter the main road.",
                "The merge involves a right turn followed by integration into the main traffic flow.",
                "Key visual cue: ego turns right from a ramp or side road and accelerates to merge with the main road traffic; the turn and merge happen as a continuous maneuver.",
            ],
        }),
        ("MergeAndDiverge_MainToService", {
            "cn": "主路进辅路",
            "def": [
                "Ego transitions from the MAIN road to a parallel SERVICE/FRONTAGE road (辅路) — the ego exits the main road to use the service road that runs alongside.",
                "The service road is a smaller, parallel road adjacent to the main road, often used for local access.",
                "Key visual cue: ego moves from the main road onto a smaller parallel road; the main road continues separately; a connecting ramp or lane transition is visible.",
            ],
        }),
        ("MergeAndDiverge_ServiceToMain", {
            "cn": "辅路进主路",
            "def": [
                "Ego transitions from a SERVICE/FRONTAGE road (辅路) back to the MAIN road — merging from the smaller parallel road onto the main road.",
                "The ego leaves the local service road and enters the higher-speed main road traffic.",
                "Key visual cue: ego accelerates from a smaller side road and merges onto the main road via a connecting ramp or merge lane.",
            ],
        }),
    ]),
    "disambiguation": [
        ("LaneLevelDiverge vs RoadLevelDiverge",
         "Lane-level: ONE lane splits into TWO lanes (a fork within the road). Road-level: the ego exits to a SEPARATE road structure (ramp, service road). Check whether the split creates two lanes on the same road or two separate road structures."),
        ("StraightMerge vs RightTurnMerge",
         "StraightMerge: ego is already moving roughly parallel to the main road and the merge lane joins. RightTurnMerge: ego turns RIGHT from a side road/ramp to merge. Check the ego's heading change during the merge."),
        ("MainToService vs RoadLevelDiverge",
         "MainToService: ego moves to a parallel SERVICE road that runs alongside the main road. RoadLevelDiverge: ego exits to a RAMP or completely different road. Check whether the destination road runs parallel to the original."),
    ],
}


# ─────────────────────────────────────────────
# PROMPT GENERATION
# ─────────────────────────────────────────────

def generate_prompt(cat_key, cat_data):
    # 过滤掉 P00 已有数据的标签
    filtered_labels = OrderedDict(
        (k, v) for k, v in cat_data["labels"].items() if k not in P00_EXCLUDE
    )

    if not filtered_labels:
        return None  # 该类别所有标签都是P00，跳过

    lines = []
    lines.append("You are an expert in autonomous driving scene annotation.")
    lines.append(f"Based on the input video, analyze the 20-second video to identify the ego vehicle's actions related to: **{cat_data['title']}**.")
    lines.append("")
    lines.append(f"CATEGORY SCOPE: {cat_data['description']}")
    lines.append("")

    # Label list
    lines.append(f"TARGET LABELS ({len(filtered_labels)} labels):")
    for label_en in filtered_labels:
        cn = filtered_labels[label_en]["cn"]
        lines.append(f"  {label_en}  ({cn})")
    lines.append("  not_applicable  (none of the above labels match)")
    lines.append("")

    lines.append(COMMON_RULES)
    lines.append(COMMON_OUTPUT_FORMAT)
    lines.append(COMMON_TERMINOLOGY)

    # Detailed definitions
    lines.append("LABEL DEFINITIONS:")
    for i, (label_en, info) in enumerate(filtered_labels.items(), 1):
        lines.append(f"{i}. {label_en} ({info['cn']}):")
        for d in info["def"]:
            lines.append(f"   • {d}")
        lines.append("")

    # Disambiguation
    if cat_data.get("disambiguation"):
        lines.append("DISAMBIGUATION RULES:")
        for i, (pair, rule) in enumerate(cat_data["disambiguation"], 1):
            lines.append(f"D{i}. {pair}:")
            lines.append(f"   • {rule}")
            lines.append("")

    lines.append(COMMON_GUIDELINES)

    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generated = 0
    total_mining = 0
    for cat_key, cat_data in CATEGORIES.items():
        prompt_text = generate_prompt(cat_key, cat_data)
        filepath = os.path.join(OUTPUT_DIR, f"{cat_key}.txt")

        if prompt_text is None:
            print(f"Skipped: {cat_key}.txt  (all labels are P00, no mining needed)")
            if os.path.exists(filepath):
                os.remove(filepath)
            continue

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        all_count = len(cat_data["labels"])
        mining_count = sum(1 for k in cat_data["labels"] if k not in P00_EXCLUDE)
        excluded = all_count - mining_count
        total_mining += mining_count
        generated += 1
        print(f"Generated: {cat_key}.txt  ({mining_count} mining labels, {excluded} P00 excluded)")

    print(f"\nTotal: {total_mining} mining labels across {generated} files (P00 {len(P00_EXCLUDE)} excluded)")


if __name__ == "__main__":
    main()
