"""
严禁靠边场景 — 文生视频 prompt 配置

可直接合入 16_generate_video.py 的 TAG_PROMPTS 字典。
来源：严禁靠边场景类别表（5大类 23个子场景）

用法:
    # 将下面的 PROHIBITED_PARKING_PROMPTS 合入 TAG_PROMPTS 后，即可用 --batch_tags 批量生成
    python scene_tag/16_generate_video.py \\
        --api_key "YOUR_API_KEY" \\
        --batch_tags "PP_NoStopSignZone,PP_BusLaneAndStation,PP_CrosswalkZone" \\
        --count 1
"""

from scene_tag import _generate_video_16 as gen  # noqa: only for reference

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


PROHIBITED_PARKING_PROMPTS = {
    # ===== 类别1: 明确交通法规禁止区域 =====
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

    # ===== 类别2: 高风险安全隐患区域 =====
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

    # ===== 类别3: 影响公共安全与秩序区域 =====
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
            "A traffic coordinator (交通协管员) in a reflective vest is directing traffic near the gate. "
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
            "SCENE: A Chinese urban residential compound (小区) main gate. "
            "Multiple private cars are PARKED or WAITING on both sides of the road near the gate, narrowing the drivable space significantly. "
            "The gate has a boom barrier (道闸) and a security booth. "
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

    # ===== 类别4: 特定功能受限区域 =====
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
            "This is a GPS-unfriendly environment. "
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

    # ===== 类别5: 易引发拥堵或冲突区域 =====
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
            "Multiple vehicles have hazard lights (四闪) flashing. Passengers are getting in and out of cars along the road. "
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
            "SCENE: The ENTRANCE of a Chinese industrial or technology PARK (园区). "
            "A gate with a boom barrier (道闸) and a security guard booth is visible ahead. "
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
