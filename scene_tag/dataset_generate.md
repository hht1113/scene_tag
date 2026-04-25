================================================================================
自动驾驶场景视频问答（VQA）数据蒸馏与数据集生成方案
（面向大参数 VLM 教师模型，参考 DriveGPT / DriveMRP 类「驾驶语言数据」思路）
================================================================================
文档版本：v1.5
适用场景 taxonomy：见本文本文「一、任务定义与标签体系」（与 data_instillation_prompt.txt 中 01–07 一致）


---
零、术语定义：教师侧 vs 学生侧

---
本方案中的「蒸馏」指：用大能力多模态模型从视频生成监督信号或伪标签，再用于训练更小或更便宜的模型。两侧角色如下。

0.1 教师侧（Teacher side）
- 定义：数据生成与标注流水线中，承担强监督来源的一方；通常是一台（或一组）大参数 VLM（也可含规则校验、人工抽检，但「教师模型」特指该 VLM）。
- 输入：短视频 clip、可选多视角元数据、固定版本的 prompt / JSON Schema。
- 输出：
  - 中间结构化理解（本文「五-A」Round-A：ego_motion、lane、road_layout、traffic_control、agents、scene_context 等）；
  - 场景 taxonomy 标签与证据（Round-B：structured_label）；
  - 面向训练的问答对（Round-C：短 Q&A、对比题等）；
  - 以及 teacher 追溯信息（模型名与版本、解码参数、prompt 模板 id、帧采样策略等），用于复现与质检。
- 特点：能力上限高、推理与 API 成本高；不部署在车载端亦可；输出需经 Schema / 一致性规则 / 抽检，降低幻觉进入数据集的风险。
  
0.2 学生侧（Student side）
- 定义：被训练、最终被部署或用于下游实验的模型，即蒸馏的学习目标。
- 典型形态：更小参数的 VLM、单视角实时模型、或仅语言头 + 视觉编码器的轻量化结构；也可指同一架构下从教师生成的数据上做 SFT / LoRA 后的 checkpoint。
- 输入：与任务一致的视频（及问题文本，若是 VQA）。
- 监督来源：主要来自教师生成的 (video, question, answer)、结构化 JSON 压缩后的句子、或 DPO/GRPO 用的偏好对与奖励信号；不在训练阶段再次调用教师（除非主动做迭代蒸馏或 co-training）。
- 特点：追求延迟、显存、吞吐与成本；能力目标是在分布内逼近教师行为（分类准确率、证据对齐、回答格式等），通常弱于教师。
  
0.3 二者关系（一句话）
- 教师侧负责「把视频变成可学习、可校验的标签与问答」；学生侧负责「在更少资源下学会复现这些行为」。
- 文档中 Round-A/B/C 属教师侧流水线；第八节 SFT/DPO/GRPO 等属学生侧训练数据用法。
  

---
一、任务定义与标签体系（锁定，不再漂移）

---
1.1 顶层类别（文中为 7 类，非 6 类；以下以列表为准）
  01_DynamicInteraction        车道动态交互
  02_TrafficLight              红绿灯通行
  03_StartStop                 发车/停车
  04_Intersection              路口通行
  05_LaneCruising              车道巡航
  06_LaneChange                变道/绕行
  07_IntersectionInteraction   路口内动态交互

1.2 每个大类下的「显式子类 + 其他子场景」
- 显式子类：用于训练时的细粒度监督与难例挖掘。
- 「其他子场景」：保留开放集能力；教师输出须说明「归入其他」的理由（简短证据链）。
  
1.3 VQA 任务形态（建议统一为三种，便于蒸馏与下游复用）
  A) 场景分类型：给定短视频片段，判断最匹配的大类 / 子类（可多标签）。
  B) 证据型：同上，并要求指出关键时空证据（时间段 + 画面要素）。
  C) 对比/排错型：给定两个候选标签或描述，选择与视频更一致的一项并说明原因。
  说明：DriveGPT / DriveMRP 类工作强调「自然语言驾驶描述 + 结构化标签」对齐；本方案将教师输出约束为「结构化 JSON + 一句自然语言摘要」，便于解析与质检。


---
二、与开源路线对齐的设计要点（概念层，非绑定某一仓库 commit）

---
2.1 DriveGPT 类思路
- 用强 VLM 将「像素级驾驶视频」转为「可学习的语言监督」：场景理解、因果、风险、交通规则。
- 蒸馏目标：学生模型或后续 RL/SFT 可用的 (video, question, answer, metadata) 四元组。
  
2.2 DriveMRP / 多模态驾驶推理类思路
- 强调时序与空间引用：答案应能对应到时间区间与物体/区域（即使只做粗粒度 timestamp）。
- 多轮澄清可生成「困难样本」：同一 clip 生成主问答 + 追问 + 反事实问。
  
2.3 通用 VLM 数据蒸馏原则
- 教师一致：全链路固定 checkpoint、解码参数、系统提示词版本号。
- 可解析：优先 JSON schema，避免自由散文作为主标签。
- 可审计：每条样本保留 teacher_model_id、prompt_hash、temperature、帧采样策略。
  

---
三、总体流水线（端到端）

---
阶段 0：资产与规范
- 定义 JSON Schema（第 4 节）、评审量表（第 7 节）、版本化命名规则（数据/模型/prompt）。
  
阶段 1：原始视频与元数据入库
- 切片：统一 clip 长度（如 4–20 s），重叠滑窗可选（用于长视频）。
- 传感器：至少 front-wide；若有环视/周视，在 meta 中标注可用视角。
- 脱敏：车牌/人脸模糊；地理敏感信息剥离或哈希。
  
阶段 2：教师 VLM 粗标注（低成本全覆盖）
- 目标：大类 + 候选子类 + 置信度 + 时间粗定位。
- 输出：进入「待质检池」。
  
阶段 3：教师 VLM 精标注（高成本仅对高价值 clip）
- 触发条件：边界样本、低置信、多标签冲突、安全相关（StartStop / IntersectionInteraction 等）。
- 输出：证据型问答、对比题、多轮对话。
  
阶段 4：规则 + 小模型质检（非必须但强烈建议）
- JSON 合法性、标签 ∈ 封闭表、时间区间在 clip 内、自相矛盾检测。
  
阶段 5：人机抽检与难例回流
- 固定抽检比例（如 2%–5%）；错误模式写入「prompt 补丁库」。
  
阶段 6：划分 train/val/test 与发布
- 按 route / 日期 / 地理桶 分组划分，避免泄漏。
  

---
四、推荐数据记录格式（每条样本）

---
4.1 最小字段
- sample_id：全局唯一
- video_uri 或 本地路径 + checksum
- clip_start_sec, clip_end_sec
- camera_ids：列表
- task_type：classification | evidence | contrastive
- question：字符串（中/英择一，全库统一）
- answer：字符串（自然语言摘要，1–3 句）
- structured_label：JSON（机器可读，见 4.2）
  
4.2 structured_label 建议结构
{
  "major_category": "06_LaneChange",
  "sub_category": "静态障碍物避让",
  "is_other_subcategory": false,
  "multi_label": false,
  "secondary_categories": [],
  "time_evidence": [{"start": 2.1, "end": 5.4, "description": "..."}],
  "visual_cues": ["锥桶占用部分车道", "自车减速向右微调"],
  "confidence": {"major": 0.86, "sub": 0.72},
  "risk_level": "low|medium|high",
  "traffic_rule_keywords": ["让行", "实线"]
}

4.3 教师追溯字段（蒸馏必备）
- teacher_model：名称 + 版本 + 量化方式
- generation_params：temperature, top_p, max_tokens
- prompt_template_id
- raw_response：可选（存对象存储，表中仅存指针）
  

---
五、Prompt 设计要点（教师侧）

---
5.1 系统提示词（固定块）
- 角色：自动驾驶场景理解专家。
- 硬性规则：
  - 仅允许从 01–07 大类中选主标签；子标签须为给定枚举或填「其他子场景」。
  - 必须输出合法 JSON（可约定 ```json 围栏）。
  - 不得编造 clip 外事件；不确定则降低 confidence 并写明「不确定原因」。
- 安全：不输出可识别个人信息；不输出精确地理位置。
  
5.2 用户提示词模板（分类任务示例骨架）
  「下面是时长 {T} 秒的驾驶视频片段（{camera}）。请完成：
1. 判断最匹配的 major_category（单选）。
2. 判断最匹配的 sub_category（单选；若无匹配则 is_other_subcategory=true 并简述）。
3. 给出 time_evidence（1–3 段，时间在 [0,{T}]）。
4. 给出 confidence。
输出严格符合 Schema。」
  
5.3 难例增强模板
- 「若 major 为 06 与 05 之间犹豫，请生成 contrastive 问答：Q/A 说明排除另一类的依据。」
- 「针对 07_IntersectionInteraction，强制描述 VRU 与自车相对运动趋势。」
  
5.4 多轮蒸馏（可选）
  Round1：分类 + JSON
  Round2：基于 Round1 的 JSON，生成 2 个「学生向」简答题（更短答案，便于小模型学习）


---
五-A、教师侧技术方案详解（自车运动 / 车道与拓扑 / 信号灯与控制 / 交通参与者 / 多轮蒸馏）

---
本章目标：把「单轮贴标签」升级为可分解、可校验、可教学生的教师流水线。核心思想是：
- 先让教师做与 taxonomy 弱耦合的通用驾驶理解（运动、拓扑、参与者），再在该上下文上做场景分类与 VQA；
- 用多轮把长推理拆成短步，降低胡编概率，并产出不同粒度的训练样本（粗 JSON → 细 JSON → 短问答）。
  
A.1 教师能力边界（必须在系统提示词中写明）
- 输入仅为给定视角视频（+可选时间戳）；禁止臆测 clip 外路况、他车意图、信号灯相位未在画面出现的变化。
- 所有断言须带 visibility（clear / partial / occluded / not_visible）与 confidence（0–1）；不可见则填 not_visible，不得用猜测补全。
- 无 CAN/里程计时，自车运动只允许视觉可推断的定性/相对描述（如「相对车道线横向偏移趋势」），并标注 uncertainty。
  
A.2 统一「驾驶场景理解」中间表示（Round-A 输出，建议与 taxonomy 解耦）
  建议固定模块化 JSON（6 个顶层键），便于程序校验、与 01–07 大类对齐及多轮拼接：
  (1) ego_motion：自车运动状态、绕行策略、复合动作、跟车行为（视觉可观测范围内）
  (2) lane_and_markings：车道线、车道功能、变道证据与自车车道关系（服务 05/06 及路口导向车道）
  (3) road_layout：道路类型与几何、路口形态与精细布局、待转区类型、路侧设施（站点/车位等）
  (4) traffic_control：交通信号灯（含绿闪/黄闪/黑灯/倒计时）、警示灯、限速标志分类、路侧标志与地面标记
  (5) traffic_agents：其他交通参与者及与自车的交互关系，含切入紧急度/前序状态、急刹强度、贴近程度、agent 自身行驶方向
  (6) scene_context：天气光照、路面状态、交通流拥堵态势、前方排队信息、传感器可见度

  说明：原「road_topology」拆为 lane_and_markings + road_layout + traffic_control，避免信号灯与车道信息挤在同一嵌套对象里导致解析混乱；下游仍可将三者合并视图称「道路拓扑」。
  
  v1.5 扩展说明：为覆盖 122 个叶子标签（7 个先行一级类 × 112 个标签）的区分需求，在原 6 模块基础上新增了以下关键维度：
  - ego_motion：绕行策略（borrow_oncoming_lane / cross_line_bypass）、复合动作（overtake / three_point_uturn）、跟车行为（following_target_type / following_stop）
  - traffic_control：黄闪判断（is_flashing_guess + flash_color_guess）、警示灯（warning_and_auxiliary_lights）、限速分类（speed_limit_context）
  - traffic_agents：切入紧急度（cut_in_detail.urgency）、agent 前序状态（stationary_then_started）、急刹强度（lead_vehicle_braking_detail）、贴近威胁度（approach_proximity_threat）、agent 自身朝向（agent_own_heading_guess）
  - road_layout：道路类型（road_type_hints）、路口精细布局（turning_lane_layout / roundabout_detail）、待转区类型（waiting_zone_type_guess）、路侧设施（roadside_facilities：bus_station / structured_parking_spots 等）
  - lane_and_markings：非机动车道标注（non_motor_lane）、变道标线类型（crossed_marking_type_guess）、是否跨入对向车道（crossed_into_oncoming_lane_guess）
  - scene_context：交通流状态（traffic_flow_state）、前方排队（queue_ahead）

A.2.0 Round-A：教师「读视频」的方式、FPS 约定与具体输出形态
  【核心结论】
- Round-A 的语义输出是：针对一个 clip（时长 T 秒）的、覆盖整段时间的一份结构化 JSON（即 A.2 各模块字段），不是对「原始视频文件的每一帧各生成一条 response」。
- 工程上教师模型实际看到的是：从该 clip 中按策略抽取的多帧图像序列（或框架支持下的 video token），属于抽帧/重采样输入；源视频 FPS 与 送入教师的等效采样率 必须区分记录。
  
  【源视频 FPS vs 教师输入 FPS】
- source_video_fps：数据采集/编码时的帧率（如 30），仅作元数据。
- teacher_input_sampling：描述如何选帧，例如：
  - uniform_fps：均匀采样，如 2 Hz（2 FPS） 表示每秒取约 2 张，10 s clip 约 20 张；
  - uniform_stride：每 N 帧取 1 张（由 source_fps 与目标密度换算）；
  - hybrid：均匀 1–2 FPS 为底 + 光流/场景切变处加密关键帧（适合灯色变化、切入）。
- 推荐在数据 manifest 中固化并写入追溯字段（见下），否则无法复现实验。
  
  【默认推荐（与第六节一致，可配置）】
- 单次 Round-A 调用：输入 = 上述采样得到的一组帧 + 文本说明「clip 总时长 T、时间零点为 clip 起点」；输出 = 整段 clip 一份 JSON。
- 默认采样：1–2 FPS 均匀采样通常足够做场景级理解；02 灯色变化 / 01 切入 / 07 短交互 可把该 clip 的采样提高到 4 FPS 或启用 hybrid，而不是全局提高到 30 FPS（成本与 vision token 爆炸）。
- JSON 内所有时间量均为 相对 clip 起点的秒 ∈ [0, T]（如 time_span、ego_motion.segments），与「第几张输入帧」通过 frame_timestamps_sec 对齐。
  
  【何时拆成多次调用】
- 若单 clip 过长（如 T>30–60 s）或模型上下文放不下：将 clip 滑窗切成子段（可重叠），每子段一次 Round-A，再用单独「融合轮」合并为一条 canonical JSON（或保留多段作为多条训练样本）。这是工程折中，非默认。
  
  【建议在样本 manifest 中记录的输入追溯（可与 Round-A JSON 并列存储）】
- clip_id, clip_start_sec, clip_end_sec, duration_sec
- source_video_fps
- teacher_sampling_policy（如 uniform_2fps）
- num_frames_sent_to_teacher
- frame_timestamps_sec：列表，与送模顺序一一对应
- frame_resolution / resize_policy
- teacher_model_id, prompt_template_id_round_a
  
  【与「整段 response」的对应关系】
- 整段：指时间覆盖整段 clip 的一份结构化输出；其内部仍用 segments / time_span 表达子区间上的变化。
- 抽帧：仅指模型输入侧不送全帧率，不改变「一次 Round-A → 一份 JSON」的推荐形态。
  
A.2.1 ego_motion（自车运动状态）— 字段要点
- 时间粒度：对整个 clip 给 summary，对关键变化给 segments（start/end 相对 clip 起点秒）。
- longitudinal（纵向趋势，定性）：
  - approximate_action：maintain_speed | accelerate | decelerate | stop | start_from_stop | emergency_stop | unknown
  - stop_urgency（仅 stop/decelerate 时填）：gradual | hard_brake | emergency | unknown
  - evidence：如「与前车间距增大/减小」「刹车灯亮起频率」「画面整体光学流主导方向」等短句。
- lateral（横向趋势）：
  - approximate_action：lane_keep | drift_left_within_lane | drift_right_within_lane |
  lane_change_left | lane_change_right | nudge_left | nudge_right |
  borrow_oncoming_lane_left | borrow_oncoming_lane_right |
  cross_line_bypass_left | cross_line_bypass_right | unknown
  - 若无法区分「变道」与「弯中车道保持」，标 unknown 并降低 confidence。
- avoidance_maneuver（绕行/避让策略，若有横向位移时填写）：
  - strategy：none | standard_lane_change | borrow_oncoming_lane | cross_line_bypass | in_lane_nudge | unknown
  - target_type_guess（被避让的对象类型）：static_vehicle | slow_vru | static_obstacle | slow_vehicle | unknown | none
  - return_to_original_lane_guess：bool | unknown（是否回到原车道，区分绕行 vs 永久变道）
- compound_maneuver_guess（复合动作，可选，出现时填写）：
  - type：none | overtake | three_point_uturn | waiting_zone_uturn | unknown
  - phase：preparing | executing | completing | unknown
  - evidence：一句话描述（如「变道超越前方慢车后回到原车道」）
- following_behavior（跟车行为，当自车稳态跟随前方目标时填写）：
  - is_following：bool
  - following_target_type：vehicle | vru | unknown | none
  - following_distance_trend：stable | closing | opening | unknown
  - following_stop_guess：bool（是否因跟车而停车）
- ego_speed_regime（无速度真值时仅用粗分级）：very_low | low | medium | high | unknown
- stability：steady | transitional（正在变道/急刹/起步等）| unknown
  
A.2.2 lane_and_markings（车道信息）— 字段要点
  只描述画面中可见的车道标线与本车相对车道关系；不要求 HDMap 级精度。
- lane_markings_visible：bool
- lane_line_types（多选）：solid_white | dashed_white | solid_yellow | double_yellow | wide_white_stop | botts_dots | unclear | none
- lane_function_hints（车道功能线索，可多选）：through | left_turn | right_turn | uturn | bus_only | bike_lane_adjacent | non_motor_lane | variable_lane | tidal_lane | shoulder | no_parking_zone | unclear
  （新增 non_motor_lane 用于标注自车或相邻非机动车道；no_parking_zone 用于标注禁停区标记）
- ego_lane_relation：
  - estimated_lane_count_in_view：integer 或 "min-max" 字符串
  - ego_lane_index_from_left_guess：integer 或 unknown（自车所在车道为从左数第几条，看不清填 unknown）
  - ego_lane_position_in_lane：center | closer_to_left_line | closer_to_right_line | straddling_line | unknown
- lane_change_evidence（服务于 06_LaneChange / 绕行）：
  - lane_change_in_progress_guess：bool
  - direction：left | right | unknown
  - crossed_into_oncoming_lane_guess：bool | unknown（是否跨入对向车道，区分「借道绕行」vs「同向变道」）
  - crossed_marking_type_guess：dashed | solid | double_yellow | unknown（跨越了何种标线，区分合法变道 vs 跨实线绕行）
- merge_split_cues：merge_from_left | merge_from_right | lane_drop | lane_add | split_ahead | ramp_diverge | ramp_merge | unclear | none（可多选）
  （新增 ramp_diverge / ramp_merge 用于服务 MergeAndDiverge 类标签）
- curb_and_boundary：guardrail | curb_visible | parking_cars_line | sidewalk_visible | none | unclear（可多选）
- confidence；uncertainty_notes（如「雨夜标线反光弱」）
  
A.2.3 road_layout（道路几何与路口静态布局）— 字段要点
  与信号灯解耦：此处聚焦「路怎么走、路口长什么样」，不重复灯色细节（灯色放在 A.2.4）。
- road_type_hints（道路类型，可多选）：
  main_road | non_motor_lane | service_road | rural_road | urban_road | ramp |
  parking_area | residential | unknown
  （服务于 StartStop_StartFromNonMotorLane、LaneCruising_RuralRoad 等标签的区分）
- road_geometry_cues（多选）：straight | gentle_curve | sharp_curve | chicane | narrow_road |
  intersection_approach | intersection_interior | ramp_merge | tunnel_like | bridge_deck | unclear
- road_width_impression：wide_multi_lane | normal | narrow_single_lane | very_narrow | unknown
  （服务于 LaneCruising_NarrowSpace 等标签）
- intersection_topology_guess：none | T_junction | cross | multi_leg | roundabout | misaligned_cross | offset_left_turn | unclear
- roundabout_detail（仅 intersection_topology_guess=roundabout 时填写）：
  - lane_count_guess：1 | 2 | 3+ | unknown（区分单车道小环岛与多车道环岛）
- turning_lane_layout（转弯车道布局，路口处填写）：
  - dedicated_left_turn_lane_visible：bool | unknown（是否有左转专用道）
  - dedicated_right_turn_lane_visible：bool | unknown（是否有右转专用道）
  - parallel_turning_visible：bool | unknown（是否多车道同方向转弯，即并行左转/右转）
  - right_turn_adjacent_non_motor_lane：bool | unknown（右转车道右侧是否有直行非机动车道）
- ego_maneuver_slot_guess（自车所处/将执行的机动槽，弱标签）：go_straight | left_turn | right_turn | uturn | three_point_uturn | lane_select_unclear | unknown
- static_markings（路面静态标记，可见才写）：
  - stop_line_visible | crosswalk_visible | yield_marking_visible | directional_arrows_on_pavement | speed_bump_visible | no_parking_marking_visible | unclear
- waiting_zone（待转区，与 02 子类「待转区红绿灯起停」强相关）：
  - waiting_zone_visible：bool
  - ego_relative_to_waiting_zone：inside | approaching | not_applicable | unknown
  - waiting_zone_type_guess：straight | left_turn | text | combined_signal | image | unknown
    （text = 地面有文字标注；combined_signal = 有专门灯控；image = 地面有图示标记）
  - waiting_zone_marking_content_guess：短字符串或 unknown（如「左转待转」「直行待转」等地面文字内容）
- roadside_facilities（路侧设施，可见才填，可多选）：
  bus_station | taxi_stand | structured_parking_spots | unstructured_roadside_parking |
  charging_station | gas_station | none | unknown
  （服务于 StartStop_StopAtStation、StartStop_ParkInStructuredSpot 等标签的区分）
- railway_crossing_guess | toll_gate_guess：bool（可选，出现则标注 visibility）
- static_infrastructure（非灯控）：barrier | construction_zone | traffic_cone | water_horse | accident_scene | work_zone_signs 等标签列表（可见才写）
- confidence
  
A.2.4 traffic_control（交通信号灯、倒计时与相关控制信息）— 字段要点
  专门服务 02_TrafficLight 及与灯控强相关的 04_Intersection；所有灯色、箭头、闪烁类判断必须带 visibility 与 time_span（若在 clip 内发生变化须分段描述）。
- traffic_light：
  - any_traffic_light_visible：bool
  - heads：列表，每项描述一个灯头或一组同义灯头（教师自行分配 tl1, tl2…）：
    - position_hint：overhead_ahead | left_side | right_side | mast_arm | unknown
    - aspect_guess：red | yellow | green | red_yellow | off | unknown
    - is_arrow：bool；arrow_direction_guess：straight | left | right | uturn | none | unknown
    - is_flashing_guess：bool（灯是否在闪烁，含绿闪和黄闪）
    - flash_color_guess（仅 is_flashing_guess=true 时填）：green | yellow | unknown（区分绿闪与黄闪）
    - countdown_visible：bool；countdown_seconds_guess：integer 或 unknown
    - visibility：clear | partial | occluded | not_visible
    - time_span：[[t0,t1], ...] 内该 head 的可见外观变化区间
  - ego_relevance_guess：which_head_ids 列表或 unknown（哪几个灯头约束自车当前车道/转向）
- warning_and_auxiliary_lights（警示灯与辅助灯，非标准交通信号灯）：
  - warning_light_visible：bool（黄色/琥珀色持续闪烁的警示灯，常见于施工区、学校区）
  - warning_light_position_hint：overhead | roadside | vehicle_mounted | unknown
  - mobile_signal_visible：bool（临时移动式信号灯）
  - mobile_signal_aspect_guess：red | yellow | green | off | unknown
- traffic_signs（路侧标志，与灯控互补；可见才列）：
  - signs：列表，含 type_guess：stop | yield | no_entry | speed_limit | lane_direction | no_turn | merge | school_zone | no_parking | other | unknown
  - speed_limit_context（仅 type_guess=speed_limit 时填）：road_permanent | school_zone | construction_zone | intersection_approach | unknown
  - visibility；time_span（临时标志可限时）
- road_text_markings（地面文字/公交专用道刻字等）：visible bool；text_guess 短字符串或 unknown
- temporary_traffic_control：traffic_police_visible | portable_light_visible | manual_flag_visible（bool，可见才 true）
- right_of_way_visual_cues（非意图推断，仅画面直接支持的弱线索）：
  - opposing_traffic_has_green_guess：bool | unknown
  - pedestrian_signal_visible_guess：bool | unknown
  - notes：一句客观说明（如「仅见侧面灯头，自车道灯头被遮挡」）
- confidence；consistency_hint_for_round_b：可选字符串（供 Round-B 检查与 02/04 标签一致性）
  
A.2.5 traffic_agents（其他车辆 / VRU / 骑行者）— 字段要点
- agents：列表，每一项描述一个可区分的交通体（教师自行分配临时 id：a1,a2… 每 clip 内唯一即可）。
- 每个 agent 建议字段：
  - category：passenger_car | truck_bus | two_wheeler | pedestrian | cyclist | animal | generic_vehicle | static_small_object | unknown
    （static_small_object 用于锥桶、水马、落石、遗撒等 SOD 类目标，与 road_layout.static_infrastructure 互补但此处强调「自车对该物体有交互反应」）
  - ego_relative_position：front | front_left | front_right | left_adjacent | right_adjacent | rear | rear_left | rear_right | unknown
  - agent_own_heading_guess（该 agent 自身行驶方向，相对道路）：
    same_direction | opposing | perpendicular_from_left | perpendicular_from_right | stationary | unknown
    （服务于 IntersectionInteraction 中区分「左转车横穿」与「直行车横穿」等）
  - motion_trend_relative_ego：approaching | receding | parallel_same_dir | parallel_opposing | crossing | stationary | merging_toward_ego | unknown
  - approach_proximity_threat（贴近/逼近程度，服务于 LaneApproach 及 IntersectionInteraction 的 Approach 类标签）：
    high | moderate | low | none | unknown
  - interaction_with_ego（与自车的交互语义，可多选）：
    none_apparent | cut_in | cut_out | lead_vehicle | steady_following |
    oncoming | cross_path | pedestrian_near_crosswalk | cyclist_merge |
    side_pass | yield_relation_unclear | gap_opening_then_cut_in
  - cut_in_detail（仅 interaction_with_ego 包含 cut_in 时填写）：
    - urgency：emergency | normal | slow | unknown（紧急切入 / 常规 / 缓速切入）
    - agent_prior_state_guess：moving | stationary_then_started | unknown（起步切入 = stationary_then_started）
    - consecutive_lane_change_guess：bool | unknown（是否从 2+ 车道外连续变道切入）
  - lead_vehicle_braking_detail（仅 interaction_with_ego 包含 lead_vehicle 时填写）：
    - braking_intensity：sudden_hard | gradual | none | unknown（区分「前车急刹」vs 正常减速）
    - ego_reaction：hard_brake | moderate_brake | no_reaction | unknown
  - visibility：clear | partial | occluded
  - time_span：该 agent 在 clip 内显著出现的时间段列表
  - note：一句话客观描述（不含意图揣测；意图仅允许「画面直接支持」的弱表述，如「正在横穿斑马线」）
    
A.2.6 scene_context（场景环境、交通流状态与传感器可见度）— 字段要点（推荐启用）
  用于解释低置信度样本、辅助质检与难例挖掘，不用于编造未见物体。
- time_of_day_guess：day | dusk_dawn | night | unknown
- weather_visibility：clear | rain | fog | snow | heavy_glare | backlit | unknown
- ego_wiper_active_guess：bool | unknown（雨刷摆动有时可辨）
- road_surface_cues（改为数组，可多选）：wet | icy_suspected | construction_debris | flooded | unknown | none
- camera_issues（数组，可多选）：dirty_lens | strong_reflection | compression_artifact | unknown | none
- ambient_light_level：bright | normal | dark | unknown
- traffic_flow_state（交通流状态，整段 clip 的主观印象）：
  - overall：free_flow | moderate | congested | gridlock | unknown
  - segments（可选，若 clip 内交通流变化明显则分段描述）：
    列表 [{"start": t0, "end": t1, "state": "congested"}, ...]
- queue_ahead（前方排队信息，服务于 LaneCruising_StaticVehicleQueueCongestion / CongestedFollowing 等）：
  - queue_visible：bool
  - queue_type_guess：static_vehicle_queue | slow_moving_queue | mixed | unknown | none
  - estimated_queue_length：short_1_3_vehicles | medium_4_8_vehicles | long_8_plus | unknown | none
  - ego_in_queue：bool | unknown（自车是否在排队队列中）
- overall_visibility_notes：自由短句（与旧版字段兼容时可合并至此）
  
A.3 多轮蒸馏协议（推荐默认 4 轮 + 可选第 5 轮）

  【Round-A】感知与状态分解（禁止输出 01–07 大类标签）
    - 目的：减少「先猜类别再编理由」的偏见；为后续轮次提供条件上下文。
    - 输出：包含 ego_motion + lane_and_markings + road_layout + traffic_control + traffic_agents +（推荐）scene_context 的 JSON；若 token 紧张可对 traffic_control.heads / traffic_agents 做摘要，但不得删除灯头与车道功能冲突相关字段。
    - 解码建议：temperature 0.2–0.5；要求 JSON mode / 严格 schema。

  【Round-B】场景分类与证据（条件于 Round-A）
    - 把 Round-A 的 JSON 全文附在用户消息中，要求：
        * 输出第 4.2 节的 structured_label（major/sub、time_evidence、confidence 等）；
        * 一致性约束（加强）：
            - 若 traffic_control.traffic_light.any_traffic_light_visible=false 或全部为 not_visible，则 Round-B 不得将 02_TrafficLight 作为主类，除非 structured_label 中 explicit_rationale 说明「灯不可见但地面标线/标志明确表明灯控路口」且 consistency_check.passed=true（此类样本占比应极低，建议人工抽审）。
            - 若 lane_and_markings 与 traffic_control 对「左转/直行导向」冲突，须降低 confidence 或拆分 multi_label。
            - 子类与结构化字段对齐示例：选「待转区红绿灯起停」→ road_layout.waiting_zone 应有 visible/inside 等支撑；选「左转绿闪通行」→ traffic_control 某 head 的 is_flashing_green_guess=true 或 aspect 证据充分。
    - 解码建议：temperature 0.2–0.4。

  【Round-C】学生向 VQA 生成（条件于 Round-A + Round-B）
    - 自动生成 K 个（如 K=4–8）短问答，覆盖：
        (c1) 自车运动：「片段中自车纵向/横向趋势如何？」
        (c2) 车道/路口：「可见车道线类型、车道功能线索、待转区/停止线/斑马线？」
        (c2b) 灯控：「可见灯头位置、圆灯/箭头、是否绿闪/倒计时、是否与自车相关？」（无灯则答 not_visible）
        (c3) 交互：「与自车最相关的交通参与者是谁、关系如何？」
        (c4) 场景归类：「最符合哪一大类/子类，关键证据时间段？」
        (c5) 对比：易混类二选一（从规则库按 major 邻接表抽取）
    - 答案长度控制：学生答案建议 1–2 句或 ≤ 40 汉字（可配置），并附 teacher_short_rationale（可截断不入训练）。
    - 解码建议：temperature 0.5–0.7 略增多样性；须 JSON 列表 [{question, answer, skill_tag}, ...]。

  【Round-D】一致性审查（可选，低成本规则 + 二次调用）
    - 输入：Round-A/B/C 的合并 JSON。
    - 任务：仅输出 {passed: bool, issues: [str], suggested_fix: null|object}；不重新看视频则只做文本一致性；若预算允许可同视频第二次短调用做「挑错」。
    - 未通过：整样本降级（丢弃或仅保留 Round-A 作预训练）。

  【Round-E】反事实 / 困难扩展（可选，仅对高价值 clip）
    - 例：「若未出现某 agent，场景分类是否改变？」「若信号灯被遮挡，应如何标注 visibility？」
    - 用于 GRPO/DPO 的对比对或思维链数据，占比宜小（如 ≤5%），避免噪声过大。

A.4 与 01–07 taxonomy 的映射策略（教师侧）
- Round-B 前可由规则库给出「候选 major 集合」（如出现 crosswalk + turning_vehicle → 优先 04/07），仅作为提示，不强制。
- 对易混对强制 Round-C 生成 contrastive 题：
  - 05_LaneCruising vs 06_LaneChange（横向位移动机是否为「绕行/变道」）
  - 04_Intersection vs 07_IntersectionInteraction（几何通过 vs 路口内动态交互主体）
  - 02_TrafficLight vs 04_Intersection（信号控制是否主导场景叙事）
- 「其他子场景」须在 structured_label 中给出 exclude_reasons：列举 Top-2 被排除子类及理由。
  
A.5 解码与工程要点
- JSON Schema 校验：每轮失败则重试（最多 N 次，指数退避）；仍失败记录 error_code。
- 上下文长度：Round-B/C 附 Round-A 时可摘要 Round-A（如保留 traffic_control.heads 与 lane_and_markings.lane_function_hints + agents 列表）以省 token，但须保留 灯头变化 time_span 与 关键车道证据。
- 同 clip 多视角：先逐视角 Round-A，再融合（第五轮「融合」或规则 merge）后 Round-B；避免简单拼接导致幻觉。
- 缓存：同一 clip、同 teacher 版本、同帧哈希的 Round-A 结果可缓存，重复跑分类只算 Round-B/C。
  
A.6 扩展 JSON 示例（Round-A 片段示意，可与 4.2 节合并存储于 teacher_rounds.round_a）

示例场景：自车在雨天接近十字路口，左转箭头红灯倒计时中，前方有同向跟车目标，斑马线处有行人横穿，右侧有 VRU 缓速切入，路口有左转待转区。

{
  "ego_motion": {
    "summary": {
      "longitudinal": "decelerate",
      "stop_urgency": "gradual",
      "lateral": "nudge_right",
      "speed_regime": "low",
      "stability": "transitional"
    },
    "segments": [
      {"start": 0.0, "end": 3.0, "longitudinal": "maintain_speed", "lateral": "lane_keep"},
      {"start": 3.0, "end": 8.0, "longitudinal": "decelerate", "lateral": "nudge_right"}
    ],
    "avoidance_maneuver": {
      "strategy": "in_lane_nudge",
      "target_type_guess": "slow_vru",
      "return_to_original_lane_guess": true
    },
    "compound_maneuver_guess": {
      "type": "none",
      "phase": "unknown",
      "evidence": ""
    },
    "following_behavior": {
      "is_following": true,
      "following_target_type": "vehicle",
      "following_distance_trend": "closing",
      "following_stop_guess": false
    },
    "confidence": 0.74,
    "uncertainty_notes": "无速度计，仅依据光学流与车道线相对运动"
  },
  "lane_and_markings": {
    "lane_markings_visible": true,
    "lane_line_types": ["dashed_white", "solid_white"],
    "lane_function_hints": ["through", "left_turn"],
    "ego_lane_relation": {
      "estimated_lane_count_in_view": 3,
      "ego_lane_index_from_left_guess": 2,
      "ego_lane_position_in_lane": "center"
    },
    "lane_change_evidence": {
      "lane_change_in_progress_guess": false,
      "direction": "unknown",
      "crossed_into_oncoming_lane_guess": false,
      "crossed_marking_type_guess": "unknown"
    },
    "merge_split_cues": ["none"],
    "curb_and_boundary": ["sidewalk_visible"],
    "confidence": 0.78,
    "uncertainty_notes": "远处导向箭头不完整"
  },
  "road_layout": {
    "road_type_hints": ["urban_road", "main_road"],
    "road_geometry_cues": ["intersection_approach"],
    "road_width_impression": "wide_multi_lane",
    "intersection_topology_guess": "cross",
    "roundabout_detail": null,
    "turning_lane_layout": {
      "dedicated_left_turn_lane_visible": true,
      "dedicated_right_turn_lane_visible": false,
      "parallel_turning_visible": false,
      "right_turn_adjacent_non_motor_lane": false
    },
    "ego_maneuver_slot_guess": "go_straight",
    "static_markings": {
      "stop_line_visible": true,
      "crosswalk_visible": true,
      "yield_marking_visible": false,
      "directional_arrows_on_pavement": true,
      "speed_bump_visible": false,
      "no_parking_marking_visible": false
    },
    "waiting_zone": {
      "waiting_zone_visible": true,
      "ego_relative_to_waiting_zone": "approaching",
      "waiting_zone_type_guess": "left_turn",
      "waiting_zone_marking_content_guess": "左转待转"
    },
    "roadside_facilities": ["none"],
    "static_infrastructure": [],
    "confidence": 0.76
  },
  "traffic_control": {
    "traffic_light": {
      "any_traffic_light_visible": true,
      "heads": [
        {
          "id": "tl1",
          "position_hint": "overhead_ahead",
          "aspect_guess": "red",
          "is_arrow": true,
          "arrow_direction_guess": "left",
          "is_flashing_guess": false,
          "flash_color_guess": null,
          "countdown_visible": true,
          "countdown_seconds_guess": 12,
          "visibility": "clear",
          "time_span": [[0.0, 8.0]]
        }
      ],
      "ego_relevance_guess": ["tl1"]
    },
    "warning_and_auxiliary_lights": {
      "warning_light_visible": false,
      "warning_light_position_hint": "unknown",
      "mobile_signal_visible": false,
      "mobile_signal_aspect_guess": "unknown"
    },
    "traffic_signs": {
      "signs": [
        {
          "type_guess": "lane_direction",
          "speed_limit_context": null,
          "visibility": "partial",
          "time_span": [[0.0, 8.0]]
        }
      ]
    },
    "road_text_markings": {"visible": false, "text_guess": "unknown"},
    "temporary_traffic_control": {
      "traffic_police_visible": false,
      "portable_light_visible": false,
      "manual_flag_visible": false
    },
    "right_of_way_visual_cues": {
      "opposing_traffic_has_green_guess": "unknown",
      "pedestrian_signal_visible_guess": false,
      "notes": "仅清晰看到本向箭头红灯，对向灯头未入画"
    },
    "confidence": 0.82,
    "consistency_hint_for_round_b": "待转区可见且自车接近，灯控与路口叙事一致"
  },
  "traffic_agents": {
    "agents": [
      {
        "id": "a1",
        "category": "passenger_car",
        "ego_relative_position": "front",
        "agent_own_heading_guess": "same_direction",
        "motion_trend_relative_ego": "parallel_same_dir",
        "approach_proximity_threat": "none",
        "interaction_with_ego": ["lead_vehicle", "steady_following"],
        "cut_in_detail": null,
        "lead_vehicle_braking_detail": {
          "braking_intensity": "gradual",
          "ego_reaction": "moderate_brake"
        },
        "visibility": "clear",
        "time_span": [[0.0, 8.0]],
        "note": "正前方同向行驶车辆，间距逐渐减小"
      },
      {
        "id": "a2",
        "category": "pedestrian",
        "ego_relative_position": "front_right",
        "agent_own_heading_guess": "perpendicular_from_right",
        "motion_trend_relative_ego": "crossing",
        "approach_proximity_threat": "moderate",
        "interaction_with_ego": ["cross_path", "pedestrian_near_crosswalk"],
        "cut_in_detail": null,
        "lead_vehicle_braking_detail": null,
        "visibility": "partial",
        "time_span": [[4.5, 7.2]],
        "note": "斑马线附近行人从右侧横穿，出现部分遮挡"
      },
      {
        "id": "a3",
        "category": "two_wheeler",
        "ego_relative_position": "right_adjacent",
        "agent_own_heading_guess": "same_direction",
        "motion_trend_relative_ego": "merging_toward_ego",
        "approach_proximity_threat": "moderate",
        "interaction_with_ego": ["cut_in"],
        "cut_in_detail": {
          "urgency": "slow",
          "agent_prior_state_guess": "moving",
          "consecutive_lane_change_guess": false
        },
        "lead_vehicle_braking_detail": null,
        "visibility": "clear",
        "time_span": [[5.0, 7.5]],
        "note": "右侧电动车缓慢横移进入自车车道"
      }
    ],
    "confidence": 0.69
  },
  "scene_context": {
    "time_of_day_guess": "day",
    "weather_visibility": "rain",
    "ego_wiper_active_guess": true,
    "road_surface_cues": ["wet"],
    "camera_issues": ["compression_artifact"],
    "ambient_light_level": "normal",
    "traffic_flow_state": {
      "overall": "moderate",
      "segments": []
    },
    "queue_ahead": {
      "queue_visible": false,
      "queue_type_guess": "none",
      "estimated_queue_length": "none",
      "ego_in_queue": false
    },
    "overall_visibility_notes": "雨天，远处细节偏弱"
  }
}

A.7 落地检查清单（教师侧）
  [ ] 为 Round-A/B/C/D 分别冻结 prompt_template_id 与 schema 版本
  [ ] 每条样本记录 teacher 输入：source_video_fps、sampling_policy、num_frames、frame_timestamps_sec（与 A.2.0 一致）
  [ ] 校验器：时间区间 ⊆ clip；agents.id 唯一；interaction 枚举闭合
  [ ] traffic_control.heads[].id 唯一；灯色/箭头/闪烁/倒计时字段与 visibility 同时出现
  [ ] is_flashing_guess=true 时 flash_color_guess 不得为 null；is_flashing_guess=false 时 flash_color_guess 须为 null
  [ ] lane_and_markings.ego_lane_index_from_left_guess 与 estimated_lane_count_in_view 逻辑自洽（若可知）
  [ ] cut_in_detail 仅在 interaction_with_ego 含 cut_in 时出现，否则为 null
  [ ] lead_vehicle_braking_detail 仅在 interaction_with_ego 含 lead_vehicle 时出现，否则为 null
  [ ] avoidance_maneuver.strategy 与 ego_motion.lateral 逻辑一致（如 strategy=borrow_oncoming_lane → lateral 应含 borrow_oncoming_lane_*）
  [ ] compound_maneuver_guess.type=overtake → 须有 return_to_original_lane_guess=true 的支撑
  [ ] road_layout.road_type_hints 不得为空列表
  [ ] waiting_zone.waiting_zone_type_guess 仅在 waiting_zone_visible=true 时填写
  [ ] roadside_facilities 与 StartStop 标签交叉校验：StartStop_StopAtStation → roadside_facilities 含 bus_station
  [ ] traffic_flow_state.overall 与 queue_ahead.queue_visible 逻辑一致（queue_visible=true 时 overall 不应为 free_flow）
  [ ] Round-B 强制 consistency_check：无灯不可主标 02（除极少数「纯标线/标志灯控路口」例外并抽审）
  [ ] 统计各轮 token 与失败率，优先优化 Round-A（缓存收益最大）
  [ ] 易混类对比题覆盖率监控（按 major 邻接表）
  [ ] 按大类抽检：02 对照 traffic_control；05/06 对照 lane_and_markings + avoidance_maneuver；07 对照 traffic_agents+road_layout
  [ ] 新增字段覆盖率监控：cut_in_detail.urgency、flash_color_guess、road_type_hints、traffic_flow_state 在各大类中的非 unknown/null 占比


---
六、帧/视频输入策略（影响成本与效果）

---
  与五-A「A.2.0」一致：源视频 FPS ≠ 教师输入 FPS；Round-A 默认「整段 clip 一次调用、一份 JSON」，输入为抽帧后的序列。本节为执行层摘要。

6.1 采样
- 教师侧等效输入 FPS：默认 1–2 FPS 均匀采样即可覆盖多数 01–07 场景识别；需要捕捉快速变化（灯色切换、切入起始帧）时对该类样本或该 clip 提到 4 FPS 或使用「均匀 + 关键帧」混合策略。
- 源视频 FPS（如 30）仅用于解码与 stride 换算，不必整段以源 FPS 送教师。
- 分辨率：与教师训练分辨率对齐；不足则 letterbox，避免形变。
  
6.2 时序打包
- 优先「均匀采样 + 关键帧」：用简单光流/场景切分在变化大的时刻加密帧（可选流水线）。
- 将 frame_timestamps_sec 与 JSON 内 time_span 对齐，便于质检与调试。
  
6.3 与 vLLM / API 限制对齐
- 控制 vision token：必要时减少帧数或降低分辨率，优先保证时间覆盖与关键事件帧。
  

---
七、质量控制（QC）与过滤规则

---
7.1 自动过滤
- JSON 解析失败 → 丢弃或重试（有限次数）。
- major_category 不在 01–07 → 丢弃。
- confidence.major < 阈值（如 0.5）→ 进入人工或二次教师（更强 prompt）。
- time_evidence 超出 clip → 修正或丢弃。
  
7.2 分布与平衡
- 按 major_category 目标占比分层抽样生成；避免「其他子场景」占比过高。
- 对稀有子类（如特定 VRU 交互）单独过采样生成，但需防模板化复读（见 7.3）。
  
7.3 去重与多样性
- 文本侧：SimHash / MinHash 近似去重。
- 视频侧：同一路段连续 clip 限制每日上限，防止记忆路段。
  
7.4 人工抽检量表（简版）
- 正确性 1–5；证据是否支持标签 1–5；是否幻觉 1–5；是否可教学生 1–5。
  

---
八、训练集构造建议（供下游 SFT / DPO / GRPO）

---
8.1 SFT 正样本
- (video, question, short_answer) 来自 structured_label 压缩句式 + 证据一句。
  
8.2 偏好对（可选）
- 同 clip 上：高置信教师答案 vs 低置信或故意错误答案（规则合成），用于 DPO。
  
8.3 负样本与边界
- 易混类别对：06 vs 05、04 vs 07、02 vs 04（灯控路口）等，专门生成 contrastive 题。
  

---
九、工具与工程落地（最小可行栈）

---
9.1 任务编排
- 队列：Ray / Slurm / K8s Job；失败重试、断点续跑。
  
9.2 存储
- 视频：对象存储 + manifest（parquet/jsonl）。
- 标注：jsonl + schema 校验（pydantic/jsonschema）。
  
9.3 成本估算
- 记录每千条样本的 token / 秒视频 / GPU 时；按 major 分层统计。
  

---
十、评估与验收指标（数据集层面）

---
10.1 自动指标
- Schema 通过率、平均 confidence、类别熵、平均证据段数。
  
10.2 小规模金标
- 每类至少 N 条人工金标，用于估计教师标签准确率与校准曲线。
  
10.3 下游探测任务
- 用小 VLM 做 linear probe 或轻量 LoRA：看是否学得动（防「数据虚胖」）。
  

---
十一、风险与合规

---
- 数据授权与车内录音录像合规。
- 教师输出不可作为真实世界决策依据；标注为 research / training only。
- 模型版本冻结，避免「边标边升级」导致分布漂移。
  

---
十二、执行清单（可勾选）

---
[ ] 冻结 taxonomy 与 JSON Schema v1
[ ] 定 clip 长度、FPS、分辨率、语言（中/英）
[ ] 实现 jsonl 写入 + schema 校验 + 重试
[ ] 跑 1k 条试点，统计错误类型，修订系统提示词
[ ] 扩大全量；分层 QC；导出 train/val/test
[ ] 发布数据卡（Data Card）：来源、偏差、限制
[ ] 启用教师多轮协议（Round-A/B/C，可选 D/E）与分轮 schema 校验
[ ] 配置易混类对比题与 Round-B 一致性（consistency_check）规则

================================================================================
附录：与 data_instillation_prompt.txt 中场景列表一一对应（便于脚本解析）
================================================================================
01_DynamicInteraction | 邻车道车辆紧急切入 | 邻车道VRU缓速切入 | 其他子场景
02_TrafficLight       | 待转区红绿灯起停   | 左转绿闪通行     | 其他子场景
03_StartStop          | 非机动车道发车     | 主路紧急停车     | 其他子场景
04_Intersection       | 有保护左转         | 无保护路口直行   | 其他子场景
05_LaneCruising       | 大曲率弯道巡航     | 窄空间巡航       | 其他子场景
06_LaneChange         | 静态障碍物避让     | 非机动车道抑制变道 | 其他子场景
07_IntersectionInteraction | 路口直行右转VRU切入 | 路口逆行VRU贴近 | 其他子场景

================================================================================
结束
================================================================================
