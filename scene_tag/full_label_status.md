# 全量标签分级现状总表

> 生成日期：2026-03-27
> 达标标准：抽查 Precision ≥ 70%
> 微调模型：Qwen3-VL-30B（P00 阶段）
> 冷启动挖掘模型：Qwen3-VL-235B-FP8（后续阶段）

## 状态说明

| 符号 | 含义 |
|:---:|---|
| ✅ | P00 达标（抽查 ≥ 70%） |
| ⏳ | P00 接近达标（60-69%） |
| ❌ | P00 未达标（< 60%） |
| 📋 | 无数据，需 235B 冷启动挖掘 |

## 统计概览

| 优先级 | 标签数 | ✅达标 | ⏳接近 | ❌未达标 | 📋需挖掘 | FW 🟢 | FW 🟡 | FW 🔴 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| P00（已微调） | 12 | 6 | 2 | 4 | 0 | 8 | 3 | 1 |
| P0 新增 | 35 | 0 | 0 | 0 | 35 | 27 | 8 | 0 |
| P1 新增 | 54 | 0 | 0 | 0 | 54 | 33 | 17 | 4 |
| P2 新增 | 12 | 0 | 0 | 0 | 12 | 8 | 1 | 3 |
| 前司补充 | 9 | 0 | 0 | 0 | 9 | 7 | 2 | 0 |
| **合计** | **122** | **6** | **2** | **4** | **110** | **83** | **31** | **8** |

---

## 一、P00 级（12 个，已有 Qwen3-VL-30B 微调数据）

| # | 一级类 | 标签（中文） | 英文标签 | FW | 测试Prec | 抽查Prec | 状态 |
|:---:|---|---|---|:---:|:---:|:---:|:---:|
| 1 | 路口通行 | 普通U-turn | Intersection_StandardUTurn | 🟢 | 82.6% | 85% | ✅ |
| 2 | 车道动态交互 | 邻车道车辆常规切入 | DynamicInteraction_StandardVehicleCutIn | 🟢 | 76.0% | 75% | ✅ |
| 3 | 发车/停车 | 靠边停车 | StartStop_ParkRoadside | 🟡 | 75.0% | 30% | ❌ |
| 4 | 车道巡航 | 普通车道巡航 | LaneCruising_Straight | 🟢 | 70.6% | 95% | ✅ |
| 5 | 变道/绕行 | 慢速VRU避让 | LaneChange_AvoidSlowVRU | 🟡 | 68.0% | 45% | ❌ |
| 6 | 发车/停车 | 主路发车 | StartStop_StartFromMainRoad | 🟢 | 64.3% | 95% | ✅ |
| 7 | 红绿灯通行 | 直行红绿灯起停 | TrafficLight_StraightStopOrGo | 🟢 | 62.5% | 80% | ✅ |
| 8 | 红绿灯通行 | 左转红绿灯起停 | TrafficLight_LeftTurnStopOrGo | 🟡 | 53.7% | 15% | ❌ |
| 9 | 车道动态交互 | 车道内VRU横穿 | DynamicInteraction_VRUInLaneCrossing | 🟡 | 42.9% | 60% | ⏳ |
| 10 | 变道/绕行 | 静态障碍车避让 | LaneChange_AvoidStaticVehicle | 🟡 | 41.2% | 65% | ⏳ |
| 11 | 变道/绕行 | 前方路口导航变道 | LaneChange_NavForIntersection | 🔴 | 41.2% | 35% | ❌ |
| 12 | 车道动态交互 | 车道内车辆横穿 | DynamicInteraction_VehicleInLaneCrossing | 🟢 | 40.5% | 80%* | ✅* |

> *#12 P00 测试的是"车道内车辆横穿"（80%），P0 细化为"车道内车辆横穿"，不完全直接对应。

---

## 二、P0 级新增（35 个，📋 需冷启动挖掘）

### 车道动态交互（5 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 13 | 邻车道车辆紧急切入 | DynamicInteraction_EmergencyVehicleCutIn | 🟢 |
| 14 | 邻车道车辆起步切入 | DynamicInteraction_StartupVehicleCutIn | 🟡 |
| 15 | 连续变道车辆切入 | DynamicInteraction_ConsecutiveLaneChangeCutIn | 🟡 |
| 16 | 邻车道VRU缓速切入 | DynamicInteraction_SlowVRUCutIn | 🟢 |
| 17 | 邻车道VRU紧急切入 | DynamicInteraction_EmergencyVRUCutIn | 🟢 |

### 红绿灯通行（3 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 18 | 右转红绿灯起停 | TrafficLight_RightTurnStopOrGo | 🟢 |
| 19 | 掉头红绿灯起停 | TrafficLight_UTurnStopOrGo | 🟢 |
| 20 | 待转区红绿灯起停 | TrafficLight_WaitingZoneStopOrGo | 🟡 |

### 发车/停车（4 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 21 | 非机动车道发车 | StartStop_StartFromNonMotorLane | 🟢 |
| 22 | 主路紧急停车 | StartStop_EmergencyStopOnMainRoad | 🟢 |
| 23 | 站点停车 | StartStop_StopAtStation | 🟢 |
| 24 | 路侧结构化车位泊入 | StartStop_ParkInStructuredSpot | 🟢 |

### 路口通行（10 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 25 | 有保护左转 | Intersection_ProtectedLeftTurn | 🟡 |
| 26 | 无保护左转 | Intersection_UnprotectedLeftTurn | 🟡 |
| 27 | 有保护路口直行 | Intersection_ProtectedStraight | 🟢 |
| 28 | 无保护路口直行 | Intersection_UnprotectedStraight | 🟡 |
| 29 | 错位非对齐路口直行 | Intersection_MisalignedStraight | 🟢 |
| 30 | 拥堵路口直行 | Intersection_CongestedStraight | 🟢 |
| 31 | 有保护右转 | Intersection_ProtectedRightTurn | 🟢 |
| 32 | 右转专用道 | Intersection_DedicatedRightTurnLane | 🟢 |
| 33 | 并行右转 | Intersection_ParallelRightTurn | 🟡 |
| 34 | 右转右侧直行非机动车道 | Intersection_RightTurnWithNonMotorLane | 🟢 |

### 车道巡航（5 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 35 | 大曲率弯道巡航 | LaneCruising_SharpCurve | 🟢 |
| 36 | 跟车拥堵巡航 | LaneCruising_CongestedFollowing | 🟢 |
| 37 | 道路限速巡航 | LaneCruising_RoadSpeedLimit | 🟡 |
| 38 | 场景限速巡航 | LaneCruising_SceneSpeedLimit | 🟡 |
| 39 | 路口限速巡航 | LaneCruising_IntersectionSpeedLimit | 🟡 |

### 路口内动态交互（8 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 40 | 路口直行VRU横穿 | IntersectionInteraction_StraightVRUCrossing | 🟢 |
| 41 | 路口直行左转车辆横穿 | IntersectionInteraction_StraightLeftTurnVehicleCrossing | 🟢 |
| 42 | 路口左转VRU横穿 | IntersectionInteraction_LeftTurnVRUCrossing | 🟢 |
| 43 | 路口左转直行车辆横穿 | IntersectionInteraction_LeftTurnStraightVehicleCrossing | 🟢 |
| 44 | 路口右转VRU横穿 | IntersectionInteraction_RightTurnVRUCrossing | 🟢 |
| 45 | 路口右转直行车辆横穿 | IntersectionInteraction_RightTurnStraightVehicleCrossing | 🟢 |
| 46 | 路口直行右转VRU切入 | IntersectionInteraction_StraightRightTurnVRUCutIn | 🟢 |
| 47 | 路口直行右转车辆切入 | IntersectionInteraction_StraightRightTurnVehicleCutIn | 🟢 |

---

## 三、P1 级新增（54 个，📋 需冷启动挖掘）

### 车道贴近（4 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 48 | 邻车道VRU贴近 | LaneApproach_AdjacentVRU | 🟡 |
| 49 | 邻车道车辆贴近 | LaneApproach_AdjacentVehicle | 🟡 |
| 50 | 逆向车道VRU贴近 | LaneApproach_OncomingVRU | 🟢 |
| 51 | 逆向车道车辆贴近 | LaneApproach_OncomingVehicle | 🟢 |

### 红绿灯通行 — 闪灯通行（6 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 52 | 直行绿闪通行 | TrafficLight_StraightGreenFlash | 🟡 |
| 53 | 左转绿闪通行 | TrafficLight_LeftTurnGreenFlash | 🟡 |
| 54 | 右转绿闪通行 | TrafficLight_RightTurnGreenFlash | 🟡 |
| 55 | 直行长黄闪通行 | TrafficLight_StraightYellowFlash | 🟢 |
| 56 | 左转长黄闪通行 | TrafficLight_LeftTurnYellowFlash | 🟡 |
| 57 | 右转长黄闪通行 | TrafficLight_RightTurnYellowFlash | 🟡 |

### 红绿灯通行 — 黑灯通行（3 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 58 | 直行黑灯通行 | TrafficLight_StraightDarkLight | 🟢 |
| 59 | 左转黑灯通行 | TrafficLight_LeftTurnDarkLight | 🟡 |
| 60 | 右转黑灯通行 | TrafficLight_RightTurnDarkLight | 🟢 |

### 红绿灯通行 — 其他红绿灯（3 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 61 | 移动红绿灯通行 | TrafficLight_MobileSignal | 🟢 |
| 62 | 警示灯通行 | TrafficLight_WarningLight | 🟢 |
| 63 | 遮挡红绿灯通行 | TrafficLight_OccludedSignal | 🟢 |

### 车道巡航 — 场景巡航（5 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 64 | 窄空间巡航 | LaneCruising_NarrowSpace | 🟢 |
| 65 | 乡村道路巡航 | LaneCruising_RuralRoad | 🟢 |
| 66 | 减速带巡航 | LaneCruising_SpeedBump | 🟢 |
| 67 | 施工区巡航 | LaneCruising_ConstructionZone | 🟢 |
| 68 | 斑马线巡航 | LaneCruising_ZebraCrossing | 🟢 |

### 车道巡航 — 拥堵巡航（2 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 69 | 联排静止车拥堵巡航 | LaneCruising_StaticVehicleQueueCongestion | 🟢 |
| 70 | 其他拥堵巡航 | LaneCruising_OtherCongestion | 🟢 |

### 车道巡航 — 语义巡航（4 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 71 | 可变车道巡航 | LaneCruising_VariableLane | 🟡 |
| 72 | 公交车道巡航 | LaneCruising_BusLane | 🟢 |
| 73 | 潮汐车道巡航 | LaneCruising_TidalLane | 🟡 |
| 74 | 禁停区巡航 | LaneCruising_NoParkingZone | 🟢 |

### 变道/绕行 — 导航变道（2 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 75 | 短距离连续导航变道 | LaneChange_ShortConsecutiveNav | 🔴 |
| 76 | 拥堵导航变道 | LaneChange_CongestedNav | 🔴 |

### 变道/绕行 — 车道内避让（1 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 77 | 静态障碍物避让 | LaneChange_AvoidStaticObstacle | 🟢 |

### 变道/绕行 — 借道避让（4 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 78 | 慢速VRU避让（借道） | LaneChange_BorrowLaneAvoidSlowVRU | 🟢 |
| 79 | 静态障碍车避让（借道） | LaneChange_BorrowLaneAvoidStaticVehicle | 🟢 |
| 80 | 静态障碍物避让（借道） | LaneChange_BorrowLaneAvoidStaticObstacle | 🟢 |
| 81 | 对向车道障碍车借道避让 | LaneChange_BorrowOncomingLaneAvoidVehicle | 🟢 |

### 变道/绕行 — 跨线绕行（2 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 82 | 联排静止车跨线绕行 | LaneChange_CrossLineBypassStaticVehicles | 🟢 |
| 83 | 联排静态障碍物跨线绕行 | LaneChange_CrossLineBypassStaticObstacles | 🟢 |

### 路口通行（11 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 84 | 并行左转 | Intersection_ParallelLeftTurn | 🟡 |
| 85 | 左转专用道 | Intersection_DedicatedLeftTurnLane | 🟢 |
| 86 | 待转区U-turn | Intersection_WaitingZoneUTurn | 🟡 |
| 87 | 三点U-turn | Intersection_ThreePointUTurn | 🟢 |
| 88 | 直行待转区 | Intersection_StraightWaitingZone | 🟢 |
| 89 | 左转待转区 | Intersection_LeftTurnWaitingZone | 🟢 |
| 90 | 文字型待转区 | Intersection_TextWaitingZone | 🟢 |
| 91 | 组合灯控待转区 | Intersection_CombinedSignalWaitingZone | 🟡 |
| 92 | 图像型待转区 | Intersection_ImageWaitingZone | 🟢 |
| 93 | 单车道小环岛 | Intersection_SingleLaneRoundabout | 🟢 |
| 94 | 多车道环岛 | Intersection_MultiLaneRoundabout | 🟢 |

### 路口内动态交互 — 路口内贴近（7 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 95 | 路口侧后VRU贴近 | IntersectionInteraction_RearSideVRUApproach | 🔴 |
| 96 | 路口并行VRU贴近 | IntersectionInteraction_ParallelVRUApproach | 🟡 |
| 97 | 路口逆行VRU贴近 | IntersectionInteraction_OncomingVRUApproach | 🟢 |
| 98 | 路口侧后车辆贴近 | IntersectionInteraction_RearSideVehicleApproach | 🔴 |
| 99 | 路口并行直行车贴近 | IntersectionInteraction_ParallelStraightVehicleApproach | 🟡 |
| 100 | 路口并行左转车贴近 | IntersectionInteraction_ParallelLeftTurnVehicleApproach | 🟡 |
| 101 | 路口并行右转车贴近 | IntersectionInteraction_ParallelRightTurnVehicleApproach | 🟡 |

---

## 四、P2 级新增（12 个，📋 需冷启动挖掘）

### 变道/绕行 — 效率变道（3 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 102 | 慢速车效率变道 | LaneChange_SlowVehicleEfficiency | 🟡 |
| 103 | 慢速VRU效率变道 | LaneChange_SlowVRUEfficiency | 🟢 |
| 104 | 静止障碍物效率变道 | LaneChange_StaticObstacleEfficiency | 🟢 |

### 变道/绕行 — 抑制变道（3 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 105 | 拥堵排队抑制变道 | LaneChange_CongestedQueueSuppressed | 🔴 |
| 106 | 非机动车道抑制变道 | LaneChange_NonMotorLaneSuppressed | 🔴 |
| 107 | 公交停泊港抑制变道 | LaneChange_BusStopSuppressed | 🔴 |

### 主辅路/分合流（6 个）

| # | 标签（中文） | 英文标签 | FW |
|:---:|---|---|:---:|
| 108 | 车道级分流 | MergeAndDiverge_LaneLevelDiverge | 🟢 |
| 109 | 道路级分流 | MergeAndDiverge_RoadLevelDiverge | 🟢 |
| 110 | 直行合流汇入 | MergeAndDiverge_StraightMerge | 🟢 |
| 111 | 右转merge | MergeAndDiverge_RightTurnMerge | 🟢 |
| 112 | 主路进辅路 | MergeAndDiverge_MainToService | 🟢 |
| 113 | 辅路进主路 | MergeAndDiverge_ServiceToMain | 🟢 |

---

## 五、前司补充（9 个，📋 需冷启动挖掘）

| # | 一级类 | 标签（中文） | 英文标签 | FW | 前司标签来源 | 补充理由 |
|:---:|---|---|---|:---:|---|---|
| 114 | 车道动态交互 | 前车切出 | DynamicInteraction_LeadVehicleCutOut | 🟢 | cutout_vehicle (41) | 有"切入"无"切出"，对称场景 |
| 115 | 车道动态交互 | 豁口切入 | DynamicInteraction_GapOpeningCutIn | 🟢 | opening_cut_in (82) | 豁口处车辆/VRU突然出现 |
| 116 | 车道动态交互 | 前车急刹 | DynamicInteraction_LeadVehicleSuddenBrake | 🟡 | front_vehicle_brake_suddenly (23) | P00 树中曾有此节点 |
| 117 | 车道动态交互 | 静态障碍物反应(SOD) | DynamicInteraction_StaticObjectReaction | 🟡 | SOD_react (19) | SOD=静态OD，区别于动态目标检测 |
| 118 | 车道巡航 | 跟VRU稳态行驶 | LaneCruising_FollowingVRU | 🟢 | following_VRU_steady (31) | 跟 VRU 低速行驶，与巡航不同 |
| 119 | 车道巡航 | 稳态跟车 | LaneCruising_SteadyFollowing | 🟢 | following_steady (24) | 有"拥堵跟车"无"正常跟车" |
| 120 | 发车/停车 | 跟车停车 | StartStop_FollowingStop | 🟢 | following_stop (38) | 非信号灯/非靠边的跟车停车 |
| 121 | 变道/绕行 | 超车 | LaneChange_Overtake | 🟢 | overtake (33) | 完整"变道→超越→回道"复合动作 |
| 122 | 路口通行 | T型无信号路口汇入 | Intersection_TJunctionUnprotectedMerge | 🟢 | T_no_tld_merge_to_main | 无信号 T 口汇入，现有标签不覆盖 |

---

## 六、按一级类汇总

| 一级类 | 英文前缀 | P00 | P0新 | P1 | P2 | 补充 | 总计 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 车道动态交互 | `DynamicInteraction_` | 3 | 5 | 0 | 0 | **4** | **12** |
| 红绿灯通行 | `TrafficLight_` | 2 | 3 | 12 | 0 | 0 | **17** |
| 发车/停车 | `StartStop_` | 2 | 4 | 0 | 0 | **1** | **7** |
| 路口通行 | `Intersection_` | 1 | 10 | 11 | 0 | **1** | **23** |
| 车道巡航 | `LaneCruising_` | 1 | 5 | 11 | 0 | **2** | **19** |
| 变道/绕行 | `LaneChange_` | 3 | 0 | 9 | 6 | **1** | **19** |
| 路口内动态交互 | `IntersectionInteraction_` | 0 | 8 | 7 | 0 | 0 | **15** |
| 车道贴近 | `LaneApproach_` | 0 | 0 | 4 | 0 | 0 | **4** |
| 主辅路/分合流 | `MergeAndDiverge_` | 0 | 0 | 0 | 6 | 0 | **6** |
| **合计** | | **12** | **35** | **54** | **12** | **9** | **122** |

---

## 七、FW 可行性分布

| | 🟢 好做 | 🟡 不容易 | 🔴 做不了 | 合计 |
|---|:---:|:---:|:---:|:---:|
| P00 | 8 | 3 | 1 | 12 |
| P0 新增 | 27 | 8 | 0 | 35 |
| P1 | 33 | 17 | 4 | 54 |
| P2 | 8 | 1 | 3 | 12 |
| 补充 | 7 | 2 | 0 | 9 |
| **合计** | **83 (68%)** | **31 (25%)** | **8 (7%)** | **122** |

### 🔴 FW 做不了的 8 个标签（需融合其他数据源）

| # | 标签 | 优先级 | 缺什么 |
|:---:|---|:---:|---|
| 11 | 前方路口导航变道 | P00 | 导航意图 |
| 75 | 短距离连续导航变道 | P1 | 导航意图 |
| 76 | 拥堵导航变道 | P1 | 导航意图 |
| 95 | 路口侧后VRU贴近 | P1 | 侧后视角 |
| 98 | 路口侧后车辆贴近 | P1 | 侧后视角 |
| 105 | 拥堵排队抑制变道 | P2 | 规划决策信息 |
| 106 | 非机动车道抑制变道 | P2 | 规划决策信息 |
| 107 | 公交停泊港抑制变道 | P2 | 规划决策信息 |

---

*分析日期：2026-03-27*
