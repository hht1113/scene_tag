# 驾驶行为标注 审核+修改 平台 使用文档

> 最后更新：2026-04-21

## 1. 平台概述

`13_review.py` 是一个纯 Python 单文件 Web 应用（无第三方依赖），提供驾驶行为标注数据的审核与修改功能。

### 核心功能
- **审核模式**：逐视频/逐段审核标注结果，标记 正确 / 错误 / 待定
- **修改模式**：修改标签、调整时间、增删标注段
- **统计面板**：实时显示审核进度、各标签准确率（Precision）、预估可挖掘数量
- **导出功能**：导出审核 CSV / 训练数据 JSON

---

## 2. 启动方式

```bash
# 基本启动（默认端口 9000）
python 13_review.py

# 指定端口
python 13_review.py --port 8080

# 指定监听地址
python 13_review.py --host 0.0.0.0 --port 9000
```

启动后在浏览器打开 `http://<服务器IP>:<端口>` 即可。

如需通过 cloudflare tunnel 暴露外网访问：
```bash
cloudflared tunnel --url http://localhost:9000
```

---

## 3. JSON 数据格式要求

平台支持 **三种输入格式**，加载时自动检测和转换：

### 格式一：标准 segments 格式（旧格式）

```json
[
  {
    "video_path": "/mnt/pfs/houhaotian/web_videos_480p/.../slice_20_40.mp4",
    "segments": [
      {
        "label": "LaneCruising_Straight",
        "major_category": "05_车道巡航",
        "start": 0,
        "end": 15,
        "confidence": 85,
        "label_cn": "直线巡航",
        "description": "描述文字"
      }
    ]
  }
]
```

### 格式二：structured_label 格式（模型推理输出）

这是 `round_b_results_235b_eval100` 系列文件的格式，也是模型 `raw_generation` 解析后的结果：

```json
[
  {
    "sample_index": null,
    "video_path": "/mnt/pfs/houhaotian/web_videos_480p/.../slice_20_40.mp4",
    "label_en": null,
    "all_labels": null,
    "structured_label": {
      "tags": [
        {
          "major_category": "05_LaneCruising",
          "sub_category": "直线巡航",
          "sub_category_en": "LaneCruising_Straight",
          "is_other_subcategory": false,
          "time_evidence": [
            {
              "start": 0.0,
              "end": 20.0,
              "description": "自车在直行车道上稳定行驶"
            }
          ],
          "visual_cues": ["清晰车道线", "前方无障碍物"],
          "confidence": {
            "major": 0.95,
            "sub": 0.9
          },
          "risk_level": "low",
          "traffic_rule_keywords": ["直行", "巡航"]
        }
      ]
    },
    "raw_generation": "```json\n{...}\n```"
  }
]
```

### 格式三：顶层 tags 格式（v8 标注规则输出）

这是 `round_b_label_results_merged_v8_annotation` 系列文件的格式：

```json
[
  {
    "video_path": "/mnt/pfs/houhaotian/web_videos_480p/.../slice_20_40.mp4",
    "tags": [
      {
        "major_category": "05_LaneCruising",
        "sub_category": "场景限速巡航",
        "sub_category_en": "LaneCruising_SceneSpeedLimit",
        "time_evidence": [
          {"start": 0.0, "end": 20.0, "description": "..."}
        ],
        "confidence": {"major": 0.9, "sub": 0.85}
      }
    ]
  }
]
```

---

## 4. 字段映射（JSON → 前端显示）

| 前端显示位置 | 对应 JSON 字段 | 说明 |
|-------------|--------------|------|
| 视频播放 | `video_path` | 服务器本地绝对路径，平台通过 `/video/` API 读取并流式传输 |
| 标签名（中文） | `sub_category_en` → 内置 `LABEL_CN` 映射表 | 英文标签 key 自动映射为中文 |
| 标签名（英文） | `sub_category_en` 或 `label` | 显示在中文名右侧 |
| 时间段 | `time_evidence[].start` / `time_evidence[].end` | 秒数，范围 0-20 |
| 置信度 | `confidence.sub`（优先）或 `confidence.major` | 转换为百分比显示 |
| 时间线颜色条 | 由 `major_category` 大类决定颜色 | 7 大类对应 7 种颜色 |
| 描述文字 | `time_evidence[].description` | 暂不在前端显示，但保存在数据中 |

### 大类颜色映射

| 大类编号 | 大类名称 | 颜色 |
|---------|---------|------|
| 01 | 动态交互 | 🟠 橙色 `#e67e22` |
| 02 | 红绿灯 | 🔴 红色 `#e74c3c` |
| 03 | 起停 | 🟣 紫色 `#9b59b6` |
| 04 | 路口通行 | 🟢 青色 `#1abc9c` |
| 05 | 车道巡航 | ⚪ 灰色 `#95a5a6` |
| 06 | 变道 | 🔵 蓝色 `#3498db` |
| 07 | 路口交互 | 🟢 绿色 `#2ecc71` |

### major_category 必须满足的过滤条件

`major_category` 字段必须以 `01_` ~ `07_` 开头，否则该标注段会被过滤掉不显示。以下格式均合法：
- `01_动态交互` — 中文格式
- `01_DynamicInteraction` — 英文格式
- 只要以 `01_`、`02_`、`03_`、`04_`、`05_`、`06_`、`07_` 开头即可

### sub_category_en 必须在已知标签列表中

`sub_category_en`（或 `label`）的值必须匹配平台内置的标签英文名列表，否则不会显示中文名。完整列表见平台底部的"可用标签参考"区域。

---

## 5. 操作指南

### 5.1 加载数据
1. 打开平台首页
2. 在输入框粘贴 JSON 文件的服务器端绝对路径
3. 点击"加载数据"
4. 历史加载路径会自动保存（最多 10 条）

### 5.2 审核流程
1. 默认进入**审核模式**
2. 播放视频，对照标注段判断是否正确
3. 对每个标注段点击 ✓正确 / ?待定 / ✗错误
4. 快捷键：`A`=正确 `S`=待定 `D`=错误 `Q`=全部正确
5. `←→` 切换上/下一个视频，`Space` 播放/暂停
6. 审核结果自动保存到 `_review.json` 文件

### 5.3 修改流程
1. 点击"修改"切换到修改模式（或按 `M`）
2. 选中标注段后可修改标签、调整时间
3. 点击"+ 添加标注段"可新增
4. 点击"删除"可删除错误段
5. 修改会记录在审核文件中，不改动原始标注文件

### 5.4 导出
- **导出 CSV**：全部标注段导出为 CSV 文件
- **导出训练数据**：仅导出标记为"正确"的段，按标签分类保存为 JSON

---

## 6. 审核结果文件格式

审核结果保存为 `<原始文件名>_review.json`，结构如下：

```json
{
  "<video_path>": {
    "segments": {
      "0": "correct",
      "1": "wrong",
      "2": "unsure"
    },
    "comment": "备注文字",
    "modifications": {
      "1": {"label": "NewLabel", "start": 2, "end": 10}
    },
    "added_segments": [
      {"label": "LaneCruising_Straight", "start": 5, "end": 15}
    ],
    "deleted_segments": [3]
  }
}
```

---

## 7. 常见问题

### Q: 加载后标注段不显示？
- 检查 `major_category` 是否以 `01_`~`07_` 开头
- 检查 `sub_category_en`（或 `label`）是否在平台支持的标签列表中
- JSON 格式是否为数组 `[{...}, {...}]`

### Q: 视频无法播放？
- 确认 `video_path` 是服务器端的绝对路径
- 确认文件存在且可读
- 视频格式需为 mp4

### Q: 如何分配多人审核？
- 将大文件按索引切分为多个 part 文件（如 `_part01.json` ~ `_part13.json`）
- 每人负责不同 part 文件
- 审核结果独立保存为各自的 `_review.json`

---

## 8. 支持的完整标签列表

### 01_动态交互 (12个)
| 英文标签 | 中文名称 |
|---------|---------|
| DynamicInteraction_VRUInLaneCrossing | VRU车道内横穿 |
| DynamicInteraction_VehicleInLaneCrossing | 车辆车道内横穿 |
| DynamicInteraction_StandardVehicleCutIn | 标准车辆切入 |
| DynamicInteraction_EmergencyVehicleCutIn | 紧急车辆切入 |
| DynamicInteraction_StartupVehicleCutIn | 起步车辆切入 |
| DynamicInteraction_ConsecutiveLaneChangeCutIn | 连续变道切入 |
| DynamicInteraction_SlowVRUCutIn | 慢速VRU切入 |
| DynamicInteraction_EmergencyVRUCutIn | 紧急VRU切入 |
| DynamicInteraction_LeadVehicleCutOut | 前车切出 |
| DynamicInteraction_GapOpeningCutIn | 空隙切入 |
| DynamicInteraction_LeadVehicleSuddenBrake | 前车急刹 |
| DynamicInteraction_StaticObjectReaction | 静态障碍物反应 |

### 02_红绿灯 (17个)
| 英文标签 | 中文名称 |
|---------|---------|
| TrafficLight_StraightStopOrGo | 直行红绿灯起停 |
| TrafficLight_LeftTurnStopOrGo | 左转红绿灯起停 |
| TrafficLight_RightTurnStopOrGo | 右转红绿灯起停 |
| TrafficLight_UTurnStopOrGo | 掉头红绿灯起停 |
| TrafficLight_WaitingZoneStopOrGo | 待转区红绿灯起停 |
| TrafficLight_StraightGreenFlash | 直行绿闪通行 |
| TrafficLight_LeftTurnGreenFlash | 左转绿闪通行 |
| TrafficLight_RightTurnGreenFlash | 右转绿闪通行 |
| TrafficLight_StraightYellowFlash | 直行长黄闪通行 |
| TrafficLight_LeftTurnYellowFlash | 左转长黄闪通行 |
| TrafficLight_RightTurnYellowFlash | 右转长黄闪通行 |
| TrafficLight_StraightDarkLight | 直行黑灯通行 |
| TrafficLight_LeftTurnDarkLight | 左转黑灯通行 |
| TrafficLight_RightTurnDarkLight | 右转黑灯通行 |
| TrafficLight_MobileSignal | 移动红绿灯通行 |
| TrafficLight_WarningLight | 警示灯通行 |
| TrafficLight_OccludedSignal | 遮挡红绿灯通行 |

### 03_起停 (7个)
| 英文标签 | 中文名称 |
|---------|---------|
| StartStop_StartFromMainRoad | 主路发车起步 |
| StartStop_ParkRoadside | 靠边停车 |
| StartStop_StartFromNonMotorLane | 非机动车道发车 |
| StartStop_EmergencyStopOnMainRoad | 主路紧急停车 |
| StartStop_StopAtStation | 站点停车 |
| StartStop_ParkInStructuredSpot | 结构化车位泊入 |
| StartStop_FollowingStop | 跟车停车 |

### 04_路口通行 (21个)
| 英文标签 | 中文名称 |
|---------|---------|
| Intersection_ProtectedLeftTurn | 有保护左转 |
| Intersection_UnprotectedLeftTurn | 无保护左转 |
| Intersection_ParallelLeftTurn | 并行左转 |
| Intersection_DedicatedLeftTurnLane | 左转专用道 |
| Intersection_ProtectedStraight | 有保护直行 |
| Intersection_UnprotectedStraight | 无信号路口直行 |
| Intersection_MisalignedStraight | 错位路口直行 |
| Intersection_CongestedStraight | 拥堵路口直行 |
| Intersection_ProtectedRightTurn | 有保护右转 |
| Intersection_DedicatedRightTurnLane | 右转专用道 |
| Intersection_ParallelRightTurn | 并行右转 |
| Intersection_RightTurnWithNonMotorLane | 右转伴非机动车道 |
| Intersection_StandardUTurn | 标准掉头 |
| Intersection_WaitingZoneUTurn | 待转区掉头 |
| Intersection_ThreePointUTurn | 三点掉头 |
| Intersection_StraightWaitingZone | 直行待转区 |
| Intersection_LeftTurnWaitingZone | 左转待转区 |
| Intersection_TextWaitingZone | 文字待转区 |
| Intersection_CombinedSignalWaitingZone | 组合灯控待转区 |
| Intersection_ImageWaitingZone | 图标待转区 |
| Intersection_SingleLaneRoundabout | 单车道环岛 |
| Intersection_MultiLaneRoundabout | 多车道环岛 |
| Intersection_TJunctionUnprotectedMerge | T型无信号汇入 |

### 05_车道巡航 (19个)
| 英文标签 | 中文名称 |
|---------|---------|
| LaneCruising_Straight | 直线巡航 |
| LaneCruising_SharpCurve | 大曲率弯道巡航 |
| LaneCruising_NarrowSpace | 窄空间巡航 |
| LaneCruising_RuralRoad | 乡村道路巡航 |
| LaneCruising_SpeedBump | 减速带巡航 |
| LaneCruising_ConstructionZone | 施工区巡航 |
| LaneCruising_ZebraCrossing | 斑马线巡航 |
| LaneCruising_CongestedFollowing | 拥堵跟车巡航 |
| LaneCruising_StaticVehicleQueueCongestion | 联排静止车拥堵巡航 |
| LaneCruising_OtherCongestion | 其他拥堵巡航 |
| LaneCruising_RoadSpeedLimit | 道路限速巡航 |
| LaneCruising_SceneSpeedLimit | 场景限速巡航 |
| LaneCruising_IntersectionSpeedLimit | 路口限速巡航 |
| LaneCruising_VariableLane | 可变车道巡航 |
| LaneCruising_BusLane | 公交专用道巡航 |
| LaneCruising_TidalLane | 潮汐车道巡航 |
| LaneCruising_NoParkingZone | 禁停区巡航 |
| LaneCruising_FollowingVRU | 跟随VRU行驶 |
| LaneCruising_SteadyFollowing | 稳态跟车巡航 |

### 06_变道 (19个)
| 英文标签 | 中文名称 |
|---------|---------|
| LaneChange_NavForIntersection | 路口导航变道 |
| LaneChange_AvoidSlowVRU | 避让慢速VRU变道 |
| LaneChange_AvoidStaticVehicle | 避让静止车辆变道 |
| LaneChange_AvoidStaticObstacle | 避让静态障碍变道 |
| LaneChange_BorrowLaneAvoidSlowVRU | 借道避慢速VRU |
| LaneChange_BorrowLaneAvoidStaticVehicle | 借道避静止车辆 |
| LaneChange_BorrowLaneAvoidStaticObstacle | 借道避静态障碍 |
| LaneChange_BorrowOncomingLaneAvoidVehicle | 借对向车道避车 |
| LaneChange_CrossLineBypassStaticVehicles | 跨线绕行静止车辆 |
| LaneChange_CrossLineBypassStaticObstacles | 跨线绕行静态障碍 |
| LaneChange_ShortConsecutiveNav | 短距离连续导航变道 |
| LaneChange_CongestedNav | 拥堵导航变道 |
| LaneChange_SlowVehicleEfficiency | 慢速车效率变道 |
| LaneChange_SlowVRUEfficiency | 慢速VRU效率变道 |
| LaneChange_StaticObstacleEfficiency | 静态障碍效率变道 |
| LaneChange_CongestedQueueSuppressed | 拥堵排队抑制变道 |
| LaneChange_NonMotorLaneSuppressed | 非机动车道抑制变道 |
| LaneChange_BusStopSuppressed | 公交站抑制变道 |
| LaneChange_Overtake | 超车 |

### 07_路口交互 (15个)
| 英文标签 | 中文名称 |
|---------|---------|
| IntersectionInteraction_StraightVRUCrossing | 直行VRU横穿 |
| IntersectionInteraction_StraightLeftTurnVehicleCrossing | 直行左转车辆横穿 |
| IntersectionInteraction_LeftTurnVRUCrossing | 左转VRU横穿 |
| IntersectionInteraction_LeftTurnStraightVehicleCrossing | 左转直行车辆横穿 |
| IntersectionInteraction_RightTurnVRUCrossing | 右转VRU横穿 |
| IntersectionInteraction_RightTurnStraightVehicleCrossing | 右转直行车辆横穿 |
| IntersectionInteraction_StraightRightTurnVRUCutIn | 直行右转VRU切入 |
| IntersectionInteraction_StraightRightTurnVehicleCutIn | 直行右转车辆切入 |
| IntersectionInteraction_RearSideVRUApproach | 侧后VRU贴近 |
| IntersectionInteraction_ParallelVRUApproach | 并行VRU贴近 |
| IntersectionInteraction_OncomingVRUApproach | 对向VRU贴近 |
| IntersectionInteraction_RearSideVehicleApproach | 侧后车辆贴近 |
| IntersectionInteraction_ParallelStraightVehicleApproach | 并行直行车贴近 |
| IntersectionInteraction_ParallelLeftTurnVehicleApproach | 并行左转车贴近 |
| IntersectionInteraction_ParallelRightTurnVehicleApproach | 并行右转车贴近 |

---

## 9. 实验结果可视化分析用法

要用本平台可视化分析实验结果，需将模型推理结果转换为上述任一格式。

### 快速对接：模型推理输出 → 平台输入

如果模型输出包含 `structured_label` 或 `raw_generation` 字段，平台会自动解析，无需额外转换。

关键要求：
1. 最外层必须是 JSON 数组 `[{...}, {...}]`
2. 每条必须有 `video_path` 字段
3. 标注数据放在 `structured_label.tags`、`tags`、或 `segments` 中（三选一）
4. `major_category` 必须以 `01_` ~ `07_` 开头
5. `sub_category_en` 或 `label` 必须匹配已知标签英文名

### 数据转换脚本示例

```python
import json

def convert_model_output_to_review_format(input_path, output_path):
    """将模型推理输出转换为审核平台可用格式"""
    with open(input_path, 'r') as f:
        data = json.load(f)

    result = []
    for item in data:
        entry = {
            "video_path": item["video_path"],
            "structured_label": item.get("structured_label") or {"tags": item.get("tags", [])}
        }
        result.append(entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"转换完成: {len(result)} 条 → {output_path}")
```

---

## 10. 文件组织建议

```
project/
├── 13_review.py                    # 审核平台主程序
├── review_platform_usage.md        # 本使用文档
├── data/
│   ├── round_b_results_*.json      # 模型推理结果（输入）
│   ├── *_annotation_*.json         # 标注规则输出（输入）
│   └── *_review.json               # 审核结果（自动生成）
```
