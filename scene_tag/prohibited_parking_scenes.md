# 严禁靠边场景 — 文生视频实验表

> 来源：严禁靠边场景分类表（5大类 23个子场景）
> 生成时间：2026-04-08
> 用途：使用 Seedance 2.0 生成合成驾驶场景视频

## 场景总览

| 序号 | 大类 | Tag 名称 | 场景名称 | 场景描述 | 风险说明 |
|------|------|----------|----------|----------|----------|
| 1 | 1-明确交通法规禁止区域 | `PP_NoStopSignZone` | 禁停标志标线路段 | 设有禁止停车标志/标线的路段（如黄实线、网格线、禁停标志下） | — |
| 2 | 1-明确交通法规禁止区域 | `PP_BusLaneAndStation` | 公交专用车道及站台区域 | 公交专用车道内及站台区域（除非明确允许且非运营时段） | — |
| 3 | 1-明确交通法规禁止区域 | `PP_FireLaneEntrance` | 消防通道出入口 | 消防通道出入口及两侧规定范围内 | — |
| 4 | 1-明确交通法规禁止区域 | `PP_CrosswalkZone` | 人行横道及两端5米 | 人行横道（斑马线）及两端5米范围内 | — |
| 5 | 1-明确交通法规禁止区域 | `PP_IntersectionSpecialRoad` | 交叉路口及特殊路段50米内 | 交叉路口及铁路道口、急弯路、宽度不足4米的窄路、桥梁、陡坡、隧道以及距离上述地点50米以内的路段 | — |
| 6 | 2-高风险安全隐患区域 | `PP_ExpresswayMainRoad` | 高速公路及快速路主路 | 高速公路、城市快速路主路（应急车道仅限紧急故障使用） | 极易引发高速追尾、二次事故；威胁自身及他车安全 |
| 7 | 2-高风险安全隐患区域 | `PP_InsideTunnel` | 隧道内 | 隧道内（除非紧急故障） | 极易引发高速追尾、二次事故 |
| 8 | 2-高风险安全隐患区域 | `PP_LongDownhill` | 长下坡路段 | 长下坡路段 | 极易引发高速追尾、二次事故 |
| 9 | 2-高风险安全隐患区域 | `PP_PoorVisibilityZone` | 视距不良区域 | 视距不良区域（如急弯后、坡顶、大型障碍物遮挡视线处） | 极易引发高速追尾、二次事故 |
| 10 | 2-高风险安全隐患区域 | `PP_NonMotorLaneSidewalk` | 非机动车道或人行道 | 非机动车道/人行道（除非当地法规明确允许且车辆尺寸符合） | 极易引发高速追尾、二次事故 |
| 11 | 3-影响公共安全与秩序区域 | `PP_FireHydrantZone` | 消防栓周边5米 | 消防栓、消火栓周边规定范围内（通常5米） | — |
| 12 | 3-影响公共安全与秩序区域 | `PP_HospitalEntrance` | 医院急救中心出入口 | 医院、急救中心主要出入口及应急通道 | — |
| 13 | 3-影响公共安全与秩序区域 | `PP_SchoolEntrance` | 学校幼儿园出入口 | 学校、幼儿园上学/放学时段主要出入口附近 | — |
| 14 | 3-影响公共安全与秩序区域 | `PP_EventVenueEntrance` | 大型活动场所出入口 | 大型活动场所/场馆出入口及周边管制区域（活动期间） | — |
| 15 | 3-影响公共安全与秩序区域 | `PP_MetroTransitHub` | 地铁站公交枢纽出入口 | 地铁站、大型公交枢纽主要出入口及出租车/网约车专用上客区 | — |
| 16 | 3-影响公共安全与秩序区域 | `PP_ResidentialGateCrowded` | 小区门口多车聚集 | 多台车集中在小区门口 | — |
| 17 | 4-特定功能受限区域 | `PP_MilitaryGovZone` | 军事及重要机关周边 | 军事管理区、重要国家机关、外交机构周边敏感区域 | 涉及国家安全；车辆可能失联或无法精确定位 |
| 18 | 4-特定功能受限区域 | `PP_GPSUnstableZone` | 信号屏蔽或GPS不稳定区域 | 信号屏蔽或GPS定位严重不稳定区域（如地下设施周边、强电磁干扰区） | 车辆可能失联或无法精确定位，导致失控风险 |
| 19 | 5-易引发拥堵或冲突区域 | `PP_SingleLaneMainRoad` | 主干道仅剩一条车道 | 主干道仅剩一条车道的路段 | — |
| 20 | 5-易引发拥堵或冲突区域 | `PP_HighTrafficIntersection` | 交通流量极大路口 | 交通流量极大的路口附近 | — |
| 21 | 5-易引发拥堵或冲突区域 | `PP_TaxiRideshareWaiting` | 出租车网约车密集候客区 | 出租车/网约车密集候客区 | — |
| 22 | 5-易引发拥堵或冲突区域 | `PP_LoadingUnloadingZone` | 装卸货专用车位 | 装卸货专用车位（除非明确允许无人车使用且未被占用） | — |
| 23 | 5-易引发拥堵或冲突区域 | `PP_IndustrialParkEntrance` | 园区出入口 | 园区出入口（工业园区、科技园区等出入口）⚠️重点关注 | — |

## 使用方法

### 批量生成全部场景视频

```bash
python scene_tag/16_generate_video.py \
    --api_key "YOUR_ARK_API_KEY" \
    --batch_tags "PP_NoStopSignZone,PP_BusLaneAndStation,PP_FireLaneEntrance,PP_CrosswalkZone,PP_IntersectionSpecialRoad,PP_ExpresswayMainRoad,PP_InsideTunnel,PP_LongDownhill,PP_PoorVisibilityZone,PP_NonMotorLaneSidewalk,PP_FireHydrantZone,PP_HospitalEntrance,PP_SchoolEntrance,PP_EventVenueEntrance,PP_MetroTransitHub,PP_ResidentialGateCrowded,PP_MilitaryGovZone,PP_GPSUnstableZone,PP_SingleLaneMainRoad,PP_HighTrafficIntersection,PP_TaxiRideshareWaiting,PP_LoadingUnloadingZone,PP_IndustrialParkEntrance" \
    --count 1
```

### 生成单个场景视频

```bash
python scene_tag/16_generate_video.py \
    --api_key "YOUR_ARK_API_KEY" \
    --tag PP_IndustrialParkEntrance \
    --count 3
```

## 大类统计

| 大类序号 | 大类名称 | 子场景数 |
|----------|----------|----------|
| 1 | 明确交通法规禁止区域 | 5 |
| 2 | 高风险安全隐患区域 | 5 |
| 3 | 影响公共安全与秩序区域 | 6 |
| 4 | 特定功能受限区域 | 2 |
| 5 | 易引发拥堵或冲突区域 | 5 |
| **合计** | | **23** |
