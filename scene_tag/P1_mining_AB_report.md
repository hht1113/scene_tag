# P1 标签挖掘方案 A/B 对比实验报告

## 4.0 SFT 模型评估

| 项目 | 说明 |
|---|---|
| 模型名称 | 30B全参-multilabel_v3数据集 |
| 基座模型 | Qwen3-VL-30B-A3B-Instruct |
| 模型路径 | `/mnt/pfs/houhaotian/saves/Qwen3-VL-30B-A3B-Instruct/full/sft_segment_multilabel_v3_8gpu/checkpoint-282` |
| 训练数据 | qwen3_sft_train_segment_multilabel_v3.json（3003 条，12 类+else，均衡采样） |
| Prompt 版本 | system_prompt_v3（新增消歧规则：UTurn/LeftTurn 互斥、方向判断、VRU 严格触发） |
| 训练参数 | 3 epochs, lr=1e-5, batch=32 (1×4×8 卡), DeepSpeed ZeRO-3, 全参微调 |
| 训练耗时 | ~8.7 小时（31206 秒） |
| **mAP** | **0.6706**（↑ 0.0486，旧版 0.6220） |
| Micro P / R / F1 | 0.6733 / 0.6733 / 0.6733 |

### Precision ≥ 70% 的标签（5 个）

| 标签 | Precision | Recall | F1 | AP |
|---|:---:|:---:|:---:|:---:|
| StartStop_ParkRoadside | **88.24%** | 57.69% | 69.77% | 0.5769 |
| Intersection_StandardUTurn | **86.67%** | 83.87% | 85.25% | 0.8387 |
| StartStop_StartFromMainRoad | **85.71%** | 44.44% | 58.54% | 0.4444 |
| DynamicInteraction_VehicleInLaneCrossing | **81.48%** | 73.33% | 77.19% | 0.7333 |
| LaneChange_AvoidSlowVRU | **80.00%** | 53.33% | 64.00% | 0.5333 |

### Precision < 70% 的标签（5 个）

| 标签 | Precision | Recall | F1 | AP |
|---|:---:|:---:|:---:|:---:|
| DynamicInteraction_StandardVehicleCutIn | 63.16% | 80.00% | 70.59% | 0.8000 |
| TrafficLight_StraightStopOrGo | 60.98% | 69.44% | 64.94% | 0.6944 |
| DynamicInteraction_VRUInLaneCrossing | 57.14% | 57.14% | 57.14% | 0.5714 |
| LaneChange_AvoidStaticVehicle | 56.82% | 83.33% | 67.57% | 0.8333 |
| TrafficLight_LeftTurnStopOrGo | 50.00% | 68.00% | 57.63% | 0.6800 |

> 与旧版模型（30B全参-动态采样数据集, checkpoint-226, mAP=0.6220）相比，v3 模型 mAP **提升至 0.6706（+7.8%）**。v3 使用了更严格的 system_prompt（新增消歧规则），在分类难度增大的情况下仍取得了效果提升。

---

## 4.1 挖掘池

| 项目 | 说明 |
|---|---|
| 数据源 | 验证集 300 条 20 秒视频（2FPS，40帧/条） |
| 视频来源 | 路口场景行车视频（前视摄像头） |
| 帧采样 | 每条视频均匀抽取 12 帧（每 ~1.67 秒一帧） |
| 推理模型 | Qwen3-VL-30B-A3B-Instruct（multilabel_v3 SFT, checkpoint-282） |
| 目标标签 | 12 类驾驶行为 + else（system_prompt_v3，含消歧规则） |

> 本次实验为方案验证，使用 300 条验证集视频做 A/B 对比。正式挖掘将使用路口视频挖掘池（约 10 万+ 条 20 秒视频）。

---

## 4.2 挖掘速度对比

| 指标 | 方案 A（Baseline 单次推理） | 方案 B（P1 分层+投票） |
|---|:---:|:---:|
| 推理策略 | 一次性 36 标签分类 | 事实抽取→粗分类→细分，×3 轮采样 |
| Temperature | 0（确定性输出） | 0.4（3 次独立采样） |
| 每条 VLM 调用数 | **1 次** | **6 次**（2步 × 3轮） |
| 单条耗时 | ~2 秒 | ~27 秒 |
| 300 条总耗时 | ~10 分钟 | ~136 分钟（2.3 小时） |
| 推理成本比 | **1×** | **6×** |

> 方案 B 成本约为方案 A 的 6 倍，但可通过 8B 模型做第一步（事实+粗分类）、235B 只做细分来降至约 4 倍。

---

## 4.3 挖掘效果对比

### 4.3.1 整体结果

| 指标 | 方案 A（Baseline） | 方案 B（P1） | 对比 |
|---|:---:|:---:|:---:|
| 已审标签类别数 | 10 类 | 13 类 | B 多审 3 类 |
| 已审样本数 | 118 条 | 147 条 | — |
| 正确样本数 | 17 条 | 36 条 | B 多 **112%** |
| **整体 Precision** | **14.4%** | **24.5%** | **↑ 10.1pp** |
| 标签覆盖（总） | 16 类 | 21 类 | B 多覆盖 5 类 |
| 头部标签集中度 | 63%（前2类占比） | 46%（前2类占比） | B 分布更均匀 |

### 4.3.2 逐标签 Precision 对比（双方都审核的标签）

| 标签 | 方案 A Precision | 方案 B Precision | 变化 |
|---|:---:|:---:|:---:|
| IntersectionInteraction_StraightVRUCrossing | 38.5% | **81.8%** | **↑ 43.3pp** |
| LaneCruising_RoadSpeedLimit | 0.0% | **40.0%** | **↑ 40.0pp** |
| LaneCruising_Congestion | 18.8% | **40.0%** | **↑ 21.2pp** |
| Intersection_ProtectedStraight | 6.7% | **18.8%** | **↑ 12.1pp** |
| LaneCruising_SharpCurve | 0.0% | 7.7% | ↑ 7.7pp |
| LaneCruising_SceneSpeedLimit | 50.0% | **55.0%** | ↑ 5.0pp |
| Intersection_UnprotectedStraight | 0.0% | 0.0% | — |
| Intersection_ProtectedRightTurn | 14.3% | 0.0% | ↓ 14.3pp |

> **8 个共同标签中，6 个提升、1 个持平、1 个下降。**

### 4.3.3 方案 B 独有的已审标签

| 标签 | 数量 | Precision | 说明 |
|---|:---:|:---:|---|
| LaneCruising_IntersectionSpeedLimit | 6 | 16.7% | 方案 A 未召回 |
| TrafficLight_RightTurnStopOrGo | 13 | 15.4% | 方案 A 未审核（仅 5 条） |
| Intersection_RightTurnLane | 11 | 9.1% | 方案 A 未召回 |
| TrafficLight_WaitingAreaStopOrGo | 7 | 0.0% | 方案 A 未审核（仅 4 条） |
| DynamicInteraction_StartingVehicleCutIn | 7 | 0.0% | 方案 A 未召回 |

### 4.3.4 方案 A 的核心问题

1. **"万金油"标签偏好**：`UnprotectedStraight`（96 条）和 `SharpCurve`（93 条）合占 63%，审核后 Precision 均为 **0%** —— 模型面对 36 标签走捷径，大量样本被错误归类到这两个高频标签
2. **有效样本极少**：300 条中仅 17 条正确（5.7%），人工筛选比约 **18:1**
3. **标签覆盖窄**：仅覆盖 16 类，36 类标签中 20 类完全未召回

### 4.3.5 方案 B 的改进总结

| 改进点 | 具体表现 |
|---|---|
| **Precision 提升** | 整体 14.4% → 24.5%（+70%），最佳标签 `StraightVRUCrossing` 从 38.5% → 81.8% |
| **标签覆盖更广** | 16 类 → 21 类，独有召回 5 个新标签 |
| **分布更均匀** | 消除了方案 A 中 63% 集中在 2 个标签的问题 |
| **可解释性** | 每条结果附带结构化事实（facts），人工审核时可快速判断合理性 |
| **质量分级** | 投票机制自带 high/medium/low 置信度，可按等级优先使用 |

### 4.3.6 待改进

| 问题 | 说明 |
|---|---|
| 整体 Precision 仍偏低（24.5%） | 需要优化 prompt，特别是容易混淆的标签（如 Protected/Unprotected 的区分） |
| 部分标签 Precision 为 0% | `UnprotectedStraight`、`ProtectedRightTurn`、`StartingVehicleCutIn` 等需重点优化 |
| 推理成本 ×6 | 可通过小模型做第一步、大模型做细分来降低成本 |
| 挖掘池规模不足 | 本次仅 300 条验证，正式挖掘需扩大到 5000-10000 条 |
