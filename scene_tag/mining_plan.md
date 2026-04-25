# 标签挖掘计划

> 生成日期：2026-03-27（v2 修正推理耗时估算）

---

## 一、挖掘池分析

| 项目 | 数值 |
|---|---|
| 挖掘池路径 | `/mnt/pfs/houhaotian/junction_videos_segment/raw_clips/` |
| 车辆数 | 57 台 |
| **总 segment 数** | **154,548 个** |
| 每 segment 时长 | 20 秒 |
| 帧率 | 2 FPS（40 帧/segment） |
| 分辨率 | 1920×1080（推理时下采样至 256×256） |
| 单 segment 大小 | ~11 MB（原始），推理 payload ~0.8MB |

### 数据偏向性

挖掘池名为 `junction_videos_segment`（路口视频切片），数据**偏向路口场景**：

| 一级类 | 预估命中率 | 本池是否适合 |
|---|:---:|:---:|
| 路口通行 | 30-50% | ✅ 非常适合 |
| 红绿灯通行 | 20-40% | ✅ 非常适合 |
| 路口内动态交互 | 10-20% | ✅ 适合 |
| 车道动态交互 | 10-20% | ✅ 适合 |
| 车道巡航 | 5-15% | ⚠️ 部分适合 |
| 变道/绕行 | 5-10% | ⚠️ 部分适合 |
| 发车/停车 | 3-8% | ⚠️ 部分适合 |
| 车道贴近 | 2-5% | ⚠️ 偏少 |
| ~~主辅路/分合流~~ | ~~<1-3%~~ | ❌ **暂不挖掘**，需高速数据池 |

> 决策：**主辅路/分合流 (P2, 6个标签)** 在路口数据池中命中率极低，暂不挖掘，记录待后续补充高速/快速路数据池。

---

## 二、推理时间估算（修正版）

### 单 segment 推理耗时

| 环节 | 耗时 |
|---|:---:|
| 图像编码（40帧 256×256 → ~10K image tokens） | 3-8s |
| Prefill（prompt ~2K tokens + images） | 2-5s |
| Decode（输出 ~300-500 tokens） | 5-10s |
| **单 segment 总耗时** | **~10-20s** |

> 之前 P1 mining 管线用了 **triple sample**（3 次采样），所以每 segment 30-60s。
> 新的 `12_distillation.py` 只调 **1 次 API**，实际 **~15s/segment**。

### 吞吐量估算（按 15s/segment）

| 并发数 | 每小时 | 每天（24h） | 每周 |
|:---:|:---:|:---:|:---:|
| 1 | 240 | 5,760 | 40,320 |
| 2 | 480 | 11,520 | 80,640 |
| 4 | 960 | 23,040 | 161,280 |

### 各规模挖掘耗时

| 方案 | segments | 并发=1 | 并发=2 | 并发=4 |
|---|:---:|:---:|:---:|:---:|
| 1K 试探 | 1,000 | **4.2h** | 2.1h | 1h |
| 5K 试探 | 5,000 | **20.8h** | 10.4h | 5.2h |
| 10K 挖掘 | 10,000 | **1.7天** | 0.9天 | 0.4天 |
| 30K 挖掘 | 30,000 | **5.2天** | 2.6天 | 1.3天 |
| 全量（单prompt） | 154,548 | **26.8天** | 13.4天 | 6.7天 |

---

## 三、挖掘范围与优先级

### 不挖掘的标签（排除项）

| 排除类别 | 标签数 | 原因 |
|---|:---:|---|
| P00 已有数据的 12 个标签 | 12 | 已有充足微调数据，不需要额外挖掘 |
| FW🔴 标签（需导航/侧后/规划信息） | 8 | 纯视觉无法挖掘 |
| 主辅路/分合流（P2） | 6 | 路口数据池命中率 <1%，暂不挖掘 |
| **排除小计** | **26** | |

### 实际需挖掘的标签

| 优先级 | 挖掘目标 | 标签数 | Prompt 文件 |
|:---:|---|:---:|---|
| **第 1 批** | P0 新增 — 路口通行 + 红绿灯通行 | 13 | `04_Intersection.txt` + `02_TrafficLight.txt` |
| **第 2 批** | P0 新增 — 车道动态交互 + 路口内动态交互 | 13 | `01_DynamicInteraction.txt` + `07_IntersectionInteraction.txt` |
| **第 3 批** | P0 新增 — 车道巡航 + 变道/绕行 + 发车/停车 | 9 | `05_LaneCruising.txt` + `06_LaneChange.txt` + `03_StartStop.txt` |
| **第 4 批** | P1 新增 FW🟢 标签 | 33 | 同上对应 prompt |
| **第 5 批** | P1 FW🟡 + 前司补充标签 | 26 | 同上对应 prompt |
| | **需挖掘合计** | **~96** | 7 个 prompt 文件（排除 08/09） |

---

## 四、分阶段挖掘计划

### 阶段 1：试探挖掘（~1 天）

**目的**：在 1,000 个随机 segment 上跑高优 prompt，验证命中率和 prompt 质量。

| 批次 | Prompt | 标签数 | 耗时（并发=1） |
|:---:|---|:---:|:---:|
| 1a | `04_Intersection.txt` | 23 | ~4.2h |
| 1b | `02_TrafficLight.txt` | 17 | ~4.2h |
| 1c | `01_DynamicInteraction.txt` | 12 | ~4.2h |
| | **合计** | | **~12.5h** |

**产出**：
- 每个标签的实际命中率（命中率 × 154K = 全池可挖量）
- Prompt 质量抽查（每标签 20 条）
- 调整 prompt 后进入阶段 2

```bash
# 随机抽样 1K segment
find /mnt/pfs/houhaotian/junction_videos_segment/raw_clips/ -name "*.mp4" | \
  shuf -n 1000 > pilot_1000_videos.txt
```

### 阶段 2：P0 新增标签主力挖掘（~3-7 天）

基于阶段 1 的命中率，按需扩量。**目标：每个标签 100-300 个高质量样本**。

| 批次 | Prompt 文件 | 扫描量 | 耗时（并发=1） | 耗时（并发=2） |
|:---:|---|:---:|:---:|:---:|
| 2a | `04_Intersection.txt` | 10,000 | 1.7天 | 0.9天 |
| 2b | `02_TrafficLight.txt` | 10,000 | 1.7天 | 0.9天 |
| 2c | `01_DynamicInteraction.txt` + `07_IntersectionInteraction.txt` | 15,000 | 2.6天 | 1.3天 |
| 2d | `05_LaneCruising.txt` + `06_LaneChange.txt` + `03_StartStop.txt` | 10,000 | 1.7天 | 0.9天 |
| | **合计** | ~30,000* | **~7.7天** | **~4天** |

> *segment 可跨批次复用：同一 segment 可以用不同 prompt 跑，每次只提取该类别的标签。但实际去重后约 20K-30K 唯一 segment。

### 阶段 3：P1 标签扩展（~3-5 天）

P0 完成后推进 P1。P1 标签已包含在各 prompt 文件中（同一 prompt 覆盖 P0+P1），可直接在阶段 2 结果中提取 P1 标签命中。如命中量不足，追加扫描。

### 阶段 4：长尾标签补充

命中率 <1% 的标签（如 停车开门切入、T 型路口汇入 等），考虑：
1. 在剩余 ~120K 未扫描 segment 中全量搜索
2. 补充非路口数据池

---

## 五、执行时间线

| 时间 | 阶段 | 动作 | segments | 耗时 |
|---|---|---|:---:|:---:|
| **D1** | 阶段 1 | 试探挖掘（1K × 3 prompts） | 1,000 | ~12.5h |
| **D2** | 评估 | 抽查命中率、调整 prompt | — | 0.5 天 |
| **D2-D5** | 阶段 2a-b | 路口通行 + 红绿灯通行挖掘 | 20,000 | ~3.5 天 |
| **D5-D8** | 阶段 2c-d | 动态交互 + 巡航 + 变道 | 25,000 | ~4.3 天 |
| **D9+** | 阶段 3 | P1 标签补充 | 按需 | 按需 |

**总预计耗时（并发=1）**：约 **8-10 天**
**提高并发=2 后**：约 **4-5 天**

---

## 六、关键决策总结

| 问题 | 决策 |
|---|---|
| P00 未达标标签要挖掘吗？ | **不需要**，已有充足数据 |
| 主辅路/分合流要挖掘吗？ | **暂不挖掘**，路口池命中率太低，待补充高速数据池 |
| 要全量挖掘吗？ | **不全量**，按需 20K-30K segments（约全池 15-20%） |
| 先挖哪些？ | 路口通行 → 红绿灯通行 → 车道动态交互 → 其余 |
| 单 segment 耗时？ | **~15 秒**（单次 API 调用，40 帧 256×256） |

---

## 七、挖掘脚本适配要点

当前 `12_distillation.py` 需要适配以支持新的标签挖掘：

1. **替换 SYSTEM_PROMPT**：从 `prompt_txt/` 目录读取对应类别的 prompt 文件
2. **增加 `--prompt_file` 参数**：指定使用哪个类别的 prompt
3. **更新标签白名单**：`CATEGORY_LABELS` 需动态从 prompt 文件中提取
4. **调整 `--output` 命名**：按类别命名输出文件，如 `results/01_DynamicInteraction.json`
5. **采样功能**：增加 `--sample_count` 参数支持随机抽样

示例命令：
```bash
# 阶段 1 试探挖掘
python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_list pilot_1000_videos.txt \
    --prompt_file scene_tag/prompt_txt/04_Intersection.txt \
    --output results/pilot_04_Intersection.json

# 阶段 2 扩量挖掘
python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_dir /mnt/pfs/houhaotian/junction_videos_segment/raw_clips/ \
    --max_videos 10000 \
    --prompt_file scene_tag/prompt_txt/02_TrafficLight.txt \
    --output results/main_02_TrafficLight.json
```

---

## 八、10K 抽样说明

已执行分层抽样，从 154K 中抽取 ~10K segment：

| 项目 | 值 |
|---|---|
| 抽样数量 | 9,998 segments |
| 抽样比例 | 6.5% |
| 抽样方法 | 按车辆ID分层等比例随机抽样 |
| 随机种子 | 42（可复现） |
| 每辆车抽取 | 39-381 条（按该车segment数等比例） |
| 列表文件 | `scene_tag/mining_10k_video_list.txt` |
| 统计文件 | `scene_tag/mining_10k_stats.json` |
| 生成脚本 | `scene_tag/sample_10k_segments.py` |

如需扩大挖掘范围，修改 `TARGET_COUNT` 后重新运行即可。

---

## 九、多标签去重合并

### 问题

本挖掘池与 P00 训练数据来自**同一批车辆**（车辆ID完全重叠），部分 segment 可能已有 P00 标注。后续需要处理：

1. **同 segment 多标签合并**：同一个 20s 视频可能被多个 prompt 命中不同标签，需合并为多标签标注
2. **P00 已有标注去重**：与 P00 训练集的 segment 匹配，避免重复标注冲突
3. **时间段重叠处理**：同一 segment 内不同标签的时间段可能重叠，需保留合理的重叠

### 合并策略

```
原始挖掘结果（按 prompt 分类别）
  ↓
Step 1: 按 video_path 聚合所有 prompt 的标注结果
  ↓
Step 2: 与 P00 训练集匹配（按 slice_key / video_path）
  ↓
Step 3: 合并标签 → 多标签 SFT 格式
  ↓
Step 4: 人工判读抽查
```

---

## 十、人工判读

### 判读文件

| 文件 | 用途 |
|---|---|
| `scene_tag/review_sheet.md` | 完整判读表（56 个标签 × 20 条 = 1,120 条） |
| `scene_tag/review_template.md` | 判读说明 + 错误原因代码表 |

### 判读流程

1. 每个标签挖掘完成后，从命中结果中随机抽 20 条
2. 逐条播放视频，核对标签和时间段
3. 记录判定（✅/❌/⚠️）和错误原因
4. **人工准确率 = 正确数 / 20 = 模型直接推理准确率**

### 常用错误原因

| 代码 | 描述 |
|---|---|
| MIX-01 | 应为其他标签（标签混淆） |
| MIX-03 | 应为 not_applicable（场景不匹配） |
| SCENE-01 | 目标存在但未与ego交互（如邻车道有车通过但没有切入） |
| SCENE-02 | 目标不存在（幻觉/误识别） |
| SCENE-04 | ego静止等红灯，正常交通流不应标为interaction |
| OBJ-01 | VRU/Vehicle 分类错误 |
| TIME-01 | 标注时间段偏移 |

---

## 十一、启动顺序

按 prompt 优先级，使用 10K 抽样列表启动：

```bash
# 第1批：路口通行（命中率最高，ROI最大）
python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_list scene_tag/mining_10k_video_list.txt \
    --prompt_file scene_tag/prompt_txt/04_Intersection.txt \
    --output scene_tag/results/mining_04_Intersection.json

# 第2批：红绿灯通行
python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_list scene_tag/mining_10k_video_list.txt \
    --prompt_file scene_tag/prompt_txt/02_TrafficLight.txt \
    --output scene_tag/results/mining_02_TrafficLight.json

# 第3批：车道动态交互
python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_list scene_tag/mining_10k_video_list.txt \
    --prompt_file scene_tag/prompt_txt/01_DynamicInteraction.txt \
    --output scene_tag/results/mining_01_DynamicInteraction.json

# 第4批：路口内动态交互
python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_list scene_tag/mining_10k_video_list.txt \
    --prompt_file scene_tag/prompt_txt/07_IntersectionInteraction.txt \
    --output scene_tag/results/mining_07_IntersectionInteraction.json

# 第5批：车道巡航 + 变道/绕行 + 发车/停车
python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_list scene_tag/mining_10k_video_list.txt \
    --prompt_file scene_tag/prompt_txt/05_LaneCruising.txt \
    --output scene_tag/results/mining_05_LaneCruising.json

python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_list scene_tag/mining_10k_video_list.txt \
    --prompt_file scene_tag/prompt_txt/06_LaneChange.txt \
    --output scene_tag/results/mining_06_LaneChange.json

python 12_distillation.py \
    --api_base http://localhost:8000/v1 \
    --video_list scene_tag/mining_10k_video_list.txt \
    --prompt_file scene_tag/prompt_txt/03_StartStop.txt \
    --output scene_tag/results/mining_03_StartStop.json
```

### 10K 抽样 × 7 个 prompt 的总耗时

| 并发 | 耗时 |
|:---:|:---:|
| 1 | 10K × 7 × 15s = **~29h（1.2天）** |
| 2 | **~14.5h** |
| 4 | **~7.3h** |

---

*计划日期：2026-03-27 v3*
