"""
挖掘池构建脚本 v2：直接从帧生成 3 段 20s 视频（跳过 60s 中间步骤）

优化:
  - 直接从帧按时间戳分段，每段单独编码为 20s 视频，省一次编码
  - 自动跳过已完成的目录
  - 提高并行度到 24 workers
"""

import json
import os
import random
import subprocess
import logging
from pathlib import Path
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/root/workspace/build_mining_pool.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

RAWDATA_DIR = "/mnt/pfs/rawdata"
CAMERA_NAME = "front_wide"
SAMPLE_COUNT = 99999  # 全量处理所有可用trip（实际约10947个）
OUTPUT_SLICE_DIR = "/mnt/pfs/sampled_videos_5k/slices_20s"
OUTPUT_DATASET = "/root/workspace/LLaMA-Factory/data/mining_pool_all.json"
NATIVE_FPS = 10
SLICE_SECONDS = 20
FRAMES_PER_SLICE = NATIVE_FPS * SLICE_SECONDS  # 200 frames per 20s
MAX_WORKERS = 24

SYSTEM_PROMPT = """You are an expert in autonomous driving scene annotation.
Based on the input video and the question about the ego vehicle's behavior, analyze the 20-second video to identify the ego vehicle's actions with strict precision, focusing on predefined driving maneuver categories.

DRIVING MANEUVER CATEGORIES:
You MUST use ONLY these predefined labels for the ego vehicle's actions:

TrafficLight_StraightStopOrGo
TrafficLight_LeftTurnStopOrGo
LaneChange_NavForIntersection
LaneChange_AvoidSlowVRU
LaneChange_AvoidStaticVehicle
DynamicInteraction_VRUInLaneCrossing
DynamicInteraction_VehicleInLaneCrossing
DynamicInteraction_StandardVehicleCutIn
StartStop_StartFromMainRoad
StartStop_ParkRoadside
Intersection_StandardUTurn
LaneCruising_Straight
else (ONLY when NO label above matches, meaning the ego vehicle's action does not fit any of the predefined categories)

LABELING RULES:
1. Assign a label ONLY if the action clearly matches the definition of one of the predefined categories
2. NEVER force-match ambiguous scenes to predefined labels
3. Use "else" when:
   - The ego vehicle's action does not match any predefined category
   - The scene is ambiguous or uncertain (confidence < 90%)
   - No clearly identifiable maneuver occurs
4. For "else" segments: Cover ONLY time periods with NO identifiable predefined maneuver
5. Time segments MUST be contiguous and non-overlapping, covering the entire 20-second video
6. Minimum segment duration: 1.0 second. Ignore shorter or transient actions
7. Base times on video timeline (0.0 to 20.0 seconds)

OUTPUT FORMAT:
<driving_maneuver>action_label</driving_maneuver> from <start_time>XX.X</start_time> to <end_time>YY.Y</end_time> seconds
- Use one of the predefined category labels or "else" for each time segment
- Multiple segments: Separate with " and " in chronological order
- Time precision: 1 decimal place (e.g., 5.0, 23.5)
- NO additional text or explanations—only output the formatted segments

CATEGORY DEFINITIONS:
1. TrafficLight_StraightStopOrGo: Ego vehicle stops or starts at a traffic light for straight-line movement
2. TrafficLight_LeftTurnStopOrGo: Ego vehicle stops or starts at a traffic light for left-turn movement
3. LaneChange_NavForIntersection: Lane change for navigation purposes approaching an intersection
4. LaneChange_AvoidSlowVRU: Lane change to avoid slow-moving vulnerable road users (pedestrians, cyclists)
5. LaneChange_AvoidStaticVehicle: Lane change to avoid stationary vehicles
6. DynamicInteraction_VRUInLaneCrossing: Interaction with vulnerable road users crossing the ego's lane
7. DynamicInteraction_VehicleInLaneCrossing: Interaction with other vehicles crossing the ego's lane
8. DynamicInteraction_StandardVehicleCutIn: Another vehicle cuts in front of the ego vehicle
9. StartStop_StartFromMainRoad: Starting from a stopped position on a main road
10. StartStop_ParkRoadside: Parking or stopping at roadside
11. Intersection_StandardUTurn: Making a U-turn at an intersection
12. LaneCruising_Straight: Straight-line cruising without notable events
13. else: Default for all other behaviors not covered by the predefined categories

IMPORTANT GUIDELINES:
1. Analyze the entire 20-second video thoroughly
2. Match actions to the most specific appropriate category
3. If multiple categories could apply, choose the one that best describes the primary action
4. Ensure time segments accurately reflect when each maneuver occurs
5. Maintain chronological order in output
"""

INSTRUCTION = "<video>\nWhat is the ego vehicle's behavior in this 20-second video clip?"


def find_ad_model_dirs():
    pattern = os.path.join(RAWDATA_DIR, "ad_model_ds_*")
    dirs = sorted(glob(pattern))
    logger.info(f"找到 {len(dirs)} 个 ad_model_ds 目录")
    return dirs


def find_front_wide_dir(ad_dir):
    to_anno = os.path.join(ad_dir, "to_anno")
    if not os.path.isdir(to_anno):
        return None
    for car_dir in os.listdir(to_anno):
        car_path = os.path.join(to_anno, car_dir)
        if not os.path.isdir(car_path):
            continue
        for record_dir in os.listdir(car_path):
            fw = os.path.join(car_path, record_dir, CAMERA_NAME)
            if os.path.isdir(fw):
                return fw
    return None


def sort_frames(frame_dir):
    frames = [f for f in os.listdir(frame_dir) if f.endswith(".webp")]
    frames.sort(key=lambda x: float(x.replace(".webp", "").split("_")[0]))
    return [os.path.join(frame_dir, f) for f in frames]


def frames_to_slice(frame_paths, output_path, fps=NATIVE_FPS):
    """直接从帧列表编码为一个 20s 视频"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    list_file = output_path + ".tmp.txt"
    with open(list_file, "w") as f:
        for fp in frame_paths:
            f.write(f"file '{fp}'\n")
            f.write(f"duration {1.0 / fps}\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-loglevel", "error",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        os.remove(list_file)
        if result.returncode != 0:
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        if os.path.exists(list_file):
            os.remove(list_file)
        return False


def process_one(ad_dir):
    dir_name = os.path.basename(ad_dir)
    slice_dir = os.path.join(OUTPUT_SLICE_DIR, dir_name)

    # 跳过已完成
    existing = glob(os.path.join(slice_dir, "slice_*.mp4"))
    valid_existing = [s for s in existing if os.path.getsize(s) > 1000]
    if len(valid_existing) >= 3:
        return valid_existing

    fw_dir = find_front_wide_dir(ad_dir)
    if fw_dir is None:
        return []

    all_frames = sort_frames(fw_dir)
    if len(all_frames) < FRAMES_PER_SLICE:
        return []

    os.makedirs(slice_dir, exist_ok=True)
    slices = []

    for seg_idx, start_frame in enumerate(range(0, len(all_frames), FRAMES_PER_SLICE)):
        end_frame = start_frame + FRAMES_PER_SLICE
        if end_frame > len(all_frames):
            break

        seg_start_sec = seg_idx * SLICE_SECONDS
        seg_end_sec = seg_start_sec + SLICE_SECONDS
        slice_path = os.path.join(slice_dir, f"slice_{seg_start_sec}_{seg_end_sec}.mp4")

        if os.path.exists(slice_path) and os.path.getsize(slice_path) > 1000:
            slices.append(slice_path)
            continue

        chunk = all_frames[start_frame:end_frame]
        ok = frames_to_slice(chunk, slice_path)
        if ok:
            slices.append(slice_path)

    return slices


def main():
    logger.info("=" * 60)
    logger.info("  挖掘池构建 v2 (直接帧→20s, 跳过已完成)")
    logger.info("=" * 60)

    all_dirs = find_ad_model_dirs()
    if len(all_dirs) < SAMPLE_COUNT:
        sampled = all_dirs
    else:
        sampled = random.sample(all_dirs, SAMPLE_COUNT)
    logger.info(f"采样 {len(sampled)} 个目录")

    # 统计已完成
    already_done = 0
    for d in sampled:
        dn = os.path.basename(d)
        existing = glob(os.path.join(OUTPUT_SLICE_DIR, dn, "slice_*.mp4"))
        valid = [s for s in existing if os.path.exists(s) and os.path.getsize(s) > 1000]
        if len(valid) >= 3:
            already_done += 1
    logger.info(f"已完成: {already_done}, 待处理: {len(sampled) - already_done}")

    os.makedirs(OUTPUT_SLICE_DIR, exist_ok=True)

    all_slices = []
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, d): d for d in sampled}
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理视频"):
            try:
                slices = future.result()
                if slices:
                    all_slices.extend(slices)
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"处理失败: {e}")
                failed += 1

    logger.info(f"成功: {len(sampled) - failed}, 失败: {failed}")
    logger.info(f"总切片数: {len(all_slices)}")

    # 构造推理数据集
    dataset = []
    for sp in sorted(all_slices):
        if not os.path.exists(sp):
            continue
        dataset.append({
            "instruction": INSTRUCTION,
            "input": "",
            "output": "",
            "videos": [sp],
            "system": SYSTEM_PROMPT,
        })

    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"推理数据集: {OUTPUT_DATASET}, 样本数: {len(dataset)}")

    info_path = "/root/workspace/LLaMA-Factory/data/dataset_info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["mining_pool_5000"] = {
        "file_name": "mining_pool_5000.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "videos": "videos",
            "system": "system",
        },
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info(f"  完成! 切片: {len(all_slices)}, 数据集: {len(dataset)} 条")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
