"""
挖掘池扩充脚本 v3：把 rawdata 中所有未切片的 trip 切成 20s 视频，
合并已有切片，生成完整挖掘池。

与 build_mining_pool.py (v2) 的区别：
  - 处理全量 rawdata（不做随机采样）
  - 自动跳过已切片的 trip（增量式）
  - 生成挖掘池时排除训练/测试集用过的视频
  - 输出统计报告

数据来源：
  /mnt/pfs/rawdata/ad_model_ds_*  (25528 个 trip)
    └── to_anno/<car_id>/<record_id>/front_wide/ (webp 帧序列, 10fps)

切片输出：
  /mnt/pfs/sampled_videos_5k/slices_20s/<trip_name>/slice_0_20.mp4 ...

挖掘池输出：
  /root/workspace/LLaMA-Factory/data/mining_pool_expanded.json

Usage:
  # 第一步：切片（耗时，需 ffmpeg，建议 tmux）
  python build_mining_pool_v3.py --step slice --workers 24

  # 第二步：构建挖掘池 JSON（秒级）
  python build_mining_pool_v3.py --step build

  # 一步到位
  python build_mining_pool_v3.py --step all --workers 24
"""

import argparse
import json
import os
import subprocess
import logging
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/root/workspace/build_mining_pool_v3.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

RAWDATA_DIR = "/mnt/pfs/rawdata"
CAMERA_NAME = "front_wide"
OUTPUT_SLICE_DIR = "/mnt/pfs/sampled_videos_5k/slices_20s"
OUTPUT_POOL = "/root/workspace/LLaMA-Factory/data/mining_pool_expanded.json"
NATIVE_FPS = 10
SLICE_SECONDS = 20
FRAMES_PER_SLICE = NATIVE_FPS * SLICE_SECONDS  # 200

TRAIN_FILES = [
    "/root/workspace/LLaMA-Factory/data/qwen3_sft_train_segment_upsample.json",
    "/root/workspace/LLaMA-Factory/data/qwen3_sft_test_segment_upsample.json",
]

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
"""

INSTRUCTION = "<video>\nWhat is the ego vehicle's behavior in this 20-second video clip?"


def load_train_video_set():
    """加载训练/测试集用过的所有视频路径，用于排除。"""
    videos = set()
    for f in TRAIN_FILES:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            data = json.load(fh)
        for d in data:
            for v in d.get("videos", []):
                videos.add(v)
    logger.info(f"训练/测试集共 {len(videos)} 条视频需排除")
    return videos


def find_ad_model_dirs():
    pattern = os.path.join(RAWDATA_DIR, "ad_model_ds_*")
    dirs = sorted(glob(pattern))
    logger.info(f"rawdata 中找到 {len(dirs)} 个 ad_model_ds_* 目录")
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
        "-r", str(fps), "-loglevel", "error",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        os.remove(list_file)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception:
        if os.path.exists(list_file):
            os.remove(list_file)
        return False


def process_one(ad_dir):
    dir_name = os.path.basename(ad_dir)
    slice_dir = os.path.join(OUTPUT_SLICE_DIR, dir_name)

    existing = glob(os.path.join(slice_dir, "slice_*.mp4"))
    valid_existing = [s for s in existing if os.path.getsize(s) > 1000]
    if valid_existing:
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
        if frames_to_slice(chunk, slice_path):
            slices.append(slice_path)
    return slices


def step_slice(workers=24):
    """Step 1: 切片所有未处理的 rawdata trip。"""
    all_dirs = find_ad_model_dirs()

    already_done = 0
    to_process = []
    for d in all_dirs:
        dn = os.path.basename(d)
        existing = glob(os.path.join(OUTPUT_SLICE_DIR, dn, "slice_*.mp4"))
        valid = [s for s in existing if os.path.exists(s) and os.path.getsize(s) > 1000]
        if valid:
            already_done += 1
        else:
            to_process.append(d)

    logger.info(f"已切片: {already_done}, 待处理: {len(to_process)}")
    if not to_process:
        logger.info("无需处理，全部已完成")
        return

    os.makedirs(OUTPUT_SLICE_DIR, exist_ok=True)
    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, d): d for d in to_process}
        for future in tqdm(as_completed(futures), total=len(futures), desc="切片中"):
            try:
                slices = future.result()
                if slices:
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                logger.error(f"处理失败: {e}")
                fail += 1

    logger.info(f"新增成功: {ok}, 失败: {fail}")


def step_build():
    """Step 2: 扫描所有切片目录，排除训练数据，生成挖掘池 JSON。"""
    train_videos = load_train_video_set()

    all_slice_dirs = sorted(glob(os.path.join(OUTPUT_SLICE_DIR, "*")))
    logger.info(f"扫描切片目录: {len(all_slice_dirs)} 个")

    dataset = []
    skipped_train = 0
    for sd in tqdm(all_slice_dirs, desc="构建挖掘池"):
        if not os.path.isdir(sd):
            continue
        mp4s = sorted(glob(os.path.join(sd, "slice_*.mp4")))
        for mp4 in mp4s:
            if os.path.getsize(mp4) < 1000:
                continue
            if mp4 in train_videos:
                skipped_train += 1
                continue
            dataset.append({
                "instruction": INSTRUCTION,
                "input": "",
                "output": "",
                "videos": [mp4],
                "system": SYSTEM_PROMPT,
            })

    logger.info(f"挖掘池总条数: {len(dataset)}, 排除训练集: {skipped_train}")

    with open(OUTPUT_POOL, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {OUTPUT_POOL}")

    print(f"\n{'='*60}")
    print(f"  挖掘池构建完成")
    print(f"{'='*60}")
    print(f"  总条数:       {len(dataset)}")
    print(f"  排除训练集:   {skipped_train}")
    print(f"  输出文件:     {OUTPUT_POOL}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="挖掘池扩充 v3")
    parser.add_argument("--step", choices=["slice", "build", "all"], default="all")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    if args.step in ("slice", "all"):
        step_slice(workers=args.workers)
    if args.step in ("build", "all"):
        step_build()


if __name__ == "__main__":
    main()
