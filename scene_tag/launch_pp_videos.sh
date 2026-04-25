#!/bin/bash
# ============================================================
# 严禁靠边场景 — 批量文生视频
#
# 用法:
#   bash scene_tag/launch_pp_videos.sh YOUR_ARK_API_KEY
# ============================================================

set -e

API_KEY="${1}"
OUTPUT_DIR="/mnt/pfs/houhaotian/prohibited_parking_videos"
COUNT=1

if [ -z "$API_KEY" ]; then
    echo "用法: bash $0 <ARK_API_KEY>"
    echo ""
    echo "示例: bash scene_tag/launch_pp_videos.sh xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
    exit 1
fi

PP_TAGS="PP_NoStopSignZone,PP_BusLaneAndStation,PP_FireLaneEntrance,PP_CrosswalkZone,PP_IntersectionSpecialRoad,PP_ExpresswayMainRoad,PP_InsideTunnel,PP_LongDownhill,PP_PoorVisibilityZone,PP_NonMotorLaneSidewalk,PP_FireHydrantZone,PP_HospitalEntrance,PP_SchoolEntrance,PP_EventVenueEntrance,PP_MetroTransitHub,PP_ResidentialGateCrowded,PP_MilitaryGovZone,PP_GPSUnstableZone,PP_SingleLaneMainRoad,PP_HighTrafficIntersection,PP_TaxiRideshareWaiting,PP_LoadingUnloadingZone,PP_IndustrialParkEntrance"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  严禁靠边场景 — Seedance 2.0 视频生成"
echo "  场景数: 23"
echo "  每个场景: ${COUNT} 条"
echo "  输出目录: $OUTPUT_DIR"
echo "============================================"

python3 scene_tag/16_generate_video.py \
    --api_key "$API_KEY" \
    --batch_tags "$PP_TAGS" \
    --count "$COUNT" \
    --output_dir "$OUTPUT_DIR" \
    --duration 10 \
    --resolution 720p

echo ""
echo "============================================"
echo "  全部完成! 视频保存在: $OUTPUT_DIR"
echo "============================================"
