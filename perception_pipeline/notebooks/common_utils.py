import os

# 1080p dims hardcoded
BBOX_COORD_WIDTH = 1920
BBOX_COORD_HEIGHT = 1080

# Crop-size expectaions for ReID/image embedding quality
BASE_MIN_CROP_SHORT_SIDE = 128
BASE_MIN_CROP_LONG_SIDE = 256
BASE_MIN_CROP_AREA = 128 * 256

CAMERA_ID = "cam_0"

CAMERA_PIPELINE_FILES = {
    "cam_0": "cam_0_222",
    "cam_1": "cam_1_222",
    "cam_2": "cam_2_eve_lightson",
    "cam_3": "cam_3_39511_lowres",
    "cam_4": "cam_4_lowres",
    "cam_5": "cam_5_night",
    "cam_6": "cam_6_39211_harshsun",
    "cam_7": "cam_7_40871_rain",
    "cam_8": "cam_8_night_side",
    "cam_9": "cam_9_highway_carflow",
    "cam_10": "cam_10_night_cross_pedestrain",
    "cam_11": "cam_11_crossjun_carflow",
    "cam_12": "cam_12_foreground_occ",
    "cam_13": "cam_13_sideview_rain",
    "PLQ": "PLQ",
    "SG_VOI": "SG_VOI"
}
INPUT_FILE_NAME = CAMERA_PIPELINE_FILES[CAMERA_ID]

EXPORT_ROOT = "/opt/nvidia/deepstream/deepstream-8.0/samples/playground/notebooks/topk"
JSONL_PATH = os.path.join(EXPORT_ROOT, f"vehicle_candidates_{CAMERA_ID}.jsonl")
INPUT_FILE = f"../../models/data/{INPUT_FILE_NAME}.h264"
FRAME_DIR = os.path.join(EXPORT_ROOT, "frames")
OUTPUT_TOPK_ROOT = os.path.join(FRAME_DIR, f"output_topk")
RESULTS_PATH = os.path.join(EXPORT_ROOT, f"topk_results_{CAMERA_ID}.pkl")

# We could either upscale the source input video to reference frames dims
# or scale down the bbox to match the original input video dims.
# LOW Resolution Toggles
# handle benchmarking with lower resolution videos
TOGGLE_LOW_RES_INPUT = False

LOW_RES_MODE = "upscale_frame"
# LOW_RES_MODE = "scale_bbox"

# Frame reference for upscaling when LOW_RES_MODE = "scale_bbox" is ON
REFERENCE_FRAME_WIDTH = 1920
REFERENCE_FRAME_HEIGHT = 1080

# with TOGGLE_LOW_RES_INPUT toggle, 
# it would still reduce by 50% i.e. min_short, min_long, min_area == 64.0 128.0 16384.0