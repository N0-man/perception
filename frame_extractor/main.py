import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Set

import cv2


FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_INDEX_BASE = 0
JPEG_QUALITY = 95


def parse_frame_index_from_uri(frame_uri: str) -> Optional[int]:
    match = re.search(r"frame_(\d+)", frame_uri)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\.jpg", frame_uri)
    if match:
        return int(match.group(1))
    return None


def collect_unique_frames(jsonl_path: Path) -> Dict[str, Set[int]]:
    camera_frames: Dict[str, Set[int]] = {}
    
    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            camera_id = row.get("camera_id")
            if not camera_id:
                continue
            
            frame_uri = row.get("frame_uri")
            if not frame_uri:
                continue
            
            frame_idx = parse_frame_index_from_uri(frame_uri)
            if frame_idx is None:
                continue
            
            if camera_id not in camera_frames:
                camera_frames[camera_id] = set()
            camera_frames[camera_id].add(frame_idx)
    
    return camera_frames


def extract_frames_for_camera(
    video_path: Path,
    output_dir: Path,
    camera_id: str,
    wanted_frames: Set[int],
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
    jpeg_quality: int = JPEG_QUALITY,
) -> int:
    cam_out_dir = output_dir / camera_id
    cam_out_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    wanted_sorted = sorted(wanted_frames)
    wanted_iter = iter(wanted_sorted)
    next_target = next(wanted_iter, None)
    
    current_idx = FRAME_INDEX_BASE
    saved = 0
    
    while next_target is not None:
        ok, frame = cap.read()
        if not ok:
            break
        
        if current_idx == next_target:
            h, w = frame.shape[:2]
            if w != frame_width or h != frame_height:
                frame = cv2.resize(
                    frame,
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            
            out_path = cam_out_dir / f"frame_{current_idx}.jpg"
            success = cv2.imwrite(
                str(out_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            
            if success:
                saved += 1
            
            next_target = next(wanted_iter, None)
        
        current_idx += 1
    
    cap.release()
    return saved


def extract_frames(
    jsonl_path: Path,
    video_path: Path,
    output_dir: Path,
    camera_id: Optional[str] = None,
    frame_width: int = FRAME_WIDTH,
    frame_height: int = FRAME_HEIGHT,
    jpeg_quality: int = JPEG_QUALITY,
) -> Dict[str, int]:
    camera_frames = collect_unique_frames(jsonl_path)
    
    if camera_id:
        if camera_id not in camera_frames:
            print(f"No frames found for camera: {camera_id}")
            return {}
        camera_frames = {camera_id: camera_frames[camera_id]}
    
    results = {}
    for cam_id, frames in camera_frames.items():
        print(f"Extracting {len(frames)} frames for camera: {cam_id}")
        saved = extract_frames_for_camera(
            video_path=video_path,
            output_dir=output_dir,
            camera_id=cam_id,
            wanted_frames=frames,
            frame_width=frame_width,
            frame_height=frame_height,
            jpeg_quality=jpeg_quality,
        )
        results[cam_id] = saved
        print(f"Saved {saved}/{len(frames)} frames for camera: {cam_id}")
    
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique frames from video based on kafka_output.jsonl"
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("output/kafka_output.jsonl"),
        help="Path to kafka_output.jsonl file",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input video file (mp4)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/frames"),
        help="Output directory for frames",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Filter by specific camera ID",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=FRAME_WIDTH,
        help=f"Frame width (default: {FRAME_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=FRAME_HEIGHT,
        help=f"Frame height (default: {FRAME_HEIGHT})",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=JPEG_QUALITY,
        help=f"JPEG quality (default: {JPEG_QUALITY})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    if not args.jsonl.exists():
        print(f"Error: JSONL file not found: {args.jsonl}", file=sys.stderr)
        return 1
    
    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}", file=sys.stderr)
        return 1
    
    results = extract_frames(
        jsonl_path=args.jsonl,
        video_path=args.video,
        output_dir=args.output,
        camera_id=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        jpeg_quality=args.quality,
    )
    
    total = sum(results.values())
    print(f"Total frames extracted: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
