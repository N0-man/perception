import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import cv2

from . import crop as crop_utils
from . import selection
from .config import Config


def parse_frame_index(frame_uri: str) -> Optional[int]:
    match = re.search(r"frame_(\d+)", frame_uri)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\.jpg", frame_uri)
    if match:
        return int(match.group(1))
    return None


def resolve_frame_path(
    observation: Dict[str, Any],
    frames_dir: Path,
    camera_id: str,
) -> Optional[Path]:
    frame_uri = observation.get("frame_uri")
    if not frame_uri:
        return None

    frame_idx = parse_frame_index(frame_uri)
    if frame_idx is None:
        return None

    observation["frame_idx"] = frame_idx

    camera_frames_dir = frames_dir / camera_id
    return crop_utils.frame_path_from_idx(camera_frames_dir, frame_idx)


def process_track_for_topk(
    track: Dict[str, Any],
    frames_dir: Path,
    config: Config,
) -> Dict[str, Any]:
    track_id = track["track_id"]
    camera_id = track.get("camera_id", "unknown")
    observations = track.get("observations", [])

    candidates = []
    for obs in observations:
        frame_path = resolve_frame_path(obs, frames_dir, camera_id)
        if frame_path is None:
            continue

        candidate = selection.build_candidate(obs, frame_path, config)
        if candidate is None:
            continue

        candidate["frame_idx"] = obs.get("frame_idx")
        candidates.append(candidate)

    if not candidates:
        return {
            **track,
            "top_k_candidates": [],
            "top_k_selected": [],
        }

    scored = selection.score_candidates(candidates, config.scoring)
    selected = selection.select_top_k_diverse(scored, config.selection)

    for rank, sel in enumerate(selected, 1):
        sel["rank"] = rank

    return {
        **track,
        "top_k_candidates": scored,
        "top_k_selected": selected,
    }


def save_top_k_crops(
    track: Dict[str, Any],
    output_dir: Path,
    jpeg_quality: int = 95,
) -> List[Dict[str, Any]]:
    track_id = track["track_id"]
    selected = track.get("top_k_selected", [])

    if not selected:
        return []

    saved_crops = []
    for sel in selected:
        frame_path = sel.get("frame_path")
        if not frame_path:
            continue

        img = crop_utils.load_frame(Path(frame_path))
        if img is None:
            continue

        bbox = sel.get("bbox")
        if not bbox:
            continue

        rank = sel.get("rank", 0)
        frame_idx = sel.get("frame_idx", 0)

        crop_name = f"{track_id}_rank{rank:02d}_frame{frame_idx}.jpg"
        crop_path = output_dir / crop_name

        if crop_utils.save_crop(img, bbox, crop_path, jpeg_quality):
            saved_crops.append({
                "track_id": track_id,
                "rank": rank,
                "frame_idx": frame_idx,
                "crop_path": str(crop_path),
                "final_score": sel.get("final_score"),
            })

    return saved_crops


def process_tracks_for_topk(
    tracks: Iterator[Dict[str, Any]],
    frames_dir: Path,
    config: Config,
) -> Iterator[Dict[str, Any]]:
    for track in tracks:
        yield process_track_for_topk(track, frames_dir, config)
