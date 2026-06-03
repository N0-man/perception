import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import src.bbox as bbox_utils
import src.crop as crop_utils
import src.scoring as scoring
from src.config import Config, ScoringConfig, SelectionConfig


def get_edge_penalty(observation: Dict[str, Any], config: Config) -> float:
    if "edge_penalty" in observation:
        return scoring.clamp01(observation["edge_penalty"])

    if "edge_margin_px" in observation:
        margin = float(observation["edge_margin_px"])
        if margin < 0:
            return 1.0
        if margin >= config.scoring.edge_margin_px:
            return 0.0
        return 1.0 - (margin / float(config.scoring.edge_margin_px))

    return 0.0


def get_overlap_penalties(observation: Dict[str, Any]) -> Tuple[float, float]:
    foreground = float(observation.get("max_foreground_overlap_coverage_by_other", 0.0))
    bbox_overlap = float(observation.get("max_bbox_overlap_coverage_by_other", 0.0))
    bbox_overlap_penalty = bbox_overlap ** 2
    return foreground, bbox_overlap_penalty


def scale_bbox_to_frame(
    bbox: List[int],
    img_w: int,
    img_h: int,
    coord_width: int,
    coord_height: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Scale bbox from coordinate space to actual frame dimensions."""
    if img_w == coord_width and img_h == coord_height:
        return bbox, {
            "bbox_coord_scaling_enabled": False,
            "scale_x": 1.0,
            "scale_y": 1.0,
        }

    sx = img_w / float(coord_width)
    sy = img_h / float(coord_height)

    x1, y1, x2, y2 = bbox
    scaled_bbox = [
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    ]

    return scaled_bbox, {
        "bbox_coord_scaling_enabled": True,
        "scale_x": sx,
        "scale_y": sy,
    }


def build_candidate(
    observation: Dict[str, Any],
    frame_path: Path,
    config: Config,
) -> Optional[Dict[str, Any]]:
    img = crop_utils.load_frame(frame_path)
    if img is None:
        return None

    obs_bbox = observation.get("bbox")
    if not obs_bbox:
        return None

    crop_img, crop_info = crop_utils.safe_crop(img, obs_bbox)
    if crop_img is None:
        return None

    ch, cw = crop_img.shape[:2]

    if crop_utils.is_too_small(cw, ch):
        return None

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    edge_penalty = get_edge_penalty(observation, config)
    fg_overlap, bbox_overlap_penalty = get_overlap_penalties(observation)

    det_conf = float(observation.get("det_conf", 0.0))
    visible_ratio = crop_info["visible_area_ratio"]

    if det_conf < config.scoring.min_det_conf:
        return None
    if fg_overlap > config.scoring.max_foreground_overlap:
        return None
    if edge_penalty > config.scoring.max_edge_penalty:
        return None
    if visible_ratio < config.scoring.min_visible_area_ratio:
        return None

    return {
        "frame_path": str(frame_path),
        "frame_uri": observation.get("frame_uri"),
        "ntp_timestamp": observation.get("ntp_timestamp"),
        "object_id": observation.get("object_id"),
        "bbox": obs_bbox,
        "bbox_clipped": crop_info["bbox_clipped"],
        "bbox_was_clipped": crop_info["was_clipped"],
        "visible_area_ratio": visible_ratio,
        "crop_w": cw,
        "crop_h": ch,
        "bbox_area": cw * ch,
        "bbox_aspect_ratio": cw / float(max(1, ch)),
        "det_conf": det_conf,
        "tracker_confidence": observation.get("tracker_confidence"),
        "lap_var": scoring.laplacian_variance(gray),
        "tenengrad": scoring.tenengrad_score(gray),
        "exposure": scoring.exposure_score(gray),
        "edge_penalty": edge_penalty,
        "foreground_overlap_penalty": fg_overlap,
        "bbox_overlap_penalty": bbox_overlap_penalty,
        "num_context_boxes": observation.get("num_context_boxes", 0),
        "num_near_duplicate_boxes": observation.get("num_near_duplicate_boxes", 0),
        "num_foreground_context_boxes": observation.get("num_foreground_context_boxes", 0),
    }


def score_candidates(
    candidates: List[Dict[str, Any]],
    config: ScoringConfig,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    size_raw = [math.sqrt(c["bbox_area"]) for c in candidates]
    aspect_raw = [scoring.vehicle_aspect_score(c.get("bbox_aspect_ratio", 1.0)) for c in candidates]

    size_norm = scoring.normalize_saturating_size(size_raw, percentile=75)

    for i, c in enumerate(candidates):
        focus_score = scoring.compute_focus_score(
            c["lap_var"],
            c["tenengrad"],
            config.lap_reference,
            config.tenengrad_reference,
        )

        c["size_score"] = float(size_norm[i])
        c["focus_score"] = float(focus_score)
        c["det_score"] = scoring.det_conf_score(c["det_conf"], config.det_conf_good)
        c["exp_score"] = scoring.clamp01(c["exposure"])
        c["aspect_score"] = float(aspect_raw[i])

        c["quality_dampener"] = scoring.compute_quality_dampener(
            c["foreground_overlap_penalty"],
            c["bbox_overlap_penalty"],
            c["edge_penalty"],
            c["visible_area_ratio"],
        )

        c["adjusted_size_score"] = c["size_score"] * c["quality_dampener"]

        c["final_score"] = scoring.compute_final_score(
            c["adjusted_size_score"],
            c["focus_score"],
            c["det_score"],
            c["exp_score"],
            c["aspect_score"],
            c["foreground_overlap_penalty"],
            c["bbox_overlap_penalty"],
            c["edge_penalty"],
        )

    return candidates


def _is_far_enough(
    candidate: Dict[str, Any],
    selected: List[Dict[str, Any]],
    min_temporal_gap: int,
    frame_key: str = "frame_idx",
) -> bool:
    cand_frame = candidate.get(frame_key)
    if cand_frame is None:
        return True

    for s in selected:
        s_frame = s.get(frame_key)
        if s_frame is not None and abs(cand_frame - s_frame) < min_temporal_gap:
            return False
    return True


def select_top_k_diverse(
    candidates: List[Dict[str, Any]],
    config: SelectionConfig,
    frame_key: str = "frame_idx",
) -> List[Dict[str, Any]]:
    candidates = sorted(candidates, key=lambda x: x.get("final_score", 0), reverse=True)

    selected = []
    selected_keys = set()

    for c in candidates:
        if c.get("final_score", 0) < config.min_score:
            continue
        if not _is_far_enough(c, selected, config.strict_temporal_gap, frame_key):
            continue

        key = (c.get(frame_key), tuple(c.get("bbox", [])))
        selected.append(c)
        selected_keys.add(key)

        if len(selected) >= config.top_k:
            return sorted(selected, key=lambda x: x.get("final_score", 0), reverse=True)

    for c in candidates:
        key = (c.get(frame_key), tuple(c.get("bbox", [])))
        if key in selected_keys:
            continue
        if c.get("final_score", 0) < config.min_score:
            continue
        if not _is_far_enough(c, selected, config.relaxed_temporal_gap, frame_key):
            continue

        selected.append(c)
        selected_keys.add(key)

        if len(selected) >= config.top_k:
            break

    return sorted(selected, key=lambda x: x.get("final_score", 0), reverse=True)
