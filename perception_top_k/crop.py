from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


MIN_CROP_SHORT_SIDE = 128
MIN_CROP_LONG_SIDE = 256
MIN_CROP_AREA = MIN_CROP_SHORT_SIDE * MIN_CROP_LONG_SIDE


def is_too_small(crop_w: int, crop_h: int) -> bool:
    short_side = min(crop_w, crop_h)
    area = crop_w * crop_h
    return short_side < MIN_CROP_SHORT_SIDE or area < MIN_CROP_AREA


def safe_crop(
    img: np.ndarray,
    bbox: List[int],
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Check if bbox is completely outside the image
    if x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
        return None, {
            "bbox_clipped": [0, 0, 0, 0],
            "was_clipped": True,
            "visible_area_ratio": 0.0,
        }

    x1_clipped = max(0, min(x1, w - 1))
    y1_clipped = max(0, min(y1, h - 1))
    x2_clipped = max(0, min(x2, w))
    y2_clipped = max(0, min(y2, h))

    if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
        return None, {
            "bbox_clipped": [x1_clipped, y1_clipped, x2_clipped, y2_clipped],
            "was_clipped": True,
            "visible_area_ratio": 0.0,
        }

    original_area = max(1, max(0, x2 - x1) * max(0, y2 - y1))
    visible_area = (x2_clipped - x1_clipped) * (y2_clipped - y1_clipped)

    was_clipped = (
        x1 != x1_clipped
        or y1 != y1_clipped
        or x2 != x2_clipped
        or y2 != y2_clipped
    )

    crop = img[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

    return crop, {
        "bbox_clipped": [x1_clipped, y1_clipped, x2_clipped, y2_clipped],
        "was_clipped": was_clipped,
        "visible_area_ratio": min(1.0, visible_area / float(original_area)),
    }


def frame_path_from_idx(frames_dir: Path, frame_idx: int) -> Optional[Path]:
    candidates = [
        frames_dir / f"frame_{frame_idx}.jpg",
        frames_dir / f"frame_{frame_idx:08d}.jpg",
        frames_dir / f"{frame_idx:08d}.jpg",
        frames_dir / f"{frame_idx}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_frame(frame_path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(frame_path))
    return img


def save_crop(
    img: np.ndarray,
    bbox: List[int],
    output_path: Path,
    jpeg_quality: int = 95,
) -> bool:
    crop, crop_info = safe_crop(img, bbox)
    if crop is None or crop.size == 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(
        str(output_path),
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
