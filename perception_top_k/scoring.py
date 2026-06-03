import math
from typing import List

import numpy as np


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def laplacian_variance(gray: np.ndarray) -> float:
    import cv2
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(blur, cv2.CV_64F)
    return float(lap.var())


def tenengrad_score(gray: np.ndarray) -> float:
    import cv2
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag2 = gx * gx + gy * gy
    return float(np.mean(gmag2))


def exposure_score(gray: np.ndarray) -> float:
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    mean_score = 1.0 - min(abs(mean - 128.0) / 128.0, 1.0)
    std_score = min(std / 64.0, 1.0)
    return 0.6 * mean_score + 0.4 * std_score


def det_conf_score(det_conf: float, det_conf_good: float = 0.80) -> float:
    return min(1.0, max(0.0, float(det_conf)) / det_conf_good)


def saturating_score(value: float, reference: float) -> float:
    return min(1.0, float(value) / float(reference))


def vehicle_aspect_score(aspect_ratio: float) -> float:
    if 0.8 <= aspect_ratio <= 3.5:
        return 1.0
    if 0.6 <= aspect_ratio <= 4.5:
        return 0.5
    return 0.0


def normalize_saturating_size(size_raw: List[float], percentile: int = 75) -> np.ndarray:
    values = np.asarray(size_raw, dtype=np.float64)
    if len(values) == 0:
        return values
    ref = np.percentile(values, percentile)
    if ref <= 0:
        return np.ones_like(values) * 0.5
    return np.clip(values / ref, 0.0, 1.0)


def compute_focus_score(
    lap_var: float,
    tenengrad: float,
    lap_reference: float = 120.0,
    tenengrad_reference: float = 12000.0,
) -> float:
    lap_score = saturating_score(lap_var, lap_reference)
    ten_score = saturating_score(tenengrad, tenengrad_reference)
    return 0.5 * lap_score + 0.5 * ten_score


def compute_quality_dampener(
    foreground_overlap_penalty: float,
    bbox_overlap_penalty: float,
    edge_penalty: float,
    visible_area_ratio: float,
) -> float:
    dampener = (
        1.0
        - 0.60 * foreground_overlap_penalty
        - 0.20 * bbox_overlap_penalty
        - 0.60 * edge_penalty
        - 0.40 * (1.0 - visible_area_ratio)
    )
    return max(0.15, dampener)


def compute_final_score(
    adjusted_size_score: float,
    focus_score: float,
    det_score: float,
    exp_score: float,
    aspect_score: float,
    foreground_overlap_penalty: float,
    bbox_overlap_penalty: float,
    edge_penalty: float,
) -> float:
    return (
        0.40 * adjusted_size_score +
        0.05 * focus_score +
        0.25 * det_score +
        0.10 * exp_score +
        0.10 * aspect_score -
        0.25 * foreground_overlap_penalty -
        0.10 * bbox_overlap_penalty -
        0.20 * edge_penalty
    )
