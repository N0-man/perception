import math
from typing import List, Tuple


BBox = List[int]


def width(box: BBox) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1)


def height(box: BBox) -> int:
    x1, y1, x2, y2 = box
    return max(0, y2 - y1)


def area(box: BBox) -> int:
    return width(box) * height(box)


def center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def aspect_ratio(box: BBox) -> float:
    w = width(box)
    h = height(box)
    return w / float(max(1, h))


def diagonal(box: BBox) -> float:
    w = width(box)
    h = height(box)
    return max(1.0, math.sqrt(w * w + h * h))


def edge_margin_px(box: BBox, frame_w: int, frame_h: int) -> int:
    x1, y1, x2, y2 = box
    return min(x1, y1, frame_w - x2, frame_h - y2)


def edge_margin_ratio(box: BBox, frame_w: int, frame_h: int) -> float:
    margin = edge_margin_px(box, frame_w, frame_h)
    return max(0.0, margin / float(max(1, min(frame_w, frame_h))))


def edge_penalty(margin_px: int, threshold_px: int = 32) -> float:
    if margin_px < 0:
        return 1.0
    if margin_px >= threshold_px:
        return 0.0
    return 1.0 - (margin_px / float(threshold_px))


def area_similarity(box_a: BBox, box_b: BBox) -> float:
    area_a = area(box_a)
    area_b = area(box_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(area_a, area_b) / float(max(area_a, area_b))


def center_distance_px(box_a: BBox, box_b: BBox) -> float:
    ax, ay = center(box_a)
    bx, by = center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def area_change_ratio(box_a: BBox, box_b: BBox) -> float:
    area_a = area(box_a)
    area_b = area(box_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return abs(area_b - area_a) / float(max(area_a, area_b))


def expand(box: BBox, expand_ratio: float = 0.15) -> BBox:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * expand_ratio)
    pad_y = int(h * expand_ratio)
    return [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]


def intersects(box_a: BBox, box_b: BBox) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def intersection_area(box_a: BBox, box_b: BBox) -> int:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def iou(box_a: BBox, box_b: BBox) -> float:
    inter = intersection_area(box_a, box_b)
    area_a = area(box_a)
    area_b = area(box_b)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def target_coverage(target_box: BBox, other_box: BBox) -> float:
    target_area = area(target_box)
    if target_area <= 0:
        return 0.0
    return intersection_area(target_box, other_box) / float(target_area)


def vertical_overlap_ratio(target_box: BBox, other_box: BBox) -> float:
    _, ay1, _, ay2 = target_box
    _, by1, _, by2 = other_box
    inter_y1 = max(ay1, by1)
    inter_y2 = min(ay2, by2)
    inter_h = max(0, inter_y2 - inter_y1)
    target_h = max(1, ay2 - ay1)
    return inter_h / float(target_h)


def horizontal_overlap_ratio(target_box: BBox, other_box: BBox) -> float:
    ax1, _, ax2, _ = target_box
    bx1, _, bx2, _ = other_box
    inter_x1 = max(ax1, bx1)
    inter_x2 = min(ax2, bx2)
    inter_w = max(0, inter_x2 - inter_x1)
    target_w = max(1, ax2 - ax1)
    return inter_w / float(target_w)
