from typing import Any, Dict, List

import src.bbox as bbox
from src.config import ContextConfig


def is_likely_foreground(
    target_box: bbox.BBox,
    other_box: bbox.BBox,
    config: ContextConfig
) -> bool:
    bottom_y_delta = other_box[3] - target_box[3]
    v_overlap = bbox.vertical_overlap_ratio(target_box, other_box)
    h_overlap = bbox.horizontal_overlap_ratio(target_box, other_box)
    coverage = bbox.target_coverage(target_box, other_box)
    
    return (
        bottom_y_delta > config.foreground_bottom_y_delta and
        v_overlap > config.foreground_vertical_overlap_threshold and
        h_overlap > config.foreground_horizontal_overlap_threshold and
        coverage > config.foreground_target_coverage_threshold
    )


def max_context_value(context_boxes: List[Dict], key: str, default: float = 0.0) -> float:
    if not context_boxes:
        return default
    return max(float(box.get(key, default)) for box in context_boxes)


def collect_for_object(
    target: Dict[str, Any],
    frame_objects: List[Dict[str, Any]],
    config: ContextConfig
) -> Dict[str, Any]:
    target_id = target["object_id"]
    target_box = target["bbox"]
    expanded_box = bbox.expand(target_box, expand_ratio=config.expand_ratio)
    
    context_boxes = []
    near_duplicate_boxes = []
    
    for other in frame_objects:
        other_id = other["object_id"]
        if other_id == target_id:
            continue
        
        other_box = other["bbox"]
        box_iou = bbox.iou(target_box, other_box)
        coverage = bbox.target_coverage(target_box, other_box)
        area_sim = bbox.area_similarity(target_box, other_box)
        
        is_near_duplicate = (
            box_iou >= config.near_duplicate_iou_threshold and
            area_sim >= config.near_duplicate_area_ratio_threshold
        )
        
        if is_near_duplicate:
            near_duplicate_boxes.append({
                "object_id": other_id,
                "class_id": other.get("class_id"),
                "class_name": other.get("class_name"),
                "bbox": other_box,
                "iou_with_target": box_iou,
                "bbox_area_similarity": area_sim,
                "bbox_overlap_coverage_by_other": coverage,
            })
            continue
        
        if box_iou > 0.0 or bbox.intersects(expanded_box, other_box):
            bottom_y_delta = other_box[3] - target_box[3]
            v_overlap = bbox.vertical_overlap_ratio(target_box, other_box)
            is_foreground = is_likely_foreground(target_box, other_box, config)
            
            target_area = bbox.area(target_box)
            other_area = bbox.area(other_box)
            area_ratio = other_area / float(max(1, target_area))
            
            context_boxes.append({
                "object_id": other_id,
                "class_id": other.get("class_id"),
                "class_name": other.get("class_name"),
                "bbox": other_box,
                "iou_with_target": box_iou,
                "bbox_overlap_coverage_by_other": coverage,
                "bottom_y_delta": bottom_y_delta,
                "area_ratio_other_to_target": area_ratio,
                "vertical_overlap_ratio": v_overlap,
                "other_is_likely_foreground": is_foreground,
            })
    
    max_overlap = max_context_value(context_boxes, "bbox_overlap_coverage_by_other")
    foreground_overlaps = [
        c["bbox_overlap_coverage_by_other"]
        for c in context_boxes
        if c.get("other_is_likely_foreground")
    ]
    max_fg_overlap = max(foreground_overlaps) if foreground_overlaps else 0.0
    
    return {
        "context_boxes": context_boxes,
        "near_duplicate_boxes": near_duplicate_boxes,
        "num_context_boxes": len(context_boxes),
        "num_near_duplicate_boxes": len(near_duplicate_boxes),
        "num_foreground_context_boxes": sum(
            1 for c in context_boxes if c.get("other_is_likely_foreground")
        ),
        "max_bbox_overlap_coverage_by_other": max_overlap,
        "max_foreground_overlap_coverage_by_other": max_fg_overlap,
    }
