from typing import Any, Dict, List

from . import bbox
from . import context
from .config import Config


def compute_object_metadata(
    obj: Dict[str, Any],
    frame_w: int,
    frame_h: int,
    config: Config
) -> Dict[str, Any]:
    box = obj["bbox"]
    
    margin_px = bbox.edge_margin_px(box, frame_w, frame_h)
    
    return {
        "bbox_width": bbox.width(box),
        "bbox_height": bbox.height(box),
        "bbox_area": bbox.area(box),
        "bbox_area_ratio": bbox.area(box) / float(max(1, frame_w * frame_h)),
        "bbox_aspect_ratio": bbox.aspect_ratio(box),
        "bbox_center": list(bbox.center(box)),
        "edge_margin_px": margin_px,
        "edge_margin_ratio": bbox.edge_margin_ratio(box, frame_w, frame_h),
        "edge_penalty": bbox.edge_penalty(margin_px, config.bbox.edge_penalty_threshold_px),
    }


def compute_context_metadata(
    target: Dict[str, Any],
    frame_objects: List[Dict[str, Any]],
    config: Config
) -> Dict[str, Any]:
    return context.collect_for_object(target, frame_objects, config.context)


def enrich_object(
    obj: Dict[str, Any],
    frame_objects: List[Dict[str, Any]],
    frame_w: int,
    frame_h: int,
    config: Config
) -> Dict[str, Any]:
    enriched = dict(obj)
    
    obj_meta = compute_object_metadata(obj, frame_w, frame_h, config)
    enriched.update(obj_meta)
    
    target_with_area = {**obj, "bbox_area": obj_meta["bbox_area"]}
    objects_with_area = []
    for o in frame_objects:
        box = o["bbox"]
        objects_with_area.append({
            **o,
            "bbox_area": bbox.area(box),
        })
    
    ctx_meta = compute_context_metadata(target_with_area, objects_with_area, config)
    enriched.update(ctx_meta)
    
    return enriched
