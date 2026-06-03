import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from .config import Config
from . import metadata


class TrackBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.tracks: Dict[str, Dict[str, Any]] = {}
    
    def process_frame(self, frame: Dict[str, Any]) -> None:
        camera_id = frame.get("camera_id", "unknown")
        ntp_timestamp = frame.get("ntp_timestamp")
        frame_uri = frame.get("frame_uri")
        objects = frame.get("objects", [])
        
        frame_w = self.config.bbox.coord_width
        frame_h = self.config.bbox.coord_height
        
        for obj in objects:
            class_name = obj.get("class_name", "").lower()
            if class_name not in self.config.target_labels:
                continue
            
            track_id = obj.get("track_id")
            if not track_id:
                continue
            
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "track_id": track_id,
                    "camera_id": camera_id,
                    "class_id": obj.get("class_id"),
                    "class_name": class_name,
                    "observations": [],
                }
            
            enriched = metadata.enrich_object(obj, objects, frame_w, frame_h, self.config)
            
            observation = {
                "object_id": obj.get("object_id"),
                "frame_uri": frame_uri,
                "ntp_timestamp": ntp_timestamp,
                "bbox": obj.get("bbox"),
                "det_conf": obj.get("det_conf"),
                "tracker_confidence": obj.get("tracker_confidence"),
                "bbox_width": enriched.get("bbox_width"),
                "bbox_height": enriched.get("bbox_height"),
                "bbox_area": enriched.get("bbox_area"),
                "bbox_area_ratio": enriched.get("bbox_area_ratio"),
                "bbox_aspect_ratio": enriched.get("bbox_aspect_ratio"),
                "bbox_center": enriched.get("bbox_center"),
                "edge_margin_px": enriched.get("edge_margin_px"),
                "edge_margin_ratio": enriched.get("edge_margin_ratio"),
                "edge_penalty": enriched.get("edge_penalty"),
                "context_boxes": enriched.get("context_boxes"),
                "near_duplicate_boxes": enriched.get("near_duplicate_boxes"),
                "num_context_boxes": enriched.get("num_context_boxes"),
                "num_near_duplicate_boxes": enriched.get("num_near_duplicate_boxes"),
                "num_foreground_context_boxes": enriched.get("num_foreground_context_boxes"),
                "max_bbox_overlap_coverage_by_other": enriched.get("max_bbox_overlap_coverage_by_other"),
                "max_foreground_overlap_coverage_by_other": enriched.get("max_foreground_overlap_coverage_by_other"),
            }
            
            self.tracks[track_id]["observations"].append(observation)
    
    def get_tracks(self) -> Dict[str, Dict[str, Any]]:
        return self.tracks
    
    def iter_tracks(self) -> Iterator[Dict[str, Any]]:
        for track in self.tracks.values():
            yield track


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_tracks(input_path: Path, config: Config) -> TrackBuilder:
    builder = TrackBuilder(config)
    for frame in read_jsonl(input_path):
        builder.process_frame(frame)
    return builder
