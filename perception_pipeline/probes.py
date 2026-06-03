import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Ensure GStreamer is initialized
if not Gst.is_initialized():
    Gst.init(None)

import pyds

from .config import PGIE_GIE_ID, VEHICLE_CLASS_IDS, INVALID_TRACK_ID


class MetadataWriter(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...

class PerformanceStats:
    def __init__(self, log_interval: float = 5.0):
        self.log_interval = log_interval
        self._start_time = time.perf_counter()
        self._last_log_time = self._start_time
        
        # Per-camera counters
        self._frames_processed: dict[str, int] = defaultdict(int)
        self._frames_written: dict[str, int] = defaultdict(int)
        self._objects_detected: dict[str, int] = defaultdict(int)
        
        # Counters since last log (for interval FPS)
        self._interval_frames: dict[str, int] = defaultdict(int)
        self._interval_written: dict[str, int] = defaultdict(int)
        self._interval_objects: dict[str, int] = defaultdict(int)
    
    def record_frame(self, camera_id: str, num_objects: int, written: bool) -> None:
        self._frames_processed[camera_id] += 1
        self._interval_frames[camera_id] += 1
        self._objects_detected[camera_id] += num_objects
        self._interval_objects[camera_id] += num_objects
        
        if written:
            self._frames_written[camera_id] += 1
            self._interval_written[camera_id] += 1
    
    def maybe_log(self) -> None:
        now = time.perf_counter()
        elapsed_since_log = now - self._last_log_time
        
        if elapsed_since_log >= self.log_interval:
            self._log_stats(elapsed_since_log)
            self._last_log_time = now
            self._reset_interval_counters()
    
    def _log_stats(self, interval_seconds: float) -> None:
        total_elapsed = time.perf_counter() - self._start_time
        
        total_processed = sum(self._frames_processed.values())
        total_written = sum(self._frames_written.values())
        total_objects = sum(self._objects_detected.values())
        
        # Summary line with totals
        print(f"\n[PERF] Stats @ {total_elapsed:.1f}s (last {interval_seconds:.1f}s interval)")
        print(f"TOTAL so far - Frames: {total_processed}, File Records: {total_written}, Objects detected: {total_objects}")
        print(f"{'─' * 56}")
        print(f"{'Camera':<12} {'Frames':>10} {'Objects':>10} {'Records':>10} {'Obj/s':>10}")
        print(f"{'─' * 56}")
        
        for camera_id in sorted(self._frames_processed.keys()):
            interval_frames = self._interval_frames[camera_id]
            interval_objects = self._interval_objects[camera_id]
            interval_written = self._interval_written[camera_id]
            
            obj_per_sec = interval_objects / interval_seconds if interval_seconds > 0 else 0
            
            print(f"{camera_id:<12} {interval_frames:>10} {interval_objects:>10} {interval_written:>10} {obj_per_sec:>10.1f}")
        
        total_interval_frames = sum(self._interval_frames.values())
        total_interval_objects = sum(self._interval_objects.values())
        total_interval_written = sum(self._interval_written.values())
        total_obj_per_sec = total_interval_objects / interval_seconds if interval_seconds > 0 else 0
        
        print(f"{'─' * 56}")
        print(f"{'Interval':<12} {total_interval_frames:>10} {total_interval_objects:>10} {total_interval_written:>10} {total_obj_per_sec:>10.1f}")
        print()
    
    def _reset_interval_counters(self) -> None:
        self._interval_frames.clear()
        self._interval_written.clear()
        self._interval_objects.clear()
    
    def log_final(self) -> None:
        total_elapsed = time.perf_counter() - self._start_time
        
        print(f"\n{'=' * 60}")
        print(f"[PERF] FINAL SUMMARY - Total runtime: {total_elapsed:.2f}s")
        print(f"{'=' * 60}")
        
        total_processed = sum(self._frames_processed.values())
        total_written = sum(self._frames_written.values())
        total_objects = sum(self._objects_detected.values())
        
        avg_obj_per_sec = total_objects / total_elapsed if total_elapsed > 0 else 0
        
        for camera_id in sorted(self._frames_processed.keys()):
            processed = self._frames_processed[camera_id]
            written = self._frames_written[camera_id]
            objects = self._objects_detected[camera_id]
            cam_obj_per_sec = objects / total_elapsed if total_elapsed > 0 else 0
            
            print(f"  {camera_id}: {processed} frames, {written} records, {objects} objects, {cam_obj_per_sec:.1f} obj/s")
        
        print(f"{'─' * 60}")
        print(f"  TOTAL: {total_processed} frames, {total_written} records, {total_objects} objects, {avg_obj_per_sec:.1f} obj/s")
        print(f"{'=' * 60}\n")


@dataclass
class ProbeConfig:
    writer: MetadataWriter
    source_id_to_camera_id: dict[int, str]
    only_vehicles: bool = True
    sample_every_n_frames: int = 1
    stats: Optional[PerformanceStats] = None

def rect_to_xyxy(rect) -> list[int]:
    x1 = int(rect.left)
    y1 = int(rect.top)
    x2 = int(rect.left + rect.width)
    y2 = int(rect.top + rect.height)
    return [x1, y1, x2, y2]


def compute_bbox_geometry(rect) -> dict[str, Any]:
    x1, y1, x2, y2 = rect_to_xyxy(rect)
    
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area = width * height
    
    return {
        "bbox": [x1, y1, x2, y2],
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "bbox_center": [x1 + width / 2.0, y1 + height / 2.0],
    }

def get_object_label(obj_meta) -> str:
    try:
        label = obj_meta.obj_label
        return label if label else ""
    except Exception:
        return ""


def get_frame_ntp_timestamp(frame_meta) -> Optional[float]:
    try:
        ntp_ns = int(frame_meta.ntp_timestamp)
        if ntp_ns > 0:
            return ntp_ns / 1e9
    except Exception:
        pass
    return None


def build_object_record(obj_meta, camera_id: str) -> dict[str, Any]:
    geometry = compute_bbox_geometry(obj_meta.rect_params)
    object_id = int(obj_meta.object_id)
    
    # Create composite track_id for multi-camera scenarios
    track_id = None
    if object_id != INVALID_TRACK_ID and object_id >= 0:
        track_id = f"{camera_id}_{object_id}"
    
    return {
        "object_id": object_id,
        "track_id": track_id,
        # "unique_component_id": int(obj_meta.unique_component_id),
        "class_id": int(obj_meta.class_id),
        "class_name": get_object_label(obj_meta),
        **geometry,
        "det_conf": float(obj_meta.confidence),
        "tracker_confidence": float(obj_meta.tracker_confidence),
    }


def build_frame_record(
    frame_meta,
    camera_id: str,
    # source_id: int,
    objects: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "camera_id": camera_id,
        # "source_id": source_id,
        "frame_uri": f"./frame_{int(frame_meta.frame_num)}.jpg",
        "ntp_timestamp": get_frame_ntp_timestamp(frame_meta),
        "objects": objects,
    }

def iterate_nvds_list(nvds_list):
    current = nvds_list
    while current is not None:
        try:
            yield current.data
            current = current.next
        except StopIteration:
            break

def create_metadata_export_probe(config: ProbeConfig):
    
    def probe_callback(pad: Gst.Pad, info: Gst.PadProbeInfo, user_data) -> Gst.PadProbeReturn:
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        for frame_data in iterate_nvds_list(batch_meta.frame_meta_list):
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_data)
            except StopIteration:
                continue
            
            frame_idx = int(frame_meta.frame_num)
            
            # Frame sampling
            if config.sample_every_n_frames > 1:
                if frame_idx % config.sample_every_n_frames != 0:
                    continue
            
            source_id = int(frame_meta.source_id)
            camera_id = config.source_id_to_camera_id.get(
                source_id, f"source_{source_id}"
            )
            
            objects = []
            for obj_data in iterate_nvds_list(frame_meta.obj_meta_list):
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(obj_data)
                except StopIteration:
                    continue
                
                if int(obj_meta.unique_component_id) != PGIE_GIE_ID:
                    continue
                
                if config.only_vehicles:
                    if int(obj_meta.class_id) not in VEHICLE_CLASS_IDS:
                        continue
                
                objects.append(build_object_record(obj_meta, camera_id))
            
            if objects:
                frame_record = build_frame_record(
                    frame_meta, camera_id, objects
                )
                config.writer.write(frame_record)
            
            # Record stats
            if config.stats:
                config.stats.record_frame(
                    camera_id=camera_id,
                    num_objects=len(objects),
                    written=bool(objects)
                )
        
        if config.stats:
            config.stats.maybe_log()
        
        return Gst.PadProbeReturn.OK
    
    return probe_callback


def create_debug_probe(name: str = "debug"):
    
    def probe_callback(pad: Gst.Pad, info: Gst.PadProbeInfo, user_data) -> Gst.PadProbeReturn:
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta:
            frame_count = 0
            for _ in iterate_nvds_list(batch_meta.frame_meta_list):
                frame_count += 1
            print(f"[{name}] Batch with {frame_count} frames")
        
        return Gst.PadProbeReturn.OK
    
    return probe_callback
