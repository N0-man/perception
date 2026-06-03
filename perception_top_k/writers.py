import json
from pathlib import Path
from typing import Any, Dict, Iterator


def clean_track_for_output(track: Dict[str, Any]) -> Dict[str, Any]:
    output = {
        "track_id": track.get("track_id"),
        "camera_id": track.get("camera_id"),
        "class_id": track.get("class_id"),
        "class_name": track.get("class_name"),
        "observations": track.get("observations", []),
    }

    if "top_k_selected" in track:
        selected = track["top_k_selected"]
        output["top_k"] = [
            {
                "rank": s.get("rank"),
                "frame_idx": s.get("frame_idx"),
                "frame_uri": s.get("frame_uri"),
                "ntp_timestamp": s.get("ntp_timestamp"),
                "bbox": s.get("bbox"),
                "crop_w": s.get("crop_w"),
                "crop_h": s.get("crop_h"),
                "final_score": s.get("final_score"),
                "size_score": s.get("size_score"),
                "adjusted_size_score": s.get("adjusted_size_score"),
                "quality_dampener": s.get("quality_dampener"),
                "focus_score": s.get("focus_score"),
                "det_score": s.get("det_score"),
                "exp_score": s.get("exp_score"),
                "aspect_score": s.get("aspect_score"),
                "edge_penalty": s.get("edge_penalty"),
                "foreground_overlap_penalty": s.get("foreground_overlap_penalty"),
                "bbox_overlap_penalty": s.get("bbox_overlap_penalty"),
                "visible_area_ratio": s.get("visible_area_ratio"),
            }
            for s in selected
        ]

    return output


def write_tracks_jsonl(tracks: Iterator[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as f:
        for track in tracks:
            cleaned = clean_track_for_output(track)
            f.write(json.dumps(cleaned) + "\n")
            count += 1

    return count
