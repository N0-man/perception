# Perception Top-K

Track-level metadata enrichment and top-K crop selection from perception pipeline output.

## Overview

This module reads frame-level detection data from `kafka_output.jsonl` and:

1. Builds track-level objects with enriched metadata
2. Computes visual quality scores (sharpness, exposure, focus)
3. Selects top-K diverse crops per track
4. Saves crops and enhanced track metadata

## Usage

### Basic Track Building

```bash
python3 -m perception_top_k --input output/kafka_output.jsonl --output output/perception_tracks.jsonl
```

### Full Top-K Pipeline

```bash
python3 -m perception_top_k \
    --input output/kafka_output.jsonl \
    --output output/perception_tracks.jsonl \
    --frames-dir output/frames \
    --crops-output output/top_k \
    --top-k 2
```

## Options

- `--input`: Path to kafka_output.jsonl file (default: output/kafka_output.jsonl)
- `--output`: Path to output tracks JSONL (default: output/perception_tracks.jsonl)
- `--frames-dir`: Directory containing extracted frames (required for top-k)
- `--crops-output`: Directory to save top-k crops
- `--top-k`: Number of top crops per track (default: 2)

## Input Format

Expects JSONL with frame-level entries:

```json
{
  "camera_id": "cam_0",
  "ntp_timestamp": 1779678593.265,
  "frame_uri": "./frame_0.jpg",
  "objects": [
    {
      "object_id": 0,
      "track_id": "cam_0_0",
      "class_id": 2,
      "class_name": "car",
      "bbox": [100, 200, 300, 400],
      "det_conf": 0.95,
      "tracker_confidence": 0.85
    }
  ]
}
```

## Output Format

### perception_tracks.jsonl

```json
{
  "track_id": "cam_0_0",
  "camera_id": "cam_0",
  "class_id": 2,
  "class_name": "car",
  "observations": [...],
  "top_k": [
    {
      "rank": 1,
      "frame_idx": 100,
      "bbox": [100, 200, 300, 400],
      "final_score": 0.85,
      "size_score": 0.75,
      "focus_score": 0.80,
      "det_score": 0.95,
      "exp_score": 0.70,
      "edge_penalty": 0.05
    }
  ]
}
```

### Crop Naming

Crops are saved with the pattern:
```
{track_id}_rank{rank:02d}_frame{frame_idx:08d}.jpg
```

Example: `cam_0_1_rank01_frame00000100.jpg`

## Scoring Components

- **Size Score**: Normalized bbox area (larger is better, p75 reference)
- **Focus Score**: Laplacian variance + Tenengrad (sharpness)
- **Detection Score**: Detector confidence (saturates at 0.80)
- **Exposure Score**: Image brightness and contrast
- **Aspect Score**: Vehicle aspect ratio validation
- **Edge Penalty**: Proximity to frame edge
- **Overlap Penalties**: Foreground occlusion and bbox overlap

## Testing

```bash
cd playground
python3 -m pytest perception_top_k/tests/ -v
```
