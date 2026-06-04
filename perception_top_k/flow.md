# Top-K Pipeline Flow

Input → Tracks → Metadata → Candidates → Scoring → Selection → Crops

---

## 1. Ingest JSONL and Build Tracks

**`tracks.py` → `build_tracks()`, `TrackBuilder.process_frame()`**

- Read JSONL frames from DeepStream output (`kafka_output.jsonl`)
- Keep only objects matching `target_labels`: car, bus, truck, motorcycle, bicycle
- Group by `track_id` — each track collects its observations across frames
- Each observation stores: bbox `[x1, y1, x2, y2]`, det_conf, tracker_confidence, frame_uri

One `.h264` = one camera, one DeepStream run = one camera. Frames are 0-indexed. The JSONL bbox is already `x1, y1, x2, y2` — DeepStream's native `left, top, w, h` gets converted inside the perception pipeline before the JSONL is written.

---

## 2. Enrich with Bbox Metadata

**`metadata.py` → `enrich_object()`, `compute_object_metadata()`**  
**`bbox.py` → `area()`, `aspect_ratio()`, `edge_margin_px()`, `edge_penalty()`**

- `[x1, y1, x2, y2]` → width, height, area, aspect_ratio, center
- `edge_margin_px`: `min(x1, y1, frame_w - x2, frame_h - y2)` — closest distance to any frame edge
- `edge_penalty`: 1.0 if touching/outside edge, 0.0 if margin ≥ 32px, linear in between
- `bbox_area_ratio`: bbox area / frame area (1920×1080)

Run 8: edge penalty was too weak — a large clipped crop was outscoring a smaller clean one. A big bbox near the edge is probably a partial vehicle, so it still gets penalized.

---

## 3. Compute Context / Occlusion Metadata

**`metadata.py` → `compute_context_metadata()`**  
**`context.py` → `collect_for_object()`, `is_likely_foreground()`**  
**`bbox.py` → `expand()`, `intersects()`, `iou()`, `target_coverage()`, `vertical_overlap_ratio()`**

- Expand target bbox by 15% to catch nearby objects just outside the box
- For each other object in the same frame:
  - **Near-duplicate** (IoU ≥ 0.90 AND area_sim ≥ 0.85): goes into `near_duplicate_boxes`, not counted as occlusion — likely the same physical vehicle detected twice
  - **Context** (IoU > 0 OR expanded box overlaps other): goes into `context_boxes`
  - `target_coverage` = intersection_area / target_area — how much of the target is covered by the other object
  - `is_likely_foreground`: all four must hold:
    - other's bottom_y > target's bottom_y by ≥ **8px** (lower in image = closer to camera)
    - vertical overlap with target > **20%**
    - horizontal overlap with target > **20%**
    - other covers > **5%** of target area

Outputs per observation:
- `max_bbox_overlap_coverage_by_other` — highest coverage across all context objects
- `max_foreground_overlap_coverage_by_other` — same but only foreground objects
- `num_context_boxes`, `num_near_duplicate_boxes`, `num_foreground_context_boxes`

Don't use IoU for occlusion — it's symmetric. A bus overlapping a small car gets a low IoU but the car's `target_coverage` is high. That was the real failure case that caused the switch from IoU.

Foreground detection is a heuristic. HTVS cameras aren't consistently placed — bridges, bus stops, side angles. The "lower = closer" assumption only holds for typical road-facing cameras.

---

## 4. Build Crop Candidates

**`topk.py` → `process_track_for_topk()`**  
**`selection.py` → `build_candidate()`**  
**`crop.py` → `load_frame()`, `safe_crop()`, `is_too_small()`**

- Resolve frame path from `frame_uri`, load image
- Clip bbox to frame bounds → `visible_area_ratio`
- Drop if crop is too small: short side < 128px or area < 32K
- Convert to grayscale for quality scoring

Drop candidate early if:
- `det_conf < 0.20`
- `foreground_overlap > 0.85`
- `edge_penalty > 0.65`
- `visible_area_ratio < 0.80`

`bbox_overlap_penalty` gets squared here: `bbox_overlap ** 2`. A raw value of 0.226 becomes 0.051. Run 10 showed that small overlaps were being punished too hard without squaring.

---

## 5. Compute Visual Quality Scores

**`scoring.py` → `laplacian_variance()`, `tenengrad_score()`, `exposure_score()`**  
**`selection.py` → `build_candidate()` (calls scoring functions on grayscale crop)**

- `laplacian_variance`: Gaussian blur → Laplacian → variance. High = sharp.
- `tenengrad_score`: Sobel gradients → mean squared magnitude. High = sharp.
- `exposure_score`: gray mean near 128 + reasonable std = well exposed.

Laplacian needs the blur first — it's sensitive to noise. It can still be tricked by license plate edges, grilles, background text, JPEG artifacts. Tenengrad uses mean not sum, so bigger crops don't automatically win. Both are used together to balance out.

---

## 6. Score Candidates

**`selection.py` → `score_candidates()`**  
**`scoring.py` → `compute_focus_score()`, `compute_quality_dampener()`, `compute_final_score()`**

Normalize:
- `size_score`: sqrt(bbox_area), normalized against 75th percentile of the track (saturating)
- `aspect_score`: 1.0 if 0.8–3.5, 0.5 if 0.6–4.5, 0.0 otherwise

Derived scores:
- `focus_score = 0.5 * lap_score + 0.5 * tenengrad_score` (both saturating against references)
- `det_score = det_conf / 0.80` (caps at 1.0)
- `exp_score = exposure_score` (clamped 0–1)

Quality dampener — applied to size first, before the final sum. `bbox_overlap_penalty` here is already the squared value from step 4.
```
dampener = 1.0
  - 0.60 * foreground_overlap_penalty
  - 0.20 * bbox_overlap_penalty        # squared value
  - 0.60 * edge_penalty
  - 0.40 * (1 - visible_area_ratio)

dampener = max(0.15, dampener)         # don't fully zero out size
```

**Final score:**
```
0.40 * adjusted_size_score    (size × dampener)
+ 0.05 * focus_score
+ 0.25 * det_score
+ 0.10 * exp_score
+ 0.10 * aspect_score
- 0.25 * foreground_overlap_penalty
- 0.10 * bbox_overlap_penalty
- 0.20 * edge_penalty
```

Run 2: occluded but larger crops were winning. Fixed by dampening size with quality penalties before the final sum.  
Run 8: large clipped crops were still winning. Fixed by putting edge_penalty inside the dampener at 0.60 weight.  
Run 10: mild bbox_overlap (0.226) was punished too hard. Fixed by squaring (`0.226² = 0.051`).

---

## 7. Select Top-K Diverse

**`selection.py` → `select_top_k_diverse()`, `_is_far_enough()`**

- Sort by `final_score` descending
- First pass: pick candidates at least **30 frames apart** (`strict_temporal_gap`)
- Second pass: if still under `top_k`, relax to **15 frames** (`relaxed_temporal_gap`)
- Skip anything below `final_score < 0.55` (`min_score`)
- Default `top_k = 2`

Without the temporal gap you'd just pick the same frame k times. The second pass exists because short tracks may not have enough frames spread 30 apart.

---

## 8. Save Crops

**`topk.py` → `save_top_k_crops()`**  
**`crop.py` → `save_crop()`**

- Load frame, crop bbox, save as `{track_id}_rank{N}_frame{idx}.jpg`
- Output: track_id, rank, frame_idx, crop_path, final_score

---

## Key Notes

| | |
|---|---|
| Occlusion | `target_coverage` not IoU — IoU is symmetric, misses the car-under-bus case |
| Foreground | lower bbox bottom ≈ closer to camera, but only a heuristic |
| Edge | large bbox near edge is still probably partial — still penalized |
| Size | biggest weight (0.40), but dampened first by occlusion/edge/clipping |
| Sharpness | laplacian + tenengrad together; laplacian alone gets fooled too easily |
| Exposure | vehicles are black or white legitimately — kept at low weight (0.10) |
| Diversity | temporal gap so the two crops aren't the same moment |
