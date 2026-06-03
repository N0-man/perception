# Top-K Pipeline Notes

Reference notes extracted from `top_k_identify_new.ipynb` and `top_k_pipeline_new.ipynb`.

---

## Post Ingestion Identification: Save Top K Crops Per Track

1. Extract unique frames referenced by JSONL
2. Load JSONL rows grouped by track_id
3. Build crop candidates
4. Compute visual quality features
5. Score candidates
6. Select top-k diverse crops
7. Save crops

---

## Ingestion Pipeline Assumptions

- One .h264 input = one camera.
- One DeepStream ingestion run = one camera.
- DeepStream frame numbering starts at 0.
- OpenCV frame extraction starts at 0.

---

## Scoring Helpers

### Laplacian (Details and Sharpness)

Laplacian measures high-frequency detail. Sharp images tend to have higher Laplacian variance.

**NOTE:**
- Blur first because Laplacian is noise-sensitive
- Laplacian variance can be fooled by:
  - License plate edges
  - Background text
  - Vehicle grille
  - Small noisy crops
  - JPEG artifacts

### Tenengrad Score (Sharpness)

Tenengrad uses Sobel gradients i.e. sharp crops usually have stronger gradients.
We use `mean` instead of `sum` because it avoids simply rewarding larger crops.

**Note:** Tenengrad can reward high-edge but semantically bad crops. Add saturating threshold.

### Exposure Score

Scores whether the crop is reasonably exposed.

**Note:** Vehicles can be black or white. A black car crop may have lower mean but still be valid. Keep this in mind during scoring i.e. use `0.10 * c["exp_score"]` or even lower.

---

## Bbox Helper Functions

### `_get_bbox`
Converts DeepStream bbox format `left, top, width, height` into more convenient crop/ranking format `x1, y1, x2, y2`. Assumes that `x2 not> frame_w` OR `y2 not> frame_h`. i.e. we configure nvinfer with `crop-objects-to-roi-boundary`.

### `_bbox_area`
Computes bbox area. If x2 < x1 then returns 0.

### Edge Margin Ratio
Measures how far the bbox is from the nearest frame edge.

```
frame = 1920 x 1080
bbox = [100, 50, 400, 300]

left margin   = 100
top margin    = 50
right margin  = 1920 - 400 = 1520
bottom margin = 1080 - 300 = 780

nearest margin = 50
edge_margin_ratio = 50 / 1080 = 0.046
```

A bbox touching the frame edge gets `edge_margin_ratio = 0` - we need this because edge crops near the camera are bigger but partial.

---

## Overlap Bbox Helpers

- **IoU:** box similarity / duplicate detection
- **Target coverage:** crop contamination / possible occlusion
- **Foreground overlap:** stronger possible occlusion signal

### `_expand_bbox`
Makes a larger version of target object bbox. Helps find nearby overlapping objects that might impact crop visibility/quality.

```
target bbox = [100, 100, 200, 200]
crop width = 100
crop height = 100
expand_ratio = 0.15 (configurable, 15%)

expanded bbox = [85, 85, 215, 215]
```

### `_boxes_intersect`
Returns `True` if two boxes overlap at all.

They do **not** intersect if:
- A is fully left of B
- B is fully left of A
- A is fully above B
- B is fully above A

**Note:** If boxes only touch at the edge, returns False because of `<=`.

### `_intersection_area`
Computes overlapping area between two boxes. Used to compute target coverage.

```
target vehicle = [100, 100, 200, 200]
closeby vehicle = [150, 150, 250, 250]

intersection = [150, 150, 200, 200]
area = 50 * 50 = 2500
```

### Target Coverage
How much of the target bbox is covered by the other bbox? This is asymmetric, unlike IoU.

**Learning:** IoU wasn't reliable because it's meant for reidentification or identifying duplicates. Issue: "a bus overlapped a small car significantly but still had low IoU during testing".

```
target vehicle bbox area = 10,000
other object overlaps 3,000 pixels of target

target_coverage = 3000 / 10000 = 0.30
```

Exposed as `"bbox_overlap_coverage_by_other": target_coverage` metadata.

### `_bbox_iou`
Computes standard Intersection-over-Union.

IoU is one of the DeepStream tracker options and could be useful for:
- Detecting duplicate boxes
- Measuring box similarity
- Catching possible ID switch / duplicate tracker outputs

**Should NOT be used for occlusion score** because IoU is symmetric. Use `_target_coverage` for occlusion.

### Vertical Overlaps
Measures how much of the target's vertical height overlaps with the other object. Only used if the vehicle overlaps or intersects.

```
target vehicle = 100px tall
other vehicle overlaps 40 pixels vertically

vertical_overlap_ratio = 0.40 (40%)
```

### Likely Foreground Object
This is **possible** foreground contamination, not a confirmed foreground occlusion.

**Challenge for HTVS:** Placement of road cameras isn't consistent - could record incoming vehicles, outgoing on pedestrian bridge, on the side of bus stop etc. Perspective of vehicles isn't consistent.

**Assumption:** In image coordinates, a lower bbox bottom usually means closer to camera.
- smaller y = higher in the image = farther away
- larger y = lower in the image = closer to camera

```
target vehicle bottom y = 500
other object bottom y = 509

bottom_y_delta = 9
```

Returns True if:
- Other object's bbox bottom is at least 8px lower than target's vehicle bbox bottom (depth clue)
- Other object box overlaps more than 20% of target's vehicle height
- Other object box overlaps more than 20% of target's vehicle width
- Other object covers at least 5% of target bbox area

---

## Scoring Candidates

Score components:
- size
- det_conf
- focus
- exposure
- aspect
- foreground overlap
- bbox overlap
- edge

### Learning on Size Score (Gelang Serai Market Video)

Size is useful, but not if the crop is occluded, contaminated, or clipped.

- **Run2 feedback:** Included occluded vehicles but with larger size and more clear
  - **Suggestion:** Keep size important, but reduce its benefit when crop has high overlap/context contamination.

- **Run 8:** Exposed that edge penalty was too weak if image size bbox is large (large clipped crop > smaller complete crop)
  - **Solution:** Damp it first - a clipped crop no longer gets full benefit from being large.

- **Run 10:** Smaller images were scored higher even after having larger and clear crop because `bbox_overlap_penalty` is harsh (bbox_overlap=0.226 shouldn't be punished)
  - **Solution:** Damping the overlap with simple normalization: `c["bbox_overlap_penalty"] ** 2` -- so bbox_overlap=0.226 would be soft=0.051

```python
quality_dampener = (
    1.0
    - 0.60 * c["foreground_overlap_penalty"]
    - 0.20 * c["bbox_overlap_penalty"]
    - 0.60 * c["edge_penalty"]
    - 0.40 * (1.0 - c["visible_area_ratio"])
)

quality_dampener = max(0.15, quality_dampener)
c["quality_dampener"] = quality_dampener
c["adjusted_size_score"] = c["size_score"] * quality_dampener
```

---

## Context Boxes

**context_boxes:**
- Nearby/overlapping objects used for occlusion/crop contamination scoring

**near_duplicate_boxes:**
- Boxes that probably represent the same physical object due to detector/tracker/class flip
- Not counted as context occlusion, but preserved for debugging

---

## Ingestion Probe Metadata

The following metadata is more for debugging and future evals, not directly used for scoring or track selection:

```
camera_id
class_id
timestamp
timestamp_source
video_timecode
frame_width/frame_height
sampling metadata
context_boxes
near_duplicate_boxes
num_context_boxes
num_near_duplicate_boxes
num_foreground_context_boxes
```

---

## Refactoring Notes

- `safe_crop` returns clip metadata
- `center_score` prefers ingestion metadata
- `edge_penalty` prefers ingestion metadata
