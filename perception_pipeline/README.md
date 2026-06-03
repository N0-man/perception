# Perception Pipeline

Simulating 3 RTSP (you will need some sample rtsp.mp4)

```
docker run --rm -it \
  --network=host \
  -e MTX_HLSADDRESS=:9001 \
  -e MTX_WEBRTCADDRESS=:9002 \
  -e MTX_APIADDRESS=:9003 \
  bluenviron/mediamtx
```

```
ffmpeg -re -stream_loop -1 -i rtsp.mp4 \
  -vf scale=1920:1080 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 30 -keyint_min 30 -bf 0 \
  -f rtsp -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/cam_0
```

```
ffmpeg -re -stream_loop -1 -i rtsp.mp4 \
  -vf scale=1920:1080 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 30 -keyint_min 30 -bf 0 \
  -f rtsp -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/cam_1
```

```
ffmpeg -re -stream_loop -1 -i rtsp.mp4 \
  -vf scale=1920:1080 \
  -c:v libx264 -preset ultrafast -tune zerolatency \
  -g 30 -keyint_min 30 -bf 0 \
  -f rtsp -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/cam_2
```



## Benchmark

```
watch -n 1 nvidia-smi
```

```
nvidia-smi dmon -s pucvmet
```

```
nvtop
```

## Pipeline design

based on the no of RTSP, the RTSPSource would be dynamically created and also update the `StreamMuxConfig(batch_size=len(sources))`

pgie is still hardcoded to 1 due to onnx export issue

```
source-bin-00 ─┐
source-bin-01 ─┤
source-bin-02 ─┼→ nvstreammux(batch=N)
...            │        ↓
source-bin-N  ─┘      PGIE (object detection)
                        ↓
                      tracker
                        ↓
                      metadata probe → JSONL output
                        ↓
                      SGIE (license plate detection)
                        ↓
                      fakesink
```

## Module Structure

```
perception_pipeline/
├── __init__.py          # Package exports
├── __main__.py          # `python -m` entry point
├── config.py            # Configuration dataclasses and constants
├── elements.py          # GStreamer element factory functions
├── probes.py            # Metadata extraction probe callbacks
├── writers.py           # Output writers (JSONL)
├── pipeline.py          # Main pipeline orchestration
├── main.py              # CLI entry point
└── tests/
    └── test_pipeline.py # Unit tests
```

## Usage

### Configuring RTSP Streams

Edit `RTSP_STREAMS` in `config.py` to change the default streams:

```python
RTSP_STREAMS = [
    ("cam_0", "rtsp://127.0.0.1:8554/cam_0"),
    ("cam_1", "rtsp://127.0.0.1:8554/cam_1"),
    ("cam_2", "rtsp://127.0.0.1:8554/cam_2"),
]
```

The pipeline checks RTSP availability at startup and dynamically sets `batch_size` based on available streams.

### From Command Line

```bash
# Run with defaults (3 RTSP streams)
python3 -m perception_pipeline

# Specify output file
python -m perception_pipeline --output /path/to/output.jsonl

# Process every 5th frame
python -m perception_pipeline --sample-rate 5

# Export all classes (not just vehicles)
python -m perception_pipeline --all-classes
```

## Configuration

### RTSPSource

```python
RTSPSource(
    camera_id="cam_0",      # Unique camera identifier
    uri="rtsp://host/path"  # RTSP stream URI
)
```

### PipelineConfig

```python
PipelineConfig(
    sources=[...],                    # List of RTSPSource
    pgie_config=Path("..."),          # Primary inference config
    sgie_config=Path("..."),          # Secondary inference config
    tracker_config=Path("..."),       # Tracker config
    streammux=StreamMuxConfig(...),   # Muxer settings
    metadata_export=MetadataExportConfig(...),  # Export settings
    output=OutputConfig(...),         # Output settings
)
```

### MetadataExportConfig

```python
MetadataExportConfig(
    only_vehicles=True,        # Filter to vehicle classes only
    sample_every_n_frames=1,   # Frame sampling rate
)
```

## Output Format

The pipeline outputs JSONL (JSON Lines) with one record per frame:

```json
{
  "camera_id": "cam_0",
  "source_id": 0,
  "frame_idx": 1234,
  "ntp_timestamp": 1716300000.123,
  "objects": [
    {
      "object_id": 42,
      "track_id": "cam_0_42",
      "unique_component_id": 1,
      "class_id": 2,
      "class_name": "car",
      "bbox": [100, 200, 300, 400],
      "bbox_width": 200,
      "bbox_height": 200,
      "bbox_area": 40000,
      "bbox_center": [200.0, 300.0],
      "det_conf": 0.95,
      "tracker_confidence": 0.88
    }
  ]
}
```

## Testing

```bash
pytest perception_pipeline/tests/ -v
```
