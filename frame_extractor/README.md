# Frame Extractor

Extract unique frames as JPEG from video based on `kafka_output.jsonl` references.

## Usage

```bash
python3 -m frame_extractor --jsonl output/kafka_output.jsonl --video models/data/cam_0_222.h264 --output output/frames
```

## Options

- `--jsonl`: Path to kafka_output.jsonl file (default: output/kafka_output.jsonl)
- `--video`: Path to input video file (mp4) - required
- `--output`: Output directory for frames (default: output/frames)
- `--camera`: Filter by specific camera ID
- `--width`: Frame width (default: 1920)
- `--height`: Frame height (default: 1080)
- `--quality`: JPEG quality (default: 95)
