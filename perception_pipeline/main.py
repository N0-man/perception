import argparse
import signal
import sys
import time
from pathlib import Path

from .config import PipelineConfig, RTSPSource, MP4Source, OutputConfig, RTSP_STREAMS, TOGGLE_MP4, MP4_FILE
from .pipeline import PerceptionPipeline, PipelineError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-camera RTSP perception pipeline"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/kafka_output.jsonl"),
        help="Path to output JSONL file (default: output/kafka_output.jsonl)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, process all)"
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Export all object classes, not just vehicles"
    )
    return parser.parse_args()


def create_default_config(args: argparse.Namespace) -> PipelineConfig:
    base_path = Path(__file__).parent.parent
    
    if TOGGLE_MP4:
        sources = [MP4Source(camera_id="cam_0", file_path=base_path / MP4_FILE)]
    else:
        sources = [
            RTSPSource(camera_id=cam_id, uri=uri)
            for cam_id, uri in RTSP_STREAMS
        ]
    
    config = PipelineConfig(
        sources=sources,
        pgie_config=base_path / "models/configs/config_infer_primary_yolo26.txt",
        sgie_config=base_path / "models/configs/config_infer_secondary_lpdnet_tao.txt",
        tracker_config=base_path / "models/configs/tracker.txt",
        output=OutputConfig(
            jsonl_path=args.output,
            flush_every=30,
        ),
    )
    
    # Apply command line overrides
    config.metadata_export.sample_every_n_frames = args.sample_rate
    config.metadata_export.only_vehicles = not args.all_classes
    
    return config


def main() -> int:
    args = parse_args()
    
    print("=" * 60)
    print("Perception Pipeline - Multi-Camera RTSP")
    print("=" * 60)
    
    try:
        config = create_default_config(args)
        
        print(f"\nConfiguration:")
        print(f"  Mode: {'MP4' if TOGGLE_MP4 else 'RTSP'}")
        print(f"  Sources: {len(config.sources)}")
        for src in config.sources:
            if isinstance(src, MP4Source):
                print(f"    - {src.camera_id}: {src.file_path}")
            else:
                print(f"    - {src.camera_id}: {src.uri}")
        print(f"  Output: {config.output.jsonl_path}")
        print(f"  Sample rate: every {config.metadata_export.sample_every_n_frames} frame(s)")
        print(f"  Vehicles only: {config.metadata_export.only_vehicles}")
        print()
        
        # Build and run pipeline
        pipeline = PerceptionPipeline(config)
        
        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\nReceived interrupt signal, stopping...")
            pipeline.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        start_time = time.perf_counter()
        
        pipeline.build()
        pipeline.run()
        
        elapsed = time.perf_counter() - start_time
        print(f"\nPipeline finished in {elapsed:.2f} seconds")
        
        if pipeline.writer:
            print(f"Records written: {pipeline.writer.record_count}")
        
        return 0
        
    except PipelineError as e:
        print(f"Pipeline error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
