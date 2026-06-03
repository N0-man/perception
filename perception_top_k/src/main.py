import argparse
import sys
from pathlib import Path
from typing import Optional

from src.config import Config
from src.tracks import build_tracks
from src.writers import write_tracks_jsonl
import src.topk as topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build track-level metadata from perception pipeline output"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/kafka_output.jsonl"),
        help="Path to input JSONL file (default: output/kafka_output.jsonl)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/perception_tracks.jsonl"),
        help="Path to output JSONL file (default: output/perception_tracks.jsonl)"
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory containing extracted frames (required for top-k)"
    )
    parser.add_argument(
        "--crops-output",
        type=Path,
        default=None,
        help="Directory to save top-k crops"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of top crops per track (default: 2)"
    )
    return parser.parse_args()


def run(config: Config) -> int:
    if not config.input_path.exists():
        print(f"Error: Input file not found: {config.input_path}", file=sys.stderr)
        return 1

    print(f"Reading from: {config.input_path}")
    builder = build_tracks(config.input_path, config)

    tracks = list(builder.iter_tracks())
    print(f"Built {len(tracks)} tracks")

    if config.frames_dir and config.frames_dir.exists():
        print(f"Processing top-k with frames from: {config.frames_dir}")
        processed_tracks = []
        for track in tracks:
            processed = topk.process_track_for_topk(track, config.frames_dir, config)
            processed_tracks.append(processed)

            if config.crops_output_dir:
                camera_id = track.get("camera_id", "unknown")
                camera_output = config.crops_output_dir / camera_id
                saved = topk.save_top_k_crops(processed, camera_output)
                if saved:
                    print(f"  Saved {len(saved)} crops for track: {track['track_id']}")

        tracks = processed_tracks

    track_count = write_tracks_jsonl(iter(tracks), config.output_path)
    print(f"Written {track_count} tracks to: {config.output_path}")

    return 0


def main() -> int:
    args = parse_args()

    selection_config = None
    if args.top_k != 2:
        from .config import SelectionConfig
        selection_config = SelectionConfig(top_k=args.top_k)

    config = Config(
        input_path=args.input,
        output_path=args.output,
        frames_dir=args.frames_dir,
        crops_output_dir=args.crops_output,
    )

    if selection_config:
        config = Config(
            input_path=args.input,
            output_path=args.output,
            frames_dir=args.frames_dir,
            crops_output_dir=args.crops_output,
            selection=selection_config,
        )

    return run(config)


if __name__ == "__main__":
    sys.exit(main())
