import json
import tempfile
from pathlib import Path

import pytest
from perception_top_k.src.config import Config
from perception_top_k.src.tracks import TrackBuilder, read_jsonl, build_tracks


@pytest.fixture
def default_config():
    return Config()


@pytest.fixture
def sample_frame():
    return {
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
                "tracker_confidence": 0.85,
            },
            {
                "object_id": 1,
                "track_id": "cam_0_1",
                "class_id": 7,
                "class_name": "truck",
                "bbox": [500, 200, 700, 450],
                "det_conf": 0.88,
                "tracker_confidence": 0.90,
            },
        ],
    }


class TestTrackBuilder:
    def test_creates_tracks(self, default_config, sample_frame):
        builder = TrackBuilder(default_config)
        builder.process_frame(sample_frame)
        
        tracks = builder.get_tracks()
        assert len(tracks) == 2
        assert "cam_0_0" in tracks
        assert "cam_0_1" in tracks
    
    def test_track_metadata(self, default_config, sample_frame):
        builder = TrackBuilder(default_config)
        builder.process_frame(sample_frame)
        
        track = builder.get_tracks()["cam_0_0"]
        assert track["track_id"] == "cam_0_0"
        assert track["camera_id"] == "cam_0"
        assert track["class_id"] == 2
        assert track["class_name"] == "car"
    
    def test_observations_added(self, default_config, sample_frame):
        builder = TrackBuilder(default_config)
        builder.process_frame(sample_frame)
        
        track = builder.get_tracks()["cam_0_0"]
        assert len(track["observations"]) == 1
        
        obs = track["observations"][0]
        assert obs["object_id"] == 0
        assert obs["frame_uri"] == "./frame_0.jpg"
        assert obs["det_conf"] == 0.95
    
    def test_enriched_metadata(self, default_config, sample_frame):
        builder = TrackBuilder(default_config)
        builder.process_frame(sample_frame)
        
        obs = builder.get_tracks()["cam_0_0"]["observations"][0]
        assert "bbox_width" in obs
        assert "bbox_height" in obs
        assert "bbox_area" in obs
        assert "edge_margin_px" in obs
        assert "num_context_boxes" in obs
    
    def test_multiple_frames_same_track(self, default_config, sample_frame):
        builder = TrackBuilder(default_config)
        builder.process_frame(sample_frame)
        
        frame2 = {
            "camera_id": "cam_0",
            "ntp_timestamp": 1779678593.300,
            "frame_uri": "./frame_1.jpg",
            "objects": [
                {
                    "object_id": 0,
                    "track_id": "cam_0_0",
                    "class_id": 2,
                    "class_name": "car",
                    "bbox": [110, 210, 310, 410],
                    "det_conf": 0.93,
                    "tracker_confidence": 0.87,
                },
            ],
        }
        builder.process_frame(frame2)
        
        track = builder.get_tracks()["cam_0_0"]
        assert len(track["observations"]) == 2
    
    def test_filters_non_target_labels(self, default_config):
        builder = TrackBuilder(default_config)
        frame = {
            "camera_id": "cam_0",
            "objects": [
                {
                    "object_id": 0,
                    "track_id": "cam_0_0",
                    "class_id": 0,
                    "class_name": "person",
                    "bbox": [100, 200, 300, 400],
                    "det_conf": 0.95,
                },
            ],
        }
        builder.process_frame(frame)
        
        assert len(builder.get_tracks()) == 0
    
    def test_iter_tracks(self, default_config, sample_frame):
        builder = TrackBuilder(default_config)
        builder.process_frame(sample_frame)
        
        tracks = list(builder.iter_tracks())
        assert len(tracks) == 2


class TestReadJsonl:
    def test_reads_valid_jsonl(self, sample_frame):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_frame) + "\n")
            f.write(json.dumps(sample_frame) + "\n")
            f.flush()
            
            frames = list(read_jsonl(Path(f.name)))
            assert len(frames) == 2
    
    def test_skips_empty_lines(self, sample_frame):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_frame) + "\n")
            f.write("\n")
            f.write(json.dumps(sample_frame) + "\n")
            f.flush()
            
            frames = list(read_jsonl(Path(f.name)))
            assert len(frames) == 2


class TestBuildTracks:
    def test_builds_from_file(self, default_config, sample_frame):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(sample_frame) + "\n")
            f.flush()
            
            builder = build_tracks(Path(f.name), default_config)
            assert len(builder.get_tracks()) == 2
