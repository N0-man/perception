import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.topk as topk
from src.config import Config


class TestParseFrameIndex:
    def test_standard_format(self):
        assert topk.parse_frame_index("./frame_123.jpg") == 123

    def test_padded_format(self):
        assert topk.parse_frame_index("frame_00000042.jpg") == 42

    def test_simple_format(self):
        assert topk.parse_frame_index("42.jpg") == 42

    def test_nested_path(self):
        assert topk.parse_frame_index("./frames/cam_0/frame_99.jpg") == 99

    def test_invalid_format(self):
        assert topk.parse_frame_index("invalid.png") is None


class TestResolveFramePath:
    def test_finds_existing_frame(self, tmp_path):
        camera_dir = tmp_path / "cam_0"
        camera_dir.mkdir()
        frame_file = camera_dir / "frame_00000042.jpg"
        frame_file.write_text("")

        obs = {"frame_uri": "./frame_42.jpg"}
        result = topk.resolve_frame_path(obs, tmp_path, "cam_0")

        assert result == frame_file
        assert obs["frame_idx"] == 42

    def test_returns_none_for_missing_frame(self, tmp_path):
        camera_dir = tmp_path / "cam_0"
        camera_dir.mkdir()

        obs = {"frame_uri": "./frame_999.jpg"}
        result = topk.resolve_frame_path(obs, tmp_path, "cam_0")

        assert result is None

    def test_handles_missing_frame_uri(self, tmp_path):
        result = topk.resolve_frame_path({}, tmp_path, "cam_0")
        assert result is None


class TestProcessTrackForTopk:
    @pytest.fixture
    def sample_track(self):
        return {
            "track_id": "cam_0_1",
            "camera_id": "cam_0",
            "class_id": 2,
            "class_name": "car",
            "observations": [
                {
                    "object_id": 1,
                    "frame_uri": "./frame_0.jpg",
                    "ntp_timestamp": 1000.0,
                    "bbox": [100, 100, 400, 400],
                    "det_conf": 0.85,
                    "tracker_confidence": 0.90,
                    "edge_penalty": 0.1,
                    "max_foreground_overlap_coverage_by_other": 0.05,
                    "max_bbox_overlap_coverage_by_other": 0.10,
                },
            ],
        }

    @pytest.fixture
    def frames_dir_with_image(self, tmp_path):
        camera_dir = tmp_path / "cam_0"
        camera_dir.mkdir()
        frame_path = camera_dir / "frame_00000000.jpg"
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), img)
        return tmp_path

    def test_processes_track(self, sample_track, frames_dir_with_image):
        config = Config()
        result = topk.process_track_for_topk(sample_track, frames_dir_with_image, config)

        assert "top_k_candidates" in result
        assert "top_k_selected" in result
        assert result["track_id"] == "cam_0_1"

    def test_handles_missing_frames(self, sample_track, tmp_path):
        config = Config()
        result = topk.process_track_for_topk(sample_track, tmp_path, config)

        assert result["top_k_candidates"] == []
        assert result["top_k_selected"] == []


class TestSaveTopKCrops:
    @pytest.fixture
    def track_with_selected(self, tmp_path):
        camera_dir = tmp_path / "frames" / "cam_0"
        camera_dir.mkdir(parents=True)
        frame_path = camera_dir / "frame_00000010.jpg"
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_path), img)

        return {
            "track_id": "cam_0_1",
            "camera_id": "cam_0",
            "top_k_selected": [
                {
                    "rank": 1,
                    "frame_idx": 10,
                    "frame_path": str(frame_path),
                    "bbox": [100, 100, 400, 400],
                    "final_score": 0.85,
                },
            ],
        }

    def test_saves_crops(self, track_with_selected, tmp_path):
        output_dir = tmp_path / "crops"
        saved = topk.save_top_k_crops(track_with_selected, output_dir)

        assert len(saved) == 1
        assert saved[0]["track_id"] == "cam_0_1"
        assert saved[0]["rank"] == 1
        assert Path(saved[0]["crop_path"]).exists()

    def test_correct_naming(self, track_with_selected, tmp_path):
        output_dir = tmp_path / "crops"
        saved = topk.save_top_k_crops(track_with_selected, output_dir)

        crop_path = Path(saved[0]["crop_path"])
        assert crop_path.name == "cam_0_1_rank01_frame10.jpg"

    def test_handles_empty_selected(self, tmp_path):
        track = {
            "track_id": "cam_0_1",
            "top_k_selected": [],
        }
        saved = topk.save_top_k_crops(track, tmp_path / "crops")
        assert saved == []
