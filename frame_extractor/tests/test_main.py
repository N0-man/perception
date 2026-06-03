import json
import tempfile
from pathlib import Path

import pytest
from frame_extractor.main import (
    parse_frame_index_from_uri,
    collect_unique_frames,
)


class TestParseFrameIndexFromUri:
    def test_standard_format(self):
        assert parse_frame_index_from_uri("./frame_123.jpg") == 123
    
    def test_frame_format_with_leading_zeros(self):
        assert parse_frame_index_from_uri("frame_00000042.jpg") == 42
    
    def test_simple_number_format(self):
        assert parse_frame_index_from_uri("42.jpg") == 42
    
    def test_invalid_format(self):
        assert parse_frame_index_from_uri("invalid.png") is None
    
    def test_nested_path(self):
        assert parse_frame_index_from_uri("./frames/cam_0/frame_99.jpg") == 99


class TestCollectUniqueFrames:
    def test_collects_frames_by_camera(self):
        data = [
            {"camera_id": "cam_0", "frame_uri": "./frame_1.jpg", "objects": []},
            {"camera_id": "cam_0", "frame_uri": "./frame_2.jpg", "objects": []},
            {"camera_id": "cam_1", "frame_uri": "./frame_1.jpg", "objects": []},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
            f.flush()
            
            result = collect_unique_frames(Path(f.name))
            
            assert "cam_0" in result
            assert "cam_1" in result
            assert result["cam_0"] == {1, 2}
            assert result["cam_1"] == {1}
    
    def test_deduplicates_frames(self):
        data = [
            {"camera_id": "cam_0", "frame_uri": "./frame_1.jpg", "objects": []},
            {"camera_id": "cam_0", "frame_uri": "./frame_1.jpg", "objects": []},
            {"camera_id": "cam_0", "frame_uri": "./frame_1.jpg", "objects": []},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
            f.flush()
            
            result = collect_unique_frames(Path(f.name))
            
            assert result["cam_0"] == {1}
    
    def test_handles_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.flush()
            
            result = collect_unique_frames(Path(f.name))
            
            assert result == {}
    
    def test_skips_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"camera_id": "cam_0", "frame_uri": "./frame_1.jpg"}\n')
            f.write("invalid json\n")
            f.write('{"camera_id": "cam_0", "frame_uri": "./frame_2.jpg"}\n')
            f.flush()
            
            result = collect_unique_frames(Path(f.name))
            
            assert result["cam_0"] == {1, 2}
    
    def test_skips_missing_camera_id(self):
        data = [
            {"frame_uri": "./frame_1.jpg", "objects": []},
            {"camera_id": "cam_0", "frame_uri": "./frame_2.jpg", "objects": []},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
            f.flush()
            
            result = collect_unique_frames(Path(f.name))
            
            assert result["cam_0"] == {2}
