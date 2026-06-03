import json
import tempfile
from pathlib import Path

import pytest
from src.writers import write_tracks_jsonl


class TestWriteTracksJsonl:
    def test_writes_tracks(self):
        tracks = [
            {"track_id": "t1", "observations": []},
            {"track_id": "t2", "observations": []},
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            count = write_tracks_jsonl(iter(tracks), output_path)
            
            assert count == 2
            assert output_path.exists()
            
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
    
    def test_creates_parent_directories(self):
        tracks = [{"track_id": "t1", "observations": []}]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "deep" / "output.jsonl"
            write_tracks_jsonl(iter(tracks), output_path)
            
            assert output_path.exists()
    
    def test_valid_json_output(self):
        tracks = [
            {
                "track_id": "cam_0_0",
                "camera_id": "cam_0",
                "class_name": "car",
                "observations": [{"frame_uri": "./frame_0.jpg"}],
            },
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            write_tracks_jsonl(iter(tracks), output_path)
            
            with open(output_path) as f:
                line = f.readline()
            
            parsed = json.loads(line)
            assert parsed["track_id"] == "cam_0_0"
            assert len(parsed["observations"]) == 1
    
    def test_empty_iterator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            count = write_tracks_jsonl(iter([]), output_path)
            
            assert count == 0
            assert output_path.exists()
