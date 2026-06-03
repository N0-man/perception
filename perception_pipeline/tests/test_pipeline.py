import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


class TestConfig:
    
    def test_rtsp_source_immutable(self):
        from perception_pipeline.config import RTSPSource
        
        source = RTSPSource(camera_id="cam_0", uri="rtsp://localhost/stream")
        
        with pytest.raises(AttributeError):
            source.camera_id = "cam_1"
    
    def test_pipeline_config_updates_batch_size(self):
        from perception_pipeline.config import PipelineConfig, RTSPSource
        
        sources = [
            RTSPSource("cam_0", "rtsp://localhost/0"),
            RTSPSource("cam_1", "rtsp://localhost/1"),
            RTSPSource("cam_2", "rtsp://localhost/2"),
        ]
        
        config = PipelineConfig(sources=sources)
        
        assert config.streammux.batch_size == 3
    
    def test_build_source_id_mapping(self):
        from perception_pipeline.config import RTSPSource, build_source_id_mapping
        
        sources = [
            RTSPSource("camera_a", "rtsp://localhost/a"),
            RTSPSource("camera_b", "rtsp://localhost/b"),
        ]
        
        mapping = build_source_id_mapping(sources)
        
        assert mapping == {0: "camera_a", 1: "camera_b"}
    
    def test_vehicle_class_ids(self):
        from perception_pipeline.config import VEHICLE_CLASS_IDS
        
        assert isinstance(VEHICLE_CLASS_IDS, frozenset)
        assert 2 in VEHICLE_CLASS_IDS
        assert 7 in VEHICLE_CLASS_IDS
    
    def test_streammux_config_defaults(self):
        from perception_pipeline.config import StreamMuxConfig, FRAME_WIDTH, FRAME_HEIGHT
        
        config = StreamMuxConfig()
        
        assert config.width == FRAME_WIDTH
        assert config.height == FRAME_HEIGHT
        assert config.live_source is True
        assert config.batched_push_timeout == 400000
    
    def test_metadata_export_config_defaults(self):
        from perception_pipeline.config import MetadataExportConfig
        
        config = MetadataExportConfig()
        
        assert config.only_vehicles is True
        assert config.sample_every_n_frames == 1


class TestRtspAvailability:
    
    @patch('perception_pipeline.elements.socket.socket')
    def test_check_rtsp_available_success(self, mock_socket_class):
        from perception_pipeline.elements import check_rtsp_available
        
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket
        
        result = check_rtsp_available("rtsp://127.0.0.1:8554/stream")
        
        assert result is True
        mock_socket.connect_ex.assert_called_once_with(("127.0.0.1", 8554))
        mock_socket.close.assert_called_once()
    
    @patch('perception_pipeline.elements.socket.socket')
    def test_check_rtsp_available_failure(self, mock_socket_class):
        from perception_pipeline.elements import check_rtsp_available
        
        mock_socket = MagicMock()
        mock_socket.connect_ex.return_value = 111
        mock_socket_class.return_value = mock_socket
        
        result = check_rtsp_available("rtsp://127.0.0.1:8554/stream")
        
        assert result is False
    
    @patch('perception_pipeline.elements.socket.socket')
    def test_check_rtsp_available_exception(self, mock_socket_class):
        from perception_pipeline.elements import check_rtsp_available
        
        mock_socket_class.side_effect = Exception("Network error")
        
        result = check_rtsp_available("rtsp://127.0.0.1:8554/stream")
        
        assert result is False
    
    def test_check_rtsp_available_parses_default_port(self):
        from perception_pipeline.elements import check_rtsp_available
        
        with patch('perception_pipeline.elements.socket.socket') as mock_socket_class:
            mock_socket = MagicMock()
            mock_socket.connect_ex.return_value = 0
            mock_socket_class.return_value = mock_socket
            
            check_rtsp_available("rtsp://192.168.1.1/stream")
            
            mock_socket.connect_ex.assert_called_once_with(("192.168.1.1", 554))
    
    @patch('perception_pipeline.elements.check_rtsp_available')
    def test_check_available_sources_filters_correctly(self, mock_check):
        from perception_pipeline.config import RTSPSource
        from perception_pipeline.elements import check_available_sources
        
        mock_check.side_effect = [True, False, True]
        
        sources = [
            RTSPSource("cam_0", "rtsp://localhost/0"),
            RTSPSource("cam_1", "rtsp://localhost/1"),
            RTSPSource("cam_2", "rtsp://localhost/2"),
        ]
        
        available = check_available_sources(sources)
        
        assert len(available) == 2
        assert available[0] == (0, sources[0])
        assert available[1] == (2, sources[2])
    
    @patch('perception_pipeline.elements.check_rtsp_available')
    def test_check_available_sources_none_available(self, mock_check):
        from perception_pipeline.config import RTSPSource
        from perception_pipeline.elements import check_available_sources
        
        mock_check.return_value = False
        
        sources = [
            RTSPSource("cam_0", "rtsp://localhost/0"),
        ]
        
        available = check_available_sources(sources)
        
        assert len(available) == 0


class TestJsonlWriter:
    
    def test_writes_json_lines(self):
        from perception_pipeline.writers import JsonlWriter
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            writer = JsonlWriter(output_path, flush_every=1)
            
            writer.write({"frame_idx": 0, "objects": []})
            writer.write({"frame_idx": 1, "objects": [{"id": 1}]})
            writer.close()
            
            with open(output_path, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2
            assert json.loads(lines[0]) == {"frame_idx": 0, "objects": []}
            assert json.loads(lines[1]) == {"frame_idx": 1, "objects": [{"id": 1}]}
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_record_count(self):
        from perception_pipeline.writers import JsonlWriter
        
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            writer = JsonlWriter(output_path)
            
            assert writer.record_count == 0
            
            writer.write({"test": 1})
            writer.write({"test": 2})
            
            assert writer.record_count == 2
            writer.close()
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_context_manager(self):
        from perception_pipeline.writers import JsonlWriter
        
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            output_path = f.name
        
        try:
            with JsonlWriter(output_path) as writer:
                writer.write({"data": "test"})
            
            with open(output_path, 'r') as f:
                content = f.read()
            
            assert '"data":"test"' in content or '"data": "test"' in content
        finally:
            Path(output_path).unlink(missing_ok=True)
    
    def test_creates_parent_directories(self):
        from perception_pipeline.writers import JsonlWriter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "output.jsonl"
            
            writer = JsonlWriter(output_path)
            writer.write({"test": True})
            writer.close()
            
            assert output_path.exists()


class TestNullWriter:
    
    def test_counts_records(self):
        from perception_pipeline.writers import NullWriter
        
        writer = NullWriter()
        
        writer.write({"data": 1})
        writer.write({"data": 2})
        
        assert writer.record_count == 2
    
    def test_no_side_effects(self):
        from perception_pipeline.writers import NullWriter
        
        writer = NullWriter()
        
        writer.write({"data": "anything"})
        writer.flush()
        writer.close()


class TestPerformanceStats:
    
    def test_record_frame_increments_counters(self):
        from perception_pipeline.probes import PerformanceStats
        
        stats = PerformanceStats(log_interval=999)
        
        stats.record_frame("cam_0", num_objects=3, written=True)
        stats.record_frame("cam_0", num_objects=2, written=False)
        stats.record_frame("cam_1", num_objects=1, written=True)
        
        assert stats._frames_processed["cam_0"] == 2
        assert stats._frames_processed["cam_1"] == 1
        assert stats._frames_written["cam_0"] == 1
        assert stats._frames_written["cam_1"] == 1
        assert stats._objects_detected["cam_0"] == 5
        assert stats._objects_detected["cam_1"] == 1
    
    def test_interval_counters_reset(self):
        from perception_pipeline.probes import PerformanceStats
        
        stats = PerformanceStats(log_interval=999)
        
        stats.record_frame("cam_0", num_objects=5, written=True)
        assert stats._interval_frames["cam_0"] == 1
        
        stats._reset_interval_counters()
        
        assert stats._interval_frames["cam_0"] == 0
        assert stats._frames_processed["cam_0"] == 1


class TestProbeUtilities:
    
    def test_rect_to_xyxy(self):
        from perception_pipeline.probes import rect_to_xyxy
        
        rect = MagicMock()
        rect.left = 100
        rect.top = 200
        rect.width = 50
        rect.height = 30
        
        result = rect_to_xyxy(rect)
        
        assert result == [100, 200, 150, 230]
    
    def test_compute_bbox_geometry(self):
        from perception_pipeline.probes import compute_bbox_geometry
        
        rect = MagicMock()
        rect.left = 10
        rect.top = 20
        rect.width = 100
        rect.height = 50
        
        result = compute_bbox_geometry(rect)
        
        assert result["bbox"] == [10, 20, 110, 70]
        assert result["bbox_width"] == 100
        assert result["bbox_height"] == 50
        assert result["bbox_area"] == 5000
        assert result["bbox_center"] == [60.0, 45.0]
    
    def test_get_object_label_success(self):
        from perception_pipeline.probes import get_object_label
        
        obj_meta = MagicMock()
        obj_meta.obj_label = "car"
        
        assert get_object_label(obj_meta) == "car"
    
    def test_get_object_label_empty(self):
        from perception_pipeline.probes import get_object_label
        
        obj_meta = MagicMock()
        obj_meta.obj_label = None
        
        assert get_object_label(obj_meta) == ""
    
    def test_get_object_label_exception(self):
        from perception_pipeline.probes import get_object_label
        
        obj_meta = MagicMock()
        type(obj_meta).obj_label = property(lambda self: (_ for _ in ()).throw(RuntimeError()))
        
        assert get_object_label(obj_meta) == ""
    
    def test_get_frame_ntp_timestamp_valid(self):
        from perception_pipeline.probes import get_frame_ntp_timestamp
        
        frame_meta = MagicMock()
        frame_meta.ntp_timestamp = 1716300000123456789
        
        result = get_frame_ntp_timestamp(frame_meta)
        
        assert result == pytest.approx(1716300000.123456789, rel=1e-6)
    
    def test_get_frame_ntp_timestamp_zero(self):
        from perception_pipeline.probes import get_frame_ntp_timestamp
        
        frame_meta = MagicMock()
        frame_meta.ntp_timestamp = 0
        
        result = get_frame_ntp_timestamp(frame_meta)
        
        assert result is None
    
    def test_build_frame_record(self):
        from perception_pipeline.probes import build_frame_record
        
        frame_meta = MagicMock()
        frame_meta.ntp_timestamp = 1716300000000000000
        
        objects = [{"id": 1}, {"id": 2}]
        
        result = build_frame_record(frame_meta, "cam_0", objects)
        
        assert result["camera_id"] == "cam_0"
        assert result["objects"] == objects
        assert "ntp_timestamp" in result
    
    def test_build_object_record(self):
        from perception_pipeline.probes import build_object_record
        from perception_pipeline.config import INVALID_TRACK_ID
        
        obj_meta = MagicMock()
        obj_meta.object_id = 42
        obj_meta.unique_component_id = 1
        obj_meta.class_id = 2
        obj_meta.obj_label = "car"
        obj_meta.confidence = 0.95
        obj_meta.tracker_confidence = 0.88
        
        rect = MagicMock()
        rect.left = 100
        rect.top = 200
        rect.width = 50
        rect.height = 30
        obj_meta.rect_params = rect
        
        result = build_object_record(obj_meta, "cam_0")
        
        assert result["object_id"] == 42
        assert result["track_id"] == "cam_0_42"
        assert result["class_id"] == 2
        assert result["class_name"] == "car"
        assert result["det_conf"] == 0.95
        assert result["bbox"] == [100, 200, 150, 230]
    
    def test_build_object_record_invalid_track_id(self):
        from perception_pipeline.probes import build_object_record
        from perception_pipeline.config import INVALID_TRACK_ID
        
        obj_meta = MagicMock()
        obj_meta.object_id = INVALID_TRACK_ID
        obj_meta.unique_component_id = 1
        obj_meta.class_id = 2
        obj_meta.obj_label = "car"
        obj_meta.confidence = 0.95
        obj_meta.tracker_confidence = 0.88
        
        rect = MagicMock()
        rect.left = 0
        rect.top = 0
        rect.width = 100
        rect.height = 100
        obj_meta.rect_params = rect
        
        result = build_object_record(obj_meta, "cam_0")
        
        assert result["object_id"] == INVALID_TRACK_ID
        assert result["track_id"] is None


class TestProbeConfig:
    
    def test_probe_config_defaults(self):
        from perception_pipeline.probes import ProbeConfig
        from perception_pipeline.writers import NullWriter
        
        writer = NullWriter()
        config = ProbeConfig(
            writer=writer,
            source_id_to_camera_id={0: "cam_0"},
        )
        
        assert config.only_vehicles is True
        assert config.sample_every_n_frames == 1
        assert config.stats is None


@pytest.mark.integration
class TestPipelineIntegration:
    
    @pytest.fixture
    def gst_available(self):
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            Gst.init(None)
            return True
        except Exception:
            pytest.skip("GStreamer not available")
    
    def test_pipeline_builds(self, gst_available):
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
