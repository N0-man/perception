import pytest
import src.context as context
from src.config import ContextConfig


@pytest.fixture
def default_config():
    return ContextConfig()


class TestIsLikelyForeground:
    def test_foreground_object(self, default_config):
        target = [100, 100, 200, 200]
        other = [90, 110, 190, 250]
        assert context.is_likely_foreground(target, other, default_config) is True
    
    def test_background_object(self, default_config):
        target = [100, 100, 200, 200]
        other = [90, 90, 190, 150]
        assert context.is_likely_foreground(target, other, default_config) is False
    
    def test_no_overlap(self, default_config):
        target = [100, 100, 200, 200]
        other = [300, 300, 400, 400]
        assert context.is_likely_foreground(target, other, default_config) is False


class TestMaxContextValue:
    def test_with_values(self):
        boxes = [
            {"overlap": 0.1},
            {"overlap": 0.5},
            {"overlap": 0.3},
        ]
        assert context.max_context_value(boxes, "overlap") == 0.5
    
    def test_empty_list(self):
        assert context.max_context_value([], "overlap", default=0.0) == 0.0
    
    def test_missing_key(self):
        boxes = [{"other": 0.5}]
        assert context.max_context_value(boxes, "overlap", default=0.0) == 0.0


class TestCollectForObject:
    @pytest.fixture
    def sample_frame_objects(self):
        return [
            {
                "object_id": 1,
                "class_id": 2,
                "class_name": "car",
                "bbox": [100, 100, 200, 200],
            },
            {
                "object_id": 2,
                "class_id": 2,
                "class_name": "car",
                "bbox": [150, 100, 250, 200],
            },
            {
                "object_id": 3,
                "class_id": 2,
                "class_name": "car",
                "bbox": [500, 500, 600, 600],
            },
        ]
    
    def test_collect_context_boxes(self, sample_frame_objects, default_config):
        target = sample_frame_objects[0]
        result = context.collect_for_object(target, sample_frame_objects, default_config)
        
        assert "context_boxes" in result
        assert "near_duplicate_boxes" in result
        assert "num_context_boxes" in result
        assert result["num_context_boxes"] == 1
    
    def test_excludes_self(self, sample_frame_objects, default_config):
        target = sample_frame_objects[0]
        result = context.collect_for_object(target, sample_frame_objects, default_config)
        
        context_ids = [c["object_id"] for c in result["context_boxes"]]
        assert target["object_id"] not in context_ids
    
    def test_non_overlapping_object_excluded(self, sample_frame_objects, default_config):
        target = sample_frame_objects[0]
        result = context.collect_for_object(target, sample_frame_objects, default_config)
        
        context_ids = [c["object_id"] for c in result["context_boxes"]]
        assert 3 not in context_ids
    
    def test_near_duplicate_detection(self, default_config):
        target = {
            "object_id": 1,
            "class_id": 2,
            "class_name": "car",
            "bbox": [100, 100, 200, 200],
        }
        frame_objects = [
            target,
            {
                "object_id": 2,
                "class_id": 2,
                "class_name": "car",
                "bbox": [100, 100, 200, 200],
            },
        ]
        
        result = context.collect_for_object(target, frame_objects, default_config)
        assert result["num_near_duplicate_boxes"] == 1
    
    def test_max_overlap_calculation(self, sample_frame_objects, default_config):
        target = sample_frame_objects[0]
        result = context.collect_for_object(target, sample_frame_objects, default_config)
        
        assert "max_bbox_overlap_coverage_by_other" in result
        assert result["max_bbox_overlap_coverage_by_other"] >= 0.0
    
    def test_foreground_overlap_calculation(self, default_config):
        target = {
            "object_id": 1,
            "class_id": 2,
            "class_name": "car",
            "bbox": [100, 100, 200, 200],
        }
        foreground_obj = {
            "object_id": 2,
            "class_id": 2,
            "class_name": "car",
            "bbox": [90, 110, 190, 250],
        }
        frame_objects = [target, foreground_obj]
        
        result = context.collect_for_object(target, frame_objects, default_config)
        
        assert result["num_foreground_context_boxes"] >= 0
        assert "max_foreground_overlap_coverage_by_other" in result
