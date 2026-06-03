import pytest
from perception_top_k.src import bbox


class TestWidth:
    def test_positive_width(self):
        assert bbox.width([0, 0, 100, 50]) == 100
    
    def test_zero_width(self):
        assert bbox.width([50, 0, 50, 100]) == 0
    
    def test_negative_width_returns_zero(self):
        assert bbox.width([100, 0, 50, 100]) == 0


class TestHeight:
    def test_positive_height(self):
        assert bbox.height([0, 0, 100, 50]) == 50
    
    def test_zero_height(self):
        assert bbox.height([0, 50, 100, 50]) == 0
    
    def test_negative_height_returns_zero(self):
        assert bbox.height([0, 100, 100, 50]) == 0


class TestArea:
    def test_positive_area(self):
        assert bbox.area([0, 0, 100, 50]) == 5000
    
    def test_zero_area(self):
        assert bbox.area([50, 50, 50, 100]) == 0


class TestCenter:
    def test_center_calculation(self):
        cx, cy = bbox.center([0, 0, 100, 100])
        assert cx == 50.0
        assert cy == 50.0
    
    def test_center_non_origin(self):
        cx, cy = bbox.center([100, 200, 200, 400])
        assert cx == 150.0
        assert cy == 300.0


class TestAspectRatio:
    def test_square(self):
        assert bbox.aspect_ratio([0, 0, 100, 100]) == 1.0
    
    def test_wide(self):
        assert bbox.aspect_ratio([0, 0, 200, 100]) == 2.0
    
    def test_tall(self):
        assert bbox.aspect_ratio([0, 0, 100, 200]) == 0.5


class TestDiagonal:
    def test_diagonal_3_4_5(self):
        result = bbox.diagonal([0, 0, 3, 4])
        assert result == 5.0
    
    def test_minimum_diagonal(self):
        result = bbox.diagonal([0, 0, 0, 0])
        assert result == 1.0


class TestEdgeMarginPx:
    def test_all_margins_positive(self):
        result = bbox.edge_margin_px([100, 50, 400, 300], 1920, 1080)
        assert result == 50
    
    def test_touching_left_edge(self):
        result = bbox.edge_margin_px([0, 100, 100, 200], 1920, 1080)
        assert result == 0
    
    def test_touching_right_edge(self):
        result = bbox.edge_margin_px([1820, 100, 1920, 200], 1920, 1080)
        assert result == 0


class TestEdgeMarginRatio:
    def test_ratio_calculation(self):
        result = bbox.edge_margin_ratio([100, 50, 400, 300], 1920, 1080)
        assert result == pytest.approx(50.0 / 1080, rel=1e-3)
    
    def test_zero_ratio_at_edge(self):
        result = bbox.edge_margin_ratio([0, 100, 100, 200], 1920, 1080)
        assert result == 0.0


class TestEdgePenalty:
    def test_no_penalty_far_from_edge(self):
        assert bbox.edge_penalty(100, 32) == 0.0
    
    def test_full_penalty_at_edge(self):
        assert bbox.edge_penalty(0, 32) == 1.0
    
    def test_full_penalty_outside(self):
        assert bbox.edge_penalty(-10, 32) == 1.0
    
    def test_partial_penalty(self):
        assert bbox.edge_penalty(16, 32) == 0.5


class TestAreaSimilarity:
    def test_same_area(self):
        result = bbox.area_similarity([0, 0, 100, 100], [200, 200, 300, 300])
        assert result == 1.0
    
    def test_different_areas(self):
        result = bbox.area_similarity([0, 0, 100, 100], [0, 0, 50, 50])
        assert result == pytest.approx(0.25, rel=1e-3)
    
    def test_zero_area(self):
        assert bbox.area_similarity([0, 0, 0, 0], [0, 0, 100, 100]) == 0.0


class TestCenterDistancePx:
    def test_same_center(self):
        result = bbox.center_distance_px([0, 0, 100, 100], [0, 0, 100, 100])
        assert result == 0.0
    
    def test_horizontal_distance(self):
        result = bbox.center_distance_px([0, 0, 100, 100], [100, 0, 200, 100])
        assert result == 100.0
    
    def test_diagonal_distance(self):
        result = bbox.center_distance_px([0, 0, 100, 100], [30, 40, 130, 140])
        assert result == pytest.approx(50.0, rel=1e-3)


class TestAreaChangeRatio:
    def test_no_change(self):
        result = bbox.area_change_ratio([0, 0, 100, 100], [0, 0, 100, 100])
        assert result == 0.0
    
    def test_area_doubled(self):
        result = bbox.area_change_ratio([0, 0, 100, 100], [0, 0, 100, 200])
        assert result == pytest.approx(0.5, rel=1e-3)
    
    def test_zero_area(self):
        assert bbox.area_change_ratio([0, 0, 0, 0], [0, 0, 100, 100]) == 0.0


class TestExpand:
    def test_expand_by_15_percent(self):
        result = bbox.expand([100, 100, 200, 200], 0.15)
        assert result == [85, 85, 215, 215]
    
    def test_expand_by_zero(self):
        result = bbox.expand([100, 100, 200, 200], 0.0)
        assert result == [100, 100, 200, 200]


class TestIntersects:
    def test_overlapping_boxes(self):
        assert bbox.intersects([0, 0, 100, 100], [50, 50, 150, 150]) is True
    
    def test_non_overlapping_boxes(self):
        assert bbox.intersects([0, 0, 100, 100], [200, 200, 300, 300]) is False
    
    def test_touching_edges_no_overlap(self):
        assert bbox.intersects([0, 0, 100, 100], [100, 0, 200, 100]) is False


class TestIntersectionArea:
    def test_overlapping_boxes(self):
        result = bbox.intersection_area([0, 0, 100, 100], [50, 50, 150, 150])
        assert result == 2500
    
    def test_non_overlapping_boxes(self):
        result = bbox.intersection_area([0, 0, 100, 100], [200, 200, 300, 300])
        assert result == 0
    
    def test_contained_box(self):
        result = bbox.intersection_area([0, 0, 100, 100], [25, 25, 75, 75])
        assert result == 2500


class TestIou:
    def test_identical_boxes(self):
        result = bbox.iou([0, 0, 100, 100], [0, 0, 100, 100])
        assert result == 1.0
    
    def test_no_overlap(self):
        result = bbox.iou([0, 0, 100, 100], [200, 200, 300, 300])
        assert result == 0.0
    
    def test_partial_overlap(self):
        result = bbox.iou([0, 0, 100, 100], [50, 0, 150, 100])
        assert result == pytest.approx(1.0 / 3.0, rel=1e-3)


class TestTargetCoverage:
    def test_full_coverage(self):
        result = bbox.target_coverage([25, 25, 75, 75], [0, 0, 100, 100])
        assert result == 1.0
    
    def test_half_coverage(self):
        result = bbox.target_coverage([0, 0, 100, 100], [50, 0, 150, 100])
        assert result == 0.5
    
    def test_no_coverage(self):
        result = bbox.target_coverage([0, 0, 100, 100], [200, 200, 300, 300])
        assert result == 0.0


class TestVerticalOverlapRatio:
    def test_full_vertical_overlap(self):
        result = bbox.vertical_overlap_ratio([0, 0, 100, 100], [50, 0, 150, 100])
        assert result == 1.0
    
    def test_half_vertical_overlap(self):
        result = bbox.vertical_overlap_ratio([0, 0, 100, 100], [0, 50, 100, 150])
        assert result == 0.5
    
    def test_no_vertical_overlap(self):
        result = bbox.vertical_overlap_ratio([0, 0, 100, 100], [0, 200, 100, 300])
        assert result == 0.0


class TestHorizontalOverlapRatio:
    def test_full_horizontal_overlap(self):
        result = bbox.horizontal_overlap_ratio([0, 0, 100, 100], [0, 50, 100, 150])
        assert result == 1.0
    
    def test_half_horizontal_overlap(self):
        result = bbox.horizontal_overlap_ratio([0, 0, 100, 100], [50, 0, 150, 100])
        assert result == 0.5
    
    def test_no_horizontal_overlap(self):
        result = bbox.horizontal_overlap_ratio([0, 0, 100, 100], [200, 0, 300, 100])
        assert result == 0.0
