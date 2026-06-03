import math

import numpy as np
import pytest

import src.scoring as scoring


class TestClamp01:
    def test_within_range(self):
        assert scoring.clamp01(0.5) == 0.5

    def test_below_zero(self):
        assert scoring.clamp01(-0.5) == 0.0

    def test_above_one(self):
        assert scoring.clamp01(1.5) == 1.0

    def test_boundary_values(self):
        assert scoring.clamp01(0.0) == 0.0
        assert scoring.clamp01(1.0) == 1.0


class TestDetConfScore:
    def test_high_confidence(self):
        assert scoring.det_conf_score(0.80, 0.80) == 1.0

    def test_low_confidence(self):
        assert scoring.det_conf_score(0.40, 0.80) == 0.5

    def test_exceeds_good(self):
        assert scoring.det_conf_score(1.0, 0.80) == 1.0

    def test_zero_confidence(self):
        assert scoring.det_conf_score(0.0, 0.80) == 0.0


class TestSaturatingScore:
    def test_at_reference(self):
        assert scoring.saturating_score(100.0, 100.0) == 1.0

    def test_below_reference(self):
        assert scoring.saturating_score(50.0, 100.0) == 0.5

    def test_above_reference(self):
        assert scoring.saturating_score(200.0, 100.0) == 1.0


class TestVehicleAspectScore:
    def test_optimal_range(self):
        assert scoring.vehicle_aspect_score(1.5) == 1.0
        assert scoring.vehicle_aspect_score(2.0) == 1.0

    def test_acceptable_range(self):
        assert scoring.vehicle_aspect_score(0.65) == 0.5
        assert scoring.vehicle_aspect_score(4.0) == 0.5

    def test_out_of_range(self):
        assert scoring.vehicle_aspect_score(0.3) == 0.0
        assert scoring.vehicle_aspect_score(5.0) == 0.0


class TestNormalizeSaturatingSize:
    def test_normalizes_to_percentile(self):
        values = [100.0, 200.0, 300.0, 400.0]
        result = scoring.normalize_saturating_size(values, percentile=75)
        assert len(result) == 4
        assert result[0] < result[1] < result[2]

    def test_empty_list(self):
        result = scoring.normalize_saturating_size([], percentile=75)
        assert len(result) == 0

    def test_clips_to_one(self):
        values = [100.0, 200.0, 1000.0]
        result = scoring.normalize_saturating_size(values, percentile=75)
        assert all(0.0 <= v <= 1.0 for v in result)


class TestComputeFocusScore:
    def test_combines_lap_and_tenengrad(self):
        result = scoring.compute_focus_score(120.0, 12000.0)
        assert result == 1.0

    def test_partial_focus(self):
        result = scoring.compute_focus_score(60.0, 6000.0)
        assert result == 0.5


class TestComputeQualityDampener:
    def test_perfect_quality(self):
        result = scoring.compute_quality_dampener(0.0, 0.0, 0.0, 1.0)
        assert result == 1.0

    def test_poor_quality(self):
        result = scoring.compute_quality_dampener(1.0, 1.0, 1.0, 0.0)
        assert result == 0.15

    def test_minimum_floor(self):
        result = scoring.compute_quality_dampener(0.5, 0.5, 0.5, 0.5)
        assert result >= 0.15


class TestComputeFinalScore:
    def test_perfect_scores(self):
        result = scoring.compute_final_score(
            adjusted_size_score=1.0,
            focus_score=1.0,
            det_score=1.0,
            exp_score=1.0,
            aspect_score=1.0,
            foreground_overlap_penalty=0.0,
            bbox_overlap_penalty=0.0,
            edge_penalty=0.0,
        )
        assert result == pytest.approx(0.90, rel=1e-3)

    def test_with_penalties(self):
        result = scoring.compute_final_score(
            adjusted_size_score=1.0,
            focus_score=1.0,
            det_score=1.0,
            exp_score=1.0,
            aspect_score=1.0,
            foreground_overlap_penalty=1.0,
            bbox_overlap_penalty=1.0,
            edge_penalty=1.0,
        )
        assert result < 0.5
