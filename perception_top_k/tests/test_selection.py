import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from perception_top_k import selection
from perception_top_k.config import Config, ScoringConfig, SelectionConfig


@pytest.fixture
def default_config():
    return Config()


@pytest.fixture
def sample_observation():
    return {
        "object_id": 1,
        "bbox": [100, 100, 400, 400],
        "det_conf": 0.85,
        "tracker_confidence": 0.90,
        "edge_penalty": 0.1,
        "max_foreground_overlap_coverage_by_other": 0.05,
        "max_bbox_overlap_coverage_by_other": 0.10,
        "num_context_boxes": 1,
        "num_near_duplicate_boxes": 0,
        "num_foreground_context_boxes": 0,
    }


class TestGetEdgePenalty:
    def test_from_edge_penalty_field(self, default_config):
        obs = {"edge_penalty": 0.5}
        result = selection.get_edge_penalty(obs, default_config)
        assert result == 0.5

    def test_from_edge_margin_px(self, default_config):
        obs = {"edge_margin_px": 16}
        result = selection.get_edge_penalty(obs, default_config)
        assert result == 0.5

    def test_negative_margin(self, default_config):
        obs = {"edge_margin_px": -10}
        result = selection.get_edge_penalty(obs, default_config)
        assert result == 1.0

    def test_large_margin(self, default_config):
        obs = {"edge_margin_px": 100}
        result = selection.get_edge_penalty(obs, default_config)
        assert result == 0.0


class TestGetOverlapPenalties:
    def test_extracts_penalties(self):
        obs = {
            "max_foreground_overlap_coverage_by_other": 0.5,
            "max_bbox_overlap_coverage_by_other": 0.3,
        }
        fg, bbox_pen = selection.get_overlap_penalties(obs)
        assert fg == 0.5
        assert bbox_pen == pytest.approx(0.09, rel=1e-3)

    def test_defaults_to_zero(self):
        fg, bbox_pen = selection.get_overlap_penalties({})
        assert fg == 0.0
        assert bbox_pen == 0.0


class TestScoreCandidates:
    def test_scores_candidates(self, default_config):
        candidates = [
            {
                "bbox_area": 10000,
                "bbox_aspect_ratio": 1.5,
                "lap_var": 100.0,
                "tenengrad": 10000.0,
                "exposure": 0.7,
                "det_conf": 0.85,
                "foreground_overlap_penalty": 0.1,
                "bbox_overlap_penalty": 0.05,
                "edge_penalty": 0.1,
                "visible_area_ratio": 0.95,
            },
            {
                "bbox_area": 20000,
                "bbox_aspect_ratio": 2.0,
                "lap_var": 150.0,
                "tenengrad": 15000.0,
                "exposure": 0.8,
                "det_conf": 0.90,
                "foreground_overlap_penalty": 0.05,
                "bbox_overlap_penalty": 0.02,
                "edge_penalty": 0.05,
                "visible_area_ratio": 1.0,
            },
        ]

        scored = selection.score_candidates(candidates, default_config.scoring)

        assert len(scored) == 2
        for c in scored:
            assert "final_score" in c
            assert "size_score" in c
            assert "focus_score" in c
            assert "det_score" in c
            assert "exp_score" in c
            assert "aspect_score" in c
            assert "quality_dampener" in c
            assert "adjusted_size_score" in c

    def test_empty_candidates(self, default_config):
        result = selection.score_candidates([], default_config.scoring)
        assert result == []


class TestSelectTopKDiverse:
    @pytest.fixture
    def scored_candidates(self):
        return [
            {"frame_idx": 0, "bbox": [0, 0, 100, 100], "final_score": 0.9},
            {"frame_idx": 10, "bbox": [0, 0, 100, 100], "final_score": 0.85},
            {"frame_idx": 50, "bbox": [0, 0, 100, 100], "final_score": 0.80},
            {"frame_idx": 100, "bbox": [0, 0, 100, 100], "final_score": 0.75},
            {"frame_idx": 5, "bbox": [0, 0, 100, 100], "final_score": 0.70},
        ]

    def test_selects_top_k(self, scored_candidates):
        config = SelectionConfig(top_k=2, strict_temporal_gap=30, relaxed_temporal_gap=15, min_score=0.55)
        result = selection.select_top_k_diverse(scored_candidates, config)
        assert len(result) == 2

    def test_respects_temporal_gap(self, scored_candidates):
        config = SelectionConfig(top_k=3, strict_temporal_gap=30, relaxed_temporal_gap=15, min_score=0.55)
        result = selection.select_top_k_diverse(scored_candidates, config)
        frame_idxs = [r["frame_idx"] for r in result]
        for i in range(len(frame_idxs)):
            for j in range(i + 1, len(frame_idxs)):
                assert abs(frame_idxs[i] - frame_idxs[j]) >= 15

    def test_respects_min_score(self):
        candidates = [
            {"frame_idx": 0, "bbox": [0, 0, 100, 100], "final_score": 0.3},
            {"frame_idx": 50, "bbox": [0, 0, 100, 100], "final_score": 0.4},
        ]
        config = SelectionConfig(top_k=2, strict_temporal_gap=30, relaxed_temporal_gap=15, min_score=0.55)
        result = selection.select_top_k_diverse(candidates, config)
        assert len(result) == 0

    def test_sorted_by_score(self, scored_candidates):
        config = SelectionConfig(top_k=3, strict_temporal_gap=10, relaxed_temporal_gap=5, min_score=0.55)
        result = selection.select_top_k_diverse(scored_candidates, config)
        scores = [r["final_score"] for r in result]
        assert scores == sorted(scores, reverse=True)
