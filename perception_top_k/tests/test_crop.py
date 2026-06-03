import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.crop as crop


class TestIsTooSmall:
    def test_large_crop(self):
        assert crop.is_too_small(256, 256) is False

    def test_small_short_side(self):
        assert crop.is_too_small(64, 256) is True

    def test_small_area(self):
        assert crop.is_too_small(100, 100) is True

    def test_boundary(self):
        assert crop.is_too_small(128, 256) is False


class TestSafeCrop:
    @pytest.fixture
    def sample_image(self):
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def test_valid_crop(self, sample_image):
        img_crop, info = crop.safe_crop(sample_image, [100, 100, 500, 500])
        assert img_crop is not None
        assert img_crop.shape == (400, 400, 3)
        assert info["was_clipped"] is False
        assert info["visible_area_ratio"] == 1.0

    def test_clipped_crop(self, sample_image):
        img_crop, info = crop.safe_crop(sample_image, [-50, -50, 200, 200])
        assert img_crop is not None
        assert info["was_clipped"] is True
        assert info["visible_area_ratio"] < 1.0

    def test_out_of_bounds(self, sample_image):
        img_crop, info = crop.safe_crop(sample_image, [2000, 2000, 2100, 2100])
        assert img_crop is None
        assert info["visible_area_ratio"] == 0.0

    def test_zero_area(self, sample_image):
        img_crop, info = crop.safe_crop(sample_image, [100, 100, 100, 100])
        assert img_crop is None


class TestFramePathFromIdx:
    def test_finds_standard_format(self, tmp_path):
        frame_file = tmp_path / "frame_42.jpg"
        frame_file.write_text("")
        result = crop.frame_path_from_idx(tmp_path, 42)
        assert result == frame_file

    def test_finds_simple_format(self, tmp_path):
        frame_file = tmp_path / "42.jpg"
        frame_file.write_text("")
        result = crop.frame_path_from_idx(tmp_path, 42)
        assert result == frame_file

    def test_not_found(self, tmp_path):
        result = crop.frame_path_from_idx(tmp_path, 999)
        assert result is None


class TestSaveCrop:
    def test_saves_valid_crop(self, tmp_path):
        img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        output_path = tmp_path / "test_crop.jpg"
        result = crop.save_crop(img, [100, 100, 500, 500], output_path)
        assert result is True
        assert output_path.exists()

    def test_fails_invalid_crop(self, tmp_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        output_path = tmp_path / "test_crop.jpg"
        result = crop.save_crop(img, [200, 200, 300, 300], output_path)
        assert result is False

    def test_creates_parent_dirs(self, tmp_path):
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        output_path = tmp_path / "nested" / "deep" / "crop.jpg"
        result = crop.save_crop(img, [0, 0, 200, 200], output_path)
        assert result is True
        assert output_path.exists()
