"""Tests for SAM backend selection logic (no torch needed)."""
import pytest

from yolo_sam_labeler.backends import (
    detect_backend_kind, BACKEND_KIND_SAM1, BACKEND_KIND_SAM2,
)
from yolo_sam_labeler.backends.sam1 import guess_sam1_model_type
from yolo_sam_labeler.backends.sam2 import (
    looks_like_sam2, guess_sam2_config, guess_sam2_label,
)


# ===========================================================================
# Backend kind detection
# ===========================================================================


class TestDetectBackendKind:
    def test_sam1_official_filenames(self):
        assert detect_backend_kind("sam_vit_h_4b8939.pth") == BACKEND_KIND_SAM1
        assert detect_backend_kind("sam_vit_l_0b3195.pth") == BACKEND_KIND_SAM1
        assert detect_backend_kind("sam_vit_b_01ec64.pth") == BACKEND_KIND_SAM1

    def test_sam2_filenames(self):
        assert detect_backend_kind("sam2_hiera_large.pt") == BACKEND_KIND_SAM2
        assert detect_backend_kind("sam2.1_hiera_tiny.pt") == BACKEND_KIND_SAM2
        assert detect_backend_kind("sam2.1_hiera_base_plus.pt") == BACKEND_KIND_SAM2

    def test_with_full_path(self):
        assert detect_backend_kind("/data/weights/sam_vit_h_4b8939.pth") == BACKEND_KIND_SAM1
        assert detect_backend_kind("/data/weights/sam2.1_hiera_large.pt") == BACKEND_KIND_SAM2

    def test_unknown_falls_back_to_sam1(self):
        """Unknown filenames default to SAM 1 to preserve existing behavior."""
        assert detect_backend_kind("custom_weights.pth") == BACKEND_KIND_SAM1
        assert detect_backend_kind("") == BACKEND_KIND_SAM1


# ===========================================================================
# SAM 1 model type guessing
# ===========================================================================


class TestSam1Guess:
    def test_official_filenames(self):
        assert guess_sam1_model_type("sam_vit_h_4b8939.pth") == "vit_h"
        assert guess_sam1_model_type("sam_vit_l_0b3195.pth") == "vit_l"
        assert guess_sam1_model_type("sam_vit_b_01ec64.pth") == "vit_b"

    def test_partial_match(self):
        assert guess_sam1_model_type("custom_vit_h_finetuned.pth") == "vit_h"
        assert guess_sam1_model_type("my_vit_b.pth") == "vit_b"

    def test_default(self):
        assert guess_sam1_model_type("unknown.pth") == "vit_h"


# ===========================================================================
# SAM 2 helpers
# ===========================================================================


class TestSam2Helpers:
    def test_looks_like_sam2(self):
        assert looks_like_sam2("sam2_hiera_large.pt") is True
        assert looks_like_sam2("sam2.1_hiera_tiny.pt") is True
        assert looks_like_sam2("/path/to/sam2_hiera_small.pt") is True
        assert looks_like_sam2("sam_vit_h.pth") is False

    def test_guess_sam2_config(self):
        assert guess_sam2_config("sam2.1_hiera_large.pt").endswith("sam2.1_hiera_l.yaml")
        assert guess_sam2_config("sam2_hiera_small.pt").endswith("sam2_hiera_s.yaml")

    def test_guess_sam2_config_unknown(self):
        assert guess_sam2_config("custom_sam2.pt") is None

    def test_guess_sam2_label(self):
        assert guess_sam2_label("sam2.1_hiera_large.pt") == "hiera_large"
        assert guess_sam2_label("sam2_hiera_tiny.pt") == "hiera_tiny"
