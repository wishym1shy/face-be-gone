"""Unit tests for the script's geometry, blur and tracking functions."""

import types

import numpy as np
import pytest

from helpers import load_script

fbg = load_script()


def noisy_frame(h=120, w=160, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# --- expand_box --------------------------------------------------------------

def test_expand_box_grows_each_side_by_margin():
    assert fbg.expand_box(100, 100, 40, 20, 1000, 1000, margin=0.25) == (90, 95, 150, 125)


def test_expand_box_clamps_to_frame():
    assert fbg.expand_box(2, 3, 40, 40, 50, 50, margin=0.5) == (0, 0, 50, 50)


def test_expand_box_zero_margin_is_identity():
    assert fbg.expand_box(10, 20, 30, 40, 100, 100, margin=0) == (10, 20, 40, 60)


# --- blur_region -------------------------------------------------------------

@pytest.mark.parametrize("shape", ["rect", "circle"])
@pytest.mark.parametrize("pixelate", [False, True])
def test_blur_region_leaves_pixels_outside_the_box_untouched(shape, pixelate):
    frame = noisy_frame()
    original = frame.copy()
    fbg.blur_region(frame, 40, 30, 100, 90, 35, pixelate=pixelate, shape=shape)

    outside = np.ones(frame.shape[:2], bool)
    outside[30:90, 40:100] = False
    assert np.array_equal(frame[outside], original[outside])
    assert not np.array_equal(frame[30:90, 40:100], original[30:90, 40:100])


def test_blur_region_rect_changes_the_whole_box():
    frame = noisy_frame()
    original = frame.copy()
    fbg.blur_region(frame, 40, 30, 100, 90, 35, shape="rect")
    changed = np.any(frame[30:90, 40:100] != original[30:90, 40:100], axis=2)
    assert changed.mean() > 0.95


def test_blur_region_circle_keeps_box_corners():
    frame = noisy_frame()
    original = frame.copy()
    fbg.blur_region(frame, 40, 30, 100, 90, 35, shape="circle")
    # top-left corner of the box, outside the inscribed ellipse
    assert np.array_equal(frame[30:34, 40:44], original[30:34, 40:44])
    # center of the box, inside the ellipse
    assert not np.array_equal(frame[55:65, 65:75], original[55:65, 65:75])


def test_blur_region_ignores_empty_box():
    frame = noisy_frame()
    original = frame.copy()
    fbg.blur_region(frame, 50, 50, 50, 80, 35)
    assert np.array_equal(frame, original)


def test_blur_region_handles_even_strength():
    frame = noisy_frame()
    fbg.blur_region(frame, 40, 30, 100, 90, 40)  # forced odd internally; must not raise


# --- Haar backend on OpenCV 5 ----------------------------------------------

def test_haar_without_cascade_classifier_is_a_clear_error(monkeypatch):
    """opencv-python 5 has no CascadeClassifier at all."""
    monkeypatch.delattr(fbg.cv2, "CascadeClassifier", raising=False)
    with pytest.raises(fbg.FaceBeGoneError, match="OpenCV 4"):
        fbg.make_detector(0.5, backend="haar")


def test_haar_without_model_file_is_a_clear_error(monkeypatch, tmp_path):
    """opencv-contrib-python 5 has the class but not the model file."""
    monkeypatch.setattr(fbg.cv2, "data", types.SimpleNamespace(haarcascades=str(tmp_path)), raising=False)
    with pytest.raises(fbg.FaceBeGoneError, match="opencv-python<5"):
        fbg.make_detector(0.5, backend="haar")


# --- SmoothedBox -------------------------------------------------------------

def test_smoothed_box_first_update_is_the_box():
    box = fbg.SmoothedBox(alpha=0.5)
    assert box.update(10, 20, 50, 60) == (10, 20, 50, 60)


def test_smoothed_box_alpha_one_follows_the_detection():
    box = fbg.SmoothedBox(alpha=1.0)
    box.update(0, 0, 10, 10)
    assert box.update(100, 100, 120, 120) == (100, 100, 120, 120)


def test_smoothed_box_moves_part_way():
    box = fbg.SmoothedBox(alpha=0.5)
    box.update(0, 0, 10, 10)
    assert box.update(20, 0, 30, 10) == (10, 0, 20, 10)


# --- match_detections_to_tracks ---------------------------------------------

def make_track(x0, y0, x1, y1):
    track = fbg.SmoothedBox(alpha=0.5)
    track.update(x0, y0, x1, y1)
    return track


def test_matching_picks_nearest_track():
    tracks = {"a": make_track(0, 0, 10, 10), "b": make_track(100, 100, 110, 110)}
    assert fbg.match_detections_to_tracks([(98, 98, 108, 108)], tracks, max_dist=50) == {0: "b"}


def test_matching_starts_new_track_beyond_max_dist():
    tracks = {"a": make_track(0, 0, 10, 10)}
    assert fbg.match_detections_to_tracks([(200, 200, 210, 210)], tracks, max_dist=50) == {0: None}


def test_matching_uses_each_track_once():
    tracks = {"a": make_track(0, 0, 10, 10)}
    result = fbg.match_detections_to_tracks([(1, 1, 11, 11), (2, 2, 12, 12)], tracks, max_dist=50)
    assert result == {0: "a", 1: None}


# --- format_elapsed ----------------------------------------------------------

@pytest.mark.parametrize("seconds, text", [(0, "0s"), (42.4, "42s"), (65, "1m 5s"), (3725, "1h 2m 5s")])
def test_format_elapsed(seconds, text):
    assert fbg.format_elapsed(seconds) == text
