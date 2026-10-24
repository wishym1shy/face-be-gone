"""End-to-end tests: run the script on synthetic clips and inspect the output.

Tests marked xfail(strict=True) describe known bugs tracked in #5. They fail
today; once the fix lands they pass, and strict mode then fails the run until
the marker is removed, so a fix can't go unnoticed and a regression can't
slip back in afterwards."""

import subprocess

import numpy as np
import pytest

from helpers import (
    audio_streams, beep_times, env_without_ffmpeg, flash_times, frame_count,
    ffmpeg_major_version, opencv_read_info, psnr, rotation, run_script, video_stream,
)

ISSUE_5 = "output pipeline bug, tracked in #5"


def ok(result):
    assert result.returncode == 0, f"script failed:\n{result.stdout}\n{result.stderr}"
    return result


# --- basics ------------------------------------------------------------------

def test_help():
    result = ok(run_script("--help"))
    assert "--backend" in result.stdout


def test_haar_end_to_end(clip, tmp_path):
    """The real Haar detector, through the whole pipeline."""
    out = tmp_path / "out.mp4"
    ok(run_script(clip, out, "--backend", "haar"))
    src, dst = video_stream(clip), video_stream(out)
    assert (dst["width"], dst["height"]) == (src["width"], src["height"])
    # Within one frame: the exact count is test_keeps_every_frame_when_audio_ends_early's job
    assert abs(frame_count(out) - frame_count(clip)) <= 1, f"source: {opencv_read_info(clip)}"
    assert len(audio_streams(out)) == 1


def test_blur_lands_in_the_detected_box(clip, tmp_path):
    """With a fixed fake detection, only the box area should change."""
    plain, boxed = tmp_path / "plain.mp4", tmp_path / "boxed.mp4"
    ok(run_script(clip, plain, stub="none"))
    ok(run_script(clip, boxed, "--strength", "51", "--margin", "0", stub="box"))

    def middle_frame(path):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", "select=eq(n\\,30)",
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(240, 320).astype(int)

    diff = np.abs(middle_frame(plain) - middle_frame(boxed))
    inside = np.zeros_like(diff, bool)
    inside[30:90, 40:120] = True  # stub box: (w/8, h/8)-(3w/8, 3h/8)
    margin = np.zeros_like(diff, bool)
    margin[22:98, 32:128] = True  # small allowance for chroma and encoder bleed at the edge
    assert diff[inside].mean() > 3 * max(diff[~margin].mean(), 0.5)


@pytest.mark.xfail(ffmpeg_major_version() >= 9, strict=True,
                   reason=ISSUE_5 + ": -shortest drops the last frame on ffmpeg 9+ (6.1 and 7.0 keep it)")
def test_keeps_every_frame_when_audio_ends_early(short_audio_clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(short_audio_clip, out, stub="none"))
    assert frame_count(out) == frame_count(short_audio_clip), f"source: {opencv_read_info(short_audio_clip)}"


def test_rotated_footage_comes_out_upright(rotated_clip, rotated_reference, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(rotated_clip, out, stub="none"))
    stream = video_stream(out)
    displayed = (stream["width"], stream["height"])
    if rotation(stream) % 180:
        displayed = displayed[::-1]
    assert displayed == (240, 320)
    # Same orientation as ffmpeg's upright rendering; a wrong turn scores ~10 dB
    assert psnr(rotated_reference, out) > 20


# --- known bugs (#5) ---------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=ISSUE_5 + ": mp4v intermediate is copied into the output")
def test_output_quality_close_to_source(clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(clip, out, stub="none"))
    assert psnr(clip, out) > 40


@pytest.mark.xfail(strict=True, reason=ISSUE_5 + ": VFR sources are written at a constant rate")
def test_audio_stays_in_sync_on_variable_frame_rate(vfr_clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(vfr_clip, out, stub="none"))
    flashes, beeps = flash_times(out), beep_times(out)
    assert len(flashes) == 4
    for flash, beep in zip(flashes, beeps):
        assert abs(flash - beep) < 0.05, f"flash at {flash:.3f}s, beep at {beep:.3f}s"


@pytest.mark.xfail(strict=True, reason=ISSUE_5 + ": color range tag is dropped")
def test_full_range_is_preserved(full_range_clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(full_range_clip, out, stub="none"))
    assert video_stream(out).get("color_range") == "pc"
    assert psnr(full_range_clip, out) > 40


@pytest.mark.xfail(strict=True, reason=ISSUE_5 + ": only the first audio track is kept")
def test_keeps_all_audio_tracks(two_track_clip, tmp_path):
    out = tmp_path / "out.mov"
    ok(run_script(two_track_clip, out, stub="none"))
    assert len(audio_streams(out)) == 2


@pytest.mark.xfail(strict=True, reason=ISSUE_5 + ": missing ffmpeg only prints a warning")
def test_missing_ffmpeg_is_an_error(clip, tmp_path):
    out = tmp_path / "out.mp4"
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run_script(clip, out, stub="none", env=env_without_ffmpeg(empty))
    assert "Traceback" not in result.stderr
    assert result.returncode != 0
    assert "ffmpeg" in result.stderr
