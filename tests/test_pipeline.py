"""End-to-end tests: run the script on synthetic clips and inspect the output.

A known bug can be written as xfail(strict=True) naming its issue. Strict mode
fails the run once the bug is fixed, until the marker is removed, so a fix
can't go unnoticed and a regression can't slip back in afterwards."""

import re
import shutil
import subprocess

import numpy as np
import pytest

from helpers import (
    audio_streams, beep_times, env_without_ffmpeg, flash_times, frame_count,
    haar_available, opencv_read_info, probe, psnr, rotation, run_script, video_stream,
)


def ok(result):
    assert result.returncode == 0, f"script failed:\n{result.stdout}\n{result.stderr}"
    return result


def failed(result):
    assert result.returncode == 1, f"expected exit status 1:\n{result.stdout}\n{result.stderr}"
    assert "Traceback" not in result.stderr
    return result


# --- basics ------------------------------------------------------------------

def test_help():
    result = ok(run_script("--help"))
    assert "--backend" in result.stdout


def test_version():
    result = ok(run_script("--version"))
    assert re.fullmatch(r"face-be-gone \d+\.\d+\.\d+", result.stdout.strip())


@pytest.mark.skipif(not haar_available(), reason="OpenCV 5 has no Haar face model")
def test_haar_end_to_end(clip, tmp_path):
    """The real Haar detector, through the whole pipeline."""
    out = tmp_path / "out.mp4"
    ok(run_script(clip, out, "--backend", "haar"))
    src, dst = video_stream(clip), video_stream(out)
    assert (dst["width"], dst["height"]) == (src["width"], src["height"])
    assert frame_count(out) == frame_count(clip), f"source: {opencv_read_info(clip)}"
    assert len(audio_streams(out)) == 1


@pytest.mark.skipif(haar_available(), reason="only on OpenCV 5")
def test_haar_on_opencv5_is_a_clear_error(clip, tmp_path):
    out = tmp_path / "out.mp4"
    result = failed(run_script(clip, out, "--backend", "haar"))
    assert "opencv-python<5" in result.stderr
    assert not out.exists()


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


# --- output fidelity (regressions of #5) -------------------------------------

def test_output_quality_close_to_source(clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(clip, out, stub="none"))
    assert psnr(clip, out) > 40


def test_keeps_every_frame_when_audio_ends_early(short_audio_clip, tmp_path):
    """-shortest used to drop the last frame here (seen with ffmpeg 9)."""
    out = tmp_path / "out.mp4"
    ok(run_script(short_audio_clip, out, stub="none"))
    assert frame_count(out) == frame_count(short_audio_clip), f"source: {opencv_read_info(short_audio_clip)}"


def test_audio_stays_in_sync_on_variable_frame_rate(vfr_clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(vfr_clip, out, stub="none"))
    flashes, beeps = flash_times(out), beep_times(out)
    assert len(flashes) == 4
    for flash, beep in zip(flashes, beeps):
        assert abs(flash - beep) < 0.05, f"flash at {flash:.3f}s, beep at {beep:.3f}s"


def test_full_range_is_preserved(full_range_clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(full_range_clip, out, stub="none"))
    assert video_stream(out).get("color_range") == "pc"
    assert psnr(full_range_clip, out) > 40


def color_tags(path):
    stream = video_stream(path)
    return stream.get("color_space"), stream.get("color_primaries"), stream.get("color_transfer")


def test_color_tags_are_copied(clip, tmp_path):
    assert color_tags(clip) == ("bt709", "bt709", "bt709"), "test clip is missing its tags"
    out = tmp_path / "out.mp4"
    ok(run_script(clip, out, stub="none"))
    assert color_tags(out) == color_tags(clip)


def test_hdr_tags_are_copied(hdr_tagged_clip, tmp_path):
    """Losing the PQ transfer tag makes HDR footage display washed out."""
    assert color_tags(hdr_tagged_clip) == ("bt2020nc", "bt2020", "smpte2084"), "test clip is missing its tags"
    out = tmp_path / "out.mp4"
    ok(run_script(hdr_tagged_clip, out, stub="none"))
    assert color_tags(out) == color_tags(hdr_tagged_clip)


def test_keeps_all_audio_tracks(two_track_clip, tmp_path):
    out = tmp_path / "out.mov"
    ok(run_script(two_track_clip, out, stub="none"))
    assert len(audio_streams(out)) == 2


@pytest.mark.parametrize("ext, codec", [(".mov", "pcm_s16le"), (".mp4", "aac")])
def test_audio_copied_when_the_container_allows(two_track_clip, tmp_path, ext, codec):
    """PCM can be copied into .mov but has to be re-encoded for .mp4."""
    out = tmp_path / f"out{ext}"
    ok(run_script(two_track_clip, out, stub="none"))
    assert [s["codec_name"] for s in audio_streams(out)] == [codec, codec]


def test_timecode_is_kept(timecode_clip, tmp_path):
    out = tmp_path / "out.mov"
    ok(run_script(timecode_clip, out, stub="none"))
    timecodes = [s.get("tags", {}).get("timecode") for s in probe(out)["streams"]]
    assert "01:00:00:00" in timecodes


def test_anamorphic_aspect_is_kept(anamorphic_clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(anamorphic_clip, out, stub="none"))
    assert video_stream(out).get("display_aspect_ratio") == "16:9"


def test_odd_dimensions_are_padded(odd_size_clip, tmp_path):
    out = tmp_path / "out.mp4"
    ok(run_script(odd_size_clip, out, stub="none"))
    stream = video_stream(out)
    assert (stream["width"], stream["height"]) == (322, 242)


# --- codecs ------------------------------------------------------------------

@pytest.mark.parametrize("codec, ext, codec_name", [
    ("h264", ".mkv", "h264"),
    ("hevc", ".mp4", "hevc"),
    ("prores", ".mov", "prores"),
    ("dnxhr", ".mov", "dnxhd"),
])
def test_codec_choice(clip, tmp_path, codec, ext, codec_name):
    out = tmp_path / f"out{ext}"
    ok(run_script(clip, out, "--codec", codec, stub="none"))
    assert video_stream(out)["codec_name"] == codec_name
    assert frame_count(out) == frame_count(clip)


def test_crf_controls_quality(clip, tmp_path):
    high, low = tmp_path / "high.mp4", tmp_path / "low.mp4"
    ok(run_script(clip, high, "--crf", "10", stub="none"))
    ok(run_script(clip, low, "--crf", "35", stub="none"))
    assert high.stat().st_size > low.stat().st_size
    assert psnr(clip, high) > psnr(clip, low)


# --- errors ------------------------------------------------------------------

def test_codec_container_mismatch_is_an_error(clip, tmp_path):
    out = tmp_path / "out.mp4"
    result = failed(run_script(clip, out, "--codec", "prores", stub="none"))
    assert ".mov" in result.stderr
    assert not out.exists()


def test_same_input_and_output_is_an_error(clip):
    failed(run_script(clip, clip, stub="none"))


def test_missing_input_is_an_error(tmp_path):
    failed(run_script(tmp_path / "nope.mp4", tmp_path / "out.mp4", stub="none"))


def test_missing_ffmpeg_is_an_error(clip, tmp_path):
    out = tmp_path / "out.mp4"
    empty = tmp_path / "empty"
    empty.mkdir()
    result = failed(run_script(clip, out, stub="none", env=env_without_ffmpeg(empty)))
    assert "ffmpeg" in result.stderr
    assert "--allow-no-audio" in result.stderr


def test_allow_no_audio_falls_back_to_opencv(clip, tmp_path):
    out = tmp_path / "out.mp4"
    empty = tmp_path / "empty"
    empty.mkdir()
    ok(run_script(clip, out, "--allow-no-audio", stub="none", env=env_without_ffmpeg(empty)))
    assert video_stream(out)["codec_name"] == "mpeg4"
    assert audio_streams(out) == []


@pytest.mark.skipif(shutil.which("nvidia-smi") is not None, reason="needs a machine without an NVIDIA GPU")
def test_failed_encode_leaves_no_output(clip, tmp_path):
    """NVENC can't start without an NVIDIA GPU. Whether ffmpeg lacks the encoder
    (caught up front) or fails on the first frame, no file may be left behind."""
    out = tmp_path / "out.mp4"
    result = failed(run_script(clip, out, "--codec", "h264_nvenc", stub="none"))
    assert "nvenc" in result.stderr.lower()
    assert not out.exists()
