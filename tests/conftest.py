"""Synthetic test clips, generated once per test session with ffmpeg."""

import shutil

import pytest

from helpers import ffmpeg

if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.exit("ffmpeg and ffprobe must be on PATH to run the tests", returncode=1)

# A soft, camera-like picture: testsrc2's saturated bars clip in an 8-bit RGB
# round trip, which would hide real encoding differences.
SOFT = "eq=contrast=0.8:saturation=0.6,gblur=sigma=1"
COLOR_TAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def clip(media):
    """2 s, 320x240 at 29.97 fps, BT.709-tagged H.264 with an AAC tone."""
    path = media / "clip.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30000/1001:duration=2,{SOFT}",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
           "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", *COLOR_TAGS, "-c:a", "aac", path)
    return path


@pytest.fixture(scope="session")
def short_audio_clip(media):
    """Audio ends ~3 ms before the video, as it often does in camera files."""
    path = media / "short_audio.mp4"
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=30000/1001:duration=3,{SOFT}",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path)
    return path


@pytest.fixture(scope="session")
def rotated_clip(media, clip):
    """The same clip with a 90° display matrix, like vertical phone footage."""
    path = media / "rotated.mp4"
    ffmpeg("-display_rotation", "90", "-i", clip, "-c", "copy", path)
    return path


@pytest.fixture(scope="session")
def rotated_reference(media, rotated_clip):
    """ffmpeg's own upright rendering of rotated_clip, near-lossless."""
    path = media / "rotated_reference.mp4"
    ffmpeg("-i", rotated_clip, "-c:v", "libx264", "-crf", "4", "-an", path)
    return path


@pytest.fixture(scope="session")
def full_range_clip(media, clip):
    path = media / "full_range.mp4"
    ffmpeg("-i", clip, "-vf", "scale=out_range=pc,format=yuvj420p", "-color_range", "pc",
           "-c:v", "libx264", "-crf", "12", "-c:a", "copy", path)
    return path


@pytest.fixture(scope="session")
def two_track_clip(media, clip):
    """A .mov with two PCM audio tracks."""
    path = media / "two_tracks.mov"
    ffmpeg("-i", clip, "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
           "-map", "0:v", "-map", "0:a", "-map", "1:a", "-c:v", "copy", "-c:a", "pcm_s16le", path)
    return path


@pytest.fixture(scope="session")
def vfr_clip(media):
    """24 s of variable frame rate video: 12 s at 24 fps, then 12 s at 60 fps.
    A white flash and an audio beep mark every 6 s, so A/V sync can be measured."""
    path = media / "vfr.mp4"
    marker = "drawbox=c=white:t=fill:enable='lt(mod(t\\,6)\\,0.1)'"
    ffmpeg("-f", "lavfi", "-i", f"color=black:s=160x120:r=24:d=12,{marker}",
           "-f", "lavfi", "-i", f"color=black:s=160x120:r=60:d=12,{marker}",
           "-f", "lavfi", "-i", "sine=f=1000:d=24:sample_rate=48000,volume=0:enable='gte(mod(t\\,6)\\,0.1)'",
           "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
           "-fps_mode", "vfr", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", path)
    return path


@pytest.fixture(scope="session")
def timecode_clip(media, clip):
    path = media / "timecode.mov"
    ffmpeg("-i", clip, "-c", "copy", "-timecode", "01:00:00:00", path)
    return path


@pytest.fixture(scope="session")
def anamorphic_clip(media, clip):
    """720x480 with non-square pixels (SAR 32:27), displayed as 16:9."""
    path = media / "anamorphic.mp4"
    ffmpeg("-i", clip, "-vf", "scale=720:480,setsar=32/27", "-c:v", "libx264", "-c:a", "copy", path)
    return path


@pytest.fixture(scope="session")
def odd_size_clip(media, clip):
    """321x241, stored as 4:4:4 because 4:2:0 can't have odd dimensions."""
    path = media / "odd_size.mp4"
    ffmpeg("-i", clip, "-vf", "scale=321:241,format=yuv444p", "-c:v", "libx264", "-c:a", "copy", path)
    return path
