"""Unit tests for the GIF creation and optimization engine."""

from pathlib import Path

import pytest
from PIL import Image

from src.gif_engine import create_animated_gif


@pytest.fixture
def sample_frames() -> list[Image.Image]:
    """Generates three colored 100x100 synthetic test frames."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return [Image.new("RGB", (100, 100), color=c) for c in colors]


def test_create_gif_empty_sequence_raises_error(tmp_path: Path) -> None:
    """Ensure passing an empty sequence raises ValueError."""
    out = tmp_path / "empty.gif"
    with pytest.raises(ValueError, match="Cannot create GIF from empty sequence"):
        create_animated_gif([], output_path=out)


def test_create_gif_success(sample_frames: list[Image.Image], tmp_path: Path) -> None:
    """Verify standard animated GIF creation and metadata."""
    out = tmp_path / "test_success.gif"
    result = create_animated_gif(sample_frames, output_path=out, fps=10, loop=0)

    assert result.exists()
    assert result.is_file()

    # Reopen and inspect GIF properties
    with Image.open(result) as img:
        assert img.format == "GIF"
        assert getattr(img, "is_animated", False)
        assert img.n_frames == 3
        # 10 fps -> 100ms duration per frame
        assert img.info.get("duration") == 100
        assert img.info.get("loop") == 0


def test_create_gif_ping_pong(sample_frames: list[Image.Image], tmp_path: Path) -> None:
    """Verify ping-pong mode appends frames in reverse order."""
    out = tmp_path / "ping_pong.gif"
    # 3 frames: [0, 1, 2] -> Ping-pong creates [0, 1, 2, 1] (4 frames)
    result = create_animated_gif(sample_frames, output_path=out, fps=5, ping_pong=True)

    with Image.open(result) as img:
        assert img.n_frames == 4


def test_create_gif_different_dimensions(tmp_path: Path) -> None:
    """Verify engine handles frames with varying resolutions cleanly."""
    frame1 = Image.new("RGB", (100, 100), color=(255, 255, 255))
    frame2 = Image.new("RGB", (200, 150), color=(128, 128, 128))
    frame3 = Image.new("RGBA", (50, 50), color=(0, 0, 0, 255))

    out = tmp_path / "mixed_sizes.gif"
    result = create_animated_gif([frame1, frame2, frame3], output_path=out, fps=4)

    with Image.open(result) as img:
        assert img.n_frames == 3
        # All frames normalized to base frame size (100x100)
        assert img.size == (100, 100)


def test_create_gif_max_dimension_downscaling(tmp_path: Path) -> None:
    """Verify downscaling when max_dimension is enforced."""
    large_frame = Image.new("RGB", (800, 600), color=(10, 20, 30))
    out = tmp_path / "downscaled.gif"

    result = create_animated_gif([large_frame, large_frame], output_path=out, max_dimension=400)

    with Image.open(result) as img:
        assert max(img.size) <= 400
