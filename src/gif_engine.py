"""GIF Assembly and Optimization Engine using Pillow."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def create_animated_gif(
    frames: Sequence[Image.Image | str | Path],
    output_path: str | Path = "outputs/animation.gif",
    fps: int = 8,
    loop: int = 0,
    optimize: bool = True,
    ping_pong: bool = False,
    max_dimension: int | None = None,
) -> Path:
    """Stitches a list of static frames into an optimized animated GIF.

    Args:
        frames: Sequence of PIL Image instances or file paths.
        output_path: Destination path for the generated GIF file.
        fps: Target playback speed in frames per second (1 to 60).
        loop: Number of loops (0 for infinite loop).
        optimize: Whether to attempt palette compression to reduce file size.
        ping_pong: If True, appends frames in reverse order for seamless bounce animation.
        max_dimension: Optional maximum width/height to resize frames proportionally.

    Returns:
        Path to the saved GIF file.

    Raises:
        ValueError: If `frames` sequence is empty or invalid.
    """
    if not frames:
        raise ValueError("Cannot create GIF from empty sequence of frames.")

    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert all inputs to PIL Image in RGB mode
    loaded_frames: list[Image.Image] = []
    for item in frames:
        if isinstance(item, (str, Path)):
            img = Image.open(item)
        elif isinstance(item, Image.Image):
            img = item
        else:
            raise TypeError(f"Unsupported frame item type: {type(item)}")

        # Ensure uniform RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")
        loaded_frames.append(img)

    # Establish uniform frame size based on the first frame
    base_size = loaded_frames[0].size
    if max_dimension is not None:
        width, height = base_size
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            base_size = (int(width * scale), int(height * scale))

    normalized_frames: list[Image.Image] = []
    for frame in loaded_frames:
        if frame.size != base_size:
            frame = frame.resize(base_size, Image.Resampling.LANCZOS)
        normalized_frames.append(frame)

    # Handle Ping-Pong / Boomerang effect
    if ping_pong and len(normalized_frames) > 2:
        # e.g. [0, 1, 2, 3] -> [0, 1, 2, 3, 2, 1]
        reverse_frames = normalized_frames[-2:0:-1]
        normalized_frames.extend(reverse_frames)

    # Calculate frame duration in milliseconds
    fps = max(1, min(fps, 60))
    duration_ms = max(20, int(1000 / fps))

    # Pillow GIF writer requires quantization or palette conversion
    # Using adaptive palette for crisp color reproduction
    palette_frames: list[Image.Image] = [
        f.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for f in normalized_frames
    ]

    first_frame = palette_frames[0]
    append_frames = palette_frames[1:]

    logger.info(
        "Saving GIF with %d frames at %d FPS (%dms/frame) to %s",
        len(palette_frames),
        fps,
        duration_ms,
        target_path,
    )

    first_frame.save(
        target_path,
        format="GIF",
        save_all=True,
        append_images=append_frames,
        duration=duration_ms,
        loop=loop,
        optimize=optimize,
    )

    return target_path
