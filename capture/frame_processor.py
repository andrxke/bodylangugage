"""
Frame pre- and post-processing utilities.

Handles the format conversions and transformations needed between
OpenCV's BGR frames and MediaPipe's RGB input, plus optional resizing
and mirroring for a natural webcam experience.
"""

import cv2
import numpy as np

from config import CaptureConfig


def preprocess(
    frame: np.ndarray,
    config: CaptureConfig,
) -> np.ndarray:
    """Prepare a raw BGR frame for MediaPipe processing.

    Steps:
      1. Resize to target dimensions (if configured).
      2. Optionally flip horizontally (mirror mode for webcam).
      3. Convert BGR → RGB (MediaPipe expects RGB input).

    Args:
        frame:  Raw BGR frame from cv2.VideoCapture.
        config: Capture configuration with size and mirror settings.

    Returns:
        Processed RGB frame ready for MediaPipe.
    """
    # Resize if target dimensions are specified.
    if config.frame_width and config.frame_height:
        frame = cv2.resize(
            frame,
            (config.frame_width, config.frame_height),
            interpolation=cv2.INTER_LINEAR,
        )

    # Mirror for a natural webcam experience (only for live camera).
    if config.mirror and isinstance(config.source, int):
        frame = cv2.flip(frame, 1)

    # Convert BGR (OpenCV default) → RGB (MediaPipe expects this).
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return rgb_frame


def postprocess(rgb_frame: np.ndarray) -> np.ndarray:
    """Convert an RGB frame back to BGR for OpenCV display.

    Args:
        rgb_frame: Frame in RGB format.

    Returns:
        Frame in BGR format for cv2.imshow().
    """
    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)


def create_video_source(source: int | str) -> cv2.VideoCapture:
    """Create and validate a video capture source.

    Args:
        source: Camera index (int, e.g. 0 for default webcam) or
                path to a video file (str).

    Returns:
        Opened cv2.VideoCapture object.

    Raises:
        RuntimeError: If the video source cannot be opened.
    """
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        source_desc = (
            f"camera index {source}" if isinstance(source, int)
            else f"video file '{source}'"
        )
        raise RuntimeError(f"Failed to open {source_desc}")

    return cap


def get_source_fps(cap: cv2.VideoCapture) -> float:
    """Get the FPS of the video source.

    For webcams this may return 0 or an unreliable value, in which case
    we default to 30 FPS.

    Args:
        cap: Opened VideoCapture.

    Returns:
        Frames per second (float), minimum 1.0.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default assumption for webcams.
    return fps
