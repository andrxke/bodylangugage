"""
Batch video clip → landmark extraction.

Processes all video clips in a data directory through MediaPipe's
PoseLandmarker and saves the extracted landmark sequences as .npz
files alongside the original videos. This decouples the slow pose
extraction step from the fast training step.

Usage:
    # Process all clips (skips already-processed files)
    python -m training.prepare_data --data-dir data/

    # Force reprocessing of all clips
    python -m training.prepare_data --data-dir data/ --force

    # Review each clip with skeleton overlay after extraction
    python -m training.prepare_data --data-dir data/ --review
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from capture.frame_processor import get_source_fps
from capture.pose_tracker import PoseTracker
from config import CaptureConfig
from models.landmark_data import FrameLandmarks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Video file extensions to search for.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_video_clips(data_dir: Path) -> list[Path]:
    """Recursively find all video files in the data directory.

    Args:
        data_dir: Root directory to search (typically data/clips/).

    Returns:
        Sorted list of video file paths.
    """
    clips = []
    clips_dir = data_dir / "clips"

    search_dir = clips_dir if clips_dir.exists() else data_dir

    for path in sorted(search_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            clips.append(path)

    return clips


def process_clip(
    video_path: Path,
    model_path: str = "models/pose_landmarker_lite.task",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract pose landmarks from a single video clip.

    Runs every frame through MediaPipe PoseLandmarker in VIDEO mode
    and collects the landmark sequences.

    Args:
        video_path: Path to the video file.
        model_path: Path to the PoseLandmarker .task model.

    Returns:
        Tuple of (timestamps, landmarks, world_landmarks, metadata):
          - timestamps:      (T,) int64 — frame timestamps in ms
          - landmarks:       (T, 33, 4) float32 — normalized coords
          - world_landmarks: (T, 33, 3) float32 — world coords in meters
          - metadata:        dict with source info

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = get_source_fps(cap)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configure the tracker for VIDEO mode (not live stream).
    config = CaptureConfig(
        source=str(video_path),
        model_path=model_path,
        frame_width=None,   # Use native resolution.
        frame_height=None,
        mirror=False,
    )

    tracker = PoseTracker(config)
    frames: list[FrameLandmarks] = []

    frame_idx = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, bgr_frame = cap.read()
            if not ret:
                break

            # Convert to RGB for MediaPipe.
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = tracker.process_frame(rgb_frame, timestamp_ms)
            if result is not None:
                frames.append(result)

            frame_idx += 1

            # Progress logging for long videos.
            if frame_idx % 100 == 0:
                logger.info(
                    "  %s: processed %d/%d frames",
                    video_path.name,
                    frame_idx,
                    total_frames,
                )
    finally:
        tracker.close()
        cap.release()

    elapsed = time.time() - start_time

    if not frames:
        raise RuntimeError(
            f"No frames processed from {video_path}. "
            "Is the video file valid?"
        )

    # Stack into arrays.
    frame_arrays = [f.to_arrays() for f in frames]

    timestamps = np.array(
        [a["timestamp"] for a in frame_arrays], dtype=np.int64,
    )
    landmarks = np.stack(
        [a["landmarks"] for a in frame_arrays], axis=0,
    )
    world_landmarks = np.stack(
        [a["world_landmarks"] for a in frame_arrays], axis=0,
    )

    detected_count = sum(1 for f in frames if f.is_detected)

    metadata = {
        "source": str(video_path),
        "fps": fps,
        "frame_count": len(frames),
        "detected_frames": detected_count,
        "detection_rate": round(detected_count / len(frames), 3),
        "duration_seconds": round(len(frames) / fps, 2),
        "processing_time_seconds": round(elapsed, 2),
    }

    return timestamps, landmarks, world_landmarks, metadata


def save_landmarks(
    output_path: Path,
    timestamps: np.ndarray,
    landmarks: np.ndarray,
    world_landmarks: np.ndarray,
    metadata: dict,
) -> None:
    """Save extracted landmarks to a compressed .npz file.

    Args:
        output_path:     Path for the output .npz file.
        timestamps:      (T,) int64
        landmarks:       (T, 33, 4) float32
        world_landmarks: (T, 33, 3) float32
        metadata:        Dict of session info (JSON-serialized).
    """
    np.savez_compressed(
        output_path,
        timestamps=timestamps,
        landmarks=landmarks,
        world_landmarks=world_landmarks,
        metadata=json.dumps(metadata),
    )


def review_clip(npz_path: Path) -> None:
    """Play back a processed clip with skeleton overlay for visual QA.

    Args:
        npz_path: Path to the .npz landmark file.
    """
    from visualization.skeleton_renderer import SkeletonRenderer
    from config import VisualizationConfig

    data = np.load(str(npz_path), allow_pickle=True)
    landmarks = data["landmarks"]       # (T, 33, 4)
    metadata = json.loads(str(data["metadata"]))

    renderer = SkeletonRenderer(VisualizationConfig())
    fps = metadata.get("fps", 30.0)
    delay_ms = max(1, int(1000 / fps))

    logger.info(
        "Reviewing %s (%d frames, %.1fs)",
        npz_path.name,
        len(landmarks),
        metadata.get("duration_seconds", 0),
    )

    for t in range(len(landmarks)):
        # Create a blank canvas.
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Reconstruct FrameLandmarks for the renderer.
        lm = landmarks[t]
        is_nan = np.all(np.isnan(lm))

        frame_data = FrameLandmarks(
            timestamp_ms=int(data["timestamps"][t]),
            landmarks=None if is_nan else lm,
            world_landmarks=None,
        )

        canvas = renderer.draw(canvas, frame_data)

        # Show frame number.
        cv2.putText(
            canvas,
            f"Frame {t}/{len(landmarks)}",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(f"Review: {npz_path.name}", canvas)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q") or key == 27:
            break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract pose landmarks from video clips for training.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root data directory containing clips/ subfolder.",
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_lite.task",
        help="Path to the PoseLandmarker .task model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess clips that already have .npz files.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Play back each clip with skeleton overlay after extraction.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for batch landmark extraction."""
    args = parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    clips = find_video_clips(data_dir)
    if not clips:
        logger.error(
            "No video files found in %s. "
            "Place video clips in %s/clips/ directory.",
            data_dir,
            data_dir,
        )
        sys.exit(1)

    logger.info("Found %d video clips to process.", len(clips))

    processed = 0
    skipped = 0
    failed = 0

    for clip_path in clips:
        npz_path = clip_path.with_suffix(".npz")

        if npz_path.exists() and not args.force:
            logger.info("  Skipping %s (already processed)", clip_path.name)
            skipped += 1
            continue

        logger.info("Processing: %s", clip_path.name)

        try:
            timestamps, landmarks, world_landmarks, metadata = process_clip(
                clip_path, model_path=args.model,
            )

            save_landmarks(
                npz_path, timestamps, landmarks, world_landmarks, metadata,
            )

            logger.info(
                "  → Saved %s (%d frames, %.0f%% detected, %.1fs)",
                npz_path.name,
                metadata["frame_count"],
                metadata["detection_rate"] * 100,
                metadata["duration_seconds"],
            )

            if args.review:
                review_clip(npz_path)

            processed += 1

        except Exception as e:
            logger.error("  Failed to process %s: %s", clip_path.name, e)
            failed += 1

    logger.info(
        "Done: %d processed, %d skipped, %d failed (of %d total)",
        processed,
        skipped,
        failed,
        len(clips),
    )


if __name__ == "__main__":
    main()
