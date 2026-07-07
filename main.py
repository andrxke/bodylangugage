"""
Body Language Capture Pipeline — Entry Point.

Captures pose skeleton data from a webcam or video file using
MediaPipe's PoseLandmarker, optionally records sessions to .npz files,
and displays a real-time skeleton overlay.

Usage:
    # Live webcam capture with display:
    python main.py --source 0

    # Process a video file and record:
    python main.py --source presentation.mp4 --record

    # Headless recording (no display window):
    python main.py --source 0 --record --no-display

    # Use a specific model:
    python main.py --model models/pose_landmarker_full.task

Controls:
    q / ESC  — Quit
    r        — Toggle recording on/off
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

from capture.frame_processor import (
    create_video_source,
    get_source_fps,
    preprocess,
    postprocess,
)
from capture.pose_tracker import PoseTracker
from classification.gesture_classifier import GestureClassifier
from config import (
    CaptureConfig,
    ClassificationConfig,
    PepperConfig,
    RecordingConfig,
    VisualizationConfig,
)
from feedback.aggregator import FeedbackAggregator
from feedback.controller import PepperFeedbackController
from feedback.motion import create_motion_driver
from feedback.speech import create_speech_driver
from models.landmark_data import FrameLandmarks
from recording.session_recorder import SessionRecorder
from visualization.skeleton_renderer import SkeletonRenderer

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Body Language Capture Pipeline — Pose Skeleton Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Video source: camera index (integer, e.g. '0') or path to a "
            "video file. Default: 0 (default webcam)."
        ),
    )
    parser.add_argument(
        "--model",
        default="models/pose_landmarker_lite.task",
        help="Path to the PoseLandmarker .task model file.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Start recording immediately on launch.",
    )
    parser.add_argument(
        "--output-dir",
        default="recordings",
        help="Directory for .npz recording files. Default: recordings/",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run in headless mode (no visualization window).",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring of webcam feed.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width. Default: 640.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height. Default: 480.",
    )
    parser.add_argument(
        "--detection-confidence",
        type=float,
        default=0.5,
        help="Minimum pose detection confidence [0.0, 1.0]. Default: 0.5.",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence [0.0, 1.0]. Default: 0.5.",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Enable ST-GCN gesture classification.",
    )
    parser.add_argument(
        "--gesture-models",
        default="models/gesture_classifiers",
        help="Directory containing trained gesture model .pt files.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Sliding window size in frames for classification. Default: 128.",
    )

    # ---- Pepper feedback arguments ----
    parser.add_argument(
        "--pepper",
        action="store_true",
        help="Enable Pepper robot feedback after the session.",
    )
    parser.add_argument(
        "--pepper-ip",
        default="127.0.0.1",
        help="Pepper robot IP address. Default: 127.0.0.1.",
    )
    parser.add_argument(
        "--pepper-port",
        type=int,
        default=9559,
        help="Pepper NAOqi port. Default: 9559.",
    )
    parser.add_argument(
        "--no-pepper-simulate",
        action="store_true",
        help=(
            "Connect to a real Pepper robot instead of simulating. "
            "Requires the NAOqi SDK."
        ),
    )
    parser.add_argument(
        "--pepper-password",
        default="nao",
        help="SSH password for the Pepper robot. Default: nao.",
    )

    return parser.parse_args()


def _parse_source(source_str: str) -> int | str:
    """Convert the source argument to int (camera) or str (file path).

    Args:
        source_str: Raw source argument from CLI.

    Returns:
        Integer camera index or string file path.
    """
    try:
        return int(source_str)
    except ValueError:
        return source_str


def main() -> None:
    """Main capture loop.

    Sets up all pipeline components, runs the frame-processing loop,
    and handles clean shutdown.
    """
    args = parse_args()

    source = _parse_source(args.source)
    is_webcam = isinstance(source, int)

    # Build configuration objects from CLI arguments.
    capture_config = CaptureConfig(
        source=source,
        model_path=args.model,
        min_detection_confidence=args.detection_confidence,
        min_tracking_confidence=args.tracking_confidence,
        frame_width=args.width,
        frame_height=args.height,
        mirror=not args.no_mirror and is_webcam,
    )

    recording_config = RecordingConfig(
        output_dir=args.output_dir,
        enabled=args.record,
    )

    vis_config = VisualizationConfig(
        enabled=not args.no_display,
    )

    classification_config = ClassificationConfig(
        enabled=args.classify,
        model_dir=args.gesture_models,
        window_size=args.window_size,
    )

    pepper_config = PepperConfig(
        enabled=args.pepper,
        ip=args.pepper_ip,
        port=args.pepper_port,
        ssh_password=args.pepper_password,
        simulate=not args.no_pepper_simulate,
    )

    # -----------------------------------------------------------------------
    # Initialize pipeline components.
    # -----------------------------------------------------------------------

    logger.info(
        "Starting capture (source=%s, model=%s)",
        source,
        capture_config.model_path,
    )

    # Open video source.
    cap = create_video_source(source)
    source_fps = get_source_fps(cap)
    logger.info("Source FPS: %.1f", source_fps)

    # Initialize pose tracker.
    tracker = PoseTracker(capture_config)

    # Initialize recorder.
    recorder = SessionRecorder(recording_config.output_dir)
    if recording_config.enabled:
        recorder.start_session(
            source=str(source),
            model=capture_config.model_path,
            fps=source_fps,
        )

    # Initialize gesture classifier.
    classifier = None
    if classification_config.enabled:
        classifier = GestureClassifier(
            model_dir=classification_config.model_dir,
            window_size=classification_config.window_size,
            stride=classification_config.stride,
        )
        if classifier.is_ready:
            logger.info("Gesture classification enabled.")
        else:
            logger.warning(
                "Gesture classification requested but no models loaded. "
                "Run 'python -m training.train' first."
            )
            classifier = None

    # Initialize visualization.
    renderer = SkeletonRenderer(vis_config) if vis_config.enabled else None

    # Initialize feedback aggregator (always active when classifying +
    # pepper is enabled, so we can build a report at session end).
    aggregator = None
    if pepper_config.enabled and classification_config.enabled:
        aggregator = FeedbackAggregator()
        logger.info("Pepper feedback enabled (simulate=%s).", pepper_config.simulate)
    elif pepper_config.enabled and not classification_config.enabled:
        logger.warning(
            "Pepper feedback requires --classify. "
            "Add --classify to enable gesture classification."
        )

    # -----------------------------------------------------------------------
    # Main processing loop.
    # -----------------------------------------------------------------------

    frame_index = 0
    loop_start_time = time.time()

    try:
        while cap.isOpened():
            ret, raw_frame = cap.read()
            if not ret:
                if not is_webcam:
                    logger.info("End of video file reached.")
                else:
                    logger.warning("Failed to read frame from webcam.")
                break

            # Compute timestamp in milliseconds.
            if is_webcam:
                timestamp_ms = int((time.time() - loop_start_time) * 1000)
            else:
                timestamp_ms = int((frame_index / source_fps) * 1000)

            # Preprocess the frame (resize, mirror, BGR→RGB).
            rgb_frame = preprocess(raw_frame, capture_config)

            # Run pose detection.
            frame_data = tracker.process_frame(rgb_frame, timestamp_ms)

            # Classify gesture (if enabled).
            gesture_results = None
            if classifier is not None and frame_data is not None:
                gesture_results = classifier.update(frame_data)

            # Feed the aggregator for Pepper feedback.
            if aggregator is not None:
                aggregator.update(gesture_results)

            # Record if active.
            if recorder.is_recording and frame_data is not None:
                recorder.record_frame(frame_data, gesture_results=gesture_results)

            # Visualize.
            if renderer is not None:
                # Convert back to BGR for display.
                display_frame = postprocess(rgb_frame)

                display_frame = renderer.draw(
                    display_frame,
                    frame_data,
                    is_recording=recorder.is_recording,
                    gesture_results=gesture_results,
                )

                cv2.imshow("Body Language Capture", display_frame)

                # Handle keyboard input.
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:  # 'q' or ESC
                    logger.info("Quit requested by user.")
                    break

                elif key == ord("r"):
                    # Toggle recording.
                    if recorder.is_recording:
                        path = recorder.stop_session()
                        logger.info("Recording saved: %s", path)
                    else:
                        recorder.start_session(
                            source=str(source),
                            model=capture_config.model_path,
                            fps=source_fps,
                        )
                        logger.info("Recording started.")

            frame_index += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")

    finally:
        # ---------------------------------------------------------------
        # Clean shutdown.
        # ---------------------------------------------------------------
        if recorder.is_recording:
            path = recorder.stop_session()
            logger.info("Recording saved on shutdown: %s", path)

        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

        # Print session summary.
        elapsed = time.time() - loop_start_time
        avg_fps = frame_index / elapsed if elapsed > 0 else 0
        logger.info(
            "Session complete: %d frames in %.1fs (avg %.1f FPS)",
            frame_index,
            elapsed,
            avg_fps,
        )

        # ---------------------------------------------------------------
        # Pepper feedback delivery (post-session).
        # ---------------------------------------------------------------
        if aggregator is not None:
            report = aggregator.build_report()

            if report.total_windows > 0:
                logger.info("Delivering Pepper feedback...")
                try:
                    motion_driver = create_motion_driver(
                        ip=pepper_config.ip,
                        port=pepper_config.port,
                        simulate=pepper_config.simulate,
                        password=pepper_config.ssh_password,
                    )
                    speech_driver = create_speech_driver(
                        ip=pepper_config.ip,
                        port=pepper_config.port,
                        simulate=pepper_config.simulate,
                        password=pepper_config.ssh_password,
                    )
                    feedback_controller = PepperFeedbackController(
                        motion_driver=motion_driver,
                        speech_driver=speech_driver,
                        pose_settle_time=pepper_config.pose_settle_time,
                        speech_pause=pepper_config.speech_pause,
                        motion_speed=pepper_config.motion_speed,
                    )
                    feedback_controller.deliver_feedback(report)
                    feedback_controller.close()
                except Exception:
                    logger.exception("Pepper feedback delivery failed.")
            else:
                logger.info(
                    "No classification data collected — "
                    "skipping Pepper feedback."
                )


if __name__ == "__main__":
    main()
