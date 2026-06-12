"""
Download the MediaPipe PoseLandmarker .task model bundle.

Downloads the model file from Google's public storage and saves it
to the local models/ directory. Supports three complexity levels:
  - lite:  Fastest, lower accuracy (~3 MB)
  - full:  Balanced speed and accuracy (~6 MB)
  - heavy: Highest accuracy, slower (~26 MB)

Usage:
    python download_model.py                        # Downloads lite model
    python download_model.py --complexity full       # Downloads full model
    python download_model.py --complexity heavy      # Downloads heavy model
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Google's public URLs for the PoseLandmarker .task model bundles.
MODEL_URLS = {
    "lite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task"
    ),
    "full": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task"
    ),
    "heavy": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/"
        "pose_landmarker_heavy.task"
    ),
}


def download_model(complexity: str, output_dir: str = "models") -> str:
    """Download the PoseLandmarker model for the given complexity.

    Args:
        complexity: One of 'lite', 'full', or 'heavy'.
        output_dir: Directory to save the model file.

    Returns:
        Path to the downloaded model file.

    Raises:
        ValueError: If an invalid complexity level is specified.
        RuntimeError: If the download fails.
    """
    if complexity not in MODEL_URLS:
        raise ValueError(
            f"Invalid complexity '{complexity}'. "
            f"Choose from: {', '.join(MODEL_URLS.keys())}"
        )

    url = MODEL_URLS[complexity]
    filename = f"pose_landmarker_{complexity}.task"
    output_path = Path(output_dir) / filename

    # Create output directory if needed.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded.
    if output_path.exists():
        logger.info("Model already exists: %s", output_path)
        return str(output_path)

    logger.info("Downloading %s model from:", complexity)
    logger.info("  %s", url)
    logger.info("  → %s", output_path)

    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}") from e

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Download complete (%.1f MB)", file_size_mb)

    return str(output_path)


def main() -> None:
    """Parse arguments and download the model."""
    parser = argparse.ArgumentParser(
        description="Download MediaPipe PoseLandmarker model",
    )
    parser.add_argument(
        "--complexity",
        choices=["lite", "full", "heavy"],
        default="lite",
        help="Model complexity level. Default: lite.",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory. Default: models/",
    )

    args = parser.parse_args()

    try:
        path = download_model(args.complexity, args.output_dir)
        print(f"\nModel ready: {path}")
    except Exception as e:
        logger.error("Error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
