# Body Language Capture Pipeline

A modular Python pipeline that captures a presenter's body skeleton from a webcam or video file using MediaPipe's PoseLandmarker, stores sessions in NumPy `.npz` format, and structures data for downstream comparison to professional speakers and Pepper robot reenactment.

## Project Goals

This is **Stage 1** of a larger pipeline:

1. **✅ Stage 1 — Capture**: Extract the presenter's pose skeleton (this project)
2. **Stage 2 — Analysis**: Compare body language to professional speakers
3. **Stage 3 — Feedback**: Generate actionable feedback for the presenter
4. **Stage 4 — Reenactment**: Drive Pepper the robot to reenact the presenter's gestures

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Video       │───▶│  Pose        │───▶│  Frame       │
│  Source      │    │  Tracker     │    │  Landmarks   │
│  (webcam/    │    │  (MediaPipe  │    │  (33 pts ×   │
│   video)     │    │  PoseLander) │    │   3D coords) │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                    ┌──────────────┐            │
                    │  Skeleton    │◀───────────┤
                    │  Renderer    │            │
                    │  (OpenCV)    │            │
                    └──────────────┘            │
                                               │
                    ┌──────────────┐            │        ┌──────────────┐
                    │  Session     │◀───────────┤───────▶│  Pepper      │
                    │  Recorder    │            │        │  Joint       │
                    │  (.npz)      │            │        │  Mapping     │
                    └──────────────┘            │        └──────────────┘
                                               │
                                        ┌──────▼───────┐
                                        │  Future:     │
                                        │  Analysis &  │
                                        │  Feedback    │
                                        └──────────────┘
```

## Setup

### Prerequisites

- Python 3.10+
- Webcam (for live capture)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd bodylangugage

# Install dependencies
pip install -r requirements.txt

# Download the PoseLandmarker model
python download_model.py                    # Lite model (fastest)
python download_model.py --complexity full  # Full model (balanced)
python download_model.py --complexity heavy # Heavy model (most accurate)
```

## Usage

### Live Webcam Capture

```bash
# Basic webcam capture with skeleton overlay
python main.py --source 0

# Start recording immediately
python main.py --source 0 --record

# Disable mirror mode
python main.py --source 0 --no-mirror
```

### Video File Processing

```bash
# Process a video file
python main.py --source presentation.mp4

# Process and record
python main.py --source presentation.mp4 --record
```

### Headless Mode (No Display)

```bash
python main.py --source 0 --record --no-display
```

### Controls

| Key   | Action                           |
|-------|----------------------------------|
| `q`   | Quit                             |
| `ESC` | Quit                             |
| `r`   | Toggle recording on/off          |

## Recording Format (.npz)

Sessions are saved as compressed NumPy `.npz` files with the following arrays:

| Array             | Shape       | Dtype   | Description                            |
|-------------------|-------------|---------|----------------------------------------|
| `timestamps`      | `(N,)`      | int64   | Frame timestamps in milliseconds       |
| `landmarks`       | `(N, 33, 4)`| float32 | Normalized x, y, z, visibility         |
| `world_landmarks` | `(N, 33, 3)`| float32 | Real-world coordinates in meters       |
| `metadata`        | scalar      | str     | JSON-encoded session info              |

### Loading a Recording

```python
import numpy as np
import json

data = np.load("recordings/session_20250527_120000.npz", allow_pickle=True)

timestamps = data["timestamps"]           # (N,) ms timestamps
landmarks = data["landmarks"]             # (N, 33, 4) normalized
world_landmarks = data["world_landmarks"] # (N, 33, 3) meters
metadata = json.loads(str(data["metadata"]))

print(f"Frames: {len(timestamps)}")
print(f"Duration: {metadata['duration_seconds']}s")
```

## Pepper Robot Integration

The `models/pepper_joint_mapping.py` module converts the 33 pose landmarks to Pepper's 12 controllable joint angles:

```python
from models.pepper_joint_mapping import compute_pepper_angles

# After loading or capturing landmarks:
angles = compute_pepper_angles(world_landmarks[frame_idx])
# Returns: {'HeadYaw': 0.1, 'LShoulderPitch': -0.5, ...}

# Future: Send to Pepper via NAOqi
# pepper_motion.setAngles(list(angles.keys()), list(angles.values()), 0.2)
```

## Gesture Classification Training

The pipeline includes a Spatial-Temporal Graph Convolutional Network (ST-GCN) model to classify gestures (e.g., facing the audience, arms crossed, or arms hidden) directly from pose landmark sequences.

### 1. Dataset Setup

Prepare a training data directory (e.g., `data/`) with a subdirectory `clips/` containing raw videos, and a `labels.csv` manifest at the root:

```text
data/
├── clips/
│   ├── clip_001.mp4
│   ├── clip_002.mp4
│   └── ...
└── labels.csv
```

The `labels.csv` file should contain the filenames of the clips and binary labels (`0` or `1`) for target gesture types (`facing`, `arms_crossed`, `arms_hidden`):

```csv
filename,facing,arms_crossed,arms_hidden
clip_001.mp4,1,0,0
clip_002.mp4,1,1,0
clip_003.mp4,0,0,1
```

### 2. Extract Landmarks (Pre-processing)

Extract pose landmarks from all video clips in the data directory and save them as compressed `.npz` files alongside the source videos. This decouples slow pose extraction from fast training:

```bash
python -m training.prepare_data --data-dir data/
```

* **Useful flags:**
  * `--force`: Reprocess clips that already have `.npz` files.
  * `--review`: Play back each clip with a skeleton overlay after extraction to verify accuracy.

### 3. Train the Model

Train the ST-GCN classifier for a specific gesture category (one of: `facing`, `arms_crossed`, or `arms_hidden`):

```bash
python -m training.train --data-dir data/ --gesture facing
```

* **Useful flags:**
  * `--epochs 100`: Maximum training epochs (default: 100).
  * `--batch-size 8`: Training batch size (default: 8).
  * `--lr 0.001`: Initial learning rate (default: 0.001).
  * `--patience 15`: Early stopping patience (default: 15).
  * `--resume`: Resume training from the latest checkpoint.

Trained model weights and training configs will be saved to `models/gesture_classifiers/`.

### 4. Running Inference

To run the live webcam capture pipeline with your trained gesture classification models enabled:

```bash
python main.py --source 0 --classify
```

---

## Project Structure

```
bodylangugage/
├── main.py                         # Entry point
├── config.py                       # Central configuration
├── download_model.py               # Model download script
├── requirements.txt                # Dependencies
├── capture/
│   ├── pose_tracker.py             # MediaPipe PoseLandmarker wrapper
│   └── frame_processor.py          # Frame pre/post-processing
├── models/
│   ├── landmark_data.py            # FrameLandmarks data class
│   └── pepper_joint_mapping.py     # Landmark → Pepper joint angles
├── recording/
│   └── session_recorder.py         # .npz session serialization
├── visualization/
│   └── skeleton_renderer.py        # Skeleton overlay rendering
└── utils/
    └── angle_utils.py              # 3D vector math utilities
```

## License

MIT
