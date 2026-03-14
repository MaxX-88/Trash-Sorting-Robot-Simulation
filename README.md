# Trash Sorting Robot Simulation

PyBullet simulation with computer vision and multi-arm robot control. YOLO detects objects on a conveyor belt; configurable Kuka arms pick trash and drop it in a recycling bin.

## Overview

- **Multi-arm simulation**: Configurable number of Kuka arms along a conveyor. Belt length and camera positions scale with arm count.
- **Computer vision**: YOLO detects objects. Runs at configurable intervals for speed (e.g. every 5 frames).
- **Synthetic data**: Scripts to generate labeled images with YCB objects for training.
- **URDF support**: YCB and custom URDFs for simulation and dataset creation.
- **Robot control**: FSM-based logic, IK for end-effector motion, debug GUI with sliders. All parameters in `SimConfig` (`src/simulation/config.py`).

## Project Structure

```
assets/           # URDFs and 3D models (YCB, variants, trash bin)
data/             # Datasets, processed images, class lists
models/           # Model weights and training scripts
output/           # Generated images, run outputs
scripts/          # Data generation, model testing, utilities
src/              # Simulation, control, utils
tests/            # Simulation and model tests
```

## Main Components

- `src/configurable_arms.py`: Main entry point. Multi-arm sim with configurable YOLO interval, render interval, and display. Saves videos to `configurable_arms_vids/`.
- `src/single_arm.py`: Single-arm sim (simpler, one Kuka).
- `src/simulation/`: Config, conveyor, trash bin, object loader.
- `src/utils/`: Camera, debug GUI, PyBullet helpers, YCB loading.
- `scripts/generate_synthetic_ycb.py`, `scripts/generate_dataset.py`: Synthetic dataset generation.
- `models/trash_detector/`: YOLO training.
- `tests/`: Arm movement, model inference, URDF tests.

## Setup

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- Optional: GPU for faster YOLO inference

### Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/MaxX-88/Trash-Sorting-Robot-Simulation
   cd robotics-project
   ```

2. Create conda env from `environment.yml`:

   ```bash
   conda env create -f environment.yml
   conda activate kuka
   ```

   Or with pip: `pip install -r requirements.txt` (if you keep it for pip-only setups).

## Usage

### Run multi-arm simulation (recommended)

```bash
python src/configurable_arms.py
```

Options:

- `--num-arms N` – Number of arms (default: 3)
- `--no-video` – Disable video capture
- `--yolo-interval N` – Run YOLO every N frames (default: 5)
- `--no-display` – No cv2 windows (faster)
- `--render-interval N` – Render cameras every N frames (default: 1)

Example: 5 arms, no display, YOLO every 10 frames:

```bash
python src/configurable_arms_fast.py --num-arms 5 --no-display --yolo-interval 10
```

### Run single-arm simulation

```bash
python src/single_arm.py
```

### Generate synthetic dataset

```bash
python scripts/generate_dataset.py
```

## Customization

- **URDFs**: Add to `assets/urdf/ycb/` or `assets/urdf/ycb_variants/`.
- **Simulation**: Edit `SimConfig` in `src/simulation/config.py` (spawn intervals, trash/recycling classes, pitch adjust, etc.).
- **Model**: Train with synthetic data and Ultralytics YOLO.

## Acknowledgements

- [kwonathan/ycb_urdfs](https://github.com/kwonathan/ycb_urdfs) for YCB URDF files.
