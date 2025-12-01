# Trash Sorting Robot Simulation

This project demonstrates a robotics simulation that combines computer vision and robot control using PyBullet. It includes synthetic data generation, a sophisticated robot control system, a virtual environment, and a trash detection model using YOLO.

## Project Overview

- **Simulation**: A robotic arm interacts with multiple objects on a conveyor belt in a virtual PyBullet environment. The system supports spawning, tracking, and managing several objects at once.
- **Synthetic Data Generation**: Scripts to generate labeled images using YCB object models for training detection models.
  - Can generate your own synthetic data for other purposes
- **Computer Vision**: Uses a YOLO model to detect objects for robotic manipulation.
  - The system predicts when an object will reach the pickup point and commands the arm to intercept and grasp it using a simulated suction gripper.
  - Scripts to train your own custom YOLO model
- **URDF Support**: Loads and manipulates YCB and custom URDF objects for simulation and dataset creation.
- **Robot Control System**: Modular, extensible framework using PyBullet for simulation and a finite state machine (FSM) for high-level logic. All simulation parameters (object spawn interval, pitch adjust list, recycling/trash classes, etc.) are centralized in a `SimConfig` class for easy customization.
  - Inverse Kinematics (IK):
    - Inverse kinematics to move the end-effector to target positions.
    - Helper functions to abstract motion control.
  - Debug GUI:
    - Real-time sliders and overlays to tune parameters and visualization of the FSM state.

## Project Structure

```
assets/           # URDFs and 3D models (YCB, variants, trash bin, etc.)
data/             # Datasets, processed images/labels, class lists
models/           # Model weights and training scripts
output/           # Generated images, run outputs
scripts/          # Data generation, model testing, and utility scripts
src/              # Main source code (simulation, control, utils)
tests/            # Test scripts for simulation and models
```

## Main Components

- `src/main.py`: Main entry point. Runs the full simulation with vision and robot control. Handles object spawning, tracking, and logic for distinguishing recycling vs. trash objects.
- `src/utils/`: Camera, debug GUI, PyBullet helpers, and YCB object loading utilities.
- `src/simulation/`: Conveyor and trash bin simulation modules.
- `scripts/generate_synthetic_ycb.py`: Generate synthetic datasets with random YCB objects.
- `scripts/generate_dataset.py`: Additional dataset generation utilities.
- `models/trash_detector/`: YOLO model weights and training code.
- `tests/`: Scripts to test arm movement, model inference, and URDF loading.

## Environment Setup

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) GPU for faster YOLO inference

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MaxX-88/Trash-Sorting-Robot-Simulation
   cd robotics-project
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Recommendations:**
- Use miniconda/anaconda for managing dependencies
- Install pybullet with conda-forge if normal installation fails


## Usage

### Run the Main Simulation

```bash
python src/main.py
```

- This launches the PyBullet simulation, loads the robot, conveyor, and objects, and starts the vision-based pick-and-place loop.

### Generate Synthetic Dataset

```bash
python scripts/generate_dataset.py
```

- Generates labeled images and YOLO-format labels using random YCB objects.

## Customization

- **Add new URDFs**: Place them in `assets/urdf/ycb/` or `assets/urdf/ycb_variants/`.
- **Change simulation parameters**: Edit `SimConfig` in `src/simulation/config.py` to adjust object spawn intervals, recycling/trash class lists, pitch adjustment, and more.
- **Train your own model**: Use the synthetic dataset and Ultralytics YOLO.

## Acknowledgements

- [kwonathan/ycb_urdfs](https://github.com/kwonathan/ycb_urdfs) for YCB URDF files.
