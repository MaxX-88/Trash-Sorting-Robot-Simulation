"""
trash detector training with explicit preprocessing and augmentation config
"""
import os
from pathlib import Path

from ultralytics import YOLO

# resolve paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
OUTPUT_PROJECT = PROJECT_ROOT / "models" / "trash_detector"

# preprocessing/normalization
IMAGE_SIZE = 512
NORMALIZE_MEAN = [0.0, 0.0, 0.0]
NORMALIZE_STD = [1.0, 1.0, 1.0]
PADDING_STRATEGY = "letterbox"
BGR_TO_RGB = True  # ultralytics expects rgb internally

# augmentation pipeline (explicit values, matches ultralytics defaults)
AUGMENTATION = {
    "hsv_h": 0.015,      # hue jitter
    "hsv_s": 0.7,        # saturation
    "hsv_v": 0.4,        # value / brightness
    "degrees": 0.0,      # rotation
    "translate": 0.1,    # translation
    "scale": 0.5,        # scale gain
    "shear": 0.0,        # shear
    "perspective": 0.0,  # perspective
    "flipud": 0.0,       # vertical flip
    "fliplr": 0.5,       # horizontal flip
    "mosaic": 1.0,       # mosaic prob
    "mixup": 0.0,        # mixup prob
    "erasing": 0.4,      # random erasing
    "copy_paste": 0.0,   # copy-paste
}

# optimizer and scheduler
OPTIMIZER = "auto"       # sgd or adam based on model
LR0 = 0.01              # initial learning rate
LRF = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1

# loss weights
BOX_LOSS_WEIGHT = 7.5
CLS_LOSS_WEIGHT = 0.5
DFL_LOSS_WEIGHT = 1.5

# training
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 100
AMP = True              # mixed precision
CACHE = False           # cache images to ram/disk
WORKERS = 8
SEED = 0
DETERMINISTIC = True


def get_training_config():
    """build full training config from module constants"""
    return {
        "data": str(DATA_YAML),
        "epochs": EPOCHS,
        "batch": BATCH_SIZE,
        "imgsz": IMAGE_SIZE,
        "project": str(OUTPUT_PROJECT),
        "name": "yolo_train",
        "exist_ok": True,
        "pretrained": True,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "lrf": LRF,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "warmup_momentum": WARMUP_MOMENTUM,
        "warmup_bias_lr": WARMUP_BIAS_LR,
        "box": BOX_LOSS_WEIGHT,
        "cls": CLS_LOSS_WEIGHT,
        "dfl": DFL_LOSS_WEIGHT,
        "patience": PATIENCE,
        "amp": AMP,
        "cache": CACHE,
        "workers": WORKERS,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        **AUGMENTATION,
    }


if __name__ == "__main__":
    assert DATA_YAML.exists(), f"data yaml not found: {DATA_YAML}"

    model = YOLO("yolo11n.pt")
    config = get_training_config()

    results = model.train(**config)

# windows training for gpu, can also migrate to google colab by changing the data path
