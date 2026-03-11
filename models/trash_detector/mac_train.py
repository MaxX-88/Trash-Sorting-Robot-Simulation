from ultralytics import YOLO
import torch


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO("yolo11n.pt")
    model.to(device)
    results = model.train(
        data="../../dataset/data.yaml",
        epochs=100,
        imgsz=512,
        batch=8,
        project="../../output/runs",
        name="yolo_train_mbp",
        resume=True
    )
#for mac silicon