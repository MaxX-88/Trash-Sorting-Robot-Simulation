import cv2
from ultralytics import YOLO


MODEL_PATH = "models/trash_detector/weights/new_best_model.pt"
IMAGE_PATH = "/Users/maxxie/Desktop/robotics-project/data/processed/images/004_sugar_box0.jpeg"  # path to the image you want to test


model = YOLO(MODEL_PATH)


image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")


display_image = image.copy()


results = model.predict(source=image, conf=0.25, save=False)


for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = box.conf[0]
    cls = int(box.cls[0])
    label = f"{model.names[cls]} {conf:.2f}"

    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display_image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


cv2.imshow("YOLO Detection", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
