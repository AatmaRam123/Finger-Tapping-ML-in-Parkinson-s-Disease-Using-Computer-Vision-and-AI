from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="C:/Users/PC/Downloads/PDA/DATA/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    project="C:/Users/PC/Downloads/PDA/runs",
    name="finger_detector"
)