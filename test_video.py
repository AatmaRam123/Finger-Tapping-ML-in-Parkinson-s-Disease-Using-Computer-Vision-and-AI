from ultralytics import YOLO

model = YOLO("./runs/finger_detector/weights/best.pt")

model.predict(
    source="./PDAV/PD50.MOV",  # change video name
    save=True,
    conf=0.25
)