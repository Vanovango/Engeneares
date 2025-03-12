from ultralytics import YOLO


model = YOLO("yolo11n.pt")

model.predict(source="peoples.png", show=True, save=True, conf=0.9)
