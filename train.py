from ultralytics import YOLO

# Load a model
model = YOLO("D:/Data/Project/启元遥感分割竞赛/ultralytics-rs/ultralytics/cfg/models/11/yolo11n-seg-wtconv.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="D:/Data/Project/启元遥感分割竞赛/RS_data/RS_normalized/dataset.yaml", epochs=100, imgsz=640, batch=1)
# model.val(data="/root/data/RS_data/RS_normalized/dataset.yaml")