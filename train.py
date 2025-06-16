from ultralytics import YOLO

# Load a model
model = YOLO("/root//ultralytics/ultralytics/cfg/models/11/yolo11x-seg-wtconv.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="/root/data/RS_data/RS_normalized/dataset.yaml", epochs=100, imgsz=640)
# model.val(data="/root/data/RS_data/RS_normalized/dataset.yaml")