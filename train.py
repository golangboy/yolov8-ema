from ultralytics import YOLO
# build a new model from scratch
model = YOLO("ultralytics/cfg/models/v8/yolov8_dcn.yaml")
model.train(data="ultralytics/cfg/datasets/coco-pose.yaml",
            epochs=1)
print(model)
