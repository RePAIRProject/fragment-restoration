#import torch
from ultralytics import YOLO
# Model

model_name = 'train_single_annotations_300_epochs'
model = YOLO(f'/home/lucap/code/yolov5/runs/train/{model_name}/weights/best.pt')  # load a custom model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path=)
model.val()