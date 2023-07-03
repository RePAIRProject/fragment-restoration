import torch
import pdb 
import os
import cv2 
import numpy as np 
from ultralytics import YOLO

# Model
model_name = 'training_yolo_shapes_512'
#model = YOLO('yolov8n-seg.yaml').load()  # build from YAML and transfer weights
model = YOLO(f'/home/lucap/code/yolov5/runs/segment/{model_name}/weights/best.pt') 
#pdb.set_trace()
images_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/mating_set/images_cropped'
# '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/yolo_dataset_shapes/test/images'
output_folder = f'/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/mating_set/segmentation_results_{model_name}'
#vis_folder = f'/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/black_mark_visualization_for_{img_folder_name}'
os.makedirs(output_folder, exist_ok=True)
#os.makedirs(vis_folder, exist_ok=True)

for img in os.listdir(images_folder):
    
    full_path = os.path.join(images_folder, img)
    imgcv = cv2.imread(full_path)
    img_input = cv2.resize(imgcv, (512, 512))
    results = model(img_input)
   
    result = results[0]
    segmentation_mask = np.zeros((512, 512), dtype=np.float32)
    if result.masks is not None:
        for mm, mask in enumerate(result.masks):
            segmentation_mask += result.masks.data.cpu().numpy()[mm,:,:] 

    cv2.imwrite(os.path.join(output_folder, img), segmentation_mask)
    cv2.imwrite(os.path.join(output_folder, f"1class_{img[:-4]}.png"), (segmentation_mask > 0)*255)
    print('saved', img)
