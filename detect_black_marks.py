import torch
import pdb 
import os
import cv2 
import numpy as np 
# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/lucap/code/fragment-restoration/Yolov5_Black_Mark_Detection/Yolo_best_model_pretrained_best_weights.pt')
images_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/images_cropped'
output_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/black_mark_region_cropped'
vis_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/black_mark_visualization_cropped'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(vis_folder, exist_ok=True)

for img in os.listdir(images_folder):
    
    full_path = os.path.join(images_folder, img)
    imgcv = cv2.imread(full_path)
    results = model(full_path)
    bm_detection = np.zeros_like(imgcv)
    #bm_visualization = np.zeros_like(imgcv)
    #pdr = results.pandas().xyxy[0]
    bms = results.xyxy[0]
    #pdb.set_trace()
    for bm in bms:
        xmax, ymax, xmin, ymin = bm[:4]

        # white filled boxes for inpainting
        cv2.rectangle(bm_detection, (np.round(xmin.item()).astype(int), np.round(ymin.item()).astype(int)), \
            (np.round(xmax.item()).astype(int), np.round(ymax.item()).astype(int)), (255, 255, 255), cv2.FILLED)

        # red rectangles for visualization
        cv2.rectangle(imgcv, (np.round(xmin.item()).astype(int), np.round(ymin.item()).astype(int)), \
            (np.round(xmax.item()).astype(int), np.round(ymax.item()).astype(int)), (0, 0, 255))
    
    cv2.imwrite(os.path.join(output_folder, img), bm_detection)
    cv2.imwrite(os.path.join(vis_folder, img), imgcv)
    #cv2.imwrite(os.path.join(output_folder, f"{img[:-4]}_orig.png"), imgcv)
    #pdb.set_trace()