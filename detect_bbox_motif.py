import torch
import pdb 
import os
import cv2 
import numpy as np 
import matplotlib.pyplot as plt

# Model
model_name = 'train_single_annotations_300_epochs'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'/home/lucap/code/yolov5/runs/train/{model_name}/weights/best.pt')

images_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/yolo_dataset_shapes/test/images'
output_folder = f'/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/yolov5/results_bounding_boxes_{model_name}'
#vis_folder = f'/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/black_mark_visualization_for_{img_folder_name}'
os.makedirs(output_folder, exist_ok=True)
#os.makedirs(vis_folder, exist_ok=True)

for img in os.listdir(images_folder):
    
    full_path = os.path.join(images_folder, img)
    imgcv = cv2.imread(full_path)
    results = model(full_path)
    
    bbox_motifs = np.zeros_like(imgcv)
    #bm_visualization = np.zeros_like(imgcv)
    #pdr = results.pandas().xyxy[0]
    bboxes = results.xyxy[0]
    #pdb.set_trace()
    for bm in bboxes:
        #pdb.set_trace()
        xmax, ymax, xmin, ymin = bm[:4]
        class_label = int(bm[5].item())
        
        # white filled boxes for inpainting
        cv2.rectangle(bbox_motifs, (np.round(xmin.item()).astype(int), np.round(ymin.item()).astype(int)), \
            (np.round(xmax.item()).astype(int), np.round(ymax.item()).astype(int)), (class_label, class_label, class_label), cv2.FILLED)

        # # red rectangles for visualization
        # cv2.rectangle(imgcv, (np.round(xmin.item()).astype(int), np.round(ymin.item()).astype(int)), \
        #     (np.round(xmax.item()).astype(int), np.round(ymax.item()).astype(int)), (0, 0, 255), 3)
        #print(len(bboxes))
        # plt.imshow(bbox_motifs)
        # plt.show()

    #print('here we have ', len(bboxes))    
    cv2.imwrite(os.path.join(output_folder, img), bbox_motifs[:,:,0])
    #pdb.set_trace()
    #cv2.imwrite(os.path.join(vis_folder, img), imgcv)
    #cv2.imwrite(os.path.join(output_folder, f"{img[:-4]}_orig.png"), imgcv)
    #pdb.set_trace()