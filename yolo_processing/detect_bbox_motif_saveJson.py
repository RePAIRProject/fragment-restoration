import torch
import os
import cv2
import numpy as np
import json

# Model
model_name = 'train_single_annotations_300_epochs'
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path=f'/home/sinem/PycharmProjects/fragment-restoration/Models/YOLO/Detectionv5/OneBoxPerMotif/best.pt')
images_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF_published_100dpi/processed/RGB_restored'
output_folder = f'/home/sinem/PycharmProjects/fragment-restoration/Results/Yolo_motif_detection/results_bounding_boxes'
#vis_folder = f'/home/sinem/PycharmProjects/fragment-restoration/Results/Yolo_motif_detection/black_mark_visualization'

os.makedirs(output_folder, exist_ok=True)
#os.makedirs(vis_folder, exist_ok=True)

for img in os.listdir(images_folder):
    full_path = os.path.join(images_folder, img)
    imgcv = cv2.imread(full_path)
    results = model(full_path)

    bbox_motifs = np.zeros_like(imgcv)
    bboxes_data = []

    for bm in results.xyxy[0]:
        xmax, ymax, xmin, ymin = bm[:4]
        class_label = int(bm[5].item())

        # Append bounding box data
        bboxes_data.append({
            "xmin": xmin.item(),
            "ymin": ymin.item(),
            "xmax": xmax.item(),
            "ymax": ymax.item(),
            "class_label": class_label
        })

        # Red rectangles for visualization
        cv2.rectangle(imgcv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)

    # Save bounding box data as JSON
    json_data = {
        "image_name": img,
        "bounding_boxes": bboxes_data
    }
    json_output_path = os.path.join(output_folder, f"{os.path.splitext(img)[0]}.json")
    with open(json_output_path, "w") as json_file:
        json.dump(json_data, json_file)

    # Save visualization with bounding boxes drawn in red
    cv2.imwrite(os.path.join(output_folder, img), imgcv)

