import os 
import matplotlib.pyplot as plt 
import cv2 
import pdb 
import numpy as np 

unet_preds_folder = 'runs/run742347053016131_RGB_inpainted_images_3classes_200epochs_augmented_lr0.001_HSV/results_test_set'
yolo_preds_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/yolov8_seg/segmentation_results_training_yolo_shapes_512'
gt_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/segmap3c'
input_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/RGB'
test_images = np.loadtxt('/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/test.txt', dtype='str')

for test_image in test_images:
    plt.subplot(141)
    input_image = cv2.imread(os.path.join(input_folder, test_image))
    plt.imshow(cv2.resize(input_image, (512, 512)))

    plt.subplot(142)
    gt_image = cv2.imread(os.path.join(gt_folder, test_image))
    gt_resized = cv2.resize(gt_image, (512, 512))
    gt_1class = (gt_resized > 1)*255
    plt.imshow(gt_1class)

    plt.subplot(143)
    unet_pred = plt.imread(os.path.join(unet_preds_folder, f"1class_{test_image}"))
    unet_error = np.sum(np.abs(gt_1class[:,:,0]/255 - unet_pred))
    plt.title(f"UNET (error: {unet_error})")
    plt.imshow(unet_pred)

    plt.subplot(144)
    yolo_pred = plt.imread(os.path.join(yolo_preds_folder, f"1class_{test_image}"))
    yolo_error = np.sum(np.abs(gt_1class[:,:,0]/255 - yolo_pred))
    plt.title(f"YOLO (error: {yolo_error})")
    plt.imshow(yolo_pred)

    plt.show()
    #pdb.set_trace()

