import os 
import matplotlib.pyplot as plt 
import cv2 
import pdb 
import numpy as np 
import json

def mean_iou(preds, gt, classes_list=[]):
    
    #pdb.set_trace()
    if len(classes_list) > 0:

        ious = []
        for j, motif_class in enumerate(classes_list):
            motif_count = np.sum((gt == motif_class))
            if motif_count > 0:
                intersection_mc = np.sum((gt == motif_class) * (preds == motif_class))
                union_mc = np.sum(((gt == motif_class) + (preds == motif_class))>0)
                iou_mc = intersection_mc / union_mc 
                ious.append(iou_mc)
        if np.isclose(np.sum(ious), 0):
            mean_iou = 0
        else:
            mean_iou = np.mean(ious)
        return mean_iou
    else:
        return 0

def mean_pixel_accuracy(preds, gt, classes_list=[]):

    if len(classes_list) > 0:

        pas = []
        for j, motif_class in enumerate(classes_list):
            motif_count = np.sum((gt == motif_class))
            if motif_count > 0:
                pa_mc = np.sum((preds==gt) * (gt == motif_class))
                pas.append(pa_mc / motif_count) 
        if np.isclose(np.sum(pas), 0):
            mean_pa = 0
        else:
            mean_pa = np.mean(pas)
        return mean_pa
    else:
        return 0

def main():

    unet_simplified_preds_folder = 'runs/run2618025592825107_simplifiedUNET_RGB_inpainted_images512x512_13classes_200epochs_augmented_lr0.001_HSV/results_simplified_UNET_512x512_test_set'
    unet_classic_preds_folder = 'runs/run3633000501830811_classicUNET_RGB_inpainted_images512x512_13classes_200epochs_augmented_lr0.001_HSV/results_classic_UNET_512x512_test_set'
    # yolo_preds_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/yolov8_seg/segmentation_results_training_yolo_shapes_512'
    yolo_preds_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/yolov8_seg/segmentation_results_v2_inpainted_training_yolo_shapes_inpainted512'
    unet_gt_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/segmap13c'
    yolo_gt_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/segmap14c'
    input_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/RGB_inpainted'
    test_images = np.loadtxt('/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/test.txt', dtype='str')
    output_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/unet_vs_yolo_13classes'
    os.makedirs(output_folder, exist_ok=True)

    # average
    IoU_unet_simplif13 = [] #np.zeros((len(test_images)))
    IoU_unet_classic13 = [] # np.zeros((len(test_images)))
    IoU_yolo13 = [] # np.zeros((len(test_images)))

    # motif
    IoU_unet_simplif12 = [] #np.zeros((len(test_images)))
    IoU_unet_classic12 = [] # np.zeros((len(test_images)))
    IoU_yolo12 = [] # np.zeros((len(test_images)))

    # average
    PA_unet_simplif13 = [] # np.zeros((len(test_images)))
    PA_unet_classic13 = [] # np.zeros((len(test_images)))
    PA_yolo13 = [] # np.zeros((len(test_images)))

    # motif
    PA_unet_simplif12 = [] # np.zeros((len(test_images)))
    PA_unet_classic12 = [] # np.zeros((len(test_images)))
    PA_yolo12 = [] # np.zeros((len(test_images)))

    for j, test_image in enumerate(test_images):
        

        plt.figure(figsize=(32, 32))
        plt.title(f'{test_image[:-4]}: Color Image', fontsize=32)
        plt.subplot(231)
        input_image = cv2.imread(os.path.join(input_folder, test_image))
        input_resized = cv2.resize(input_image, (512, 512))
        plt.imshow(input_resized)

        plt.subplot(232)
        plt.title(f'{test_image[:-4]}: Ground Truth Mask (UNET)', fontsize=32)
        #unet_gt_image = cv2.imread(os.path.join(unet_gt_folder, test_image))
        unet_gt_image = plt.imread(os.path.join(unet_gt_folder, test_image))*255
        #pdb.set_trace()
        unet_gt_resized = cv2.resize(unet_gt_image, (512, 512), interpolation=cv2.INTER_NEAREST)
        unet_gt_resized = np.round(unet_gt_resized).astype(int)
        plt.imshow(unet_gt_resized, vmin=0, vmax=13)

        plt.subplot(233)
        plt.title(f'{test_image[:-4]}: Ground Truth Mask (YOLO)', fontsize=32)
        yolo_gt_image = plt.imread(os.path.join(yolo_gt_folder, test_image))
        yolo_gt_image = (yolo_gt_image*255)[:,:,0]
        yolo_gt_resized = cv2.resize(yolo_gt_image, (512, 512), interpolation=cv2.INTER_NEAREST)
        yolo_gt_resized = np.round(yolo_gt_resized).astype(int)
        plt.imshow(yolo_gt_resized, vmin=0, vmax=13)

        unet_simplif_pred = plt.imread(os.path.join(unet_simplified_preds_folder, f"{test_image}"))
        unet_simplif_pred = np.round(unet_simplif_pred*255).astype(int)
        unet_classic_pred = plt.imread(os.path.join(unet_classic_preds_folder, f"{test_image}"))
        unet_classic_pred = np.round(unet_classic_pred*255).astype(int)
        yolo_pred = plt.imread(os.path.join(yolo_preds_folder, f"{test_image}"))
        yolo_pred = np.round(yolo_pred*255).astype(int)

        if np.max(np.unique(unet_gt_resized)) > 0:

            # IOU
            IoU_unet_simplif13.append(mean_iou(unet_simplif_pred, unet_gt_resized, classes_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
            IoU_unet_classic13.append(mean_iou(unet_classic_pred, unet_gt_resized, classes_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
            IoU_yolo13.append(mean_iou(yolo_pred, yolo_gt_resized, classes_list=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
            # here we are changing the classes list removing the 0 class (background) !
            IoU_unet_simplif12.append(mean_iou(unet_simplif_pred, unet_gt_resized, classes_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
            IoU_unet_classic12.append(mean_iou(unet_classic_pred, unet_gt_resized, classes_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
            IoU_yolo12.append(mean_iou(yolo_pred, yolo_gt_resized, classes_list=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))

            # PA
            PA_unet_simplif13.append(mean_pixel_accuracy(unet_simplif_pred, unet_gt_resized, classes_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
            PA_unet_classic13.append(mean_pixel_accuracy(unet_classic_pred, unet_gt_resized, classes_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
            PA_yolo13.append(mean_pixel_accuracy(yolo_pred, yolo_gt_resized, classes_list=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
            # here we are changing the classes list removing the 0 class (background) !
            PA_unet_simplif12.append(mean_pixel_accuracy(unet_simplif_pred, unet_gt_resized, classes_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
            PA_unet_classic12.append(mean_pixel_accuracy(unet_classic_pred, unet_gt_resized, classes_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
            PA_yolo12.append(mean_pixel_accuracy(yolo_pred, yolo_gt_resized, classes_list=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))

            # Show results!
            plt.subplot(234)
            plt.title(f'Simplified UNET\nIOU: {IoU_unet_simplif13[-1]:.3f}, PA: {PA_unet_simplif13[-1]:.3f}', fontsize=32)
            plt.imshow(unet_simplif_pred, vmin=0, vmax=13)

            plt.subplot(235)
            plt.title(f'Classic UNET\nIOU: {IoU_unet_classic13[-1]:.3f}, PA: {PA_unet_classic13[-1]:.3f}', fontsize=32)
            plt.imshow(unet_classic_pred, vmin=0, vmax=13)

            plt.subplot(236)
            plt.title(f'Yolo\nIOU: {IoU_yolo13[-1]:.3f}, PA: {PA_yolo13[-1]:.3f}', fontsize=32)
            plt.imshow(yolo_pred, vmin=0, vmax=13)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, test_image))
            plt.close()
            print(f"{j:02d}/{len(test_images)} {test_image[:-4]} -- IOU UNET (s): {IoU_unet_simplif13[-1]:.3f}, UNET: {IoU_unet_classic13[-1]:.3f}, YOLO: {IoU_yolo13[-1]:.3f}", end='\r')

    errors = {
        "IOU_UNET_SIMPLIFIED_AVG": IoU_unet_simplif13,
        "IOU_UNET_AVG": IoU_unet_classic13,
        "IOU_YOLO_AVG": IoU_yolo13,
        "PA_UNET_SIMPLIFIED_AVG": PA_unet_simplif13,
        "PA_UNET_AVG": PA_unet_classic13,
        "PA_YOLO_AVG": PA_yolo13,
        "IOU_UNET_SIMPLIFIED_MOTIF": IoU_unet_simplif12,
        "IOU_UNET_MOTIF": IoU_unet_classic12,
        "IOU_YOLO_MOTIF": IoU_yolo12,
        "PA_UNET_SIMPLIFIED_MOTIF": PA_unet_simplif12,
        "PA_UNET_MOTIF": PA_unet_classic12,
        "PA_YOLO_MOTIF": PA_yolo12
    }

    print("#" * 65)
    print("AVERAGE (13 classes)")
    print("Mean IoU:")
    print(f" UNET (simplified): {np.mean(IoU_unet_simplif13):.3f}")
    print(f" UNET             : {np.mean(IoU_unet_classic13):.3f}")
    print(f" YOLO             : {np.mean(IoU_yolo13):.3f}")

    print("Mean Pixel Accuracy:")
    print(f" UNET (simplified): {np.mean(PA_unet_simplif13):.3f}")
    print(f" UNET             : {np.mean(PA_unet_classic13):.3f}")
    print(f" YOLO             : {np.mean(PA_yolo13):.3f}")

    print("-" * 65)
    print("MOTIF (12 classes)")

    print("Mean IoU:")
    print(f" UNET (simplified): {np.mean(IoU_unet_simplif12):.3f}")
    print(f" UNET             : {np.mean(IoU_unet_classic12):.3f}")
    print(f" YOLO             : {np.mean(IoU_yolo12):.3f}")

    print("Mean Pixel Accuracy:")
    print(f" UNET (simplified): {np.mean(PA_unet_simplif12):.3f}")
    print(f" UNET             : {np.mean(PA_unet_classic12):.3f}")
    print(f" YOLO             : {np.mean(PA_yolo12):.3f}")
    print("#" * 65)


    with open(os.path.join(output_folder, "unet_vs_yolo_13classes.json"), 'w') as jres:
        json.dump(errors, jres, indent=2)

if __name__ == '__main__':
    main()