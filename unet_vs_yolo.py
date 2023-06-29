import os 
import matplotlib.pyplot as plt 
import cv2 
import pdb 
import numpy as np 
import json

unet_preds_folder = 'runs/run16885392688472511_classicUNET_RGB_images512x512_3classes_200epochs_augmented_lr0.001_HSV/results_test_set'
yolo_preds_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/yolov8_seg/segmentation_results_training_yolo_shapes_512'
gt_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/segmap3c'
input_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/RGB'
test_images = np.loadtxt('/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/test.txt', dtype='str')
output_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/unet_vs_yolo'
os.makedirs(output_folder, exist_ok=True)

buondary_mask_root_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/mask_boundary_bands/'
shift30 = 'rdp40_shift30_moff_cropped'
img_list30 = os.listdir(os.path.join(buondary_mask_root_folder, shift30))
shift50 = 'rdp40_shift50_moff_cropped'
img_list50 = os.listdir(os.path.join(buondary_mask_root_folder, shift50))


RMSE_unet = np.zeros((len(test_images)))
RMSE_yolo = np.zeros((len(test_images)))
MAE_unet = np.zeros((len(test_images)))
MAE_yolo = np.zeros((len(test_images)))
MPE_unet = np.zeros((len(test_images)))
MPE_yolo = np.zeros((len(test_images)))

IoU_unet = np.zeros((len(test_images)))
IoU_yolo = np.zeros((len(test_images)))

MAE_unet_boundaries = np.zeros((len(test_images), 2))
MAE_yolo_boundaries = np.zeros((len(test_images), 2))
MPE_unet_boundaries = np.zeros((len(test_images), 2))
MPE_yolo_boundaries = np.zeros((len(test_images), 2))
IoU_unet_boundaries = np.zeros((len(test_images), 2))
IoU_yolo_boundaries = np.zeros((len(test_images), 2))

for j, test_image in enumerate(test_images):
    print(f"{j:02d}/{len(test_images)} - test_image", end='\r')
    plt.figure(figsize=(32, 32))
    plt.subplot(221)
    input_image = cv2.imread(os.path.join(input_folder, test_image))
    plt.imshow(cv2.resize(input_image, (512, 512)))

    plt.subplot(222)
    gt_image = cv2.imread(os.path.join(gt_folder, test_image))
    gt_resized = cv2.resize(gt_image, (512, 512))
    gt_1class = (gt_resized > 1)*255
    plt.imshow(gt_1class)

    unet_pred = plt.imread(os.path.join(unet_preds_folder, f"1class_{test_image}"))
    yolo_pred = plt.imread(os.path.join(yolo_preds_folder, f"1class_{test_image}"))

    gt_count = np.sum(gt_1class[:,:,0])
    union_gt_unet = (gt_1class[:,:,0] + unet_pred) > 0
    norm_count_unet = np.sum(union_gt_unet)
    union_gt_yolo = (gt_1class[:,:,0] + yolo_pred) > 0
    norm_count_yolo = np.sum(union_gt_yolo)

    plt.subplot(223)
    intersection_gt_unet = (gt_1class[:,:,0] > 0) * (unet_pred > 0)
    
    
    MAE_unet[j] = np.sum(np.abs(gt_1class[:,:,0]/255 - unet_pred))
    if norm_count_unet > 0:
        IoU_unet[j] = np.sum(intersection_gt_unet) / np.sum(union_gt_unet)
        MPE_unet[j] = MAE_unet[j] / norm_count_unet
    else:
        MPE_unet[j] = 0
        IoU_unet[j] = 0
    plt.title(f"UNET (MPE: {MPE_unet[j]:.3f})", fontsize=32)
    plt.imshow(unet_pred)

    plt.subplot(224)
    intersection_gt_yolo = (gt_1class[:,:,0] > 0) * (yolo_pred > 0)
    

    MAE_yolo[j] = np.sum(np.abs(gt_1class[:,:,0]/255 - yolo_pred))
    if norm_count_yolo > 0:
        IoU_yolo[j] = np.sum(intersection_gt_yolo) / np.sum(union_gt_yolo)
        MPE_yolo[j] = MAE_yolo[j] / norm_count_yolo
    else:
        MPE_yolo[j] = 0
        IoU_yolo[j] = 0
    plt.title(f"YOLO (MPE: {MPE_yolo[j]:.3f})", fontsize=32)
    plt.imshow(yolo_pred)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, test_image))
    plt.close()
    
    
    RMSE_unet[j] = np.sqrt(np.sum(np.square(gt_1class[:,:,0]/255 - unet_pred)))
    RMSE_yolo[j] = np.sqrt(np.sum(np.square(gt_1class[:,:,0]/255 - yolo_pred)))
    

    #pdb.set_trace()
    boundary_images_candidates30 = [b_path for b_path in img_list30 if test_image[:-4] in b_path]
    boundary_images_candidates50 = [b_path for b_path in img_list30 if test_image[:-4] in b_path]
    if len(boundary_images_candidates30) == 1 and gt_count:
        bs_mask = cv2.imread(os.path.join(buondary_mask_root_folder, shift30, boundary_images_candidates30[0]))
        bs_mask512 = cv2.resize(bs_mask, (512, 512))
        b_mask = (bs_mask512 > 0)
        gt_mask_bounds = (gt_1class[:,:,0]/255) * b_mask[:,:,0]
        MAE_unet_boundaries[j, 0] = np.sum(np.abs(gt_mask_bounds - unet_pred * b_mask[:,:,0]))
        MPE_unet_boundaries[j, 0] = MAE_unet_boundaries[j, 0] / np.sum(gt_mask_bounds)
        MAE_yolo_boundaries[j, 0] = np.sum(np.abs(gt_mask_bounds - yolo_pred * b_mask[:,:,0]))
        MPE_yolo_boundaries[j, 0] = MAE_yolo_boundaries[j, 0] / np.sum(gt_mask_bounds)
        union_gt_unet = (gt_mask_bounds + (unet_pred*(gt_mask_bounds>0))) > 0
        intersection_gt_unet = (gt_mask_bounds > 0) * (unet_pred > 0)
        IoU_unet_boundaries[j, 0] = np.sum(intersection_gt_unet) / np.sum(union_gt_unet)
        union_gt_yolo = (gt_mask_bounds + (yolo_pred*(gt_mask_bounds>0))) > 0
        intersection_gt_yolo = (gt_mask_bounds > 0) * (yolo_pred > 0)
        IoU_yolo_boundaries[j, 0] = np.sum(intersection_gt_yolo) / np.sum(union_gt_yolo)
    else:
        MAE_unet_boundaries[j, 0] = 0
        MPE_unet_boundaries[j, 0] = 0
        MAE_yolo_boundaries[j, 0] = 0
        MPE_yolo_boundaries[j, 0] = 0

    if len(boundary_images_candidates50) == 1 and gt_count:
        bs_mask = cv2.imread(os.path.join(buondary_mask_root_folder, shift50, boundary_images_candidates50[0]))
        bs_mask512 = cv2.resize(bs_mask, (512, 512))
        b_mask = (bs_mask512 > 0)
        gt_mask_bounds = (gt_1class[:,:,0]/255) * b_mask[:,:,0]
        MAE_unet_boundaries[j, 1] = np.sum(np.abs(gt_mask_bounds - unet_pred * b_mask[:,:,0]))
        MPE_unet_boundaries[j, 1] = MAE_unet_boundaries[j, 1] / np.sum(gt_mask_bounds)
        MAE_yolo_boundaries[j, 1] = np.sum(np.abs(gt_mask_bounds - yolo_pred * b_mask[:,:,0]))
        MPE_yolo_boundaries[j, 1] = MAE_yolo_boundaries[j, 1] / np.sum(gt_mask_bounds)
        union_gt_unet = (gt_mask_bounds + (unet_pred*(gt_mask_bounds>0))) > 0
        intersection_gt_unet = (gt_mask_bounds > 0) * (unet_pred > 0)
        IoU_unet_boundaries[j, 1] = np.sum(intersection_gt_unet) / np.sum(union_gt_unet)
        union_gt_yolo = (gt_mask_bounds + (yolo_pred*(gt_mask_bounds>0))) > 0
        intersection_gt_yolo = (gt_mask_bounds > 0) * (yolo_pred > 0)
        IoU_yolo_boundaries[j, 1] = np.sum(intersection_gt_yolo) / np.sum(union_gt_yolo)
    else:
        MAE_unet_boundaries[j, 1] = 0
        MPE_unet_boundaries[j, 1] = 0
        MAE_yolo_boundaries[j, 1] = 0
        MPE_yolo_boundaries[j, 1] = 0

errors = {
    "MAE_UNET": MAE_unet.tolist(),
    "MAE_YOLO": MAE_yolo.tolist(),
    "RMSE_UNET": RMSE_unet.tolist(),
    "RMSE_YOLO": RMSE_yolo.tolist(),
    "MPE_UNET": MPE_unet.tolist(),
    "MPE_YOLO": MPE_yolo.tolist(),
    "MAE_UNET_BOUNDS": MAE_unet_boundaries.tolist(),
    "MAE_YOLO_BOUNDS": MAE_yolo_boundaries.tolist(),
    "MPE_UNET_BOUNDS": MAE_unet_boundaries.tolist(),
    "MPE_YOLO_BOUNDS": MPE_yolo_boundaries.tolist(),
    "IoU_UNET_BOUNDS": IoU_unet_boundaries.tolist(),
    "IoU_YOLO_BOUNDS": IoU_yolo_boundaries.tolist()
}


print("1CLASS\tMAE\t\tRMSE\t\tMPE\t\tIoU")
print(f" UNET \t {np.mean(MAE_unet):.3f} \t {np.mean(RMSE_unet):.3f} \t {np.mean(MPE_unet):.3f} \t\t {np.mean(IoU_unet):.3f}")
print(f" YOLO \t {np.mean(MAE_yolo):.3f} \t {np.mean(RMSE_yolo):.3f} \t {np.mean(MPE_yolo):.3f} \t\t {np.mean(IoU_yolo):.3f}")

print("BOUNDS30\tMAE\tMPE\tIoU")
print(f" UNET \t {np.mean(MAE_unet_boundaries[:,0]):.3f} \t {np.mean(MPE_unet_boundaries[:,0]):.3f} \t {np.mean(IoU_unet_boundaries[:,0]):.3f}")
print(f" YOLO \t {np.mean(MAE_yolo_boundaries[:,0]):.3f}\t {np.mean(MPE_yolo_boundaries[:,0]):.3f} \t {np.mean(IoU_yolo_boundaries[:,0]):.3f}")

print("BOUNDS50\tMAE\tMPE\tIoU")
print(f" UNET \t {np.mean(MAE_unet_boundaries[:,1]):.3f} \t {np.mean(MPE_unet_boundaries[:,1]):.3f} \t {np.mean(IoU_unet_boundaries[:,1]):.3f}")
print(f" YOLO \t {np.mean(MAE_yolo_boundaries[:,1]):.3f}\t {np.mean(MPE_yolo_boundaries[:,1]):.3f} \t {np.mean(IoU_yolo_boundaries[:,1]):.3f}")

with open(os.path.join(output_folder, "unet_vs_yolo.json"), 'w') as jres:
    json.dump(errors, jres, indent=2)

with open(os.path.join(output_folder, "unet_vs_yolo_1CLASS.csv"), 'w') as txtres:
    txtres.write("1CLASS, MAE, MPE, IOU\n")
    txtres.write(f"UNET, {np.mean(MAE_unet):.3f}, {np.mean(MPE_unet):.3f}, {np.mean(IoU_unet):.3f}\n")
    txtres.write(f"YOLO, {np.mean(MAE_yolo):.3f}, {np.mean(MPE_yolo):.3f}, {np.mean(IoU_yolo):.3f}\n")

with open(os.path.join(output_folder, "unet_vs_yolo_BOUNDS30.csv"), 'w') as txtres:
    txtres.write("BOUNDS30, MAE, MPE, IOU\n")
    txtres.write(f"UNET, {np.mean(MAE_unet_boundaries[:,0]):.3f}, {np.mean(MPE_unet_boundaries[:,0]):.3f}, {np.mean(IoU_unet_boundaries[:,0]):.3f}\n")
    txtres.write(f"YOLO, {np.mean(MAE_yolo_boundaries[:,0]):.3f}, {np.mean(MPE_yolo_boundaries[:,0]):.3f}, {np.mean(IoU_yolo_boundaries[:,0]):.3f}\n")

with open(os.path.join(output_folder, "unet_vs_yolo_BOUNDS50.csv"), 'w') as txtres:
    txtres.write("BOUNDS50, MAE, MPE, IOU\n")
    txtres.write(f"UNET, {np.mean(MAE_unet):.3f}, {np.mean(MPE_unet):.3f}, {np.mean(IoU_unet):.3f}\n")
    txtres.write(f"YOLO, {np.mean(MAE_yolo):.3f}, {np.mean(MPE_yolo):.3f}, {np.mean(IoU_yolo):.3f}\n")

with open(os.path.join(output_folder, "unet_vs_yolo_ALL.csv"), 'w') as txtres:
    txtres.write("1CLASS, MAE, MPE, IOU\n")
    txtres.write(f"UNET, {np.mean(MAE_unet):.3f}, {np.mean(MPE_unet):.3f}, {np.mean(IoU_unet):.3f}\n")
    txtres.write(f"YOLO, {np.mean(MAE_yolo):.3f}, {np.mean(MPE_yolo):.3f}, {np.mean(IoU_yolo):.3f}\n")
    txtres.write("BOUNDS30, MAE, MPE, IOU\n")
    txtres.write(f"UNET, {np.mean(MAE_unet_boundaries[:,0]):.3f}, {np.mean(MPE_unet_boundaries[:,0]):.3f}, {np.mean(IoU_unet_boundaries[:,0]):.3f}\n")
    txtres.write(f"YOLO, {np.mean(MAE_yolo_boundaries[:,0]):.3f}, {np.mean(MPE_yolo_boundaries[:,0]):.3f}, {np.mean(IoU_yolo_boundaries[:,0]):.3f}\n")
    txtres.write("BOUNDS50, MAE, MPE, IOU\n")
    txtres.write(f"UNET, {np.mean(MAE_unet):.3f}, {np.mean(MPE_unet):.3f}, {np.mean(IoU_unet):.3f}\n")
    txtres.write(f"YOLO, {np.mean(MAE_yolo):.3f}, {np.mean(MPE_yolo):.3f}, {np.mean(IoU_yolo):.3f}\n")



    # plt.show()
    # pdb.set_trace()

