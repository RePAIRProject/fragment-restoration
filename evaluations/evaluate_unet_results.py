import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb 
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU

pred_199 = cv2.imread('/home/lucap/code/fragment-restoration/runs/run9350223490688163_simplifiedUNET_RGB_images512x512_3classes_200epochs_augmented_lr0.001_HSV/results_simplified_UNET_512x512_test_set/RPf_00199.png')
gt_199 = cv2.imread('/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/segmap3c/RPf_00199.png')
gt_199_s512=cv2.resize(gt_199, (512, 512))

plt.subplot(131)
plt.title('predictions 199', fontsize=32)
plt.imshow(pred_199*127)
plt.subplot(132)
plt.title('gt_199', fontsize=32)
plt.imshow(gt_199_s512*127)
plt.subplot(133)
inters199 = np.sum((gt_199_s512 == pred_199))
union199 = np.sum(gt_199_s512 > -1)
iou = inters199 / union199
inters199_motif = np.sum(((gt_199_s512==2)*(pred_199==2)>0))
# two ways of calculating result
union199_motif = np.sum(((gt_199_s512==2)+(pred_199==2))>0)
union199_motif2 = np.sum( (gt_199_s512==2)) + np.sum((pred_199==2) )  - np.sum( (gt_199_s512==2)*(pred_199==2) > 0)
print(f'union1: {union199_motif}, union2: {union199_motif2}')
ioumotif = inters199_motif / union199_motif
text = f'IOU (3classes) = {iou}\nIntersection = {inters199}\nUnion = {union199}\nIOU (1class) = {ioumotif}\nIntersection = {inters199_motif}\nUnion = {union199_motif}'
plt.title(text, fontsize=32)
plt.imshow(((np.abs(gt_199_s512-pred_199)>0) * (gt_199_s512==2)).astype(int)*255)
plt.show()
pdb.set_trace()
