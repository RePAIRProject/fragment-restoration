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
union199_motif = np.sum(((gt_199_s512==2)+(pred_199==2))>0)
union199_motif2 = np.sum( (gt_199_s512==2)) + np.sum((pred_199==2) )  - np.sum( (gt_199_s512==2)*(pred_199==2) > 0)
print(f'union1: {union199_motif}, union2: {union199_motif2}')
pdb.set_trace()
ioumotif = inters199_motif / union199_motif
text = f'IOU (3classes) = {iou}\nIntersection = {inters199}\nUnion = {union199}\nIOU (1class) = {ioumotif}\nIntersection = {inters199_motif}\nUnion = {union199_motif}'
plt.title(text, fontsize=32)
plt.imshow(((np.abs(gt_199_s512-pred_199)>0) * (gt_199_s512==2)).astype(int)*255)
plt.show()


#pdb.set_trace()



class CustomSingleClassMeanIoU(MeanIoU):
    def __init__(self, num_classes, class_index):
        super().__init__(num_classes=num_classes)
        self.class_index = class_index

    def update_state(self, y_true, y_pred, sample_weight=None):
        pdb.set_trace()
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        
        # Only consider y_true and y_pred where y_true is class_index
        mask = tf.equal(y_true, self.class_index)
        
        return super().update_state(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask), sample_weight)

num_classes = 3
single_class_metric = CustomSingleClassMeanIoU(num_classes, 2)

# Compute single class mean IOU
test_masks_tensor = [tf.convert_to_tensor(gt_199_s512)]
test_pred_masks_tensor = [tf.convert_to_tensor(pred_199)]
class_index = 2
single_class_metric = CustomSingleClassMeanIoU(num_classes, class_index)
single_class_metric.update_state(test_masks_tensor, test_pred_masks_tensor)
print(f"Single class Mean IOU: {single_class_metric.result().numpy()}")

pdb.set_trace()