import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import to_categorical
import os
import cv2 
import numpy as np 
import pdb 
from tensorflow.keras import layers
import random 
import json 
import argparse 
from unet import simplified_unet_model, classic_unet_model, CustomMeanIoU, iou_loss
from utils import load_images, load_masks

def main(args):

    run_folder = f'runs/{args.f}'

    with open(os.path.join(run_folder, 'parameters.json'), 'r') as parj:
        parameters = json.load(parj)
    
    IMG_SIZE = parameters['img'] 
    CLASSES = parameters["classes"]
    if parameters['model'] == 'simplified':
        model = simplified_unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3))
    else:
        model = classic_unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3))

    custom_objects = {"CustomMeanIoU": CustomMeanIoU}
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.saving.load_model(os.path.join(run_folder, 'best_unet_model_da.h5'))

    if args.i != 'test_set':
        name = args.i.split('/')[-1]
    else:
        name = 'test_set'
    output_dir = os.path.join(run_folder, f'performance_{parameters["model"]}_UNET_{IMG_SIZE}x{IMG_SIZE}_{name}')
    os.makedirs(output_dir, exist_ok=True)

    root_folder_MoFF = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/'
    rgb_folder_name = 'RGB'
    if 'inpainted' in args.f:
        rgb_folder_name += '_inpainted'
    
    if args.i != 'test_set':
        custom_folder = args.i
        test_images, img_list = load_images(folder=custom_folder, size=IMG_SIZE, color_space='HSV')
        custom_folder_masks = args.m 
        test_masks = load_masks(folder=custom_folder_masks, size=IMG_SIZE)
    else:
        rgb_folder_MoFF = os.path.join(root_folder_MoFF, rgb_folder_name)
        test_images, img_list = load_images(folder=rgb_folder_MoFF, img_list_path=os.path.join(root_folder_MoFF, 'test.txt'), size=IMG_SIZE, color_space='HSV')
        masks_folder_MoFF = os.path.join(root_folder_MoFF, f'segmap{str(CLASSES)}c')
        test_masks = load_masks(folder=masks_folder_MoFF, img_list_path=os.path.join(root_folder_MoFF, 'test.txt'), size=IMG_SIZE)
    
    test_masks_one_hot = to_categorical(test_masks, num_classes=CLASSES)

    augment_text = ''
    if parameters['augment']:
        augment_text = "with data augmentation"

    title_text = f"trained for {parameters['epochs']} {augment_text} in the {parameters['color_space']}"

    for j in range(test_images.shape[0]):

        pred = model.predict(np.expand_dims(test_images[j,:,:,:], axis=0), batch_size=1)[0,:,:,:]
        y_pred = tf.cast(tf.argmax(pred, axis=-1), tf.float64)
        y_true = tf.cast(tf.argmax(test_masks_one_hot[j, :, :, :], axis=-1), tf.float64)
        iou = iou_loss(y_true, y_pred)
        print(f"iou on the {j}-th test image: {iou:.3f}")
        if iou < 0:
            print("WARNING: iou negative!")    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Show results for a trained model')
    parser.add_argument('-f', type=str, default='', help='folder with everything (model, weights, results)')
    parser.add_argument('-i', type=str, default='test_set', help='custom input images folder (if you do not want to use the test set, which will be the default)')
    parser.add_argument('-m', type=str, default='test_set', help='custom input images folder (if you do not want to use the test set, which will be the default)')
    args = parser.parse_args()
    main(args)