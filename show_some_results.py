import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.metrics import MeanIoU
import os
import cv2 
import numpy as np 
import pdb 
from tensorflow.keras import layers
import random 
import json 
import argparse 
from unet import simplified_unet_model, classic_unet_model, CustomMeanIoU
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
    output_dir = os.path.join(run_folder, f'results_{parameters["model"]}_UNET_{IMG_SIZE}x{IMG_SIZE}_{name}')
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
    
    augment_text = ''
    if parameters['augment']:
        augment_text = "with data augmentation"

    title_text = f"trained for {parameters['epochs']} {augment_text} in the {parameters['color_space']}"

    for j in range(test_images.shape[0]):

        plt.figure(figsize=(32,8))
        plt.title(title_text, fontsize=32)
        plt.subplot(241)
        plt.imshow(test_images[j,:,:,:]) #cv2.cvtColor((*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.subplot(242)
        plt.title('Ground Truth')
        if len(test_masks.shape) == 3:
            plt.imshow(test_masks[j,:,:])
        elif len(test_masks.shape) == 4:
            plt.imshow(test_masks[j,:,:,:])
        plt.axis('off')
        plt.subplot(243)
        plt.title('Predictions (values)')
        pred = model.predict(np.expand_dims(test_images[j,:,:,:], axis=0), batch_size=1)[0,:,:,:]
        if CLASSES == 3:
            plt.imshow(pred)
        plt.axis('off')
        plt.subplot(244)
        plt.title('Predictions (labels)')
        pred_labels = tf.argmax(pred, axis=-1)
        plt.imshow(pred_labels)
        plt.axis('off')
        #plt.show()
        plt.savefig(os.path.join(output_dir, f"vis_{img_list[j]}"), dpi=300, bbox_inches='tight')
        plt.close()
        #pdb.set_trace()

        cv2.imwrite(os.path.join(output_dir, img_list[j]), pred_labels.numpy())
        cv2.imwrite(os.path.join(output_dir, f"1class_{img_list[j]}"), (pred_labels.numpy() > 1)*255)
        plt.imsave(os.path.join(output_dir, f"{img_list[j]}_plt.png"), pred_labels)
    #pdb.set_trace()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Show results for a trained model')
    parser.add_argument('-f', type=str, default='', help='folder with everything (model, weights, results)')
    parser.add_argument('-i', type=str, default='test_set', help='custom input images folder (if you do not want to use the test set, which will be the default)')
    parser.add_argument('-m', type=str, default='test_set', help='custom input images folder (if you do not want to use the test set, which will be the default)')
    args = parser.parse_args()
    main(args)