import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.metrics import MeanIoU
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
# from tensorflow.keras.utils import to_categorical
# from PIL import Image
import os
import cv2 
import numpy as np 
import pdb 
from tensorflow.keras import layers
import random 
import json 
import argparse 

def simplified_unet_model(input_size=(256, 256, 3), num_classes = 3):
    inputs = tf.keras.Input(input_size)
    
    # Encoder (contracting) path
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
    
    u8 = UpSampling2D((2, 2))(c5)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    # Create the U-Net model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class CustomMeanIoU(MeanIoU):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
    
def iou_loss(y_true, y_pred):
    def f_score(y_true, y_pred, beta=1):
        smooth = 1e-15
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        true_positive = tf.reduce_sum(y_true * y_pred)
        false_positive = tf.reduce_sum(y_pred) - true_positive
        false_negative = tf.reduce_sum(y_true) - true_positive

        return (1 + beta**2) * true_positive + smooth / ((1 + beta**2) * true_positive + beta**2 * false_negative + false_positive + smooth)

    return 1 - f_score(y_true, y_pred)

def load_images(img_list_path, folder, size=256, color_space='BGR'):
    
    img_list = np.loadtxt(img_list_path, dtype='str')
    images_array = np.zeros((len(img_list), size, size, 3))
    for j, img_name in enumerate(img_list):
        img_path = os.path.join(folder, img_name)
        imgcv = cv2.imread(img_path)
        if color_space == 'RGB':
            imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
        elif color_space == 'HSV':
            imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2HSV)
        imgcv_resized = cv2.resize(imgcv, (size, size))
        imgcv_resized = imgcv_resized / 255.0
        images_array[j, :, :, :] = imgcv_resized
    return images_array, img_list

def load_masks(img_list_path, folder, size=256):

    img_list = np.loadtxt(img_list_path, dtype='str')
    images_array = np.zeros((len(img_list), size, size))
    for j, img_name in enumerate(img_list):
        img_path = os.path.join(folder, img_name)
        imgcv = cv2.imread(img_path, 0)
        imgcv_resized = cv2.resize(imgcv, (size, size))
        images_array[j, :, :] = imgcv_resized
    return images_array

def main(args):

    run_folder = f'runs/{args.f}'
    model = simplified_unet_model()
    #pdb.set_trace()
    custom_objects = {"CustomMeanIoU": CustomMeanIoU}
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.saving.load_model(os.path.join(run_folder, 'best_unet_model_da.h5'))
        #        'UNET/Model_to_detect_3_classes_simplified_HSV_150epoch/Model_to_detect_3_classes_simplified_HSV_150epoch.h5')
        
        #
    output_dir = os.path.join(run_folder, 'results_test_set')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(run_folder, 'parameters.json'), 'r') as parj:
        parameters = json.load(parj)
    
    IMG_SIZE = parameters['img'] 
    CLASSES = parameters["classes"]

    root_folder_MoFF = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/'
    rgb_folder_name = 'RGB'
    if 'inpainted' in args.f:
        rgb_folder_name += '_inpainted'
    rgb_folder_MoFF = os.path.join(root_folder_MoFF, rgb_folder_name)
    # train_images = load_images(os.path.join(root_folder_MoFF, 'train.txt'), rgb_folder_MoFF, size=IMG_SIZE, color_space=COLOR_SPACE)
    # valid_images = load_images(os.path.join(root_folder_MoFF, 'validation.txt'), rgb_folder_MoFF, size=IMG_SIZE, color_space='HSV') #parameters["color_space"])
    test_images, img_list = load_images(os.path.join(root_folder_MoFF, 'test.txt'), rgb_folder_MoFF, size=IMG_SIZE, color_space='HSV')
    masks_folder_MoFF = os.path.join(root_folder_MoFF, f'segmap{str(CLASSES)}c')
    # train_masks = load_masks(os.path.join(root_folder_MoFF, 'train.txt'), masks_folder_MoFF, size=IMG_SIZE)
    # valid_masks = load_masks(os.path.join(root_folder_MoFF, 'validation.txt'), masks_folder_MoFF, size=IMG_SIZE)
    test_masks = load_masks(os.path.join(root_folder_MoFF, 'test.txt'), masks_folder_MoFF, size=IMG_SIZE)
    
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
    args = parser.parse_args()
    main(args)