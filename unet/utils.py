import os 
import numpy as np 
import cv2 

def load_images(folder, img_list_path=None, size=512, color_space='BGR'):
    
    if img_list_path is not None:
        img_list = np.loadtxt(img_list_path, dtype='str')
    else:
        img_list = os.listdir(folder)
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

def load_masks(folder, img_list_path=None, size=512):

    if img_list_path is not None:
        img_list = np.loadtxt(img_list_path, dtype='str')
    else:
        img_list = os.listdir(folder)
    images_array = np.zeros((len(img_list), size, size))
    for j, img_name in enumerate(img_list):
        img_path = os.path.join(folder, img_name)
        imgcv = cv2.imread(img_path, 0)
        imgcv_resized = cv2.resize(imgcv, (size, size))
        images_array[j, :, :] = imgcv_resized
    return images_array