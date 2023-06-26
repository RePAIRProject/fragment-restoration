import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re
import random
from PIL import Image
import paths_inpainting as paths
import utils


erode_bb_masks = 'False' # This erodes the detected bounding box region from Yolo (It was tried but the experiments showed that it is better not erode the mask)
eroding_kernel = np.ones((11, 11), np.uint8) # Define the kernel for erosion operation


# Create relevant directories
os.makedirs(paths.images_wh, exist_ok=True)
os.makedirs(paths.masked_bm, exist_ok=True)

# Ensure directories exist
if not os.path.exists(paths.images_folder):
    print(f"Image directory does not exist: {paths.images_folder}")
    exit(1)
if not os.path.exists(paths.bm_imgs):
    print(f"Mask directory does not exist: {paths.bm_imgs}")
    exit(1)

# Read the input images and black_mark_masks created after Yolo detection
img_names = sorted(os.listdir(paths.images_folder), key=utils.sort_key)
bm_names = sorted(os.listdir(paths.bm_imgs), key=utils.sort_key)

# Verify that there is a 1-to-1 correspondence between images and labels
if len(img_names) != len(bm_names):
    print("Number of image files does not match number of mask files.")
    exit(1)


for image_file, bm_file in zip(img_names, bm_names):
    # Load image and its foreground
    imgpath = os.path.join(paths.images_folder, image_file)
    fg_file_path = os.path.join(paths.fg_folder, f"{image_file[:-4]}_fg.png")

    rgb_image = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    fg = np.array(cv2.imread(fg_file_path, 0))
    bm_yolo = cv2.imread(os.path.join(paths.bm_imgs, bm_file), cv2.IMREAD_GRAYSCALE)

    #-------------------------------------------------------------
    # Optional
    if erode_bb_masks == 'True':
        os.makedirs('Eroded_masks_original_size', exist_ok=True)
        # Apply mask on image (this assumes mask and image are the same size)
        bm_mask = bm_yolo * (rgb_image[:, :, 0] > 0)
        # Erode the mask
        eroded_mask = cv2.erode(bm_yolo, eroding_kernel)

        # Save the eroded mask to file
        cv2.imwrite(f'Eroded_masks_original_size/{image_file[:-4]}_eroded.png', eroded_mask)
        bm_yolo = eroded_mask
    # -------------------------------------------------------------


    # mask yolo bounding box with foreground mask. In this way we remove the image background contribution into the detected bounding box regions
    bm_yolo[np.where(bm_yolo == 255)] = 1
    masked_bm = bm_yolo * fg
    plt.imsave(os.path.join(paths.masked_bm, f"{image_file}"), masked_bm, cmap='gray')



    # Paint the background to white:
    white_background_image = utils.make_background_white(rgb_image, fg)

    # Save the image to file
    white_background_image.save(os.path.join(paths.images_wh, f"{image_file[:-4]}_wh.png"))
