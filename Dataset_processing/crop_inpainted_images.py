import os
import sys
import numpy as np
import cv2
import sys

from black_mark_removal.inpainting_Criminisi import paths_inpainting as paths


os.makedirs(paths.images_inpainted_cropped, exist_ok=True)

img_files = os.listdir(paths.images_inpainted)
mask_files = os.listdir(paths.masks_folder)

for imfile in img_files:

    im_file_path = os.path.join(paths.images_inpainted, imfile)
    m_file_path = os.path.join(paths.masks_folder, f"{imfile[:-4]}_label_ground-truth_semantic.png")

    img = np.array(cv2.imread(im_file_path))
    mask = np.array(cv2.imread(m_file_path))

    binary_mask = mask > 0
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)

    # check if there are any foreground objects
    if not np.any(rows) or not np.any(cols):
        print(f"No foreground objects found in the image: {imfile}")

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cropped_img = img[y_min:y_max, x_min:x_max]

    cv2.imwrite(os.path.join(paths.images_inpainted_cropped, imfile), cropped_img)