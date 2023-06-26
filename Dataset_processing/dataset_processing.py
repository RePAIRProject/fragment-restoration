import os

import cv2
import numpy as np

import paths
import utils


#----------------------------------------------------
# Export repositories from segments.ai
#utils.export_segments_ai(paths.v_release)

#----------------------------------------------------
# Change mask labels of Decor 2 repository, overwrite by changed masks in decor2 repository in segments folder
#folder_path = os.path.join('/home/sinem/PycharmProjects/fragment-restoration/Dataset/segments/UNIVE_decor2/', paths.v_release)
#utils.relabel_decor2(folder_path)

# Merge exported repositories into images_from_segments and masks_from_segments folders
#utils.merge_segments_folders(paths.segments_images_path, paths.segments_masks_path, paths.v_release)

# Create foreground masks for the fragment region, and save them into fg folder. Assign 0 to the background of all images
#utils.create_fg(paths.segments_images_path,paths.images_folder,paths.fg_folder)

# Refine masks : changes the category id number of motifs, to increase in a sequential way
#utils.refine_masks(paths.segments_masks_path,paths.masks_folder,paths.fg_folder)

#----------------------------------------------------
# Crop images and masks
os.makedirs(paths.cropped_masks, exist_ok=True)
os.makedirs(paths.cropped_images, exist_ok=True)
os.makedirs(paths.cropped_fgs, exist_ok=True)

mask_files = os.listdir(paths.masks_folder)
img_files = os.listdir(paths.images_folder)
fg_files = os.listdir(paths.cropped_fgs)

for imfile in img_files:

    im_file_path = os.path.join(paths.images_folder, imfile)
    m_file_path = os.path.join(paths.masks_folder, f"{imfile[:-4]}_label_ground-truth_semantic.png")
    fg_file_path = os.path.join(paths.fg_folder, f"{imfile[:-4]}_fg.png")


    img = np.array(cv2.imread(im_file_path))
    mask = np.array(cv2.imread(m_file_path))
    fg = np.array(cv2.imread(fg_file_path))

    cropped_img, cropped_mask, cropped_fg = utils.crop_image(img, mask, fg, imfile)

    cv2.imwrite(os.path.join(paths.cropped_images, imfile), cropped_img)

    cv2.imwrite(os.path.join(paths.cropped_fgs, f"{imfile[:-4]}_fg.png"), cropped_fg)

    cv2.imwrite(os.path.join(paths.cropped_masks, f"{imfile[:-4]}_label_ground-truth_semantic.png"), cropped_mask)





