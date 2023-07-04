import os
import sys
import numpy as np
import cv2
import sys


import paths

def colorize_mask(input_mask, color_mapping):
    colored_mask = np.zeros((input_mask.shape[0], input_mask.shape[1], 3), dtype=np.uint8)
    for class_label, color in color_mapping.items():
        colored_mask[input_mask == class_label] = color
    return colored_mask

color_mapping = {
    0: (0, 0, 0),       # Background (Black)
    1: (245, 245, 220), # Class 1 (Beige)
    2: (0, 128, 0),     # Class 2 (Dark Green)
    3: (128, 128, 0),   # Class 3 (Olive)
    4: (0, 0, 128),     # Class 4 (Navy Blue)
    5: (128, 0, 128),   # Class 5 (Purple)
    6: (0, 128, 128),   # Class 6 (Teal)
    7: (128, 128, 128), # Class 7 (Gray)
    8: (255, 0, 0),     # Class 8 (Red)
    9: (0, 255, 0),     # Class 9 (Green)
    10: (255, 255, 0),  # Class 10 (Yellow)
    11: (0, 0, 255),    # Class 11 (Blue)
    12: (255, 0, 255),  # Class 12 (Magenta)
    13: (0, 255, 255),  # Class 13 (Cyan)
    14: (255, 128, 0)   # Class 14 (Orange)
}

os.makedirs(paths.colored_masks_folder, exist_ok=True)
mask_files = os.listdir(paths.masks_folder)

for im_file in mask_files:

    m_file_path = os.path.join(paths.masks_folder, im_file)

    mask = np.array(cv2.imread(m_file_path, 0))

    mask_colored = colorize_mask(mask, color_mapping)

    cv2.imwrite(os.path.join(paths.colored_masks_folder, im_file), mask_colored)

