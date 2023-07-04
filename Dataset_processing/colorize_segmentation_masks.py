import os
import sys
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt


import paths

def colorize_mask(input_mask, color_mapping):
    colored_mask = np.zeros((input_mask.shape[0], input_mask.shape[1], 3), dtype=np.uint8)
    for class_label, color in color_mapping.items():
        colored_mask[input_mask == class_label] = color
    return colored_mask

color_mapping = {
    0: (0, 0, 0),           # Class 0 (Image Background - Black)
    1: (255, 255, 240),     # Class 1 (Fragment Background - Light Beige)
    2: (0, 0, 255),         # Class 2 (Bluebird - Blue)
    3: (204, 204, 0),       # Class 3 (Yellow Bird - Yellow)
    4: (255, 0, 0),         # Class 4 (Red Griffon - Red)
    5: (255, 179, 179),     # Class 5 (Red Flower - Red - Lighter Tone)
    6: (0, 128, 255),       # Class 6 (Blue Flower - Blue - Different Tone)
    7: (255, 77, 77),       # Class 7 (Red Circle - Red - Different Tone)
    8: (192, 0, 0),         # Class 8 (Red Spiral - Dark Red)
    9: (0, 255, 0),         # Class 9 (Curved Green Stripe - Green)
    10: (255, 204, 229),       # Class 10 (Thin Red Stripe - Red - Different Tone)
    11: (255, 0, 255),      # Class 11 (Thick Red Stripe - Magenta)
    12: (128, 128, 255),    # Class 12 (Thin Floral Stripe - Blue - Different Tone)
    13: (0, 255, 255)       # Class 13 (Thick Floral Stripe - Cyan)
}
#(255, 153, 204),
os.makedirs(paths.colored_masks_folder, exist_ok=True)
mask_files = os.listdir(paths.masks_folder)

for im_file in mask_files:

    m_file_path = os.path.join(paths.masks_folder, im_file)
    mask = np.array(cv2.imread(m_file_path, 0))
    mask_colored = colorize_mask(mask, color_mapping)

    plt.imsave(os.path.join(paths.colored_masks_folder, im_file), mask_colored)
    #cv2.imwrite(os.path.join(paths.colored_masks_folder, im_file), mask_colored)

