
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import paths


# input:
img_files = os.listdir(paths.images_folder)

# output:
os.makedirs(paths.motifs_rgb, exist_ok=True)
os.makedirs(paths.motifs_rgb_cropped, exist_ok=True)
os.makedirs(paths.cleaned_fragments, exist_ok=True)
os.makedirs(paths.cleaned_fragments_cropped, exist_ok=True)


for img_files in img_files:

    im_file_path = os.path.join(paths.images_folder, img_files)
    m_file_path = os.path.join(paths.masks_folder, f"{img_files[:-4]}_label_ground-truth_semantic.png")
    fg_file_path = os.path.join(paths.fg_folder, f"{img_files[:-4]}_fg.png")

    img = np.array(cv2.cvtColor(cv2.imread(im_file_path), cv2.COLOR_BGR2RGB))
    mask = np.array(cv2.imread(m_file_path, 0))
    fg = np.array(cv2.imread(fg_file_path, 0))

    motif = img * np.expand_dims(mask > 1, axis=2)
    plt.imsave(os.path.join(paths.motifs_rgb, img_files), motif)

    # Cleaned fragment:
    clean_fragment = np.copy(motif)
    clean_fragment[np.logical_and(fg > 0, np.all(motif == 0, axis=2))] = [255, 255, 255]

    # Crop fragments:
    crop_bin = fg > 0
    rows = np.any(crop_bin, axis=1)
    cols = np.any(crop_bin, axis=0)

    # check if there are any foreground objects
    if not np.any(rows) or not np.any(cols):
        print(f"No foreground objects found in the image: {img_files}")

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    motif_cropped = motif[y_min:y_max, x_min:x_max]
    clean_fragment_cropped = clean_fragment[y_min:y_max, x_min:x_max]

    plt.imsave(os.path.join(paths.motifs_rgb, img_files), motif)
    plt.imsave(os.path.join(paths.motifs_rgb_cropped, img_files), motif_cropped)

    plt.imsave(os.path.join(paths.cleaned_fragments, img_files), clean_fragment)
    plt.imsave(os.path.join(paths.cleaned_fragments_cropped, img_files), clean_fragment_cropped)