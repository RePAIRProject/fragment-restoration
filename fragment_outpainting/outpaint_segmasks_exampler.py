import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import fragment_outpainting.config as conf
import glob
import cv2
import criminisi


def colorize_mask(input_mask, color_mapping):
    colored_mask = np.zeros((input_mask.shape[0], input_mask.shape[1], 3), dtype=np.uint8)
    for class_label, color in color_mapping.items():
        colored_mask[input_mask == class_label] = color
    return colored_mask


resize_input_image = 'True'
resize_scale_percent = 20  # percent of original size

criminisi_ps = 15
pix_outpaint = 15

# Outputs:
os.makedirs(conf.outpainted_gt_masks_criminisi, exist_ok=True)

masks_list = glob.glob(os.path.join(conf.masks_folder, conf.repair_group))

for mask_path in masks_list:
    sem_mask = cv2.imread(mask_path, 0)
    img_rgb = colorize_mask(sem_mask, conf.color_mapping)
    #sem_mask = cv2.cvtColor(sem_mask, cv2.COLOR_BGR2RGB)

    img_name = os.path.basename(mask_path)[:-32]
    fg = cv2.imread(os.path.join(conf.fg_folder,f"{img_name}_fg.png"), 0)

    if resize_input_image == 'True':
        # resize_scale_percent the image if it is so huge, for faster computation:
        scale_percent = 20  # percent of original size
        width = int(img_rgb.shape[1] * resize_scale_percent / 100)
        height = int(img_rgb.shape[0] * resize_scale_percent / 100)
        dim = (width, height)
        img_rgb = cv2.resize(img_rgb, dim, interpolation=cv2.INTER_AREA)
        fg = cv2.resize(fg, dim, interpolation=cv2.INTER_AREA)/255

    # erode the mask from its borders (which improves performance of outpainting)
    kernel = np.ones((5, 5))
    mask = cv2.erode(fg, kernel)

    # Criminisi:
    inverted_mask = 1-mask
    # Outpaint the HIT pixels shown in the binary mask
    outpainted = criminisi.inpaint(img_rgb.astype(np.uint8), inverted_mask.astype(np.uint8), patch_size = criminisi_ps)

    # to show the outpainted band around the fragment appy dilation to the mask and display it by masking with the outpainted image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pix_outpaint, pix_outpaint))
    dilation = cv2.dilate(mask, kernel)

    # ---------------------------------------------------------------------
    # VISUALIZE
    #plt.imshow(outpainted * dilation[:, :, None].astype(np.uint8))
    #plt.show()

    outpainted_rgb = outpainted * dilation[:, :, None].astype(np.uint8)

    fix, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(img_rgb)
    axes[1].imshow(outpainted_rgb)
    plt.show()
    plt.savefig(os.path.join(conf.outpainted_gt_masks_criminisi,f"{img_name}_vis.png"))
    plt.imsave(os.path.join(conf.outpainted_gt_masks_criminisi,f"{img_name}.png"), outpainted_rgb)

    # axes[0].set_title(f"num {img}")



