
# original input images after exporting from segments.ai and merging three repositories into single file for segments_masks and segments_images
segments_images_path = '/Dataset/segments_images'
segments_masks_path = '/Dataset/segments_masks'

# foreground, images and masks folder  after processing the segments_images and segments_masks
fg_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/fg/'
images_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/images/'
masks_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/masks/'
colored_masks_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/masks_colored/'

# folders for cropped stuff:
cropped_masks = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/masks_cropped/'
cropped_images = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/images_cropped/'
cropped_fgs = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/fg_cropped/'

# black mark regions:
bm_imgs = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/BoFF/black_mark_region/'
bm_cropped_imgs = '/home/sinem/PycharmProjects/fragment-restoration/BoFF/Dataset/black_mark_region_cropped/'

# Masked rgb images (motif info with clean - white colored fragment background)
motifs_rgb = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/motifs/'
motifs_rgb_cropped = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/motifs_cropped/'

cleaned_fragments = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/images_cleaned_intact/'
cleaned_fragments_cropped = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/MoFF/images_cleaned_intact_cropped/'


# Outpainting
repair_group = '*group_28*'
outpainted_gt_masks_criminisi = '/home/sinem/PycharmProjects/fragment-restoration/Results/Outpaint/Criminisi/'

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
