v_release = 'v4' #release version at segments.ai

# original input images after exporting from segments.ai and merging three repositories into single file for segments_masks and segments_images
segments_images_path = '/Dataset/segments_images'
segments_masks_path = '/Dataset/segments_masks'

# foreground, images and masks folder  after processing the segments_images and segments_masks
fg_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/fg/'
images_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/images/'
masks_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/masks/'

# folders for cropped stuff:
cropped_masks = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/masks_cropped/'
cropped_images = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/images_cropped/'
cropped_fgs = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/fg_cropped/'

# black mark regions:
bm_imgs = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/BOFF/black_mark_region/'
bm_cropped_imgs = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/BOFF/black_mark_region_cropped/'



images_wh = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/images_WhiteBG'
masked_bm = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/BOFF/black_mark_region_masked_with_fg'

images_inpainted = '/home/sinem/PycharmProjects/fragment-restoration/Results/inpainting_criminisi/images_bm_inpainted_ps11'
images_inpainted_cropped = '/home/sinem/PycharmProjects/fragment-restoration/Results/inpainting_criminisi/images_bm_inpainted_ps11_cropped/'
