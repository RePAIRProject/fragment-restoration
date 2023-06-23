import os 
import matplotlib.pyplot as plt 
import pdb 
from copy import copy 
import numpy as np 

moff_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF'
unprocessed_data_folder = os.path.join(moff_folder, 'unprocessed_data')

images_folder = os.path.join(unprocessed_data_folder, 'images_cropped')
seg_masks_folder = os.path.join(unprocessed_data_folder, 'masks_cropped')

imgs_paths = os.listdir(images_folder)
segs_paths = os.listdir(seg_masks_folder)

imgs_paths.sort()
segs_paths.sort()

output_img = os.path.join(moff_folder, 'RGB')
output_s3c = os.path.join(moff_folder, 'segmap3c')
output_s14c = os.path.join(moff_folder, 'segmap14c')
os.makedirs(output_img, exist_ok=True)
os.makedirs(output_s3c, exist_ok=True)
os.makedirs(output_s14c, exist_ok=True)

for img_path, seg_path in zip(imgs_paths, segs_paths):
    img_id = img_path[img_path.index('RPf')+4:img_path.index('RPf')+9]
    seg_id = img_path[seg_path.index('RPf')+4:seg_path.index('RPf')+9]
    assert(img_id == seg_id), 'misaligned imags and masks/fg'

    img = plt.imread(os.path.join(images_folder, img_path))
    seg_map = plt.imread(os.path.join(seg_masks_folder, seg_path))
    seg_map = (seg_map * 255).astype(np.uint8)

    seg_map3c = seg_map.copy()
    seg_map3c[seg_map3c > 2] = 2

    plt.imsave(os.path.join(output_img, f'RPf_{img_id}.png'), img)
    plt.imsave(os.path.join(output_s14c, f'RPf_{img_id}.png'), seg_map)
    plt.imsave(os.path.join(output_s3c, f'RPf_{img_id}.png'), seg_map3c)
    print(f'saved RPf_{img_id}.png')
    #pdb.set_trace()