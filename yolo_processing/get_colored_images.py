import os 
import matplotlib.pyplot as plt 
import numpy as np 
import pdb 

rgb_images = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/RGB_inpainted'
segmap_images = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF/segmap14c'
output = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/colored'
os.makedirs(output, exist_ok=True)
rgbs = os.listdir(rgb_images)
segmaps = os.listdir(segmap_images)

rgbs.sort()
segmaps.sort()

colors = [[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1], \
    [1, 0, 0], [0.25, 0.75, 0], [0, 0.5, 0.25], [0, 0.1, 0.9], [0.7, 0.4, 0.1], [0, 0.25, 1]]

for rgb_p, seg_p in zip(rgbs, segmaps):
    
    rgb_img = plt.imread(os.path.join(rgb_images, rgb_p))
    new_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3))
    seg_img = np.round(plt.imread(os.path.join(segmap_images, seg_p)) * 255).astype(int)

    for label in np.unique(seg_img[:,:,0]):
        area_to_be_colored = (seg_img == label)
        new_img += area_to_be_colored[:,:,:3] * colors[label] 

    plt.imsave(os.path.join(output, rgb_p), new_img)
