import cv2 as cv
import numpy as np 
import os, pdb
import matplotlib.pyplot as plt 


dataset = 'ayellet'


if dataset == 'repair':

    input_folder = '/media/lucap/big_data/datasets/repair/group_38/2D/inpainting/output'

    output_folder = '/media/lucap/big_data/datasets/repair/group_38/2D/inpainting'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    segments_root_paths = [path for path in os.listdir(input_folder) if path[-4:] == '.png']
    for segment_root_path in segments_root_paths:
        inpainted_image = cv.imread(os.path.join(input_folder, f"{segment_root_path[:-4]}.png"), cv.IMREAD_COLOR)
        inpainted_image = cv.cvtColor(inpainted_image, cv.COLOR_RGB2BGR)
        area_to_be_considered = cv.imread(os.path.join(output_folder, f"{segment_root_path[:-4]}_area.png"))
        plt.subplot(121)
        plt.imshow(inpainted_image)
        final_img = inpainted_image * (area_to_be_considered>0)
        plt.subplot(122)
        plt.imshow(final_img)
        plt.show()
        final_img = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        cv.imwrite(os.path.join(output_folder, f"{segment_root_path[:-4]}_inpainted.png"), final_img)

elif dataset == 'ayellet':
    input_folder = '/home/lucap/code/reassembly_2D_sources/Pictures/Fragments/test_inpainting_ferapontov1_s17'
    output_folder = os.path.join(input_folder, 'output')
    for image_path in os.listdir(output_folder):
        inpainted_image = cv.imread(os.path.join(output_folder, image_path), cv.IMREAD_COLOR)
        inpainted_image = cv.cvtColor(inpainted_image, cv.COLOR_RGB2BGR)
        area_to_be_considered = cv.imread(os.path.join(input_folder, f"{image_path[:-4]}_area.png"))
        #plt.subplot(121)
        #plt.imshow(inpainted_image)
        final_img = inpainted_image * (area_to_be_considered>0)
        #plt.subplot(122)
        #plt.imshow(final_img)
        #plt.show()
        final_img = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)
        cv.imwrite(os.path.join(output_folder, f"{image_path[:-4]}_inpainted.png"), final_img)