import cv2 as cv
import numpy as np 
import os, pdb

dataset = 'ayellet'
inner_kernel_size = 11
outer_kernel_size = 31
inner_kernel = np.ones((inner_kernel_size, inner_kernel_size), np.uint8)
outer_kernel = np.ones((outer_kernel_size, outer_kernel_size), np.uint8)
final_res = 512

if dataset == 'repair':
        
    input_folder = '/media/lucap/big_data/datasets/repair/group_38/2D'

    segmented_folder = 'Segmented'
    mask_folder = 'Mask'

    output_folder = '/media/lucap/big_data/datasets/repair/group_38/2D/inpainting'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    segments_root_paths = [path for path in os.listdir(input_folder) if path[-4:] == '.png']
    for segment_root_path in segments_root_paths:
        segmented_image = cv.imread(os.path.join(input_folder, segmented_folder, f"{segment_root_path[:-4]}_segmented.png"))
        segmented_image = cv.resize(segmented_image, (final_res, final_res))
        mask = cv.imread(os.path.join(input_folder, mask_folder, f"{segment_root_path[:-4]}_mask.png"), 0)
        mask = cv.resize(mask, (final_res, final_res))
        
        eroded_mask = cv.erode(mask, inner_kernel, iterations=1)
        dilated_mask = cv.dilate(mask, outer_kernel, iterations=1)
        inpaint_area = np.ones((final_res, final_res)) * 255 - eroded_mask #dilated_mask
        area_to_be_considered = dilated_mask # - eroded_mask
        cv.imwrite(os.path.join(output_folder, 'input', f"{segment_root_path[:-4]}_inpaint.png"), segmented_image)
        cv.imwrite(os.path.join(output_folder, 'input', f"{segment_root_path[:-4]}_inpaint_mask.png"), inpaint_area)
        cv.imwrite(os.path.join(output_folder, f"{segment_root_path[:-4]}_inpaint_area.png"), area_to_be_considered)

elif dataset == 'ayellet':

    input_folder = '/home/lucap/code/reassembly_2D_sources/Pictures/Fragments/test_inpainting_ferapontov1_s17'
    orig_folder = os.path.join(input_folder, 'orig')
    inpaint_input_folder = os.path.join(input_folder, 'input')
    for image_path in os.listdir(orig_folder):
        img = cv.imread(os.path.join(orig_folder, image_path))
        if img.shape[0] % 2 == 1 or img.shape[1] % 2 == 1:
            #pdb.set_trace()
            new_shape_0 = img.shape[0] - (img.shape[0] % 2)
            new_shape_1 = img.shape[1] - (img.shape[1] % 2)
            img = cv.resize(img, (new_shape_1, new_shape_0))
        max_dim = np.maximum(img.shape[0], img.shape[1])
        fin_img = np.ones((max_dim, max_dim, 3)) * 255
        diff_x = (max_dim - img.shape[0]) // 2
        diff_y = (max_dim - img.shape[1]) // 2
        #pdb.set_trace()
        fin_img[diff_x : max_dim - diff_x, diff_y : max_dim - diff_y] = img

        fin_img = cv.resize(fin_img, (final_res, final_res))
        #img = cv.resize(img, (final_res, final_res))
        inpaint_area = fin_img[:,:,0] > 254
        mask = (1 - inpaint_area) * 255
        # 1 - eroded is equal to dilate
        inpaint_area = cv.dilate(inpaint_area.astype(float), inner_kernel, iterations=2)
        area_to_be_considered = cv.dilate(mask.astype(float), outer_kernel, iterations=2)
        assert(fin_img.shape[0:2] == inpaint_area.shape), f"img {fin_img.shape}, inpaint {inpaint_area.shape}"
        #pdb.set_trace()
        cv.imwrite(os.path.join(input_folder, 'input', f"{image_path[:-4]}_inpaint.png"), fin_img)
        cv.imwrite(os.path.join(input_folder, 'input', f"{image_path[:-4]}_inpaint_mask.png"), inpaint_area.astype(float) * 255)
        cv.imwrite(os.path.join(input_folder, f"{image_path[:-4]}_inpaint_area.png"), area_to_be_considered)
