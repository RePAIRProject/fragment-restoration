import os 
import matplotlib.pyplot as plt 
import pdb 
from copy import copy 
import numpy as np 
import cv2 
import random 

def write_shape_to_file(path, list_of_shapes):

    f = open(path, "w")


    for shape in list_of_shapes:

        f.write(f"{shape[0]}")
        for j in range(1, len(shape)):
            f.write(f" {shape[j]:5.5f}")
        f.write("\n")
    f.close()

def main():
    moff_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF'
    unprocessed_data_folder = os.path.join(moff_folder, 'unprocessed_data')

    images_folder = os.path.join(unprocessed_data_folder, 'images_cropped')
    inpainted_images_folder = os.path.join(unprocessed_data_folder, 'images_bm_inpainted_cropped')
    seg_masks_folder = os.path.join(unprocessed_data_folder, 'masks_cropped')

    imgs_paths = os.listdir(images_folder)
    segs_paths = os.listdir(seg_masks_folder)

    imgs_paths.sort()
    segs_paths.sort()

    output_img = os.path.join(moff_folder, 'RGB')
    output_inp = os.path.join(moff_folder, 'RGB_inpainted')
    output_s3c = os.path.join(moff_folder, 'segmap3c')
    output_s14c = os.path.join(moff_folder, 'segmap14c')
    output_motif = os.path.join(moff_folder, 'motifs')
    output_yolo_bc = os.path.join(moff_folder, 'annotations_boxes_components')
    output_yolo_bm = os.path.join(moff_folder, 'annotations_boxes_motif')
    output_yolo_shapes = os.path.join(moff_folder, 'annotations_shape')


    os.makedirs(output_img, exist_ok=True)
    os.makedirs(output_inp, exist_ok=True)
    os.makedirs(output_s3c, exist_ok=True)
    os.makedirs(output_s14c, exist_ok=True)
    os.makedirs(output_motif, exist_ok=True)
    os.makedirs(output_yolo_bc, exist_ok=True)
    os.makedirs(output_yolo_bm, exist_ok=True)
    os.makedirs(output_yolo_shapes, exist_ok=True)

    for img_path, seg_path in zip(imgs_paths, segs_paths):
        img_id = img_path[img_path.index('RPf')+4:img_path.index('RPf')+9]
        seg_id = img_path[seg_path.index('RPf')+4:seg_path.index('RPf')+9]
        assert(img_id == seg_id), 'misaligned imags and masks/fg'

        img = plt.imread(os.path.join(images_folder, img_path))
        inp_img = plt.imread(os.path.join(inpainted_images_folder, img_path))
        seg_map = plt.imread(os.path.join(seg_masks_folder, seg_path))
        seg_map = (seg_map * 255).astype(np.uint8)

        seg_map3c = seg_map.copy()
        seg_map3c[seg_map3c > 2] = 2

        motif_map = np.zeros_like(img)
        #motif_map[:,:,3] = (seg_map > 0)[:,:]
        single_annotations = []
        annotations_motif = []
        annotations_shape = []
        for motif_label in range(2, np.max(seg_map)+1):
            cur_motif_map = (seg_map == motif_label).astype(np.uint8)
            motif_map[:,:,:3] += img * (cur_motif_map[:,:,:3])
            cv_motif = np.zeros((seg_map.shape[0], seg_map.shape[1]), np.uint8)
            cv_motif += cur_motif_map[:,:,0]
            h,w = img.shape[:2]
            #pdb.set_trace()
            contours = cv2.findContours(cv_motif, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            
            if len(contours) > 0:
                xc, yc, wc, hc = 0, 0, 0, 0
                x_min_vals = []
                y_min_vals = []
                x_max_vals = []
                y_max_vals = []
                for cntr in contours:
                    
                    x,y,wcnt,hcnt = cv2.boundingRect(cntr)
                    x_min_vals.append(x)
                    y_min_vals.append(y)
                    x_max_vals.append(x+wcnt)
                    y_max_vals.append(y+hcnt)

                    # for the single annotations
                    xc_ = (x + wcnt / 2) / w
                    yc_ = (y + hcnt / 2) / h 
                    wc_ = wcnt / w
                    hc_ = hcnt / h
                    single_annotations.append([motif_label, xc_, yc_, wc_, hc_])
                    contour_yolo = [motif_label]
                    for cntr_pt in cntr:
                        contour_yolo.append((cntr_pt / [w, h])[0][0])
                        contour_yolo.append((cntr_pt / [w, h])[0][1])
                    annotations_shape.append(contour_yolo)
                    #pdb.set_trace()
                
                x_min = np.min(x_min_vals)
                y_min = np.min(y_min_vals)
                x_max = np.max(x_max_vals)
                y_max = np.max(y_max_vals)
                xc = ((x_min + x_max) / 2) / w 
                yc = ((y_min + y_max) / 2) / h 
                wc = (x_max - x_min) / w 
                hc = (y_max - y_min) / h 
                # x /= w 
                # wc /= w
                # y /= h 
                # hc /= h 
            
            # cv2.rectangle(img, (x, y), (x+wcnt, y+hcnt), (0, 0, 255), 5)
            # plt.subplot(121)
            # plt.title(f"x: {x}, y: {y}, w: {wcnt}, h: {hcnt}")
            # plt.imshow(img) 
            # plt.subplot(122)
            # plt.title(f"x: {xc}, y: {yc}, w: {wc}, h: {hc}")
            # plt.imshow(cv_motif)
            # plt.show()
            # pdb.set_trace() #np.savetxt(os.path.join(output_yolo, f'RPf_{img_id}.txt'), annotations, fmt="%d %f:3f %d %d %d")
                annotations_motif.append([motif_label, xc, yc, wc, hc])
        if len(single_annotations) > 0:
            np.savetxt(os.path.join(output_yolo_bc, f'RPf_{img_id}.txt'), single_annotations, fmt="%d %.3f %.3f %.3f %.3f")
        else:
            open(os.path.join(output_yolo_bc, f'RPf_{img_id}.txt'), 'w').close()
        if len(annotations_motif) > 0:
            
            np.savetxt(os.path.join(output_yolo_bm, f'RPf_{img_id}.txt'), annotations_motif, fmt="%d %.3f %.3f %.3f %.3f")
        else:
            open(os.path.join(output_yolo_bm, f'RPf_{img_id}.txt'), 'w').close()
        if len(annotations_shape) > 0:
            write_shape_to_file(os.path.join(output_yolo_shapes, f'RPf_{img_id}.txt'), annotations_shape)
        else:
            open(os.path.join(output_yolo_shapes, f'RPf_{img_id}.txt'), 'w').close()
        plt.imsave(os.path.join(output_inp, f'RPf_{img_id}.png'), inp_img)   
        #plt.imsave(os.path.join(output_motif, f'RPf_{img_id}.png'), motif_map)
        #plt.imsave(os.path.join(output_img, f'RPf_{img_id}.png'), img)
        #plt.imsave(os.path.join(output_s14c, f'RPf_{img_id}.png'), seg_map)
        #plt.imsave(os.path.join(output_s3c, f'RPf_{img_id}.png'), seg_map3c)
        #pdb.set_trace()


        print(f'saved RPf_{img_id}.png')
        #pdb.set_trace()

    print("finished creating images, now the train/test/val files!")
    list_files_moff = os.listdir(os.path.join(moff_folder, 'RGB'))
    random.shuffle(list_files_moff)
    dset_length = len(list_files_moff)
    train = .8
    val = .1
    test = .1
    print(dset_length, "images in the dataset")
    train_limit = np.round(train*dset_length).astype(np.int32)
    val_limit = np.round(val*dset_length).astype(np.int32)
    #print(train_limit, val_limit)
    print(f"{train_limit} images for training\n{val_limit} images for validation\n{dset_length+1-val_limit-train_limit} images for test")
    train_set_files = list_files_moff[:train_limit]
    val_set_files = list_files_moff[train_limit:train_limit+val_limit]
    test_set_files = list_files_moff[-val_limit+1:]

    ltr = len(train_set_files)
    lv = len(val_set_files)
    lt = len(test_set_files)


    print(ltr, lv, lt, "=", (ltr+lv+lt))
    print(val_set_files[-1], test_set_files[0])
    np.savetxt(os.path.join(moff_folder, 'train.txt'), train_set_files, fmt="%s")
    np.savetxt(os.path.join(moff_folder, 'validation.txt'), val_set_files, fmt="%s")
    np.savetxt(os.path.join(moff_folder, 'test.txt'), test_set_files, fmt="%s")

if __name__ == '__main__':
    main()