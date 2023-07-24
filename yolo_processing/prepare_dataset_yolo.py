import os 
import matplotlib.pyplot as plt 
import pdb 
from copy import copy 
import numpy as np 
import shutil 

moff_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF'

dset_boxes = {}
only_motifs = False 
inpainted = True
if only_motifs:
    dset_boxes['images_folder'] = os.path.join(moff_folder, 'motifs')
else:
    if inpainted is True:
        dset_boxes['images_folder'] = os.path.join(moff_folder, 'RGB_inpainted')
    else:
        dset_boxes['images_folder'] = os.path.join(moff_folder, 'RGB')
dset_boxes['annotations_folder'] = os.path.join(moff_folder, 'annotations_boxes_motif')

dset_shapes = {}
only_motifs = False 
inpainted = False
if only_motifs:
    dset_shapes['images_folder'] = os.path.join(moff_folder, 'motifs')
else:
    if inpainted is True:
        dset_shapes['images_folder'] = os.path.join(moff_folder, 'RGB_inpainted')
    else:
        dset_shapes['images_folder'] = os.path.join(moff_folder, 'RGB')
dset_shapes['annotations_folder'] = os.path.join(moff_folder, 'annotations_shape')

train_set_files = np.loadtxt(os.path.join(moff_folder, 'train.txt'), dtype="str")
valid_set_files = np.loadtxt(os.path.join(moff_folder, 'validation.txt'), dtype="str")
test_set_files = np.loadtxt(os.path.join(moff_folder, 'test.txt'), dtype="str")

### BOXES
if inpainted is True:
    dset_boxes['output_dataset'] = os.path.join(moff_folder, 'yolo_inpainted_dataset_boxes')
else:
    dset_boxes['output_dataset'] = os.path.join(moff_folder, 'yolo_dataset_boxes')
os.makedirs(dset_boxes['output_dataset'], exist_ok=True)
if inpainted is True:
    dset_shapes['output_dataset'] = os.path.join(moff_folder, 'yolo_inpainted_dataset_shapes')
else:
    dset_shapes['output_dataset'] = os.path.join(moff_folder, 'yolo_dataset_shapes')
os.makedirs(dset_boxes['output_dataset'], exist_ok=True)

for dset in [dset_boxes, dset_shapes]:
    ds_train_folder = os.path.join(dset['output_dataset'], 'train')
    ds_train_images_folder = os.path.join(ds_train_folder, 'images')
    ds_train_labels_folder = os.path.join(ds_train_folder, 'labels')
    ds_val_folder = os.path.join(dset['output_dataset'], 'val')
    ds_val_images_folder = os.path.join(ds_val_folder, 'images')
    ds_val_labels_folder = os.path.join(ds_val_folder, 'labels')
    ds_test_folder = os.path.join(dset['output_dataset'], 'test')
    ds_test_images_folder = os.path.join(ds_test_folder, 'images')
    ds_test_labels_folder = os.path.join(ds_test_folder, 'labels')
    os.makedirs(ds_train_images_folder, exist_ok=True)
    os.makedirs(ds_train_labels_folder, exist_ok=True)
    os.makedirs(ds_val_images_folder, exist_ok=True)
    os.makedirs(ds_val_labels_folder, exist_ok=True)
    os.makedirs(ds_test_images_folder, exist_ok=True)
    os.makedirs(ds_test_labels_folder, exist_ok=True)

    for train_file_path in train_set_files:
        shutil.copy(os.path.join(dset['images_folder'], train_file_path), os.path.join(ds_train_images_folder, train_file_path))
        annotation_name = f"{train_file_path[:-4]}.txt"
        shutil.copy(os.path.join(dset['annotations_folder'], annotation_name), os.path.join(ds_train_labels_folder, annotation_name))

    for val_file_path in valid_set_files:
        shutil.copy(os.path.join(dset['images_folder'], val_file_path), os.path.join(ds_val_images_folder, val_file_path))
        annotation_name = f"{val_file_path[:-4]}.txt"
        shutil.copy(os.path.join(dset['annotations_folder'], annotation_name), os.path.join(ds_val_labels_folder, annotation_name))

    for test_file_path in test_set_files:
        shutil.copy(os.path.join(dset['images_folder'], test_file_path), os.path.join(ds_test_images_folder, test_file_path))
        annotation_name = f"{test_file_path[:-4]}.txt"
        shutil.copy(os.path.join(dset['annotations_folder'], annotation_name), os.path.join(ds_test_labels_folder, annotation_name))