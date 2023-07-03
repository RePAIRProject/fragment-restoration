import os 
import json 
import random 
import numpy as np 

moff_folder = '/media/lucap/big_data/datasets/repair/puzzle2D/motif_segmentation/MoFF'
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
#pdb.set_trace()