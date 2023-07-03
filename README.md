<!--- #[UNIVE Thesis] Aref 

Image enhancement for:
1) fresco motif classification , 
2) puzzle solving

Application of the following methods for fresco fragment restoration:
- Histogram equalization, 
- gamma correction, 
- balanced contrast enhancement technique (BCET), 
- Contrast limited adaptive histogram equalization (CLAHE) : https://towardsdatascience.com/clahe-and-thresholding-in-python-3bf690303e40 
- ICA ---> 


# MoFF (Motif on Fresco Fragment )

The `MoFF` dataset is prepared by the `prepare_MoFF.py` script and contains at the moment 3 folders:
- `RGB`: original (without any black mark removal) color images
- `segmap3c`: segmentation maps with 3 classes (background, foreground, motif)
- `segmap14c`: segmentation maps with 14 classes (background, foreground, motif1, motif2, ecc.. there are 12 motifs)
Images are cropped, but not resized.s

The train, validation and test split are contained in `.txt` files in the root folder.

# How to use the scripts

Not everything is fully cleaned and polished, but it should be understandable and usable.

### Training UNet

To run the training, use the script `train_segmentation_net.py`, which needs no additional parameters and can be run as:
```bash
python train_segmentation_net.py
```
Everything is inside there, at the beginning of the `main(arg):` function there are the parameters (yes, they could be moved to a parameter file, it would be great!) and afterwards you see the dataset folder (`root_folder_MoFF = ..`) which is also hardcoded at the moment. This also assume you have the MoFF folder in your local pc and the subfolders and train/test split txt files. Check the `prepare_MoFF.py` in case.

The training saves the results in single run subfolders (created) under the `runs` folder. The name has some random number and the parameters appended and inside you find model, graphics, sample predictions and parameters. The name of the run folder is printed on the terminal and can be copied to quickly use it for running inference.

### Inference with UNet

To run inference, there is another script called `show_some_results.py`. This requires some parameters, usually the run folder is enough. It can be passed with the `-f` parameter, so an example launch would be:
```bash
python show_some_results.py -f run16885392688472511_classicUNET_RGB_images512x512_3classes_200epochs_augmented_lr0.001_HSV
```
This takes the trained model and run inference on the test set, saving results and visualization in a subfolder called `results_test_set`. If you want to run inference on a custom folder, use `-i` for the custom image folder and `-m` for the relative masks (needed for visualization not for inference).
For example:
```bash
python show_some_results.py -f run16885392688472511_classicUNET_RGB_images512x512_3classes_200epochs_augmented_lr0.001_HSV -i '/.../datasets/repair/puzzle2D/motif_segmentation/mating_set/images_cropped' -m '/.../datasets/repair/puzzle2D/motif_segmentation/mating_set/masks_cropped'
```
This will run on `images_cropped` and `masks_cropped` folders and save the results in `results_images_cropped` within the subfolder of the run of the training (the one gave with `-f`)/


## Detecting and converting Yolo outputs

The script `detect_black_marks.py` detect the black marks on the images (with the resolution of the input image) and saves as output binary masks (white filled boxes) and visualization (red rectangles).
It requires the pretrained to work (please change paths).

The same (more or less) is valid for `detect_bbox_motif.py`, it creates the white filled bboxes from the prediction of pre-trained yolo (v5) network.

There is a third script `get_segm_yolo.py` which is slightly different, it converts the output of the yolo segmentation (so yolov8) and it requires other stuff (I used it inside the yolo repo actually, but I copied here in case it's needed). 

Also inside `yolo_prep` folder there is a script for preparation and the data config file for training yolo. Not perfect, but it should help.