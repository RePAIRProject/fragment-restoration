# Semantic Motif Segmentation for Archaelogical Fresco Fragments 
In this repository, you will find datasets, code, and pretrained models related to our work on the semantic motif segmentation of archaeological fresco fragments which will be presented in [E-Heritage workshop](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2023/) of ICCV 2023. This work is developed within the  [RePAIR European Project](https://github.com/RePAIRProject)

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Code](#code)
- [Pretrained Models](#pretrained-models)
- [Usage](#usage)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

### Citation
If you find our work or resources useful in your research, please consider citing our paper:
@inproceedings{enayati2023,
 title={Semantic Motif Segmentation for Archaelogical Fresco Fragments},
 author={Aref Enayati, Luca Palmieri, Sebastiano Vascon, Marcello Pelillo, Sinem Aslan},
 booktitle={},
 pages={},
 year={},
 month={},
 organization={IEEE}
}


## Introduction
This repository contains resources related to our research on the semantic motif segmentation of archaeological fresco fragments. Our work focuses on understanding and categorizing motifs found in ancient fresco fragments, 
serve as valuable resources for various computational tasks, such as fragment recognition, style classification and clustering, fragment reassembly.

<!---### Contents
1. Datasets
2. Restoration of manual annotations
3. Semantic segmentation of fresco fragments
4. Semantic motif segmentation
5. Evaluation codes
-->

## Datasets
### BoFF (Black-Annotations on the Fresco Fragments) Dataset

The BoFF dataset is curated for the task of restoration of manual annotations on the fragments. **[AREF, PLEASE INTRODUCE BOFF DATASET , briefly here and in the BoFF.md file. ex: example figure of few fragments and bounding box annotations, introduce annotations, how they prepared. link to download them.]** It is composed of  xx images with bounding box annotation covering manually drawn black-marks. 

To learn more about the BoFF dataset and how to use it, please refer to the [BoFF](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/Dataset/BoFF.md).

### MoFF (Motif on Fresco Fragment) Dataset

MoFF dataset is curated for the task of semantic segmentation of the motifs present on the painted surface of the fragments. It contains images of real fresco fragments obtained from the Archaeological Park of Pompeii.

Basically, it is generated through the [prepare_MoFF.py](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/Dataset_processing/prepare_MoFF.py) script. 
This script takes the original, high-resolution, unmodified images and associated annotations, accessible for download from [Link to Zenodo](). 
It produces various data folders, listed below, utilized in the experiments described in our paper.

- `RGB`: RGB color images (without any black mark removal), obtained by cropping original input images to include only the fragment region. 
- `RGB_inpained`: inpainted RGB images (YOLOv5 model trained on the BoFF training set was employed to identify manual annotations across all images in the MoFF dataset. Subsequently, the areas identified were inpainted to generate this collection of images.
- `segmap3c`: segmentation maps with 3 classes (image background, fragment background, motif)
- `segmap14c`: segmentation maps with 14 classes (image background, fragment background, motif1, motif2, etc.. there are 12 motifs)
- `motifs`: RGB images containing only motif regions against a black background, generated from the ground truth motif segmentation maps. 
- `annotations_boxes_components`: yolo-style annotations with one box for each component of any motif 
- `annotations_boxes_motif`: yolo-style annotations with one box for motif 
- `annotations_shape`: yolo-v8seg-style annotations (polygons) 

**(Q: ARE THE FOLLOWING TWO FOR THE BLACK MARK REMOVAL???)**
- `yolo_dataset_boxes`: full yolo dataset (images are duplicated, yes) for training for bounding box detection
- `yolo_dataset_shapes`: full yolo dataset (images are duplicated, yes) for training for segmentation (use yolo-v8)

Images are cropped, but not resized.

The train, validation and test split are contained in `.txt` files in the root folder. They are list of file names, the files have always the same names inside each folder. This helps to keep consistency, and test set is the same across different trainings (unet, yolo).

More information about the segmentation maps can be obtained in [MoFF](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/Dataset/MoFF.md).

# 2. Restoration of Manual Annotations
This task is performed by achieving two sub-tasks, including creating inpainting masks by detecting manual annotations in bounding boxes using YoloV5, and performing exampler-based inpainting method of Criminisi.


# 3. Fragment segmentation

# 4. Motif Segmentation

For more information about the semantic classes, please refer to the [MoFF readme file](https://github.com/RePAIRProject/fragment-restoration/tree/e-heritage/Dataset/MoFF.md).

For training and inference using Unet or YOLO, please see below.

## Training UNet

To run the training, use the script `train_segmentation_net.py`, which needs no additional parameters and can be run as:
```bash
python unet/train_segmentation_net.py
```
Everything is inside there, at the beginning of the `main(arg):` function there are the parameters (yes, they could be moved to a parameter file, it would be great!) and afterwards you see the dataset folder (`root_folder_MoFF = ..`) which is also hardcoded at the moment. This also assume you have the MoFF folder in your local pc and the subfolders and train/test split txt files. Check the `prepare_MoFF.py` in case.

The training saves the results in single run subfolders (created) under the `runs` folder. The name has some random number and the parameters appended and inside you find model, graphics, sample predictions and parameters. The name of the run folder is printed on the terminal and can be copied to quickly use it for running inference.

## Inference with UNet

To run inference, there is another script called `show_some_results.py`. This requires some parameters, usually the run folder is enough. It can be passed with the `-f` parameter, so an example launch would be:
```bash
python unet/show_some_results.py -f run16885392688472511_classicUNET_RGB_images512x512_3classes_200epochs_augmented_lr0.001_HSV
```
This takes the trained model and run inference on the test set, saving results and visualization in a subfolder called `results_test_set`. If you want to run inference on a custom folder, use `-i` for the custom image folder and `-m` for the relative masks (needed for visualization not for inference).
For example:
```bash
python unet/show_some_results.py -f run16885392688472511_classicUNET_RGB_images512x512_3classes_200epochs_augmented_lr0.001_HSV -i '/.../datasets/repair/puzzle2D/motif_segmentation/mating_set/images_cropped' -m '/.../datasets/repair/puzzle2D/motif_segmentation/mating_set/masks_cropped'
```
This will run on `images_cropped` and `masks_cropped` folders and save the results in `results_images_cropped` within the subfolder of the run of the training (the one gave with `-f`)/

**NOTE** This does not actually save any numerical results on the test set. I started to create a `performance_unet.py` for that, but `iou` (from `iou_loss(y_true, y_pred)` in `unet.py` returns negative ious! I do not have enough time to debug it, sorry!)

## Training Yolo

Please refer to the official Yolo documentation:
- [Yolov5 for detection](https://docs.ultralytics.com/yolov5/) (of course you could update to yolov8 for detection, but the models were trained using yolov5 at the moment)
- [Yolov8 for segmentation](https://docs.ultralytics.com/tasks/segment/)

Example train for detection (yolov5-like):
```bash
python train.py --batch 32 --epochs 300 --data data/repair_motif_boxes.yaml --weights yolov5s.pt 
```
Example run for segmentation (yolov8-style, yolo CLI):
```bash
yolo segment train data=data/repair_motif_seg.yaml epochs=200 batch=32 imgsz=512  
```

## Detecting and converting Yolo outputs

##### If you just want to get some results quickly, you can use the yolo commands for inference
Example run for detection (yolov5-like):
```bash
python detect.py --weights '/..path../Yolo_best_model_pretrained_best_weights.pt' --source '/..path../images_folder'
```
Example run for segmentation (yolov8-style, yolo CLI):
```bash
yolo segment predict model=path/to/best.pt source='image or image folder'
```


##### If you want to get white bounding boxes (to use the detection for downstream tasks):
The script `detect_black_marks.py` detect the black marks on the images (with the resolution of the input image) and saves as output binary masks (white filled boxes) and visualization (red rectangles).
It requires the pretrained to work (please change paths).

The same (more or less) is valid for `detect_bbox_motif.py`, it creates the white filled bboxes from the prediction of pre-trained yolo (v5) network.

There is a third script `get_segm_yolo.py` which is slightly different, it converts the output of the yolo segmentation (so yolov8) and it requires other stuff (I used it inside the yolo repo actually, but I copied here in case it's needed). 

Also inside `yolo_processing` folder there is a script for preparation and the data config file for training yolo (`prepare_dataset_yolo.py`). Not perfect, but it should help.

##### Validating Yolo
A nice way to get results on validation set is to use `evaluate_yolo_detection.py` (changing folder name) or call the yolo CLI `yolo segment val model=path/to/best.pt  # val custom model`.
