# Semantic Motif Segmentation for Archaelogical Fresco Fragments 
In this repository, we provide datasets, code, and pretrained models related to our work on the semantic motif segmentation 
of archaeological fresco fragments. This work will be presented in [E-Heritage workshop](https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2023/) 
held as part of [ICCV 2023](https://iccv2023.thecvf.com/). To access our project page, please click [here](https://repairproject.github.io/fragment-restoration/). 
This work has been carried out as part of the [RePAIR European Project](https://github.com/RePAIRProject).

https://github.com/RePAIRProject/fragment-restoration/assets/7011371/eabef996-9a59-47ff-996a-206ab2bd2f8d

If you find our work or resources useful in your research, please consider citing our paper:
```

    @InProceedings{Enayati_2023_ICCV,
    author    = {Enayati, Aref and Palmieri, Luca and Vascon, Sebastiano and Pelillo, Marcello and Aslan, Sinem},
    title     = {Semantic Motif Segmentation of Archaeological Fresco Fragments},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {1717-1725}
}
```

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Restoration of Manual Annotations](#restoration-of-manual-annotations)
- [Semantic Segmentation of Fragments](#semantic-segmentation-of-fragments)
- [Acknowledgments](#acknowledgments)

## Introduction
Our work focuses on understanding and categorizing motifs found in ancient fresco fragments, 
serve as valuable resources for various computational tasks, such as fragment recognition, style classification and clustering, fragment reassembly. 
Additionally, we have introduced an additional task concerning fragment restoration. In particular, two datasets were curated with annotations 
for these two specific tasks, and baseline models for them.

<!---### Contents
1. Datasets
2. Restoration of manual annotations
3. Semantic segmentation of fresco fragments
4. Semantic motif segmentation
5. Evaluation codes
-->

## Datasets
### BoFF (**B**lack-Annotations **o**n the **F**resco **F**ragments) Dataset

The BoFF dataset is curated for the task of restoration of manual annotations on the fragments. It is composed of 115 images of fresco fragments with bounding box annotations covering manually drawn black-marks. These black-marks indicate neighboring relationship between fragmetns and they serve as a guide to archeologists during manual reconstruction of these frescoes.  

To learn more about the BoFF dataset and how to use it, please refer to the [BoFF](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/Dataset/BoFF.md).

Make a request [here](https://docs.google.com/forms/d/e/1FAIpQLSfoCSHl5M23LeXok_iSL-yxKmK0AJShTWccjDb2Xas6F54qvw/viewform) to access the dataset and annotations.

### MoFF (Motifs on Fresco Fragment) Dataset

MoFF dataset is curated for the task of semantic segmentation of the motifs present on the painted surface of the fragments. It contains images of real fresco fragments obtained from the Archaeological Park of Pompeii. The following two annotation scenarios were employed:
- Scenario 1: 3-class annotation (image background, fragment background, motif class).
- Scenario 2: 12-class annotation (distinct motif types).

Basically, the data used n the experiments is generated through the [prepare_MoFF.py](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/Dataset_processing/prepare_MoFF.py) script. 
This script takes the original, high-resolution, unmodified images and associated annotations. 

It produces various data folders, listed below, utilized in the experiments described in our paper.

- `RGB`: RGB color images (without any black mark removal), obtained by cropping original input images to include only the fragment region. 
- `RGB_inpained`: inpainted RGB images (YOLOv5 model trained on the BoFF training set was employed to identify manual annotations across all images in the MoFF dataset. Subsequently, the areas identified were inpainted to generate this collection of images.
- `segmap3c`: segmentation maps with 3 classes (image background, fragment background, motif)
- `segmap14c`: segmentation maps with 14 classes (image background, fragment background, motif1, motif2, etc.. there are 12 motifs)
- `motifs`: RGB images containing only motif regions against a black background, generated from the ground truth motif segmentation maps. 
- `annotations_boxes_components`: yolo-style annotations with one box for each component of any motif 
- `annotations_boxes_motif`: yolo-style annotations with one box for motif 
- `annotations_shape`: yolo-v8seg-style annotations (polygons) 
- `yolo_dataset_boxes`: full yolo dataset (images are duplicated, yes) for training for bounding box detection
- `yolo_dataset_shapes`: full yolo dataset (images are duplicated, yes) for training for segmentation (use yolo-v8)

Images are cropped, but not resized.

The train, validation and test split are contained in `.txt` files in the root folder. They are list of file names, the files have always the same names inside each folder. This helps to keep consistency, and test set is the same across different trainings (unet, yolo).

More information about the segmentation maps can be obtained in [MoFF](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/Dataset/MoFF.md).

Make a request [here](https://docs.google.com/forms/d/e/1FAIpQLSfzdvuchDQ1Y-uqiRBS8C4eMO6LY7e6blzPawmuwmEXYmRXyA/viewform) to access the dataset and annotations.


## Restoration of Manual Annotations 
This task is performed by achieving two sub-tasks, including creating inpainting masks by detecting manual annotations in bounding boxes using YoloV5, and performing exampler-based inpainting method of Criminisi.
### Detecting black marks

For the purpose of training and detection of black-marks on fresco fragments using YOLOv5 on BOFF Augmented Dataset, we utilized the augmented version of the BOFF dataset. The ultimate goal was to accurately detect black-marks present on the test images.

#### Google Colab Implementation

For those wanting a comprehensive guide, we've set up a Google Colab notebook. This interactive document provides a step-by-step walkthrough covering the entire process including:
- Calculating and visualizing predictions
- Identifying True Positives (TP), False Positives (FP), and False Negatives (FN)
- Creating and verifying binary masks, which will be essential for the subsequent inpainting algorithm.

**Access the Google Colab notebook:** [Google Colab File](https://drive.google.com/file/d/1hVL2hjDGKQDPecFetvK_TDC90VhHLPdR/view?usp=drive_link)

#### Pretrained Weights

For those interested in using the model directly or benchmarking against our results, we've also made available the pretrained weights that produced the best results during our experiments.

**Download the pretrained weights:** [Best Weights](https://drive.google.com/file/d/1cnlSg_dwep9LsX2vI1BEUWVZNxG80m0Q/view?usp=drive_link)

#### Quick Model Deployment and Full Reproduction

- Quick Model Deployment: To swiftly load the pretrained model and obtain new results, refer to our dedicated Python file: [detect_black_marks.py](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/black_mark_removal/detecting_blackmarks_by_yolo/detect_black_marks.py).

- Full Process Reproduction: If you're interested in reproducing our entire process or aim to train your own model from scratch, the Google Colab file linked above is your go-to guide.


### Inpainting
More information about the implementation can be obtained in [Iterative_inpainting](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/Dataset/MoFF.md).



## Semantic Segmentation of Fragments

This section refers to the segmentation tasks. As described in the paper, we have envisioned two scenarios:
1. ***Fragment segmentation***: here we have 3 classes (background, fragment surface and *any* motif) and the focus is detecting the fragment and the motif (without the type)
2. ***Motif segmentation***: here we have 13 classes (12 different motif type and *anything else*) and the focus is on recognizing the *type* of the motif (the rest is treated as unimportant, background and fragment are merged).

For more information about the semantic classes, please refer to the [MoFF readme file](https://github.com/RePAIRProject/fragment-restoration/tree/e-heritage/Dataset/MoFF.md).

Two scenarios have an important difference conceptually, while regarding implementation, this makes a little difference. We used two different networks for the segmentation task, [UNet](#unet) and [YOLO](#yolo).

### UNet

UNet is a widely used framework for pixel-wise segmentation. This is a custom re-implementation using pytorch which follows the standard architecture. 

Everything related should be available under the `unet` folder, with `unet.py` containing the two model (the *classic* and the *simplified* one) and other scripts for training, for showing results and for evaluating performances.

#### Parameters

When training, we define a set of parameters in the `train_segmentation_net.py` script which are then saved into a parameter file in the output folder (so that you know which parameters where used).
The parameters look like this:
```python
## Parameters 
IMG_SIZE = 512 # resize images to IMG_SIZE x IMG_SIZE
EPOCHS = 200 # train max. until this number (early stopping is enabled)
BATCH_SIZE = 8 # change batch size according to your GPU
AUGMENT = True # whether to perform or not on-the-fly data augmentation
aug_geometric = True # geometric augmentation (rotation, mirror)
aug_color = False # color augmentation (use only with RGB)
COLOR_SPACE = 'HSV' # this changes the color space for all images
CLASSES = 13 # number of classes (scenario 1 --> 3 classes, scenario 2 --> 13 classes)
LEARNING_RATE = 0.001 # there is a scheduler for the decay!
MODEL = 'classic' # it can be either 'classic' or 'simplified' 
INPAINTED = True # picking inpainted images (use False for the original ones)
``` 


#### Training 

To run the training, use the script `train_segmentation_net.py`, which needs no additional parameters and can be run as:
```bash
python unet/train_segmentation_net.py
```
Everything is inside there, at the beginning of the `main(arg):` function there are the parameters (yes, they could be moved to a parameter file, it would be great!) and afterwards you see the dataset folder (`root_folder_MoFF = ..`) which is also hardcoded at the moment. This also assume you have the MoFF folder in your local pc and the subfolders and train/test split txt files. Check the `prepare_MoFF.py` in case.

The training saves the results in single run subfolders (created) under the `runs` folder. The name has some random number and the parameters appended and inside you find model, graphics, sample predictions and parameters. The name of the run folder is printed on the terminal and can be copied to quickly use it for running inference.

#### Inference

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

**NOTE:** This does not actually save any numerical results on the test set. Please check the `evaluation` folder/section for more information about numerical results. 
#### Pretrained Models

You can find the [pretrained models here](https://drive.google.com/drive/folders/19N7pfHEJQ6LPPzzAXYLJIoKPNtKRT7aj?usp=sharing). In this folder there are the pretrained models with their parameters for scenario 1 and scenario 2 for both simplified and classic UNet architecture.

### Yolo

#### Training

Please refer to the official Yolo documentation:
- [Yolov5 for motif detection](https://docs.ultralytics.com/yolov5/) (of course you could update to yolov8 for detection, but the models were trained using yolov5 at the moment)
- [Yolov8 for motif segmentation](https://docs.ultralytics.com/tasks/segment/)

Example train for detection (yolov5-like):
```bash
python train.py --batch 32 --epochs 300 --data data/repair_motif_boxes.yaml --weights yolov5s.pt 
```
Example run for segmentation (yolov8-style, yolo CLI):
```bash
yolo segment train data=data/repair_motif_seg.yaml epochs=200 batch=32 imgsz=512  
```

#### Inference 

YOLO already provides CLI for running train and inference, but it already draws the results on the image to show them. So it is very good for quickly cheking how is working, but you may need to run custom code if you want to do something with the results. 

##### YOLO commands for inference
Example run for detection (yolov5-like):
```bash
python detect.py --weights '/..path../Yolo_best_model_pretrained_best_weights.pt' --source '/..path../images_folder'
```
Example run for segmentation (yolov8-style, yolo CLI):
```bash
yolo segment predict model=path/to/best.pt source='image or image folder'
```


##### Custom code for the white bounding boxes (to use the detection for downstream tasks):
The script `detect_bbox_motif.py` detect the motif on the images (with the resolution of the input image) and saves as output binary masks (white filled boxes) and visualization (red rectangles).
It requires the pretrained model to work (please change paths).

There is another script `get_segm_yolo.py` which is slightly different, it converts the output of the yolo segmentation (so yolov8) and it requires other stuff (I used it inside the yolo repo actually, but I copied here in case it's needed). 

Also inside `yolo_processing` folder there is a script for preparation and the data config file for training yolo (`prepare_dataset_yolo.py`). Not perfect, but it should help.

#### Validation
A nice way to get results on the validation set is to use `evaluate_yolo_detection.py` (changing folder name) or call the yolo CLI `yolo segment val model=path/to/best.pt  # val custom model`.

#### Pretrained Models

You can find the [pretrained models here](https://drive.google.com/drive/folders/19jPUtnR-FvRWhXNse70guq2xXKXkXPva?usp=sharing). 
In the folder you find the v5 trained models for detection (we trained 2 using a bounding box for each motif or for each *component* of the motifs) and the v8 trained model for segmentation (it can also do detection).


### Evaluation
We provide a script for evaluation of predictions. In this case we expect you create predictions from all the images of the test set (the `test.txt` file is a list of the files in the test set).
Once you predicted (with your model) the masks for these files, store them in a folder. The name of the files should be the same as in the `test.txt` files so that the script reads them.
___
For benchmarking, the following parameters are required:

- Your prediction folder (all predicted masks should be there).
- The number of classes (valid values are 3, 12, 13, and 14).
- The path to the MoFF dataset (you should have it downloaded).
- Output folder (optional; if not specified, the script will use your current folder as the output location).

Note: While there is only one version of the ground truth masks with 14-class annotations, other class configurations are derived using a remapping function present in the code.
___
<!---*(For benchmarking we need your prediction folder (all predicted masks should be there), the number of classes (if you are evaluating for 3 or 13 classes), the path of the MoFF dataset (you should have it downloaded) and the output folder (optional, otherwise the script will use your current folder as output).)*-->

We rescale the images (as default the size is `512x512`). This is done using nearest neighbour interpolation to preserve integer class values. You can change the size using the `-s img_size` parameter.

To benchmark, you can run the script for example as:
```bash
python evaluations/benchmark_v2.py -p 'path_to_prediction_folder' -c 3 -d 'path_to_dataset/MoFF' -o 'path_to_output_folder'
```

If you did not download the full MoFF, but you only have the ground truth masks and the test.txt files (at least these are needed) you can run the script explicitly setting `-t path_to_the_test.txt` and `-gt path_to_the_gt_folder`.

Example Runs:
If you run on 3 classes, it outputs the performances like:
```bash
##############################
Performances on 3 classes
IoU (avg): 0.913
PA  (avg): 0.964
##############################
```
If you run on 13 classes, it outputs the performances like:
```bash
##############################
Performances on 13 classes
IoU (avg): 0.614
PA  (avg): 0.637
------------------------------
Performances on motif only
IoU (motif): 0.405
PA  (motif): 0.441
##############################
```

#### Reproducing the paper results
We also provide a script to reproduce our unet vs yolo benchmark. 
This assumes you already have results from YOLO and UNet models in their respective folder. 
The folder paths are hard-coded in the first lines of the file, and you can change accordingly to your local storage.
It is not necessary and provided as a reference.
It can be run as 
```bash
python evaluations/reproduce_unet_vs_yolo13c.py
```
And it should output (this is the result on our computer, with the prediction of the latest models on 512x512 images, Table 3 of the paper):
```bash
#################################################################
AVERAGE (13 classes)
Mean IoU:
 UNET (simplified): 0.569
 UNET             : 0.606
 YOLO             : 0.538
Mean Pixel Accuracy:
 UNET (simplified): 0.600
 UNET             : 0.630
 YOLO             : 0.797
-----------------------------------------------------------------
MOTIF (12 classes)
Mean IoU:
 UNET (simplified): 0.345
 UNET             : 0.416
 YOLO             : 0.582
Mean Pixel Accuracy:
 UNET (simplified): 0.392
 UNET             : 0.452
 YOLO             : 0.634
#################################################################
```

#### Evaluation of Color Space Experiments

For readers and researchers interested in the evaluation presented in Table 2 of our paper — experiments conducted with various color spaces in Scenario 1 — we've provided some resources to facilitate further exploration and reproduction of our results.

##### Trained Models

We have made available the trained models for these experiments. These can be instrumental for direct implementation or further tweaking based on specific needs.

**Access the trained models:** [Trained Models Folder](https://drive.google.com/drive/folders/11Ed831CBjengf0ZogeFeUvGustRCUa6m)

##### Google Colab for Reproduction

To further assist in reproducing our results or evaluating the model on your own data, we've prepared a Google Colab notebook. This notebook provides a streamlined approach to load the trained models, reproduce the results, and evaluate on custom datasets.

**Access the Google Colab notebook:** [Google Colab File for Reproduction](https://colab.research.google.com/drive/1fIOgDT6X8wWssAiyO8pWILENiFaf1dHw?ouid=114722430595098931105&usp=drive_link)

## Acknowledgement

This work is part of a project that has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No.964854. 

