**Overview**

This part of project focuses on the task of motif-based semantic segmentation using the UNet architecture. The goal is to accurately segment motifs within images, distinguishing them from the background and other foreground elements.

**Dataset**

The project uses a custom dataset consisting of images with motif patterns. Each image is accompanied by a ground truth mask where each motif class is labeled with a unique value. The dataset contains a variety of motif classes and a significant imbalance among them.
The Unet_prep.ipynb file contains the code on how this dataset was created.

The link to dataset:

https://drive.google.com/file/d/1EOuKdiiX1Dkfe9GmJTTr45FOUhQ0iU2D/view?usp=share_link

**Initial Approach**

Initially, a UNet architecture was trained with each motif class treated separately. However, this approach did not yield satisfactory results, likely due to the class imbalance and complex relationships between motifs.

Related notebook of this step is UNET.ipynb.

**Binary Semantic Segmentation**

To address the challenges posed by the motif classes, a binary semantic segmentation approach was adopted. The task was simplified to differentiate between the background and foreground. This approach achieved promising results with a Mean IoU of 99%, indicating accurate segmentation of mforeground from the background.

Training log, metrics, related code and  random visual results are in the folder "Model_to_detect_only_background_and_foreground".

**Integration of Pattern Motif Class**

To expand the model's capability, a new class for pattern motifs was introduced. The revised model now had three classes: background, pattern motifs, and foreground. However, the standard UNet architecture with this multi-class setup did not perform well.

Training log, metrics, related code and  random visual results are in the folder "Model_to_detect_3_class_background_pattern_foreground".

**Simplified UNet Architecture**

In an attempt to improve the performance, the UNet architecture was simplified and the number of layers and parameters were reduced. This simplified architecture showed promising results, achieving an Mean IoU of over 80% within the first 50 epochs of training.

Training log, metrics, related code and  random visual results are in the folder "Model_to_detect_3_class_background_pattern_foreground_simplifiedUNET".

**Separate Category for Most Frequent Motif Class**

An improvement was attempted by introducing a separate category for the most frequent motif class, resulting in four main categories: background, motif class 4, other motif classes (excluding class 4), and foreground. This modification led to a mean Intersection over Union (meanIOU) of about 45% after training the model for a reasonable number of epochs.

Training log, metrics, related code and visual results for all the test images are in the folder "MModel_to_detect_4_classes_50epoch".

**Extended Training**

Continuing from the previous approach, the same model was further trained for an extended period, reaching approximately 55% meanIOU after 150 epochs. This improvement indicates that the model continued to learn and refine its segmentation capabilities with prolonged training.

Training log, metrics, related code and  random visual results are in the folder "MModel_to_detect_4_classes_150epoch".


***Model Experiments with differant colour spaces***

I experimented with different color spaces for our image input, transforming the original RGB images to HSV, Lab, and YCrCB. I trained the model on images using each of these color spaces individually, and also with a combination of these color spaces. The corresponding models can be found in the following folders:

Model_to_detect_3_classes_simplified_HSV: Model trained using images transformed from RGB to HSV.
Model_to_detect_3_classes_simplified_Lab: Model trained using images transformed from RGB to Lab.
Model_to_detect_3_classes_simplified_YCrCB: Model trained using images transformed from RGB to YCrCB.

Additionally, I trained models using input images that combined multiple color spaces. Each of these combined color spaces model are found in the following folders:

Model_to_detect_3_classes_simplified_RGB_HSV_Lab_combined: Model trained using images with a combined 9-channel input (3 channels for RGB, 3 for HSV, and 3 for Lab).
Model_to_detect_3_classes_simplified_RGB_HSV_Lab_YCrBC_combined: Model trained using images with a combined 12-channel input (3 channels for RGB, 3 for HSV, 3 for Lab, and 3 for YCrBC).

**Results**

In my evaluation, the models trained with combined color spaces and the model trained with only HSV channels achieved similar performances in the first 50 epochs, with mean IoU scores close to 90%. However, the HSV-only model demonstrated slightly better metrics, indicating that the added complexity of the additional color channels may not provide significant benefits.

The addition of more color channels increases both the memory requirement and the training time of the models. Therefore, considering the trade-off between performance and computational efficiency, the model with only HSV color space appears to be the most promising candidate for further improvement.

**Next Steps**

Based on findings, I propose the following steps for further enhancing our image segmentation model:

    Contrast Enhancement: In the referenced paper, the authors used contrast enhancement during the pre-processing of images. This step could be valuable, as it appears that our current model struggles with low-contrast colors.

    Black Marks Removal: During evaluation, we found that our model sometimes detects black marks. We can improve the model's performance by testing on images where these black marks have been removed and inpainted.

    Extended Training: The training curves for our current best-performing model suggest that it is still learning. Training the model for a larger number of epochs could result in better metrics.

    Hyperparameter Tuning: As with any machine learning model, further improvements may be achieved through more extensive hyperparameter tuning.
