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


**Next Steps to improve model performance**

Moving forward, several avenues can be explored to further improve the model's performance:

Fine-tuning and Transfer Learning

Hyperparameter Tuning

Model Ensembling: Considering ensembling multiple versions of the simplified UNet model. Each model can be trained with different initialization weights or random seeds, resulting in diverse predictions. Combining the predictions from multiple models can help reduce noise and improve overall segmentation accuracy.

Class-specific data augmentation
