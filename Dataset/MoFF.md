# Motifs on the Fractured Fresco Fragments (MOFFF dataset)

The **Motifs on the Fresco Fragments** (**MOFF**) dataset is used as a benchmark for semantic segmentation, and classification tasks for fractured fresco fragments.
It contains pixel-wise annotations of 12 motif categories shown below, which were created in [Segments.ai](Segments.ai). 
![motif_categories2](https://github.com/RePAIRProject/fragment-restoration/assets/7011371/f2147c4d-977d-4fc8-9fe0-42ed7a0b896e)

### Semantic classes

The semantic classes are also defined in the `YAML` files (in the `yolo_processing` folder, `repair_motif_boxes.yaml` for bounding box annotation and `repair_motif_seg.yaml` for polygonal shapes annotations).
In the final mask images the class numbers correspond to the following categories:

  - `0: images_background` (image background)
  - `1: fragment_background` (the unadorned fragment region)
  - `2: blue_bird`, 
  - `3: yellow_bird`, 
  - `4: red_griffon`,
  - `5: red_flower`, 
  - `6: blue_flower`, ![](../../../Downloads/motif_categories2 (1).png)
  - `7: red_circle`, 
  - `8: red_spiral`, 
  - `9: curved_green_stripe` 
  - `10: thin_red_stripe`, 
  - `11: thick_red_stripe`,  
  - `12: thin_floral_stripe`, 
  - `13: thick_floral_stripe` 

Dataset is published with two schemes of pixel annotations, i.e., for 3-class and 13-class semantic segmentation presented as Scenario 1 and 2 in the paper, respectively.

<img width="423" alt="gts (1)" src="https://github.com/RePAIRProject/fragment-restoration/assets/7011371/ab2c654a-7fb1-4a08-9d36-be4ec7ba6346">
<p align="center">Illustration of pixel-wise semantic annotations for the MoFF dataset. The input image is on the left, followed by 3-class annotations for
Scenario 1 in the middle, and motif-wise annotations for Scenario 2 on the right (different colors represent distinct pixel classes)




