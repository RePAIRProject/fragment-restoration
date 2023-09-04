# Motifs on the Fractured Fresco Fragments (MOFFF dataset)

The **Motifs on the Fresco Fragments** (**MOFFF** dataset) contains 12 motif categories including semantic object classes such as _blue bird, yellow bird, red griffon, red flower, blue flower_, and structural pattern classes such as _red_circle, red spiral, thin red stripe, thick red stripe, thin floral stripe, thick floral stripe_ and _curved green stripe_. 
Each image in this dataset has pixel-level segmentation annotations. Pixel-wise annotations are created in [Segments.ai](Segments.ai). This dataset can be used as a benchmark for semantic segmentation, and classification tasks for fractured fresco fragments.


![motif_categories2](https://github.com/RePAIRProject/fragment-restoration/assets/7011371/f2147c4d-977d-4fc8-9fe0-42ed7a0b896e)

## Semantic classes

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



































## Dataset Creation
Dataset is created following the stages below:

- Three repositories were already created in [Segments.ai](Segments.ai) for pixel-wise annotations, namely, _lucap_repair_fragments_patterns_ , _sinemaslan_repair_fragments_patterns-clone_ , _UNIVE_decor2_. Segmentation masks and segmented input images are exported into Dataset/segments folder (check export_segments_ai function in dataset_processing).
- Segmentation mask class labels of decor2 repository are changed from (1,2,3,4) to (4,13,14,12) to avoid confusion when three repositories merged.
- Images and masks from three repositories are merged into single folders named 'images' and 'masks'
- Fragment foreground masks are created into fg folder
- Fragment background is included as a new category and masks are refined further by re-mapping class labels. 






