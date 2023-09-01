# Motifs on the Fractured Fresco Fragments (MOFFF dataset)

The **Motifs on the Fractured Fresco Fragments** (**MOFFF** dataset) contains 12 motif categories including semantic object classes such as _blue bird, yellow bird, red griffon, red flower, blue flower_, and structural pattern classes such as _red_circle, red spiral, thin red stripe, thick red stripe, thin floral stripe, thick floral stripe_ and _curved green stripe_. 
Each image in this dataset has pixel-level segmentation annotations. This dataset can be used as a benchmark for semantic segmentation, and classification tasks for fractured fresco fragments. 

## Dataset Creation
Dataset is created following the stages below:

- Three repositories were already created in [Segments.ai](Segments.ai) for pixel-wise annotations, namely, _lucap_repair_fragments_patterns_ , _sinemaslan_repair_fragments_patterns-clone_ , _UNIVE_decor2_. Segmentation masks and segmented input images are exported into Dataset/segments folder (check export_segments_ai function in dataset_processing).
- Segmentation mask class labels of decor2 repository are changed from (1,2,3,4) to (4,13,14,12) to avoid confusion when three repositories merged.
- Images and masks from three repositories are merged into single folders named 'images' and 'masks'
- Fragment foreground masks are created into fg folder
- Fragment background is included as a new category and masks are refined further by re-mapping class labels. 

## Semantic classes

The semantic classes are also defined in the `YAML` files (in the `yolo_processing` folder, `repair_motif_boxes.yaml` for bounding box annotation and `repair_motif_seg.yaml` for polygonal shapes annotations).
In the final mask images the class numbers correspond to the following categories:

  - `0: background` (image background, black)
  - `1: fragment_surface` fragment background (this is the fragment region)
  - `2: blue_bird`, (patterns showing a blue bird)
  - `3: yellow_bird`, (patterns showing a yellow bird)
  - `4: red_griffon`,
  - `5: red_flower`, 
  - `6: blue_flower`, 
  - `7: red_circle`, (red dots)
  - `8: red_spiral`, (spiral patterns)
  - `9: curved_green_stripe` (curved green lines)
  - `10: thin_red_stripe`, (straight narrow lines)
  - `11: thick_red_stripe`,  (larger straight lines)
  - `12: thin_floral_stripe`, (floral decoration band)
  - `13: thick_floral_stripe` (larger floral decoration band)





