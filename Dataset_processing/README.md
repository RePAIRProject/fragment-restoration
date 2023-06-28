# Motifs on the Fractured Fresco Fragments (MOFFF dataset)

The **Motifs on the Fractured Fresco Fragments** (**MOFFF** dataset) contains 12 motif categories including semantic 
object classes such as _**blue bird**, **yellow bird**, **red griffon**, **red flower**, **blue flower**_, and structural pattern classes such as _**red_circle**, **red spiral**, 
**thin red stripe**, **thick red stripe**, **thin floral stripe**, **thick floral stripe**_ and _**curved green stripe**_. Each image 
in this dataset has pixel-level segmentation annotations. This dataset can be used as a benchmark for semantic segmentation, 
and classification tasks for fractured fresco fragments. 


Dataset is created following the stages below:

- Three repositories were already created in [Segments.ai](Segments.ai) for pixel-wise annotations, namely, _lucap_repair_fragments_patterns_ , _sinemaslan_repair_fragments_patterns-clone_ , _UNIVE_decor2_. Segmentation masks and segmented input images are exported into Dataset/segments folder (check export_segments_ai function in dataset_processing).
- Segmentation mask class labels of decor2 repository are changed from (1,2,3,4) to (4,13,14,12) to avoid confusion when three repositories merged.
- Images and masks from three repositories are merged into single folders named 'images' and 'masks'
- Fragment foreground masks are created into fg folder
- Fragment background is included as a new category and masks are refined further by re-mapping class labels. In the final mask images the class numbers correspond to the following categories:
  - 0 image background
  - 1 fragment background (this is the fragment region)
  - 2 bluebird, (animal_blue_bird)
  - 3 yellow bird, (animal_yellow_bird)
  - 4 red griffon
  - 5 red flower, (flower_red)
  - 6 blue flower,  (flower_blue)
  - 7 red_circle, (red_dot)
  - 8 red spiral, (red_spiral_pattern)
  - 9 curved green stripe (curved_green_line)
  - 10 thin red stripe, (thin_straight_red_line)
  - 11 thick red stripe,  (thick_straight_red_line)
  - 12 thin floral stripe, (yd_small_flower)
  - 13 thick floral stripe (yd_big_flower)
- Final images and masks folders are added into Teams/wp3/files/Motif_Segmentation


**You can change the path definitions in paths.py for input / output files. To perform each operation uncomment it in dataset_processing.py**