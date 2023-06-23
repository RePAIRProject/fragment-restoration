# [UNIVE Thesis] Aref 

Image enhancement for:
1) fresco motif classification , 
2) puzzle solving



Application of the following methods for fresco fragment restoration:
- Histogram equalization, 
- gamma correction, 
- balanced contrast enhancement technique (BCET), 
- Contrast limited adaptive histogram equalization (CLAHE) : https://towardsdatascience.com/clahe-and-thresholding-in-python-3bf690303e40 
- ICA 


# MoFF (Motif on Fresco Fragment )

The `MoFF` dataset is prepared by the `prepare_MoFF.py` script and contains at the moment 3 folders:
- `RGB`: original (without any black mark removal) color images
- `segmap3c`: segmentation maps with 3 classes (background, foreground, motif)
- `segmap14c`: segmentation maps with 14 classes (background, foreground, motif1, motif2, ecc.. there are 12 motifs)
Images are cropped, but not resized.

## WIP: Detecting black marks

The script `detect_black_marks.py` detect the black marks on the images (with the resolution of the input image) and saves as output binary masks (white filled boxes) and visualization (red rectangles).
It requires the pretrained to work (please change paths).