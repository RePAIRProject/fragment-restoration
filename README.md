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


## WIP: Detecting black marks

The script `detect_black_marks.py` detect the black marks on the images (it seems resolution is not important) and saves as output binary masks (white filled boxes) and visualization (red rectangles).
It requires the pretrained to work (please change paths and remember, pretrained model is in the `main` branch).