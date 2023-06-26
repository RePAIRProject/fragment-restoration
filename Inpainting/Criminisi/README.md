**The paths to the folders containing the input files and the destination where the results are saved are defined in the file paths_inpainting.py**	

1. **prepare_for_matlab_inpainting.py** code is doing two main operations: 	
   - Creating input images in original sizes with background painted to WHITE color (These wil be the input images to inpainting algorithm in Matlab)	
   - Detected yolo bounding box regions are masked with the fragment foreground. In this way we get rid of the contribution of image background into the detected black mark region.	


2. **iterative_inpainting.m** gets input images (background painted to WHITE) and masks pointing black mark regions.	
   - Runs criminisi inpainting algorithm using the given patch size (ps=11 is used in the experiments, it was better than ps=9, ps=6, and ps=13)	
   - If any interfrence from the image background is found on the fragment region, it is painted by the approximate fragment color which is [172, 172, 170]