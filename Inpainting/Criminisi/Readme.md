
The iterative inpainting process is a back and forth process between python notebook (Prepare_for_inpainting.ipynb) and MATLAB script (final_eval.m). 
In the first step:
  Original images and masks are read in python notebook.

In First iteration of inpainting:
  We use the function make_background_white to prepare the images, and then to prepare the masks we create the eroded masks in python notebook and save them. 
  Then we reed these images in MATLAB script and perform the inpainting and save the results.

In second iteration of inpainting:
  We take the saved results of first inpainting and read them in python notebook and use make_background_black and keep_white_pixels to create the masks for second iteration of inpainting. 
  We save the masks and then load them in MATLAB and perform another iteration of inpainting and save the results. (In this istep the input images are the results of first iteration)

Final Step:
  We read the second iteration results in python notebook, use mak_background_black_updated_3 function to make the background black again and save the results. 
