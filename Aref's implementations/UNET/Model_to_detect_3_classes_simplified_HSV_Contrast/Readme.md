The pre-processing section of the paper we are inspiring from (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9701350) describes a contrast enhancement method that remaps the intensity values in each channel of an image to improve contrast. It does this through the use of a curve that defines the relationship between input and output values, in order to create more distinct bright and dark areas in the image. This can be interpreted as linear contrast stretching.

The implementation of this approach in our code:

    def contrast_normalization_and_enhancement(image):
      # image is assumed to be in HSV, LAB, or another color space
      # image is also assumed to be in range [0, 255]

      # Convert image to float32 before subtracting and dividing
      image = image.astype(np.float32)

      # We aim to stretch the contrast such that 1% of data is saturated at the low and high intensities
      # This can be achieved using numpy's percentile function

      # Calculate the 1st and 99th percentiles of the image data
      min_val = np.percentile(image, 1)
      max_val = np.percentile(image, 99)

      # Stretch the contrast of the image. Any values below min_val or above max_val are set to the min_val or max_val respectively.
      image = np.clip((image - min_val) * 255.0 / (max_val - min_val), 0, 255)

      # Convert back to uint8
      image = image.astype(np.uint8)

      return image

The metrics result of this approach:
![Metrics](https://github.com/RePAIRProject/fragment-restoration/blob/main/UNET/Model_to_detect_3_classes_simplified_HSV_Contrast/Metrics.png)

My suggestions on why the contrast enhancement step did not lead to a noticeable improvement in performance:

Noisy Contrast Enhancement: The method used for contrast enhancement is introducing noise or distorting the information in the image rather than highlighting the important features. This could indicate that this is not a good fit for this particular type of data.

Redundancy in Information: The HSV color space already separates color information (Hue and Saturation) from intensity (Value), so contrast enhancement might be mostly affecting the Value channel. If the model was already effectively learning from the color channels, then improving the contrast in the Value channel can not add much additional useful information.
