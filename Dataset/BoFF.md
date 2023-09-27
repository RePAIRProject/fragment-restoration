# Black-marks on the Fractured Fresco Fragments (BOFF) Dataset

The **Black-marks on the Fresco Fragments** (**BOFF**) dataset is curated to facilitate the detection and removal of black marks on fresco fragments through inpainting. It offers images of fragments along with bounding box annotations for the black marks.

## About the Dataset

The bounding box annotations were created using [roboflow.com](https://app.roboflow.com/), which provides a powerful set of tools for managing image datasets for computer vision applications.

We had 115 images of real fresco fragments at the time of development. These images are annotated and resulted in 405 black mark annotations, stored as `.txt` files with their respective names in a 'labels' folder. The black marks are found in both monochromatic and textured regions. Notably, most black marks are near the fragment boundaries, but a few are found in the center. Some fragments have no black marks on their surface.

Here are three example images with their bounding box annotations, depicted by red rectangles:
![boff](https://github.com/RePAIRProject/fragment-restoration/blob/e-heritage/web/static/images/boff.jpg)

## Dataset Versions

There are two versions of the BOFF dataset:

### Version 1 
- **Description:** Contains only the raw images without any form of pre-processing or augmentation.
- **Download Link:** [Version 1](https://docs.google.com/forms/d/e/1FAIpQLSfoCSHl5M23LeXok_iSL-yxKmK0AJShTWccjDb2Xas6F54qvw/viewform)

### Version 2 
- **Description:** This version includes augmented images in the training set, with the validation and test sets remaining untouched.
  
  The augmentation steps used are:
-       90-degree Rotations: Each source image has an equal probability of undergoing one of the following 90-degree rotations:
          No rotation (original image retained)
          Clockwise rotation
          Counter-clockwise rotation
          Upside-down (180-degree rotation)

        Random Rotations: Beyond the 90-degree rotations, each image may also undergo a random rotation between -30 and +30 degrees.
    
- **Download Link:** [Version 2](https://docs.google.com/forms/d/e/1FAIpQLSfoCSHl5M23LeXok_iSL-yxKmK0AJShTWccjDb2Xas6F54qvw/viewform)

## Dataset Statistics

### Original Dataset:

|       | Number of Images | Number of Bboxes |
|-------|------------------|------------------|
| Train | 91               | 324              |
| Valid | 12               | 42               |
| Test  | 12               | 39               |
| Total | 115              | 405              |

### Augmented Dataset:

|       | Number of Images | Number of Bboxes |
|-------|------------------|------------------|
| Train | 273              | 972              |
| Valid | 12               | 42               |
| Test  | 12               | 39               |
| Total | 297              | 1053             |

## Dataset Structure

Both versions of the dataset adhere to the following structure:

- Root directory: `test` (folder), `train` (folder), `valid` (folder), `data.yaml`, `README.dataset`, `README.roboflow`
- Inside each folder, you will find two subfolders: `images` and `labels`. 
  - `images` contains images of fresco fragments. 
  - `labels` houses the `.txt` files corresponding to each image in the 'images' folder. Each `.txt` file offers bounding box coordinates of black-marks on the relevant fresco fragments. If a `.txt` file has, for instance, 10 lines of coordinates, it indicates there are 10 bounding box annotations on that fragment.
