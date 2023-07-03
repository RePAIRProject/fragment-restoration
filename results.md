# Semantic Segmentation

Reulsts on the MoFF dataset

## UNET: 3 Classes
| Backbone | Model | Images | Train IoU | Val IoU | Color | D. Aug. | Image Size |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| UNet | Simplified | Original | 0.9198 | 0.8933 | HSV | Geom | 256 |
| UNet | Simplified | Inpainted | - | - | HSV | Geom | 256 |
| UNet | Classic | Original | 0.9357 | 0.9291 | HSV | Geom | 512 |
| UNet | Classic | Inpainted | 0.8858 | 0.9057 | HSV | Geom | 512 |
	

## UNET: 14 Classes 
| Backbone | Model | Images | Train IoU | Val IoU | Color | D. Aug. | Image Size |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| UNet | Simplified | Original | 0.2925 | 0.29 | HSV | Geom  | 256 |
| UNet | Simplified | Inpainted | - | - | HSV | Geom  | 256 |
| UNet | Classic | Original | 0.356 | 0.3057 | HSV | Geom  | 512 |
| UNet | Classic | Inpainted | - | - | HSV | Geom  | 512 |

## Yolo: Bounding box (12 classes)

| Model | Precision | Recall | mAP50 | Annotations | 
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Yolov5 | 0.84264 | 0.81682 | 0.84042 | single annotations 
| Yolov5 | 0.73498 | 0.69932 | 0.77455 | box per motif |  
| Yolov8 | 0.764 | 0.768 | 0.827 | polygon masks |

## Yolo: Pixel-wise Segmentation Masks (12 classes)

| Model | Precision | Recall | mAP50 | Annotations | 
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Yolov8 | 0.744 | 0.815 | 0.825 | polygon masks |

