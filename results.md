# Semantic Segmentation

Reulsts on the MoFF (v1, old) dataset

## 3 Classes
| Backbone | Model | Train IoU | Val IoU | Dataset | Epochs | Color | D. Aug. | LR |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| UNet | Simplified | 0.886 | 0.866 | `v1` | 150 | RGB | Geom & Color | 0.0001 |
| UNet | Simplified | 0.911 | 0.889 | `v1` | 150 | HSV | No | 0.0001 |
| UNet | Simplified | 0.908 | 0.889 | `v1` | 150 | HSV | Geom | 0.0001 |
| UNet | Simplified | 0.920 | 0.894 | `v2` | 150 | HSV | Geom | 0.001+`scheduler` |
| UNet | Simplified | 0.917 | 0.901 | `v2`+inpainted | 150 | HSV | Geom | 0.001+`scheduler` |
| UNet | Simplified | 0.936 | 0.929 | `v2`(512px) | 150 | HSV | Geom | 0.001+`scheduler` |


## 14 Classes 
| Backbone | Model | Train IoU | Val IoU | Dataset | Epochs | Color | D. Aug. | LR |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| UNet | Simplified | 0.292 | 0.290 | `v1` | 150 | HSV | Geom  | 0.0001 |
| UNet | Simplified | 0.345 | 0.309 | `v1` | 150 | HSV | Geom  | 0.001+`scheduler` |
| Yolov8 | Segmentation | - | 0.688 | `v2` | 100 | RGB | - | - |

# Bounding box (12 classes)
| Model | Precision | Recall | Dataset | Epochs |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Yolov5 | ~0.8 | ~0.8 | single annotations | 300 | 
| Yolov5 | 0.793 | 0.66 | box per motif | 300 | 

