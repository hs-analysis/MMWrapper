
# MMWrapper
This repo is unofficial and completely unassociated with the openmmlab projects
A wrapper around different OpenMMLab backends:
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MMPretrain](https://github.com/open-mmlab/mmpretrain)

Using [MMEngine](https://github.com/open-mmlab/mmengine) as the training engine.

## Quick Start
Install:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
pip install "mmsegmentation>=1.0.0"
pip install mmpretrain==1.0.0
pip install ftfy==6.2.0
pip install git+https://github.com/hs-analysis/MMWrapper.git

```python
from mmwrapper.src.api import get_runner
runner = get_runner("instance_segmentation.yaml")
runner.train()
```

## Configuration Examples

### Instance Segmentation Config
```yaml
model_name: "maskrcnn_r50"  # CHANGE THIS
checkpoint_interval: 1
keep_checkpoints: 1
num_classes: 3
in_channels: 3
backend: "cv2"
num_epochs: 300  # CHANGE THIS
image_size: !!python/tuple [512, 512]  # CHANGE THIS
val_interval: 1
resume: None
work_dir: "work_dir"  # CHANGE THIS
load_from: "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
batch_size: 2
pretrained: True
persistent_workers: False
num_workers: 0
classes: !!python/tuple ["Drüsengewebe gesund", "Adenom", "Karzinom", "3", "4"]
dataroot: "path/to/dataset"  # CHANGE THIS

train_ann_file: "train.json"
train_img_prefix:
  img: images/  # CHANGE THIS
  seg: annotations/panoptic_train2017/

val_ann_file: "valid.json"
val_img_prefix:
  img: images/  # CHANGE THIS
  seg: annotations/panoptic_val2017/

test_ann_file: "valid.json"
test_img_prefix:
  img: images/  # CHANGE THIS
  seg: annotations/panoptic_val2017/
```

### Segmentation Config
```yaml
model_name: "swin_s_upper"
num_classes: 6  # CHANGE THIS
image_size: !!python/tuple [1024, 1024]  # CHANGE THIS
work_dir: "melanieswinsupper"
classes: !!python/tuple ["0", "1", "2", "3", "4", "5"]  # CHANGE THIS
dataroot: "path/to/dataset"  # CHANGE THIS

train_img_prefix:
  img_path: train/images
  seg_map_path: train/labels

val_img_prefix:
  img_path: valid/images
  seg_map_path: valid/labels

test_img_prefix:
  img_path: valid/images
  seg_map_path: valid/labels
```

## Dataset Structure

### Object Detection/Instance Segmentation
Uses COCO format: [COCO Dataset](https://cocodataset.org/#home)

### Segmentation
```
root/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

### Classification
```
root/
├── Class1/
├── Class2/
└── Class3/
```

## Inference Examples

### Instance Segmentation
```python
import os
import numpy as np
import cv2
from mmdet.apis import init_detector, inference_detector

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_file = "path/to/config.py"
checkpoint_file = "path/to/checkpoint.pth"
device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)

colors = [(255,0,0), (125,0,0), (0,255,0), (50,255,0), (0,25,255), (0,125,0)]
thresh = 0.75

def process_image(image_path):
    image = cv2.imread(image_path)
    result = inference_detector(model, image)
    
    masks = result.pred_instances.masks.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    
    for mask, score, label, bbox in zip(masks, scores, labels, bboxes):
        if score > thresh:
            color = colors[label]
            if label == 0:
                mask = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 1)
            
            x1, y1, x2, y2 = bbox.astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    
    return image
```

### Segmentation
```python
import os
import cv2
import torch
from mmseg.apis import init_model, inference_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_path = "path/to/config.py"
checkpoint_path = "path/to/checkpoint.pth"
model = init_model(config_path, checkpoint_path)

def process_image(image_path):
    img = cv2.imread(image_path)
    result = inference_model(model, img)
    pred_mask = result.pred_sem_seg.data.permute(1,2,0).cpu().numpy() * 50
    return pred_mask
```

### Classification
```python
import os
from mmpretrain.apis import ImageClassificationInferencer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_file = "path/to/config.py"
checkpoint_file = "path/to/checkpoint.pth"
device = 'cuda:0'

inferencer = ImageClassificationInferencer(config_file, checkpoint_file, device=device)

def classify_image(image_path):
    return inferencer(image_path)
```
