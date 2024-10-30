# MMWrapper
This repo is unofficial and completely unassociated with the openmmlab projects

A wrapper around different OpenMMLab backends:
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [MMPretrain](https://github.com/open-mmlab/mmpretrain)

Using [MMEngine](https://github.com/open-mmlab/mmengine) as the training engine.

By using these different backends we have sophisticated and tested backend.

## Quick Start

### Installation
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
```

### Training
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

Note: All important paths that need to be changed are marked with #CHANGE THIS.

### Segmentation Config
```yaml
model_name: "swin_s_upper"
checkpoint_interval: 1
keep_checkpoints: 1
num_classes: 6  # CHANGE THIS
in_channels: 3
num_epochs: 5000
backend: "cv2"
image_size: !!python/tuple [1024, 1024]  # CHANGE THIS
val_interval: 1
resume: None
work_dir: "melanieswinsupper"
load_from: "https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024/mask2former_swin-t_8xb2-90k_cityscapes-512x1024_20221127_144501-36c59341.pth"
batch_size: 2
pretrained: True
persistent_workers: False
num_workers: 0
classes: !!python/tuple ["0", "1", "2", "3", "4", "5"]  # CHANGE THIS
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

## Model Configuration
The different configs are defined in:
https://github.com/hs-analysis/MMWrapper/blob/main/mmwrapper/src/configs/configs.py

This supports all architectures supported by the different backends. The only thing that needs to be added is the `<architecture>.py` file in for example https://github.com/hs-analysis/MMWrapper/tree/main/mmwrapper/src/configs/instance_segmentation for instance segmentation models.

## Inference Examples

### Instance Segmentation
```python
import os
import numpy as np
import cv2
from mmdet.apis import init_detector, inference_detector

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_file = "path/to/config.py"
checkpoint_file = "path/to/checkpoint.pth"
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)
colors = [(255,0,0), (125,0,0), (0,255,0), (50,255,0), (0,25,255), (0,125,0)]
thresh = 0.75

for file in os.listdir("path/to/images"):
    img_path = os.path.join("path/to/images", file)
    image = cv2.imread(img_path)
    result = inference_detector(model, image)
    
    masks = result.pred_instances.masks.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    
    for mask, score, label, bbox in zip(masks, scores, labels, bboxes):
        if score > thresh:
            color = colors[label]
            if label == 0:  # For label == 0, draw contours
                mask = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 1)
            
            x1, y1, x2, y2 = bbox.astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    
    cv2.imwrite(os.path.join("output", file), image)
```

### Object Detection
```python
import os
import numpy as np
import cv2
from mmdet.apis import init_detector, inference_detector

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config_file = "path/to/config.py"
checkpoint_file = "path/to/checkpoint.pth"
device = 'cuda:0'

model = init_detector(config_file, checkpoint_file, device=device)
colors = [(255,0,0), (125,0,0), (0,255,0), (50,255,0), (0,25,255), (0,125,0)]
thresh = 0.35

for file in os.listdir("path/to/images"):
    img_path = os.path.join("path/to/images", file)
    image = cv2.imread(img_path)
    result = inference_detector(model, image)
    
    scores = result.pred_instances.scores.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    
    for score, label, bbox in zip(scores, labels, bboxes):
        if score > thresh:
            color = colors[label]
            x1, y1, x2, y2 = bbox.astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    
    cv2.imwrite(os.path.join("output", file), image)
```

### Segmentation
```python
import os
import cv2
import torch
from mmseg.apis import init_model, inference_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config_path = "path/to/config.py"
checkpoint_path = "path/to/checkpoint.pth"
model = init_model(config_path, checkpoint_path)

img = cv2.imread("path/to/image.jpg")
result = inference_model(model, img)
logits = result.seg_logits.data
logits = torch.argmax(logits, dim=0).unsqueeze(0)
pred_mask = (result.pred_sem_seg.data.permute(1,2,0).cpu().numpy() + 0) * 50.
cv2.imwrite("output.png", pred_mask)
```

### Classification
```python
import os
import numpy as np
import cv2
from mmpretrain.apis import ImageClassificationInferencer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config_file = "path/to/config.py"
checkpoint_file = "path/to/checkpoint.pth"
device = 'cuda:0'

model = ImageClassificationInferencer(config_file, checkpoint_file, device=device)

for file in os.listdir("path/to/images"):
    img_path = os.path.join("path/to/images", file)
    result = model(img_path)
```
