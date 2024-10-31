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
# Configuration for instance segmentation using Mask R-CNN
model_name: "maskrcnn_r50"  # CHANGE THIS - Model architecture name. can be found in "https://github.com/hs-analysis/MMWrapper/blob/main/mmwrapper/src/configs/configs.py"

# Training parameters
checkpoint_interval: 1        # Save checkpoint every N epochs
keep_checkpoints: 1          # Number of checkpoints to keep
num_classes: 3               # Number of classes to detect/segment
in_channels: 3               # Number of input image channels (use with RGB images)
backend: "cv2"               # Image loading backend. Use "tifffile" for fluorescence images with multiple channels
num_epochs: 300             # CHANGE THIS - Total number of training epochs
image_size: !!python/tuple [512, 512]  # CHANGE THIS - Size to which all images will be resized

# Validation and checkpointing
val_interval: 1             # Run validation every N epochs
resume: None                # Path to checkpoint to resume training from. Supports resuming by providing .pth file of trained model
work_dir: "work_dir"        # CHANGE THIS - Directory where config.py, weights, and training stats will be saved

# Model initialization
load_from: "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
# Pre-trained weights path. Find model-specific weights in:
# https://github.com/open-mmlab/mmdetection/tree/main/configs
# Choose appropriate weights from the subfolder of your selected model

# Training configuration
batch_size: 2
pretrained: True
persistent_workers: False
num_workers: 0

# Class definitions
classes: !!python/tuple ["Drüsengewebe gesund", "Adenom", "Karzinom", "3", "4"]
# IMPORTANT: Class names must exactly match those defined in your COCO annotation file

# Dataset configuration
dataroot: "path/to/dataset"  # CHANGE THIS - Root directory of dataset
                            # If using HSA KIT export, just set this to the output folder

# Training dataset
train_ann_file: "train.json"  # COCO format annotation file
train_img_prefix:
  img: images/              # CHANGE THIS - Directory containing training images
  seg: annotations/panoptic_train2017/  # Directory containing segmentation masks

# Validation dataset
val_ann_file: "valid.json"
val_img_prefix:
  img: images/              # CHANGE THIS - Directory containing validation images
  seg: annotations/panoptic_val2017/

# Test dataset
test_ann_file: "valid.json"
test_img_prefix:
  img: images/              # CHANGE THIS - Directory containing test images
  seg: annotations/panoptic_val2017/
```

# Notes:
1. Only train_ann_file and img/ folder paths need to be changed for new datasets
2. If using HSA KIT export, setting dataroot to the HSA KIT output folder is sufficient
3. For fluorescence imaging with multiple channels, change backend to "tifffile"
4. Class names must match exactly with COCO annotation file
5. Pre-trained weights can be found in model-specific subfolders at:
    https://github.com/open-mmlab/mmdetection/tree/main/configs
6. Training can be resumed using resume parameter with path to existing .pth file

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

"""
Output format (result.pred_instances):
- masks: torch.Tensor, shape (N, H, W), bool type, N is number of instances
- scores: torch.Tensor, shape (N,), float type
- labels: torch.Tensor, shape (N,), int type
- bboxes: torch.Tensor, shape (N, 4), float type, format (x1, y1, x2, y2)

Example for single image:
{
    'pred_instances': InstanceData(
        'bboxes': torch.Tensor([[x1, y1, x2, y2], ...]), # shape (N, 4)
        'labels': torch.Tensor([0, 1, ...]),             # shape (N,)
        'scores': torch.Tensor([0.9, 0.8, ...]),         # shape (N,)
        'masks': torch.Tensor([[True, False, ...], ...])  # shape (N, H, W)
    )
}
"""

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
    
    masks = result.pred_instances.masks.cpu().numpy()     # shape (N, H, W)
    scores = result.pred_instances.scores.cpu().numpy()   # shape (N,)
    labels = result.pred_instances.labels.cpu().numpy()   # shape (N,)
    bboxes = result.pred_instances.bboxes.cpu().numpy()   # shape (N, 4)
    
    for mask, score, label, bbox in zip(masks, scores, labels, bboxes):
        if score > thresh:
            color = colors[label]
            if label == 0:  # For label == 0, draw contours
                mask = mask.astype(np.uint8)
                contours, * = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

"""
Output format (result.pred_instances):
- scores: torch.Tensor, shape (N,), float type
- labels: torch.Tensor, shape (N,), int type
- bboxes: torch.Tensor, shape (N, 4), float type, format (x1, y1, x2, y2)

Example for single image:
{
    'pred_instances': InstanceData(
        'bboxes': torch.Tensor([[x1, y1, x2, y2], ...]), # shape (N, 4)
        'labels': torch.Tensor([0, 1, ...]),             # shape (N,)
        'scores': torch.Tensor([0.9, 0.8, ...])          # shape (N,)
    )
}
"""

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
    
    scores = result.pred_instances.scores.cpu().numpy()   # shape (N,)
    labels = result.pred_instances.labels.cpu().numpy()   # shape (N,)
    bboxes = result.pred_instances.bboxes.cpu().numpy()   # shape (N, 4)
    
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

"""
Output format (result):
- seg_logits: torch.Tensor, shape (C, H, W), float type, raw logits
- pred_sem_seg: torch.Tensor, shape (1, H, W), int type, predicted class indices
where:
- C is number of classes
- H, W are height and width of input image

Example for single image:
{
    'seg_logits': torch.Tensor([[[0.1, 0.2...], ...]])   # shape (C, H, W)
    'pred_sem_seg': torch.Tensor([[[0, 1, ...]]])        # shape (1, H, W)
}
"""

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config_path = "path/to/config.py"
checkpoint_path = "path/to/checkpoint.pth"
model = init_model(config_path, checkpoint_path)
img = cv2.imread("path/to/image.jpg")
result = inference_model(model, img)
logits = result.seg_logits.data                          # shape (C, H, W)
logits = torch.argmax(logits, dim=0).unsqueeze(0)       # shape (1, H, W)
pred_mask = (result.pred_sem_seg.data.permute(1,2,0).cpu().numpy() + 0) * 50.  # shape (H, W, 1)
cv2.imwrite("output.png", pred_mask)
```

### Classification
```python
import os
import numpy as np
import cv2
from mmpretrain.apis import ImageClassificationInferencer

"""
Output format (list of dict, one per image):
[{
    'pred_scores': numpy.ndarray,  # shape (num_classes,), probabilities for each class
    'pred_label': int,            # single integer indicating predicted class
    'pred_score': float,          # confidence score for predicted class
    'pred_class': str             # class name string
}]

Example:
[{
    'pred_scores': array([0.1, 0.2, ...]),  # shape (num_classes,)
    'pred_label': 65,
    'pred_score': 0.6649367809295654,
    'pred_class': 'sea snake'
}]
"""

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config_file = "path/to/config.py"
checkpoint_file = "path/to/checkpoint.pth"
device = 'cuda:0'
model = ImageClassificationInferencer(config_file, checkpoint_file, device=device)
for file in os.listdir("path/to/images"):
    img_path = os.path.join("path/to/images", file)
    result = model(img_path)  # returns list of dicts, one per image
```
