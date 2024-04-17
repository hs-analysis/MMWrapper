# MMWrapper
This repo is unofficial and completely unassociated with the openmmlab projects\

It is just a wrapper around the mmdetection and mmsegmentation

## Install

### Pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
pip install "mmsegmentation>=1.0.0"
pip install mmpretrain==1.0.0
pip install ftfy
pip install git+https://github.com/hs-analysis/MMWrapper.git
