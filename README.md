# MMWrapper
This repo is unofficial and completely unassociated with the openmmlab projects
Wrapper around the openmmlab framework

## Install

### Pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
pip install "mmsegmentation>=1.0.0"
pip install git+https://github.com/PhilippMarquardt/MMWrapper.git
