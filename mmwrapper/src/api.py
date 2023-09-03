from mmengine.runner import Runner
from mmengine import Config
from mmseg.apis import init_model, inference_model
from mmdet.apis import init_detector, inference_detector


import yaml

from .configs.configs import ConfigModifierRegistry


def get_runner(config_file):
    if isinstance(config_file, dict):
        settings = config_file
    else:
        with open(config_file) as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)

    cfg = ConfigModifierRegistry.config_modifiers[settings["model_name"]](settings)
    runner = Runner.from_cfg(cfg)
    return runner


class InferenceModel:
    def __init__(self, checkpoint_file, config_file, device="cuda"):
        self.is_detection = (
            Config.fromfile(config_file)["train_pipeline"][-1]["type"]
            == "PackDetInputs"
        )
        print(Config.fromfile(config_file)["train_pipeline"][-1])
        print(self.is_detection)
        if self.is_detection:
            self.model = init_detector(config_file, checkpoint_file, device=device)
            self.forward = inference_detector
        else:
            self.model = init_model(config_file, checkpoint_file, device=device)
            self.forward = inference_model

    def predict(self, imgs):
        return self.forward(self.model, imgs)
