from mmengine.runner import Runner
from mmengine import Config
from mmseg.apis import init_model, inference_model
from mmdet.apis import init_detector, inference_detector
from mmpretrain.apis import ImageClassificationInferencer
from mmpretrain.utils.setup_env import register_all_modules as mmpretrainreload
from mmdet.utils.setup_env import register_all_modules as mmdetreload

# from mmseg.utils.setup_env import register_all_modules as mmsegreload
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

        mmpretrainreload(True)
        mmdetreload(True)
        # mmsegreload(True)
        self.is_detection = (
            Config.fromfile(config_file)["train_pipeline"][-1]["type"]
            == "PackDetInputs"
        )

        self.is_classification = (
            Config.fromfile(config_file)["default_scope"] == "mmpretrain"
        )
        print(self.is_classification)
        print(Config.fromfile(config_file)["train_pipeline"][-1])
        print(self.is_detection)
        if self.is_classification:
            self.model = ImageClassificationInferencer(
                config_file, pretrained=checkpoint_file, device=device
            )
            self.forward = self.model
            return
        if self.is_detection:
            self.model = init_detector(config_file, checkpoint_file, device=device)
            self.forward = inference_detector
        else:
            self.model = init_model(config_file, checkpoint_file, device=device)
            self.forward = inference_model

    def predict(self, imgs):
        if self.is_classification:
            return self.model(imgs)
        return self.forward(self.model, imgs)
