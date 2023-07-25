from mmengine.runner import Runner
from .configs.configs import ConfigModifierRegistry
import yaml


def get_runner(config_file):
    # open configs file yaml
    with open(config_file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    cfg = ConfigModifierRegistry.config_modifiers[settings["model_name"]](settings)
    runner = Runner.from_cfg(cfg)
    return runner


def get_inference_model():
    pass
