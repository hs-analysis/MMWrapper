from mmengine.config import Config, DictAction
import os
import datetime


class ConfigModifierRegistry:
    config_modifiers = {}

    @staticmethod
    def register(name):
        def decorator(func):
            ConfigModifierRegistry.config_modifiers[name] = func
            return func

        return decorator


def modify_config(cfg, dict_keys):
    # Iterate over the desired keys
    for key, value in dict_keys.items():
        key_parts = key.split(".")  # Split the nested key by '.'

        # Traverse the nested keys to access the desired key
        nested_dict = cfg
        for part in key_parts[:-1]:
            if part in nested_dict:
                nested_dict = nested_dict[part]
            else:
                # Key not found, create the nested key
                nested_dict[part] = {}
                nested_dict = nested_dict[part]

        # Modify the value of the key (or create the key if it doesn't exist)
        nested_dict[key_parts[-1]] = value

    return cfg


@ConfigModifierRegistry.register("resnet")
def modify_swin_unet(settings, cfg_path_end="resnet.py"):
    backend = settings["backend"]

    data_preprocessor = dict(
        mean=[123.675, 116.28, 103.53] if backend != "tifffile" else None,
        std=[123.675, 116.28, 103.53] if backend != "tifffile" else None,
        to_rgb=True if backend != "tifffile" else False,
    )

    train_dataloader = dict(
        pin_memory=False,
        persistent_workers=False,
        collate_fn=dict(type="default_collate"),
        batch_size=2,
        num_workers=0,
        dataset=dict(
            type="CustomDataset",
            data_prefix=settings["dataroot"] + "/train",
            with_label=True,
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
                dict(type="RandomFlip", prob=0.5, direction="horizontal"),
                dict(type="PackInputs"),
            ],
        ),
        sampler=dict(type="DefaultSampler", shuffle=True),
    )
    val_dataloader = dict(
        pin_memory=False,
        persistent_workers=False,
        collate_fn=dict(type="default_collate"),
        batch_size=2,
        num_workers=0,
        dataset=dict(
            type="CustomDataset",
            data_prefix=settings["dataroot"] + "/valid",
            with_label=True,
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
                dict(type="PackInputs"),
            ],
        ),
        sampler=dict(type="DefaultSampler", shuffle=True),
    )
    val_evaluator = dict(type="Accuracy", topk=(1, 1))
    test_dataloader = dict(
        pin_memory=False,
        persistent_workers=False,
        collate_fn=dict(type="default_collate"),
        batch_size=2,
        num_workers=0,
        dataset=dict(
            type="CustomDataset",
            data_prefix=settings["dataroot"] +  "/valid",
            with_label=True,
            pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
                dict(type="PackInputs"),
            ],
        ),
        sampler=dict(type="DefaultSampler", shuffle=True),
    )
    test_evaluator = dict(type="Accuracy", topk=(1, 1))

    vis_backends = [
        dict(type="LocalVisBackend"),  #
        dict(type="TensorboardVisBackend"),
    ]
    visualizer = dict(
        type="UniversalVisualizer",
        vis_backends=vis_backends,
        name="visualizer-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    swin_dict = {
        "data_preprocessor": data_preprocessor,
        "vis_backends": vis_backends,
        "visualizer": visualizer,
        "work_dir": settings["work_dir"],
        "load_from": settings["load_from"],
        # "log_level": "ERROR",
        # image_size
        "num_classes": settings["num_classes"],
        "data_root": settings["dataroot"],
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "val_evaluator": val_evaluator,
        "test_dataloader": test_dataloader,
        "test_evaluator": test_evaluator,
        "train_cfg": dict(
            type="EpochBasedTrainLoop",
            max_epochs=settings["num_epochs"],
            val_interval=settings["val_interval"],
        ),
        "model.backbone.in_channels": settings["in_channels"],
        "model.backbone.init_cfg": None,
        "model.head.num_classes": settings["num_classes"],
        "param_scheduler": dict(
            type="CosineAnnealingLR",
            by_epoch=True,
            T_max=settings["num_epochs"],
            convert_to_iter_based=True,
        ),
        "log_processor": dict(type="LogProcessor", window_size=50, by_epoch=True),
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "interval": settings["checkpoint_interval"],
                "by_epoch": True,
                "save_last": True,
                "max_keep_ckpts": 1,
                "save_best": "auto",
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "VisualizationHook"},
        },
        # batch augment
        "log_level": "ERROR",
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = Config.fromfile(os.path.join(current_dir, f"classification/{cfg_path_end}"))
    cfg = modify_config(cfg, swin_dict)
    return cfg


@ConfigModifierRegistry.register("swin_unet")
def modify_swin_unet(cfg, cfg_path_end="swin_unet.py"):
    return modfiy_swin_upper_segmentation(cfg, cfg_path_end)


@ConfigModifierRegistry.register("swin_s_upper")
def modfiy_swin_upper_segmentation(settings, cfg_path_end="swin_s_upper.py"):
    assert settings["batch_size"] >= 2, "batch size must be >= 2"
    backend = settings["backend"]
    train_pipeline = [
        dict(type="LoadImageFromFile", imdecode_backend=backend),
        dict(type="LoadAnnotations", reduce_zero_label=False),
        dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
        dict(type="RandomFlip", prob=0.5),
        # dict(type="PhotoMetricDistortion"),
        dict(type="PackSegInputs"),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile", imdecode_backend=backend),
        dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
        dict(type="LoadAnnotations", reduce_zero_label=False),
        dict(type="PackSegInputs"),
    ]

    data_preprocessor = dict(
        type="SegDataPreProcessor",
        mean=None if backend == "tifffile" else [123.675, 116.28, 103.53],
        std=None if backend == "tifffile" else [58.395, 57.12, 57.375],
        bgr_to_rgb=True if backend == "cv2" else False,
        pad_val=0,
        seg_pad_val=255,
        size=settings["image_size"],
    )
    vis_backends = [
        dict(type="LocalVisBackend"),  #
        dict(type="TensorboardVisBackend"),
    ]
    visualizer = dict(
        type="SegLocalVisualizer",
        vis_backends=vis_backends,
        name="visualizer-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tta_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None, imdecode_backend=backend),
        dict(
            type="TestTimeAug",
            transforms=[
                [
                    {"type": "Resize", "scale_factor": 0.5, "keep_ratio": True},
                    {"type": "Resize", "scale_factor": 0.75, "keep_ratio": True},
                    {"type": "Resize", "scale_factor": 1.0, "keep_ratio": True},
                    {"type": "Resize", "scale_factor": 1.25, "keep_ratio": True},
                    {"type": "Resize", "scale_factor": 1.5, "keep_ratio": True},
                    {"type": "Resize", "scale_factor": 1.75, "keep_ratio": True},
                ],
                [
                    {"type": "RandomFlip", "prob": 0.0, "direction": "horizontal"},
                    {"type": "RandomFlip", "prob": 1.0, "direction": "horizontal"},
                ],
                [{"type": "LoadAnnotations"}],
                [{"type": "PackSegInputs"}],
            ],
        ),
    ]

    swin_dict = {
        "vis_backends": vis_backends,
        "visualizer": visualizer,
        "work_dir": settings["work_dir"],
        "load_from": settings["load_from"],
        "num_classes": settings["num_classes"],
        "data_root": settings["dataroot"],
        "log_level": "ERROR",
        # image_size
        "model.decode_head.num_classes": settings["num_classes"],
        "model.auxiliary_head.num_classes": settings["num_classes"],
        "train_dataloader.num_workers": settings["num_workers"],
        "train_dataloader.persistent_workers": settings["persistent_workers"],
        "train_dataloader.batch_size": settings["batch_size"],
        "train_dataloader.dataset.data_root": settings["dataroot"],
        "train_dataloader.dataset.data_prefix": settings["train_img_prefix"],
        "train_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "train_dataloader.dataset.reduce_zero_label": False,
        "train_dataloader.sampler": dict(type="DefaultSampler", shuffle=True),
        "train_dataloader.drop_last": True,
        "train_dataloader.dataset.img_suffix": ".tif"
        if backend == "tifffile"
        else ".png",
        "val_dataloader.num_workers": settings["num_workers"],
        "val_dataloader.persistent_workers": settings["persistent_workers"],
        "val_dataloader.batch_size": 1,
        "val_dataloader.dataset.data_root": settings["dataroot"],
        "val_dataloader.dataset.data_prefix": settings["val_img_prefix"],
        "val_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "val_dataloader.dataset.img_suffix": ".tif"
        if backend == "tifffile"
        else ".png",
        "test_dataloader.num_workers": settings["num_workers"],
        "test_dataloader.persistent_workers": settings["persistent_workers"],
        "test_dataloader.batch_size": 1,
        "test_dataloader.dataset.data_root": settings["dataroot"],
        "test_dataloader.dataset.data_prefix": settings["test_img_prefix"],
        "test_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "test_dataloader.dataset.img_suffix": ".tif"
        if backend == "tifffile"
        else ".png",
        "tta_pipeline": tta_pipeline,
        "train_cfg": dict(
            type="EpochBasedTrainLoop",
            max_epochs=settings["num_epochs"],
            val_interval=1,
        ),
        "model.backbone.in_channels": settings["in_channels"],
        "model.backbone.init_cfg": dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth",
        )
        if settings["pretrained"]
        else None,
        "model.data_preprocessor": data_preprocessor,
        "param_scheduler": dict(
            type="CosineAnnealingLR",
            by_epoch=True,
            T_max=settings["num_epochs"],
            convert_to_iter_based=True,
            begin=settings["num_epochs"] // 2,
        ),
        "log_processor": dict(type="LogProcessor", window_size=50, by_epoch=True),
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "interval": settings["checkpoint_interval"],
                "save_best": "mIoU",
                "max_keep_ckpts": 1,
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "SegVisualizationHook"},
        },
        # edit train pipeline
        "train_dataloader.dataset.pipeline": train_pipeline,
        "val_dataloader.dataset.pipeline": test_pipeline,
        "test_dataloader.dataset.pipeline": test_pipeline,
        # batch augment
        "max_epochs": settings["num_epochs"],
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = Config.fromfile(os.path.join(current_dir, f"segmentation/{cfg_path_end}"))
    cfg = modify_config(cfg, swin_dict)
    return cfg


@ConfigModifierRegistry.register("mask2former_swin_t_seg")
def modify_mask2former_config(settings, cfg_path_end="mask2former_swin_t.py"):
    return modfiy_mask2former_r50_config(settings, cfg_path_end)


@ConfigModifierRegistry.register("mask2former_r50")
def modfiy_mask2former_r50_config(settings, cfg_path_end="mask2former_r50.py"):
    assert settings["batch_size"] >= 2, "batch size must be >= 2"
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", reduce_zero_label=False),
        dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
        dict(type="RandomFlip", prob=0.5),
        # dict(type="PhotoMetricDistortion"),
        dict(type="PackSegInputs"),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
        dict(type="LoadAnnotations", reduce_zero_label=False),
        dict(type="PackSegInputs"),
    ]

    data_preprocessor = dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=settings["image_size"],
        test_cfg=dict(size_divisor=32),
    )
    vis_backends = [
        dict(type="LocalVisBackend"),  #
        dict(type="TensorboardVisBackend"),
    ]
    visualizer = dict(
        type="SegLocalVisualizer",
        vis_backends=vis_backends,
        name="visualizer-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    swin_dict = {
        "vis_backends": vis_backends,
        "visualizer": visualizer,
        "work_dir": settings["work_dir"],
        "load_from": settings["load_from"],
        "num_classes": settings["num_classes"],
        "data_root": settings["dataroot"],
        "log_level": "ERROR",
        # image_size
        # "model.decode_head.num_classes": settings["num_classes"],
        # "model.auxiliary_head.num_classes": settings["num_classes"],
        "model.decode_head.loss_cls.class_weight": [1.0] * settings["num_classes"]
        + [0.1],
        "model.decode_head.num_classes": settings["num_classes"],
        "train_dataloader.num_workers": settings["num_workers"],
        "train_dataloader.persistent_workers": settings["persistent_workers"],
        "train_dataloader.batch_size": settings["batch_size"],
        "train_dataloader.dataset.data_root": settings["dataroot"],
        "train_dataloader.dataset.data_prefix": settings["train_img_prefix"],
        "train_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "train_dataloader.dataset.reduce_zero_label": False,
        "train_dataloader.sampler": dict(type="DefaultSampler", shuffle=True),
        "train_dataloader.drop_last": True,
        "val_dataloader.num_workers": settings["num_workers"],
        "val_dataloader.persistent_workers": settings["persistent_workers"],
        "val_dataloader.batch_size": 1,
        "val_dataloader.dataset.data_root": settings["dataroot"],
        "val_dataloader.dataset.data_prefix": settings["val_img_prefix"],
        "val_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "test_dataloader.num_workers": settings["num_workers"],
        "test_dataloader.persistent_workers": settings["persistent_workers"],
        "test_dataloader.batch_size": 1,
        "test_dataloader.dataset.data_root": settings["dataroot"],
        "test_dataloader.dataset.data_prefix": settings["test_img_prefix"],
        "test_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "train_cfg": dict(
            type="EpochBasedTrainLoop",
            max_epochs=settings["num_epochs"],
            val_interval=1,
        ),
        "model.backbone.in_channels": settings["in_channels"],
        "model.data_preprocessor": data_preprocessor,
        "param_scheduler": dict(
            type="CosineAnnealingLR",
            by_epoch=True,
            T_max=settings["num_epochs"],
            convert_to_iter_based=True,
            begin=settings["num_epochs"] // 2,
        ),
        "log_processor": dict(type="LogProcessor", window_size=50, by_epoch=True),
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "interval": settings["checkpoint_interval"],
                "save_best": "mIoU",
                "max_keep_ckpts": 1,
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "SegVisualizationHook"},
        },
        # edit train pipeline
        "train_dataloader.dataset.pipeline": train_pipeline,
        "val_dataloader.dataset.pipeline": test_pipeline,
        "test_dataloader.dataset.pipeline": test_pipeline,
        # batch augment
        "max_epochs": settings["num_epochs"],
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = Config.fromfile(os.path.join(current_dir, f"segmentation/{cfg_path_end}"))
    cfg = modify_config(cfg, swin_dict)
    return cfg


@ConfigModifierRegistry.register("dino_r50")
def modify_dino_config(settings, cfg_path_end="dino_r50.py"):
    backend = settings["backend"]

    init_cfg = None  # TODO: add init_cfg
    data_preprocessor = dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53] if backend != "tifffile" else None,
        std=[123.675, 116.28, 103.53] if backend != "tifffile" else None,
        bgr_to_rgb=True if backend != "tifffile" else False,
        pad_size_divisor=1,
    )

    train_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None, imdecode_backend=backend),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="RandomFlip", prob=0.5),
        dict(
            type="RandomChoice",
            transforms=[
                [
                    {
                        "type": "RandomChoiceResize",
                        "scales": [
                            (480, 1333),
                            (512, 1333),
                            (544, 1333),
                            (576, 1333),
                            (608, 1333),
                            (640, 1333),
                            (672, 1333),
                            (704, 1333),
                            (736, 1333),
                            (768, 1333),
                            (800, 1333),
                        ],
                        "keep_ratio": True,
                    }
                ],
                [
                    {
                        "type": "RandomChoiceResize",
                        "scales": [(400, 4200), (500, 4200), (600, 4200)],
                        "keep_ratio": True,
                    },
                    {
                        "type": "RandomCrop",
                        "crop_type": "absolute_range",
                        "crop_size": (384, 600),
                        "allow_negative_crop": True,
                    },
                    {
                        "type": "RandomChoiceResize",
                        "scales": [
                            (480, 1333),
                            (512, 1333),
                            (544, 1333),
                            (576, 1333),
                            (608, 1333),
                            (640, 1333),
                            (672, 1333),
                            (704, 1333),
                            (736, 1333),
                            (768, 1333),
                            (800, 1333),
                        ],
                        "keep_ratio": True,
                    },
                ],
            ],
        ),
        dict(type="PackDetInputs"),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile", backend_args=None, imdecode_backend=backend),
        dict(type="Resize", scale=(1333, 800), keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    vis_backends = [
        dict(type="LocalVisBackend"),  #
        dict(type="TensorboardVisBackend"),
    ]
    visualizer = dict(
        type="DetLocalVisualizer",
        vis_backends=vis_backends,
        name="visualizer-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    swin_dict = {
        "vis_backends": vis_backends,
        "visualizer": visualizer,
        "work_dir": settings["work_dir"],
        "load_from": settings["load_from"],
        "num_classes": settings["num_classes"],
        "data_root": settings["dataroot"],
        "log_level": "ERROR",
        # image_size
        "model.data_preprocessor": data_preprocessor,
        "model.bbox_head.num_classes": settings["num_classes"],
        "train_dataloader.num_workers": settings["num_workers"],
        "train_dataloader.persistent_workers": settings["persistent_workers"],
        "train_dataloader.batch_size": settings["batch_size"],
        "train_dataloader.dataset.data_root": settings["dataroot"],
        "train_dataloader.dataset.ann_file": settings["train_ann_file"],
        "train_dataloader.dataset.data_prefix": settings["train_img_prefix"],
        "train_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "val_dataloader.num_workers": settings["num_workers"],
        "val_dataloader.persistent_workers": settings["persistent_workers"],
        "val_dataloader.batch_size": settings["batch_size"],
        "val_dataloader.dataset.data_root": settings["dataroot"],
        "val_dataloader.dataset.ann_file": settings["val_ann_file"],
        "val_dataloader.dataset.data_prefix": settings["val_img_prefix"],
        "val_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "test_dataloader.num_workers": settings["num_workers"],
        "test_dataloader.persistent_workers": settings["persistent_workers"],
        "test_dataloader.batch_size": settings["batch_size"],
        "test_dataloader.dataset.data_root": settings["dataroot"],
        "test_dataloader.dataset.ann_file": settings["test_ann_file"],
        "test_dataloader.dataset.data_prefix": settings["test_img_prefix"],
        "test_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "val_evaluator.ann_file": os.path.join(
            settings["dataroot"], settings["val_ann_file"]
        ),
        "test_evaluator.ann_file": os.path.join(
            settings["dataroot"], settings["test_ann_file"]
        ),
        "train_cfg": dict(
            type="EpochBasedTrainLoop",
            max_epochs=settings["num_epochs"],
            val_interval=1,
        ),
        "model.backbone.init_cfg": init_cfg,
        "model.backbone.in_channels": settings["in_channels"],
        "param_scheduler": dict(
            type="CosineAnnealingLR",
            by_epoch=True,
            T_max=settings["num_epochs"],
            convert_to_iter_based=True,
        ),
        "log_processor": dict(type="LogProcessor", window_size=50, by_epoch=True),
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "interval": settings["checkpoint_interval"],
                "save_best": "auto",
                "max_keep_ckpts": 1,
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "DetVisualizationHook"},
        },
        # edit train pipeline
        "train_dataloader.dataset.pipeline": train_pipeline,
        "val_dataloader.dataset.pipeline": test_pipeline,
        "test_dataloader.dataset.pipeline": test_pipeline,
        # batch augment
        "max_epochs": settings["num_epochs"],
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = Config.fromfile(os.path.join(current_dir, f"object_detection/{cfg_path_end}"))
    cfg = modify_config(cfg, swin_dict)
    return cfg


@ConfigModifierRegistry.register("dino_r50_custom_pipeline")
def modify_dino_config(settings, cfg_path_end="dino_r50.py"):
    train_pipeline = [
        dict(type="LoadImageFromFile", to_float32=True, backend_args=None),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="RandomFlip", prob=0.5),
        dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
        dict(type="PackDetInputs"),
    ]
    test_pipeline = [
        dict(type="LoadImageFromFile", to_float32=True, backend_args=None),
        dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]

    # train_pipeline = [
    #     dict(type="LoadImageFromFile", backend_args=None),
    #     dict(type="LoadAnnotations", with_bbox=True),
    #     dict(type="RandomFlip", prob=0.5),
    #     dict(
    #         type="RandomChoice",
    #         transforms=[
    #             [
    #                 {
    #                     "type": "RandomChoiceResize",
    #                     "scales": [
    #                         (480, 1333),
    #                         (512, 1333),
    #                         (544, 1333),
    #                         (576, 1333),
    #                         (608, 1333),
    #                         (640, 1333),
    #                         (672, 1333),
    #                         (704, 1333),
    #                         (736, 1333),
    #                         (768, 1333),
    #                         (800, 1333),
    #                     ],
    #                     "keep_ratio": True,
    #                 }
    #             ],
    #             [
    #                 {
    #                     "type": "RandomChoiceResize",
    #                     "scales": [(400, 4200), (500, 4200), (600, 4200)],
    #                     "keep_ratio": True,
    #                 },
    #                 {
    #                     "type": "RandomCrop",
    #                     "crop_type": "absolute_range",
    #                     "crop_size": (384, 600),
    #                     "allow_negative_crop": True,
    #                 },
    #                 {
    #                     "type": "RandomChoiceResize",
    #                     "scales": [
    #                         (480, 1333),
    #                         (512, 1333),
    #                         (544, 1333),
    #                         (576, 1333),
    #                         (608, 1333),
    #                         (640, 1333),
    #                         (672, 1333),
    #                         (704, 1333),
    #                         (736, 1333),
    #                         (768, 1333),
    #                         (800, 1333),
    #                     ],
    #                     "keep_ratio": True,
    #                 },
    #             ],
    #         ],
    #     ),
    #     dict(type="PackDetInputs"),
    # ]
    # test_pipeline = [
    #     dict(type="LoadImageFromFile", backend_args=None),
    #     dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    #     dict(type="LoadAnnotations", with_bbox=True),
    #     dict(
    #         type="PackDetInputs",
    #         meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    #     ),
    # ]
    vis_backends = [
        dict(type="LocalVisBackend"),  #
        dict(type="TensorboardVisBackend"),
    ]
    visualizer = dict(
        type="DetLocalVisualizer",
        vis_backends=vis_backends,
        name="visualizer-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    swin_dict = {
        "vis_backends": vis_backends,
        "visualizer": visualizer,
        "work_dir": settings["work_dir"],
        "load_from": settings["load_from"],
        "num_classes": settings["num_classes"],
        "data_root": settings["dataroot"],
        "log_level": "ERROR",
        # image_size
        "model.bbox_head.num_classes": settings["num_classes"],
        "train_dataloader.num_workers": settings["num_workers"],
        "train_dataloader.persistent_workers": settings["persistent_workers"],
        "train_dataloader.batch_size": settings["batch_size"],
        "train_dataloader.dataset.data_root": settings["dataroot"],
        "train_dataloader.dataset.ann_file": settings["train_ann_file"],
        "train_dataloader.dataset.data_prefix": settings["train_img_prefix"],
        "train_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "val_dataloader.num_workers": settings["num_workers"],
        "val_dataloader.persistent_workers": settings["persistent_workers"],
        "val_dataloader.batch_size": settings["batch_size"],
        "val_dataloader.dataset.data_root": settings["dataroot"],
        "val_dataloader.dataset.ann_file": settings["val_ann_file"],
        "val_dataloader.dataset.data_prefix": settings["val_img_prefix"],
        "val_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "test_dataloader.num_workers": settings["num_workers"],
        "test_dataloader.persistent_workers": settings["persistent_workers"],
        "test_dataloader.batch_size": settings["batch_size"],
        "test_dataloader.dataset.data_root": settings["dataroot"],
        "test_dataloader.dataset.ann_file": settings["test_ann_file"],
        "test_dataloader.dataset.data_prefix": settings["test_img_prefix"],
        "test_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "val_evaluator.ann_file": os.path.join(
            settings["dataroot"], settings["val_ann_file"]
        ),
        "test_evaluator.ann_file": os.path.join(
            settings["dataroot"], settings["test_ann_file"]
        ),
        "train_cfg": dict(
            type="EpochBasedTrainLoop",
            max_epochs=settings["num_epochs"],
            val_interval=1,
        ),
        "model.backbone.in_channels": settings["in_channels"],
        # "param_scheduler": dict(
        #     type="CosineAnnealingLR",
        #     by_epoch=True,
        #     T_max=settings["num_epochs"],
        #     convert_to_iter_based=True,
        # ),
        "log_processor": dict(type="LogProcessor", window_size=50, by_epoch=True),
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "interval": settings["checkpoint_interval"],
                "save_best": "auto",
                "max_keep_ckpts": 1,
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "DetVisualizationHook"},
        },
        # edit train pipeline
        "train_dataloader.dataset.pipeline": train_pipeline,
        "val_dataloader.dataset.pipeline": test_pipeline,
        "test_dataloader.dataset.pipeline": test_pipeline,
        # batch augment
        "max_epochs": settings["num_epochs"],
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = Config.fromfile(os.path.join(current_dir, f"object_detection/{cfg_path_end}"))
    cfg = modify_config(cfg, swin_dict)
    return cfg


@ConfigModifierRegistry.register("mask2former_swin_t")
def modify_mask2former_config(settings, cfg_path_end="swin_t.py"):
    return modify_mask2former_config(settings, cfg_path_end)


@ConfigModifierRegistry.register("mask2former_swin_s")
def modify_mask2former_config(settings, cfg_path_end="swin_s.py"):
    return modify_mask2former_config(settings, cfg_path_end)


@ConfigModifierRegistry.register("mask2former_resnet")
def modify_mask2former_config(settings, cfg_path_end="swin_resnet.py"):
    return modify_mask2former_config(settings, cfg_path_end)


@ConfigModifierRegistry.register("mask2former_swin_l")
def modify_mask2former_config(settings, cfg_path_end="swin_l.py"):

    backend = settings["backend"]

    data_preprocessor = dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53] if backend != "tifffile" else None,
        std=[123.675, 116.28, 103.53] if backend != "tifffile" else None,
        bgr_to_rgb=True if backend != "tifffile" else False,
        pad_size_divisor=1,
    )

    train_pipeline = [
        dict(
            type="LoadImageFromFile",
            to_float32=True,
            backend_args=None,
            imdecode_backend=backend,
        ),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(type="RandomFlip", prob=0.5),
        dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
        dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-05, 1e-05), by_mask=True),
        dict(type="PackDetInputs"),
    ]
    test_pipeline = [
        dict(
            type="LoadImageFromFile",
            to_float32=True,
            backend_args=None,
            imdecode_backend=backend,
        ),
        dict(type="Resize", scale=settings["image_size"], keep_ratio=False),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(
            type="PackDetInputs",
            meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
        ),
    ]
    batch_augments = [
        dict(
            type="BatchFixedSizePad",
            size=settings["image_size"],
            img_pad_value=0,
            pad_mask=True,
            mask_pad_value=0,
            pad_seg=False,
        )
    ]
    vis_backends = [
        dict(type="LocalVisBackend"),  #
        dict(type="TensorboardVisBackend"),
    ]
    visualizer = dict(
        type="DetLocalVisualizer",
        vis_backends=vis_backends,
        name="visualizer-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    swin_dict = {
        "vis_backends": vis_backends,
        "visualizer": visualizer,
        "work_dir": settings["work_dir"],
        "load_from": settings["load_from"],
        # "log_level": "ERROR",
        # image_size
        "num_classes": settings["num_classes"],
        "data_root": settings["dataroot"],
        "model.panoptic_fusion_head.num_things_classes": settings["num_classes"],
        "model.panoptic_head.num_things_classes": settings["num_classes"],
        "num_things_classes": settings["num_classes"],
        "model.panoptic_head.loss_cls.class_weight": [1.0] * settings["num_classes"]
        + [0.1],
        "model.panoptic_fusion_head.num_stuff_classes": 0,
        "model.panoptic_head.num_stuff_classes": 0,
        "model.data_preprocessor": data_preprocessor,
        "num_stuff_classes": 0,
        "train_dataloader.num_workers": settings["num_workers"],
        "train_dataloader.persistent_workers": settings["persistent_workers"],
        "train_dataloader.batch_size": settings["batch_size"],
        "train_dataloader.dataset.data_root": settings["dataroot"],
        "train_dataloader.dataset.ann_file": settings["train_ann_file"],
        "train_dataloader.dataset.data_prefix": settings["train_img_prefix"],
        "train_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "val_dataloader.num_workers": settings["num_workers"],
        "val_dataloader.persistent_workers": settings["persistent_workers"],
        "val_dataloader.batch_size": settings["batch_size"],
        "val_dataloader.dataset.data_root": settings["dataroot"],
        "val_dataloader.dataset.ann_file": settings["val_ann_file"],
        "val_dataloader.dataset.data_prefix": settings["val_img_prefix"],
        "val_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "test_dataloader.num_workers": settings["num_workers"],
        "test_dataloader.persistent_workers": settings["persistent_workers"],
        "test_dataloader.batch_size": settings["batch_size"],
        "test_dataloader.dataset.data_root": settings["dataroot"],
        "test_dataloader.dataset.ann_file": settings["test_ann_file"],
        "test_dataloader.dataset.data_prefix": settings["test_img_prefix"],
        "test_dataloader.dataset.metainfo": dict(classes=settings["classes"]),
        "val_evaluator.ann_file": os.path.join(
            settings["dataroot"], settings["val_ann_file"]
        ),
        "test_evaluator.ann_file": os.path.join(
            settings["dataroot"], settings["test_ann_file"]
        ),
        "train_cfg": dict(
            type="EpochBasedTrainLoop",
            max_epochs=settings["num_epochs"],
            val_interval=settings["val_interval"],
        ),
        "model.backbone.in_channels": settings["in_channels"],
        "model.backbone.init_cfg": None,
        "param_scheduler": dict(
            type="CosineAnnealingLR",
            by_epoch=True,
            T_max=settings["num_epochs"],
            convert_to_iter_based=True,
        ),
        "log_processor": dict(type="LogProcessor", window_size=50, by_epoch=True),
        "default_hooks": {
            "timer": {"type": "IterTimerHook"},
            "logger": {"type": "LoggerHook", "interval": 50},
            "param_scheduler": {"type": "ParamSchedulerHook"},
            "checkpoint": {
                "type": "CheckpointHook",
                "interval": settings["checkpoint_interval"],
                "by_epoch": True,
                "save_last": True,
                "max_keep_ckpts": 1,
                "save_best": "auto",
            },
            "sampler_seed": {"type": "DistSamplerSeedHook"},
            "visualization": {"type": "DetVisualizationHook"},
        },
        # edit train pipeline
        "train_dataloader.dataset.pipeline": train_pipeline,
        "val_dataloader.dataset.pipeline": test_pipeline,
        "test_dataloader.dataset.pipeline": test_pipeline,
        # batch augment
        "data_preprocessor.batch_augments": batch_augments,
        "model.data_preprocessor.batch_augments": batch_augments,
    }
    current_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = Config.fromfile(
        os.path.join(current_dir, f"instance_segmentation/{cfg_path_end}")
    )
    cfg = modify_config(cfg, swin_dict)
    return cfg
