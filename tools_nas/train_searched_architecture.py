#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from nas.config.defaults_seg import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.evaluation.build import EVALUATOR_REGISTRY
from detectron2.modeling import build_model_nas
import logging
import pickle
import os
from detectron2.utils.regnet_utils import REGNET_MODEL_PRETRAINED


def build_sem_seg_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if "mass_road" in cfg.DATASETS.TRAIN[0] or "whu_building" in cfg.DATASETS.TRAIN[0]:
        # augs.append(
        #     T.RandomBrightness(
        #         intensity_min=0.1,
        #         intensity_max=0.4
        #     )
        # )
        augs.append(
            T.NormalizeVal(
                factor=255
            )
        )
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that ca se you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        dataset_key_lists = ["cityscapes", "mass_road", "whu_building"]
        name = ""
        for key in dataset_key_lists:
            if key in cfg.DATASETS.TEST[0]:
                name = key
                break
        return EVALUATOR_REGISTRY.get(f"build_{name}_evaluator")(cfg, dataset_name, output_folder)


    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        arch_name = cfg.MODEL.SEARCHED_ARCHITECTURE.split("/")[-1]
        arch_file_name = os.path.join(cfg.MODEL.SEARCHED_ARCHITECTURE, f"{arch_name}.pkl")
        with open(arch_file_name, 'rb') as f:
            arch = pickle.load(f)
        seg_model = arch.assemble_neural_network(cfg)
        model = build_model_nas(cfg, cfg.MODEL.DEVICE, seg_model)
        logger = logging.getLogger("detectron2")
        logger.info("Model:\n{}".format(model))
        return model


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = os.path.expanduser(REGNET_MODEL_PRETRAINED[cfg.MODEL.REGNETS.TYPE])
    arch_name = cfg.MODEL.SEARCHED_ARCHITECTURE.split("/")[-1]
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, f"{cfg.DATASETS.TRAIN[0]}_{arch_name}_{cfg.MODEL.REGNETS.TYPE.replace('.', '_')}")
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        # --eval-only --resume
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
