# Copyright Xidian University.
import os
import torch
import detectron2.utils.comm as comm
from detectron2.utils.registry import Registry
from detectron2.data import MetadataCatalog
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator, RemoteEvaluator


EVALUATOR_REGISTRY = Registry("EVALUATOR")
EVALUATOR_REGISTRY.__doc__ = """
Registry for evaluators, which return a evaluators for a spectific dataset.

The registered object must be a callable that accepts two arguments:
"""


@EVALUATOR_REGISTRY.register()
def build_cityscapes_evaluator(cfg, dataset_name, output_folder=None, logger_name=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "sem_seg":
        return SemSegEvaluator(
            dataset_name,
            distributed=True,
            output_dir=output_folder,
            logger_name=logger_name
        )
    if evaluator_type == "cityscapes_sem_seg":
        assert (
                torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


@EVALUATOR_REGISTRY.register()
def build_mass_road_evaluator(cfg, dataset_name, output_folder, logger_name=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return RemoteEvaluator(
        dataset_name,
        distributed=True,
        output_dir=output_folder,
        logger_name=logger_name
    )


@EVALUATOR_REGISTRY.register()
def build_whu_building_evaluator(cfg, dataset_name, output_folder, logger_name=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    return RemoteEvaluator(
        dataset_name,
        distributed=True,
        output_dir=output_folder,
        logger_name=logger_name
    )