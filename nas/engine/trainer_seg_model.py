import os
import time
import weakref
import logging
from nas.utils.logger import setup_logger_nas
from nas.utils.comm import seed_all_rng, save_config
from detectron2.evaluation.build import EVALUATOR_REGISTRY
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
import detectron2.data.transforms as T
from detectron2.engine.train_loop import SimpleTrainer, TrainerBase
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import hooks
from detectron2.engine.defaults import default_writers
from detectron2.modeling import build_model_nas
from detectron2.solver import build_optimizer
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset, print_csv_format, verify_results
from collections import OrderedDict
import pickle


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


class DefaultTrainer(TrainerBase):
    """
    Examples:
    ::
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
    """

    def __init__(self, cfg, logger_name, seg_model, device):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.logger = logging.getLogger(logger_name)
        # Assume these objects must be constructed in this order.
        self.model = model = self.build_model(cfg, logger_name, seg_model, device)
        self.optimizer = optimizer = self.build_optimizer(cfg, model)
        self.data_loader = data_loader = self.build_train_loader(cfg, logger_name)

        self._trainer = SimpleTrainer(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
            logger_name=logger_name
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.register_hooks(self.build_hooks(logger_name=logger_name))

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self, logger_name: str = None):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler()
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.

        ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model, logger_name=logger_name)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        # Here the default print/log frequency of each writer is used.
        # run writers in the end, so that evaluation metrics are written
        ret.append(hooks.PeriodicWriter(self.build_writers(logger_name=logger_name), period=20))
        return ret

    def build_writers(self, logger_name: str = None):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter, logger_name=logger_name)

    def train(self, logger_name:str=None):
        """
        Run training.
``
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter, logger_name=logger_name)
        assert hasattr(self, "_last_eval_results"), "No evaluation results obtained during training!"
        verify_results(self.cfg, self._last_eval_results)
        return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()

    @classmethod
    def build_model(cls, cfg, logger_name, seg_model, device):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model_nas(cfg, device, seg_model)
        logger = logging.getLogger(logger_name)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, logger_name=None):
        dataset_key_lists = ["cityscapes", "mass_road", "whu_building"]
        name = ""
        for key in dataset_key_lists:
            if key in cfg.DATASETS.TEST[0]:
                name = key
                break
        return EVALUATOR_REGISTRY.get(f"build_{name}_evaluator")(cfg, dataset_name, output_folder,
                                                                 logger_name=logger_name)

    @classmethod
    def build_train_loader(cls, cfg, logger_name):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg),
                                   logger_name=logger_name)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper, logger_name=logger_name)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, logger_name: str = None):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, logger_name=logger_name)

    @classmethod
    def test(cls, cfg, model, evaluators=None, logger_name: str = None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger(__name__)

        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name, logger_name=logger_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name, logger_name=logger_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, logger_name=logger_name)
            results[dataset_name] = results_i

            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i, logger_name=logger_name)
        if len(results) == 1:
            results = list(results.values())[0]
        return results


def train_seg_model(args, cfg_seg, seg_model, device, arch):
    save_dir = os.path.join(args.save_dir, arch.arch_key)
    logger_name = 'log_%s_gpu_%d' % (arch.arch_key[:10], device)

    cfg_seg.defrost()
    cfg_seg.OUTPUT_DIR = save_dir
    cfg_seg.freeze()

    logger = setup_logger_nas(output=os.path.join(save_dir, '%s.txt' % logger_name), name=logger_name)
    seed = int(str(time.time()).split('.')[0][::-1][:9])
    seed_all_rng(seed)
    logger.info(f'==========  Segmentation Model Key: {arch.arch_key},'
                f' GPU: {device}, Seed: {seed}==========')
    logger.info(f"=======Architecture information: {arch} =======")
    save_config(save_dir, cfg_seg, logger)

    trainer = DefaultTrainer(cfg_seg, logger_name, seg_model, device)
    trainer.resume_or_load(resume=True)
    results = trainer.train(logger_name=logger_name)
    results = results["sem_seg"]
    arch.miou = results["mIoU"]
    arch.iou_bg = results["IoU-bg"]
    if "mass_road" in cfg_seg.DATASETS.TRAIN[0]:
        arch.iou_target = results["IoU-road"]
    arch.eval_results = results

    model_file_name = os.path.join(save_dir, f"{arch.arch_key}.pkl")
    with open(model_file_name, "wb") as f:
        pickle.dump(arch, f)
    logger.info(f"Architecture {arch.arch_key} dumped to file {model_file_name}.")