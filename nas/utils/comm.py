# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


import random
import numpy as np
import copy
import torch.backends.cudnn as cudnn
import torch
import os
from hashlib import sha256
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from detectron2.config import CfgNode, LazyConfig
from detectron2.utils.env import seed_all_rng


def set_random_seed_with_cudnn(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def dynamic_gen_params(parames):
    algo_nums = parames[0]
    algo_params = parames[1]
    total_params = []
    for num in algo_nums:
        algo_params_temp = copy.deepcopy(algo_params)
        algo_params_temp['training_nums'] = num
        total_params.append(algo_params_temp)
    return total_params


def random_id(length):
    number = '0123456789'
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    id = ''
    for i in range(0, length, 2):
        id += random.choice(number)
        id += random.choice(alpha)
    return id


def _highlight(code, filename):
    try:
        import pygments
    except ImportError:
        return code

    from pygments.lexers import Python3Lexer, YamlLexer
    from pygments.formatters import Terminal256Formatter

    lexer = Python3Lexer() if filename.endswith(".py") else YamlLexer()
    code = pygments.highlight(code, lexer, Terminal256Formatter(style="monokai"))
    return code


def default_setup(cfgs, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode or omegaconf.DictConfig): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = args.save_dir
    if not PathManager.exists(output_dir):
        PathManager.mkdirs(output_dir)

    logger = setup_logger(output_dir, distributed_rank=0, name="nas")

    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_files") and args.config_files != "":
        for cfg_path in args.config_files:
            logger.info(
                "Contents of args.config_file={}:\n{}".format(
                    cfg_path,
                    _highlight(PathManager.open(cfg_path, "r").read(), cfg_path),
                )
            )

    if isinstance(cfgs, list):
        for idx, cfg in enumerate(cfgs):
            # Note: some of our scripts may expect the existence of
            # config.yaml in output directory
            path = os.path.join(output_dir, f"config_{cfg.SEARCH_STRATEGY.TYPE}.yaml")
            if isinstance(cfg, CfgNode):
                logger.info("Running with full config:\n{}".format(_highlight(cfg.dump(), ".yaml")))
                with PathManager.open(path, "w") as f:
                    f.write(cfg.dump())
            else:
                LazyConfig.save(cfg, path)
            logger.info("config file {} saved to {}".format(f"config_{cfg.SEARCH_STRATEGY.TYPE}.yaml", path))
        # make sure each worker has a different, yet deterministic seed if specified
        seed = cfgs[0].SEED
    else:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, f"config_{cfgs.SEARCH_STRATEGY.TYPE}.yaml")
        if isinstance(cfgs, CfgNode):
            logger.info("Running with full config:\n{}".format(_highlight(cfgs.dump(), ".yaml")))
            with PathManager.open(path, "w") as f:
                f.write(cfgs.dump())
        else:
            LazyConfig.save(cfgs, path)
        logger.info("config file {} saved to {}".format(f"config_{cfgs.SEARCH_STRATEGY.TYPE}.yaml", path))
        seed = cfgs.SEED

    seed_all_rng(None if seed < 0 else seed + 0)


def get_hashkey(op_list):
    return sha256(str(op_list).encode('utf-8')).hexdigest()


def save_config(save_dir, cfg_file, logger, verbose=False):
    path = os.path.join(save_dir, f"config_seg.yaml")
    if verbose:
        logger.info("Running with full config:\n{}".format(_highlight(cfg_file.dump(), ".yaml")))
    with PathManager.open(path, "w") as f:
        f.write(cfg_file.dump())
    logger.info("config file {} saved to {}".format(f"config_seg.yaml", path))


def queue_to_dict(queue):
    results_dict = {}
    q_len = queue.qsize()
    for idx in range(q_len):
        element = queue.get(idx)
        results_dict[element[0]] = element[1]
    return results_dict