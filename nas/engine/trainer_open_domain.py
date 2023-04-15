import logging
import time
from nas.utils.search_space import init_darts_folder, init_seg_folder
from nas.search_spaces.build import build_search_space
from nas.search_strategies.build import build_search_strategy
import numpy as np
import os
import pickle


def main_process(args, cfg):
    if cfg.SEARCH_SPACE.TYPE == "DARTS":
        init_darts_folder(args.save_dir)
    elif cfg.SEARCH_SPACE.TYPE == "SEG101":
        init_seg_folder(args.save_dir, f"{cfg.SEARCH_SPACE.TYPE}_{cfg.SEARCH_STRATEGY.TYPE}")
    elif cfg.SEARCH_SPACE.TYPE == "SEG102":
        init_seg_folder(args.save_dir, f"{cfg.SEARCH_SPACE.TYPE}_{cfg.SEARCH_STRATEGY.TYPE}")
    else:
        raise NotImplemented(f"Search space {cfg.SEARCH_SPACE.TYPE} does not support at present!")
    search_space = build_search_space(cfg)
    search_space.args = args
    logging.getLogger("nas")

    algorithms = []
    results = []
    walltimes = []

    starttime = time.time()
    strategy = build_search_strategy(cfg=cfg,
                                     search_space=search_space)
    # parse parameters
    # do search
    data = strategy.do_search()
    # print the best architectures
    result, best_arch_key_dict = strategy.compute_best_test_losses(data)
    # save results
    algo_result = np.round(result, 5)
    # add walltime and results
    walltimes.append(time.time() - starttime)
    results.append(algo_result)
    algorithms.append(cfg.SEARCH_STRATEGY.TYPE)
    save_results(save_dir=args.save_dir,
                 results=results,
                 walltimes=walltimes,
                 best_arch_key_dict=best_arch_key_dict)


def save_results(save_dir, results, walltimes, best_arch_key_dict):
    logger = logging.getLogger("nas")
    filename = os.path.join(save_dir, 'result_best_archs.pkl')
    logger.info(' * Trial summary: (results, walltimes, best_arch_key_dict)')

    logger.info(' * Saving to file {}'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump([results, walltimes, best_arch_key_dict], f)