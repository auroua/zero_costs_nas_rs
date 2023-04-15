import time
from torch.multiprocessing import Process
from detectron2.utils.env import seed_all_rng
from nas.utils.logger import setup_logger_nas
from nas.search_spaces.build import build_search_space
from nas.search_strategies.build import build_search_strategy
import numpy as np
import os
import pickle


def ansyc_multiple_process_search(args, cfgs):
    per_gpu_trials = int(args.trials/args.gpus)

    data_lists = [build_search_space(cfgs[0]) for _ in range(args.gpus)]

    p_consumers = [Process(target=data_consumers,
                           args=(args.save_dir, device, data_lists[device], cfgs, per_gpu_trials, args.gpus))
                   for device in range(args.gpus)]

    for p in p_consumers:
        p.start()

    for p in p_consumers:
        p.join()


def data_consumers(save_dir, device, search_space, cfg_list, per_gpu_trials, gpu_nums):
    file_name = 'log_%s_%d.txt' % ('gpus', device)
    logger = setup_logger_nas(output=os.path.join(save_dir, '%s.txt' % file_name), name=f"nas_gpu_{device}")
    seed = int(str(time.time()).split('.')[0][::-1][:9])
    seed_all_rng(seed)
    logger.info(f'==========  searching with gpu {device}, seed is {seed} ==========')

    for t in range(per_gpu_trials):
        trial = gpu_nums*t + device
        results, walltimes, algorithms = main(
            cfg_list=cfg_list,
            logger=logger,
            device=device,
            iteration=trial,
            search_space=search_space,
        )
        save_results(seed=seed,
                     save_dir=save_dir,
                     logger=logger,
                     trial=trial,
                     device=device,
                     results=results,
                     walltimes=walltimes,
                     algorithms=algorithms)


def main(cfg_list, logger, device, iteration, search_space):
    algorithms = []
    results = []
    walltimes = []
    for cfg in cfg_list:
        # load search strategy information
        logger.info(f'========  Begin neural architecture search. Iterations: {iteration}, '
                    f'Search Strategy: {cfg.SEARCH_STRATEGY.TYPE}, '
                    f'========')
        starttime = time.time()
        strategy = build_search_strategy(cfg=cfg,
                                         search_space=search_space,
                                         device=device)
        # parse parameters
        # do search
        data = strategy.do_search()
        # print the best architectures
        result = strategy.compute_best_test_losses(data)
        # save results
        algo_result = np.round(result, 5)
        # add walltime and results
        walltimes.append(time.time() - starttime)
        results.append(algo_result)
        algorithms.append(cfg.SEARCH_STRATEGY.TYPE)
    return results, walltimes, algorithms


def save_results(seed, save_dir, logger, trial, device, results, walltimes, algorithms):
    out_file = str(seed) + '_gpus_%d_' % device + 'iter_%d' % trial
    filename = os.path.join(save_dir, '{}.pkl'.format(out_file))
    logger.info(' * Trial summary: (results, walltimes)')
    for k in range(results[0].shape[0]):
        length = len(results)
        results_line = []
        for j in range(length):
            if j == 0:
                results_line.append(int(results[j][k, 0]))
                results_line.append(results[j][k, 1])
            else:
                results_line.append(results[j][k, 1])
        results_str = '  '.join([str(k) for k in results_line])
        logger.info(results_str)
    logger.info(walltimes)
    logger.info(' * Saving to file {}'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump([algorithms, results, walltimes], f)