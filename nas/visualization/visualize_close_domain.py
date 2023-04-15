import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

algo_mapping = {
    'random': 'Random',
    'evolution': 'REA',
    'bananas': 'BANANAS-PE',
    'bananas_f': 'BANANAS-AE',
    'gin_uncertainty_predictor': 'NPENAS-BO',
    'gin_predictor': 'NPENAS-NP',
    'gin_predictor_new': 'NPENAS-NP-FST',
    'oracle': 'NPENAS-GT',
    'BO w. GP': 'BO w. GP',
    'NASBOT': 'NASBOT'
}


def convert2np(root_path, model_lists, end=None):
    total_dicts = dict()
    for m in model_lists:
        total_dicts[m] = []
    files = os.listdir(root_path)
    if end:
        files = list(files)[:end]
    else:
        files = list(files)
    for f in files:
        if 'log' in f or 'full' in f or 'yaml' in f:
            continue
        file_path = os.path.join(root_path, f)
        nested_dicts = dict()
        for m in model_lists:
            nested_dicts[m] = []
        with open(file_path, 'rb') as nf:
            try:
                algo_names, results, walltimes = pickle.load(nf)
            except Exception as e:
                print(e)
                print(file_path)
            algorithm_results_dict = {}
            for idx, algo_info in enumerate(algo_names):
                algorithm_results_dict[model_lists[idx]] = results[idx]
            for i in range(len(results[0])):
                for idx, m in enumerate(model_lists):
                    # nested_dicts[m].append(results[idx][i][1])
                    nested_dicts[m].append(algorithm_results_dict[m][i][1])
            for m in model_lists:
                total_dicts[m].append(nested_dicts[m])
    results_np = {m: np.array(total_dicts[m]) for m in model_lists}
    return results_np


def convert2np_2(root_path, model_lists, end=None):
    total_dicts = dict()
    for m in model_lists:
        total_dicts[m] = []
    files = os.listdir(root_path)
    if end:
        files = list(files)[:end]
    else:
        files = list(files)
    for f in files:
        if 'log' in f or 'full' in f:
            continue
        file_path = os.path.join(root_path, f)
        nested_dicts = dict()
        for m in model_lists:
            nested_dicts[m] = []
        with open(file_path, 'rb') as nf:
            try:
                algorithm_params, metann_params, results, walltimes = pickle.load(nf)
            except Exception as e:
                print(e)
                print(file_path)
            for i in range(len(results[0])):
                for idx, m in enumerate(model_lists):
                    nested_dicts[m].append(results[idx][i][1])
            for m in model_lists:
                total_dicts[m].append(nested_dicts[m])
    results_np = {m: np.array(total_dicts[m]) for m in model_lists}
    return results_np


def getmean(results_np, model_lists, category='mean'):
    if category == 'mean':
        results_mean = {m: np.mean(results_np[m], axis=0) for m in model_lists}
    elif category == 'medium':
        results_mean = {m: np.median(results_np[m], axis=0) for m in model_lists}
    elif category == 'percentile':
        results_mean = {m: np.percentile(results_np[m], 50, axis=0) for m in model_lists}
    else:
        raise ValueError('this type operation is not supported!')
    return results_mean


def get_quantile(results_np, model_lists, divider=30):
    results_quantile = {m: np.percentile(results_np[m], divider, axis=0) for m in model_lists}
    return results_quantile


def get_bounder(total_mean, quantile_30, quantile_70, model_lists):
    bound_dict = dict()
    for m in model_lists:
        bound_dict[m] = np.stack([(total_mean[m]-quantile_30[m]),
                                  (quantile_70[m]-total_mean[m])], axis=0)
    return bound_dict


def draw_plot_nasbench_101(root_path, model_lists, model_masks, draw_type='ERRORBAR', verbose=1, order=True):
    # draw_type  ERRORBAR, MEANERROR
    if order:
        np_datas_dict = convert2np(root_path, model_lists=model_lists, end=None)
    else:
        np_datas_dict = convert2np_2(root_path, model_lists=model_lists, end=None)
    np_mean_dict = getmean(np_datas_dict, model_lists=model_lists)
    np_quantile_30 = get_quantile(np_datas_dict, model_lists=model_lists, divider=30)
    np_quantile_70 = get_quantile(np_datas_dict, model_lists=model_lists, divider=70)

    if verbose:
        for k, v in np_mean_dict.items():
            print(k)
            print('30 quantile value')
            print(np_quantile_30[k])
            print('mean')
            print(v)
            print('70 quantile value')
            print(np_quantile_70[k])
            print('###############')
    np_bounds = get_bounder(np_mean_dict, np_quantile_30, np_quantile_70, model_lists=model_lists)
    # get data mean
    idx = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    fig, ax = plt.subplots(1)
    if draw_type == 'ERRORBAR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.errorbar(idx, np_mean_dict[m], yerr=np_bounds[m], label=m, capsize=3, capthick=2)
    elif draw_type == 'MEANERROR':
        for j, m in enumerate(model_lists):
            if model_masks[j]:
                plt.plot(idx, np_mean_dict[m], label=m, marker='s', linewidth=1, ms=3)     # fmt='o',
    if draw_type == 'ERRORBAR':
        ax.set_yticks(np.arange(5.8, 7.4, 0.2))
    elif draw_type == 'MEANERROR':
        ax.set_yticks(np.arange(5.8, 7.2, 0.2))
    fig.set_dpi(600.0)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Test Error [%] of Best Neural Net')
    plt.legend(loc='upper right')
    # plt.grid(b=True, which='major', color='#666699', linestyle='--')
    plt.show()