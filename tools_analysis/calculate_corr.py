import os
import pickle
from nas.utils.correlation import get_kendalltau_correlation, get_spearmanr_correlation


def convert2dict(saved_datasets, eval_keys):
    with open(saved_datasets, "rb") as f:
        results = pickle.load(f)
    keys = list(results.keys())

    values_dict = {}
    for k in keys:
        result_k = results[k]
        temp_dict = {}
        break_flag = False
        for k_nested in eval_keys:
            if "eval_results" in k_nested:
                val = result_k[k_nested]["mIoU"]
                temp_dict[k_nested] = val
            else:
                if k_nested not in result_k:
                    break_flag = True
                    break
                else:
                    val = result_k[k_nested]
                    temp_dict[k_nested] = val

        if break_flag:
            continue

        for k in temp_dict:
            if k in values_dict:
                values_dict[k].append(temp_dict[k])
            else:
                values_dict[k] = [temp_dict[k]]
    return values_dict


def calculate_corr(results_dict, compare_val):
    gt = results_dict["eval_results"]
    for k in compare_val:
        kendall_val = get_kendalltau_correlation(gt, results_dict[k])[0]
        spearman_val = get_spearmanr_correlation(gt, results_dict[k])[0]
        print(f"{k}: {spearman_val}, {kendall_val}")


def combination_results(results_list, eval_keys):
    total_dict = {}
    for result in results_list:
        val_dict = convert2dict(result, eval_keys)
        for k, v in val_dict.items():
            if k in total_dict:
                total_dict[k].extend(v)
            else:
                total_dict[k] = v
    return total_dict


if __name__ == "__main__":
    pkl_dir_1 = "/home/albert_wei/fdisk_c/SEG101_Results/results_2022_4_19_13_27_01_mass_road_3_30_epochs.pkl"
    pkl_dir_2 = "/home/albert_wei/fdisk_c/SEG101_Results/results_2022_4_19_21_15_37_mass_road_4_30_epochs.pkl"
    pkl_dir_3 = "/home/albert_wei/fdisk_c/SEG101_Results/results_2022_4_19_17_25_32_mass_road_2_20_epochs.pkl"
    # pkl_dir = "/home/albert_wei/fdisk_c/SEG101_Results/results_2022_4_20_07_33_18.pkl"
    eval_keys = ['eval_results', 'grad_norm_seg', 'snip_seg', 'grasp_seg', 'fisher_seg', 'jacob_cov_seg', 'plain_seg', 'synflow_bn_seg']
    # results_dict = convert2dict(pkl_dir, eval_keys)
    # print(len(results_dict["eval_results"]))
    # calculate_corr(results_dict, eval_keys[1:])

    results_dict = combination_results([pkl_dir_1, pkl_dir_2, pkl_dir_3], eval_keys)

    calculate_corr(results_dict, eval_keys)
