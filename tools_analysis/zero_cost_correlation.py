import os
import pickle
from nas.config.defaults_seg import get_cfg
import torch
from tools_nas.train_searched_architecture import build_sem_seg_train_aug
from nas.zero_costs import measures
from nas.datasets.builder import build_dataloader
from detectron2.structures import ImageList
from detectron2.modeling.meta_arch.build import build_model_nas
from datetime import datetime
import gc


def parse_saved_model(model_dir, cfg):
    dirs = [os.path.join(model_dir, model) for model in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, model))]
    model_informantion_dict = {}
    for dir in dirs:
        arch_name = dir.split("/")[-1]
        arch_file_name = os.path.join(dir, f"{arch_name}.pkl")
        if not os.path.isfile(arch_file_name):
            continue
        with open(arch_file_name, 'rb') as f:
            arch = pickle.load(f)
            seg_model = arch.assemble_neural_network(cfg)
            eval_results_file = os.path.join(dir, "inference", "sem_seg_evaluation.pth")
            results = torch.load(eval_results_file)
            model_informantion_dict[arch_name] = {"arch": seg_model, "eval_results": results}
    return model_informantion_dict


def get_zero_costs(cfg, model_acc_dict, proxies, device):
    dataloader = build_dataloader(cfg, build_sem_seg_train_aug)
    dataloader_iter = iter(dataloader)
    keys = list(model_acc_dict.keys())
    for k in keys:
        print(f"############# {k} ##############")
        seg_model = model_acc_dict[k]["arch"]
        model = build_model_nas(cfg, device, seg_model)
        batched_inputs = next(dataloader_iter)
        del model_acc_dict[k]["arch"]
        for algo in proxies:
            try:
                val = measures.calc_measure(algo, model, device, batched_inputs, split_data=16)
            except Exception as e:
                del model
                torch.cuda.empty_cache()
                gc.collect()
                print(e)
                break
            model_acc_dict[k][algo] = val
    current_time = datetime.now()
    short_time = current_time.strftime("%H_%M_%S")
    with open(f"/home/albert_wei/fdisk_c/SEG101_Results/results_{current_time.year}_{current_time.month}_{current_time.day}_{short_time}.pkl", "wb") as f:
        pickle.dump(model_acc_dict, f)


def gen_cfg(config_file):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    return cfg


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # output_path = "/home/albert_wei/fdisk_c/SEG101_Results/npenas_seg101_mass_road_3/"
    # output_path = "/home/albert_wei/fdisk_a/train_output_remote/npenas_seg101_mass_road_20_epochs_2/"
    # output_path = "/home/albert_wei/fdisk_c/SEG101_Results/npenas_seg101_mass_road_4/"
    output_path = "/home/albert_wei/fdisk_c/SEG101_Results/npenas_seg_101_mass_road_20epochs_800mfx_1/"
    cfg_file_path = "/home/albert_wei/WorkSpaces/nas_seg_detectron2/configs/Segmentation/Seg101/Remote-SemanticSegmentation/Base-Seg101-OS16-Semantic-Zero-Cost.yaml"
    cfg = gen_cfg(cfg_file_path)
    model_eval_results = parse_saved_model(output_path, cfg)

    print(f"There {len(model_eval_results)} files to be calculate!")
    proxies = ["grad_norm_seg", "snip_seg", "grasp_seg", "fisher_seg", "jacob_cov_seg", "plain_seg", "synflow_bn_seg"]
    get_zero_costs(cfg, model_eval_results, proxies, device)