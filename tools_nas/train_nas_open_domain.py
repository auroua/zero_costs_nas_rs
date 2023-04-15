import argparse
import torch.multiprocessing as multiprocessing
from nas.utils.comm import random_id
from nas.config.defaults import get_cfg
from nas.utils.comm import default_setup
from nas.engine.trainer_open_domain import main_process


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_file(args.data_file_path)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfgs = setup(args)
    multiprocessing.set_start_method('spawn')
    main_process(args, cfgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for neural architecture in close domain.')
    parser.add_argument("--config_file", type=str,
                        default="../configs_nas/seg/seg_101_evolutionary.yaml",
                        help="Configuration files of neural architecture search algorithms.")
    parser.add_argument("--data_file_path", type=str,
                        default='../configs_nas/data.yaml',
                        help="Configuration files for different neural architecture search strategies.")
    parser.add_argument('--gpus', type=int, default=2, help='The num of gpus used for search.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/fdisk_c/2022_train_output_seg101/seg101_evolutionary_400mfx_2epochs/',
                        help='output directory')

    args = parser.parse_args()
    args.output_filename = random_id(64)
    main(args)