import argparse
import torch.multiprocessing as multiprocessing
from nas.utils.comm import random_id
from nas.config.defaults import get_cfg
from nas.utils.comm import default_setup
from nas.engine.trainer import ansyc_multiple_process_search


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfgs = list()
    for cfg_path in args.config_files:
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.merge_from_file(args.data_file_path)
        cfg.freeze()
        cfgs.append(cfg)
    default_setup(cfgs, args)
    return cfgs


def main(args):
    cfgs = setup(args)
    multiprocessing.set_start_method('spawn')
    ansyc_multiple_process_search(args, cfgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for neural architecture in close domain.')
    parser.add_argument("--trials", type=int, default=600, help="Number of trials.")
    parser.add_argument("--config_files", type=list,
                        default=['/home/albert_wei/WorkSpaces/nas_seg_detectron2/configs_nas/nasbench_101/nasbench_101_random.yaml',
                                 '/home/albert_wei/WorkSpaces/nas_seg_detectron2/configs_nas/nasbench_101/nasbench_101_evolutionary.yaml',
                                 '/home/albert_wei/WorkSpaces/nas_seg_detectron2/configs_nas/nasbench_101/nasbench_101_npenas.yaml'],
                        help="Configuration files of neural architecture search algorithms.")
    parser.add_argument("--data_file_path", type=str,
                        default='/home/albert_wei/WorkSpaces/nas_seg_detectron2/configs_nas/data.yaml',
                        help="Configuration files for different neural architecture search strategies.")
    parser.add_argument('--gpus', type=int, default=2, help='The num of gpus used for search.')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/fdisk_a/train_output_remote/nas_nasbench_101_output_dir/',
                        help='output directory')

    args = parser.parse_args()
    args.output_filename = random_id(64)
    main(args)