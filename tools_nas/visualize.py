import os
import sys
import argparse
sys.path.append(os.getcwd())
from nas.visualization.visualize_close_domain import draw_plot_nasbench_101

model_lists_nasbench = ['Random', 'REA', 'NPENAS']
model_masks_nasbench = [True, True, True]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for visualize search results.')
    parser.add_argument('--search_space', type=str, default='nasbench_101',
                        choices=['nasbench_101', 'nasbench_201', 'scalar_prior', 'evaluation_compare',
                                 'nasbench_nlp', 'nasbench_asr'],
                        help='The algorithm output folder')
    parser.add_argument('--save_dir', type=str,
                        default='/home/albert_wei/fdisk_a/train_output_remote/nas_nasbench_101_output_dir/',
                        help='The search strategy output folder')
    parser.add_argument('--train_data', type=str, default='cifar100',
                        choices=['cifar10-valid', 'cifar100', 'ImageNet16-120'],
                        help='The evaluation of dataset of NASBench-201.')
    parser.add_argument('--draw_type', type=str, default='MEANERROR', choices=['ERRORBAR', 'MEANERROR'],
                        help='Draw result with or without errorbar.')
    parser.add_argument('--show_all', type=str, default='1', help='Weather to show all results.')

    args = parser.parse_args()
    if args.search_space == 'nasbench_101':
        draw_plot_nasbench_101(args.save_dir, draw_type=args.draw_type, model_lists=model_lists_nasbench,
                               model_masks=model_masks_nasbench, order=True)
    else:
        raise ValueError('This search space does not support!')