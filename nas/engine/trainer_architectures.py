import os
from torch.multiprocessing import Process
from nas.engine.trainer_seg_model import train_seg_model
from nas.config.defaults_seg import get_cfg
from detectron2.utils.regnet_utils import REGNET_MODEL_PRETRAINED
from torch.multiprocessing import Queue
from nas.utils.comm import queue_to_dict


def ansyc_multiple_process_evaluation(args, cfg, arch_lists):
    q = Queue(len(arch_lists))
    count_list = [[] for _ in range(args.gpus)]
    [count_list[i % args.gpus].append(i) for i in range(len(arch_lists))]
    arch_idxs = list(range(len(arch_lists)))

    cfg_seg = get_cfg()
    cfg_seg.merge_from_file(cfg.SOLVER_NN.SEG.CONFIG)
    cfg_seg.MODEL.WEIGHTS = os.path.expanduser(REGNET_MODEL_PRETRAINED[cfg_seg.MODEL.REGNETS.TYPE])
    cfg_seg.freeze()

    p_consumers = [Process(target=data_consumers,
                           args=(args, cfg_seg.clone(), device,
                                 [arch_idxs[arch_idx] for arch_idx in count_list[device]],
                                 # arch_idxs[device*per_gpu_archs:(device+1)*per_gpu_archs],
                                 arch_lists, q
                                 ))
                   for device in range(args.gpus)]

    for p in p_consumers:
        p.start()

    for p in p_consumers:
        p.join()

    results_dict = queue_to_dict(q)
    for k, v in results_dict.items():
        arch_lists[k].eval_results = v
        arch_lists[k].miou = v["mIoU"]
        arch_lists[k].iou_bg = v["IoU-bg"]
        if "mass_road" in cfg_seg.DATASETS.TRAIN[0]:
            arch_lists[k].iou_target = v["IoU-road"]


def data_consumers(args, cfg_seg, device, arch_keys, arch_lists, q):
    for key in arch_keys:
        arch = arch_lists[key]
        seg_model = arch.assemble_neural_network(cfg_seg)
        train_seg_model(args, cfg_seg, seg_model, device, arch)
        q.put((key, arch.eval_results))
