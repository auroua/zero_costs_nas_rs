import math
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


def gen_batch_idx(idx_list, batch_size, drop_last=False):
    ds_len = len(idx_list)
    idx_batch_list = []

    for i in range(0, math.ceil(ds_len/batch_size)):
        if (i+1)*batch_size > ds_len and not drop_last:
            idx_batch_list.append(idx_list[i*batch_size:])
        else:
            idx_batch_list.append(idx_list[i*batch_size: (i+1)*batch_size])
    return idx_batch_list


def make_agent_optimizer(model, base_lr, weight_deacy=1e-4, bias_multiply=True):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_lr
        weight_decay = weight_deacy
        if "bias" in key:
            if bias_multiply:
                lr = base_lr*2.0
            else:
                lr = base_lr
            weight_decay = 0.0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params, base_lr, (0.0, 0.9))
    return optimizer


def lr_step(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_loss_criteria(loss_type):
    if loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_type == 'mae':
        criterion = torch.nn.L1Loss()
    else:
        raise ValueError('This loss type does not support!')
    return criterion


class CosineLR(_LRScheduler):
    def __init__(self, optimizer, epochs, train_images, batch_size):
        self.epochs = epochs
        self.train_image_num = train_images
        self.batch_size = batch_size
        self.total_steps = int(self.epochs*self.train_image_num / self.batch_size)
        super(CosineLR, self).__init__(optimizer, -1)

    def get_lr(self):
        progress_fraction = float(self._step_count+1) / self.total_steps
        lr_lists = [(0.5 * base_lr * (1 + math.cos(np.pi * progress_fraction)))
                    for base_lr in self.base_lrs]
        return lr_lists

    def set_train_images(self, new_count):
        self.train_image_num = new_count