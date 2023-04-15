# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import torch.nn.functional as F

import copy

from nas.zero_costs.measures import measure
from ..p_utils import get_layer_metric_array

@measure('grad_norm', bn=True)
def get_grad_norm_arr(net, inputs, targets, loss_fn, split_data=1, skip_grad=False):
    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

        grad_norm_arr = get_layer_metric_array(net, lambda l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param')
        
    return grad_norm_arr


@measure('grad_norm_seg', copy_net=False)
def get_grad_norm_arr_seg(net, batched_inputs, split_data=1, skip_grad=False):
    net.zero_grad()
    N = len(batched_inputs)
    metric_array = {}
    for sp in range(split_data):
        st = sp * N // split_data
        en = (sp + 1) * N // split_data

        losses = net.forward(batched_inputs[st:en])
        losses_val = sum(losses.values())
        losses_val.backward()

        get_layer_metric_array(net, lambda
            l: l.weight.grad.norm() if l.weight.grad is not None else torch.zeros_like(l.weight), mode='param',
                               metric_array=metric_array)

    return sum(metric_array.values())
