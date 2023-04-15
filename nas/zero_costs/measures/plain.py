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

from nas.zero_costs.measures import measure
from ..p_utils import get_layer_metric_array


@measure('plain', bn=True, mode='param')
def compute_plain_per_weight(net, inputs, targets, mode, loss_fn, split_data=1):

    net.zero_grad()
    N = inputs.shape[0]
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        outputs = net.forward(inputs[st:en])
        loss = loss_fn(outputs, targets[st:en])
        loss.backward()

    # select the gradients that we want to use for search/prune
    def plain(layer):
        if layer.weight.grad is not None:
            return layer.weight.grad * layer.weight
        else:
            return torch.zeros_like(layer.weight)

    grads_abs = get_layer_metric_array(net, plain, mode)
    return grads_abs


@measure('plain_seg', copy_net=False, mode='param')
def compute_plain_per_weight_seg(net, batched_inputs, mode, split_data=1):

    net.zero_grad()
    N = len(batched_inputs)
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data

        losses = net.forward(batched_inputs[st:en])
        losses_val = sum(losses.values())
        losses_val.backward()

    # select the gradients that we want to use for search/prune
    def plain(layer):
        if layer.weight.grad is not None:
            return layer.weight.grad * layer.weight
        else:
            return torch.zeros_like(layer.weight)
    metric_array = {}
    get_layer_metric_array(net, plain, mode, metric_array)
    return sum(metric_array.values())
