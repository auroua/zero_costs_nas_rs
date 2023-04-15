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
import numpy as np

from nas.zero_costs.measures import measure


def get_batch_jacobian(net, batched_inputs, device, split_data):
    # x.requires_grad_(True)

    N = len(batched_inputs)
    total_inputs_grad = []
    for sp in range(split_data):
        st=sp*N//split_data
        en=(sp+1)*N//split_data
        inputs, y = net(batched_inputs[st:en], inputs_requires_grad=True, zero_costs=True)
        y.backward(torch.ones_like(y))
        inputs.requires_grad_(False)
        total_inputs_grad.append(inputs.grad.detach())
    jacob = torch.cat(total_inputs_grad, dim=0)
    return jacob

def eval_score(jacob):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))

@measure('jacob_cov', bn=True)
def compute_jacob_cov(net, inputs, targets, split_data=1, loss_fn=None):
    device = inputs.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    jacobs, _ = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        jc = eval_score(jacobs)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc


@measure('jacob_cov_seg', copy_net=False)
def compute_jacob_cov(net, batched_inputs, split_data=1):
    device = net.device
    # Compute gradients (but don't apply them)
    net.zero_grad()

    jacobs = get_batch_jacobian(net, batched_inputs, device, split_data=split_data)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

    try:
        jc = eval_score(jacobs)
    except Exception as e:
        print(e)
        jc = np.nan

    return jc