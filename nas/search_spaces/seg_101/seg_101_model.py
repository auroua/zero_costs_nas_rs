# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from detectron2.projects.deeplab.loss import DeepLabCE
from nas.layers.losses_seg import dice_loss


class Seg101Model(nn.Module):
    """
    This class is the neural architecture search segmentation class which only need to create segmentation head.
    """
    def __init__(self,
                 stages,
                 groups,
                 stage_group_lists,
                 stage_channel_dict,
                 stage_op_dict,
                 head_0_stage_0_dict,
                 head_0_stage_1_dict,
                 head_1_stage_0_dict,
                 head_1_stage_1_dict,
                 head_0_stage_0_keys,
                 head_0_stage_1_keys,
                 head_1_stage_0_keys,
                 head_1_stage_1_keys,
                 head_0_stage_0_idxs,
                 head_0_stage_1_idxs,
                 head_1_stage_0_idxs,
                 head_1_stage_1_idxs,
                 head_0_merge,
                 head_1_merge,
                 head_merge,
                 num_classes,
                 loss_func="cross_entropy",
                 ignore_value=-1,
                 loss_weight=1.0):
        super().__init__()
        self.stages = stages
        self.groups = groups
        for s in stages:
            self.add_module(f'{s}_group', stage_group_lists[f'{s}_group'])
            self.add_module(f'{s}_channel', stage_channel_dict[f'{s}_channel'])
            self.add_module(f"{s}_feature_op", stage_op_dict[f"{s}_feature_op"])

        for head_k in head_0_stage_0_keys:
            head_merge_op = head_0_stage_0_dict[head_k]
            self.add_module(head_k, head_merge_op)

        for head_k in head_0_stage_1_keys:
            self.add_module(head_k, head_0_stage_1_dict[head_k])

        for head_k in head_1_stage_0_keys:
            self.add_module(head_k, head_1_stage_0_dict[head_k])

        for head_k in head_1_stage_1_keys:
            self.add_module(head_k, head_1_stage_1_dict[head_k])

        self.add_module('head_merge_0', head_0_merge)
        self.add_module('head_merge_1', head_1_merge)

        self.add_module("head_merge", head_merge)
        self.num_classed = num_classes

        self.head_0_stage_0_keys = head_0_stage_0_keys
        self.head_0_stage_1_keys = head_0_stage_1_keys
        self.head_1_stage_0_keys = head_1_stage_0_keys
        self.head_1_stage_1_keys = head_1_stage_1_keys

        self.head_0_stage_0_idxs = head_0_stage_0_idxs
        self.head_0_stage_1_idxs = head_0_stage_1_idxs
        self.head_1_stage_0_idxs = head_1_stage_0_idxs
        self.head_1_stage_1_idxs = head_1_stage_1_idxs

        self.ignore_value = ignore_value
        self.apply(self._init_weights)

        if loss_func == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)
        elif loss_func == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=self.ignore_value, top_k_percent_pixels=0.2)
        elif loss_func == "dice":
            self.loss = dice_loss
        else:
            raise ValueError("Unexpected loss type: %s" % loss_func)

        self.loss_weight = loss_weight

    def forward(self, features, targets, T=1.0, zero_costs=False):
        out = self.layers(features) / T
        self.common_stride = (int(targets.size(1) / out.size(2)), int(targets.size(2) / out.size(3)))

        if zero_costs:
            return out
        else:
            if self.training:
                return None, self.losses(out, targets)
            else:
                y = F.interpolate(
                    out, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                return y, {}

    def layers(self, features):
        flag = False
        for k, v in features.items():
            if isinstance(v, tuple):
                flag = True
            else:
                flag = False
        stage_channel_attention_dict = {}
        stage_op_dict = {}
        for ss in self.stages:
            if flag:
                ss_feature_attention = getattr(self, f'{ss}_group')(features[ss][0])
                stage_channel_attention_dict[ss] = getattr(self, f'{ss}_channel')(features[ss][1], ss_feature_attention)
                stage_op_dict[ss] = getattr(self, f'{ss}_feature_op')(stage_channel_attention_dict[ss])
            else:
                ss_feature_attention = getattr(self, f'{ss}_group')(features[ss])
                stage_channel_attention_dict[ss] = getattr(self, f'{ss}_channel')(features[ss], ss_feature_attention)
                stage_op_dict[ss] = getattr(self, f'{ss}_feature_op')(stage_channel_attention_dict[ss])

        head_0_stage_0_features_dict = {}
        for idx, head_k in enumerate(self.head_0_stage_0_keys):
            head_0_stage_0_features_dict[idx] = getattr(self, head_k)(stage_op_dict[self.stages[self.head_0_stage_0_idxs[idx][0]]],
                                                                      stage_op_dict[self.stages[self.head_0_stage_0_idxs[idx][1]]])
        head_0_stage_1_features_dict = {}
        for idx, head_k in enumerate(self.head_0_stage_1_keys):
            head_0_stage_1_features_dict[idx] = getattr(self, head_k)(head_0_stage_0_features_dict[self.head_0_stage_1_idxs[idx][0]],
                                                                      head_0_stage_0_features_dict[self.head_0_stage_1_idxs[idx][1]])
        head_1_stage_0_features_dict = {}
        for idx, head_k in enumerate(self.head_1_stage_0_keys):
            head_1_stage_0_features_dict[idx] = getattr(self, head_k)(stage_op_dict[self.stages[self.head_1_stage_0_idxs[idx][0]]],
                                                                      stage_op_dict[self.stages[self.head_1_stage_0_idxs[idx][1]]])

        head_1_stage_1_features_dict = {}
        for idx, head_k in enumerate(self.head_1_stage_1_keys):
            head_1_stage_1_features_dict[idx] = getattr(self, head_k)(head_1_stage_0_features_dict[self.head_1_stage_1_idxs[idx][0]],
                                                                      head_1_stage_0_features_dict[self.head_1_stage_1_idxs[idx][1]])

        head_0_feature = getattr(self, "head_merge_0")(head_0_stage_1_features_dict[0], head_0_stage_1_features_dict[1])
        head_1_feature = getattr(self, "head_merge_1")(head_1_stage_1_features_dict[0], head_1_stage_1_features_dict[1])

        out = getattr(self, "head_merge")(head_0_feature, head_1_feature)

        return out

    def losses(self, predictions, targets):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

    def _init_weights(self, m):
        """Performs ResNet-style weight initialization."""
        if isinstance(m, nn.Conv2d):
            # Note that there is no bias due to BN
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
        elif isinstance(m, nn.BatchNorm2d):
            zero_init_gamma = (
                    hasattr(m, "final_bn") and m.final_bn and False
            )
            m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.zero_()