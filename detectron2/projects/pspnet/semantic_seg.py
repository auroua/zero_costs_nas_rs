import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from detectron2.layers import Conv2d, get_norm, PSPHead
from ..deeplab.loss import DeepLabCE
import fvcore.nn.weight_init as weight_init


@SEM_SEG_HEADS_REGISTRY.register()
class PSPNetHead(nn.Module):
    @configurable
    def __init__(self,
                 input_shape,
                 avg_pool_sizes,
                 psp_dropout,
                 psp_channel,
                 norm,
                 loss_weight,
                 loss_type,
                 ignore_value,
                 num_classes,
                 with_aux,
                 aux_weight
                 ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        self.in_features = [k for k, v in input_shape]
        in_channels = [x[1].channels for x in input_shape]
        self.strides = [x[1].stride for x in input_shape]
        self.head = PSPHead(in_channels=in_channels[1], nclass=num_classes,
                            norm_layer=norm, avg_pool_sizes=avg_pool_sizes,
                            psp_channel=psp_channel)
        self.with_aux = with_aux
        self.aux_weight = aux_weight
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.ignore_value = ignore_value
        if with_aux:
            self.dsn = nn.Sequential(
                Conv2d(in_channels=in_channels[0],
                       out_channels=psp_channel,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       norm=get_norm(norm, psp_channel),
                       activation=F.relu),
                nn.Dropout2d(psp_dropout),
                Conv2d(psp_channel, num_classes, kernel_size=1, stride=1)
            )
            weight_init.c2_xavier_fill(self.dsn[0])
            weight_init.c2_xavier_fill(self.dsn[2])
        if self.loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_value)
        elif self.loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=self.ignore_value, top_k_percent_pixels=0.2)
        else:
            raise ValueError("Unexpected loss type: %s" % self.loss_type)

    def forward(self, x, targets):
        c3, c4 = x["res4"], x["res5"]
        y = self.head(c4)

        if self.with_aux:
            aux_y = self.dsn(c3)

        if self.training:
            if self.with_aux:
                return None, self.losses(y, targets, aux_y)
            else:
                return None, self.losses(y, targets)
        else:
            y = F.interpolate(y, scale_factor=self.strides[1],
                              mode='bilinear', align_corners=False)
            return y, {}

    def losses(self, predictions, targets, aux_y=None):
        predictions = F.interpolate(
            predictions, scale_factor=self.strides[1], mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets)
        if self.with_aux:
            aux_y = F.interpolate(aux_y, scale_factor=self.strides[0],
                                  mode="bilinear", align_corners=False)
            aux_loss = self.loss(aux_y, targets)
            losses = {"loss_sem_seg": self.aux_weight*aux_loss + (1-self.aux_weight)*loss}
        else:
            losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            avg_pool_sizes=cfg.MODEL.SEM_SEG_HEAD.AVG_POOL_SIZES,
            psp_dropout=cfg.MODEL.SEM_SEG_HEAD.PSP_DROPOUT,
            psp_channel=cfg.MODEL.SEM_SEG_HEAD.PSP_CHANNELS,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            loss_weight=cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            loss_type=cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            with_aux=cfg.MODEL.SEM_SEG_HEAD.WITH_AUX,
            aux_weight=cfg.MODEL.SEM_SEG_HEAD.AUX_WEIGHT
        )
        return ret