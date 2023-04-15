import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from .wrappers import Conv2d
from .batch_norm import get_norm


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, avg_pool_sizes, psp_channel, norm_layer):
        super(PyramidPooling, self).__init__()
        assert len(avg_pool_sizes) == 4, "Not enough adaptive average pool size provided."
        self.avgpool1 = nn.AdaptiveAvgPool2d(avg_pool_sizes[0])
        self.avgpool2 = nn.AdaptiveAvgPool2d(avg_pool_sizes[1])
        self.avgpool3 = nn.AdaptiveAvgPool2d(avg_pool_sizes[2])
        self.avgpool4 = nn.AdaptiveAvgPool2d(avg_pool_sizes[3])
        self.conv1 = Conv2d(in_channels, psp_channel, kernel_size=1, bias=False,
                            norm=get_norm(norm_layer, psp_channel),
                            activation=F.relu)
        self.conv2 = Conv2d(in_channels, psp_channel, kernel_size=1, bias=False,
                            norm=get_norm(norm_layer, psp_channel),
                            activation=F.relu)
        self.conv3 = Conv2d(in_channels, psp_channel, kernel_size=1, bias=False,
                            norm=get_norm(norm_layer, psp_channel),
                            activation=F.relu)
        self.conv4 = Conv2d(in_channels, psp_channel, kernel_size=1, bias=False,
                            norm=get_norm(norm_layer, psp_channel),
                            activation=F.relu)
        weight_init.c2_xavier_fill(self.conv1)
        weight_init.c2_xavier_fill(self.conv2)
        weight_init.c2_xavier_fill(self.conv3)
        weight_init.c2_xavier_fill(self.conv4)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class PSPHead(nn.Module):
    def __init__(self, in_channels, psp_channel, nclass,
                 avg_pool_sizes, norm_layer="BN"):
        super(PSPHead, self).__init__()
        self.psp = PyramidPooling(in_channels, avg_pool_sizes, psp_channel, norm_layer=norm_layer)
        self.block = nn.Sequential(
            Conv2d(in_channels*2, psp_channel, kernel_size=3, padding=1, bias=False,
                   norm=get_norm(norm_layer, psp_channel),
                   activation=F.relu),
            nn.Dropout(0.1),
            Conv2d(psp_channel, nclass, kernel_size=1)
        )
        weight_init.c2_xavier_fill(self.block[0])
        weight_init.c2_xavier_fill(self.block[2])

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)