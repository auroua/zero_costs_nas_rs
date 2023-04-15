import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import NAS_LAYER_REGISTRY


@NAS_LAYER_REGISTRY.register()
class Zero(nn.Module):
    def __init__(self, ch_num):
        super(Zero, self).__init__()
        self.ch_num = ch_num

    def forward(self, x):
        b, c, h, w = x.size()
        assert self.ch_num <= c, 'input feature map channel nums should larger than channel num variable'
        if self.ch_num == c:
            return x.mul(0.)
        return x[:, :self.ch_num, :, :].mul(0.)


@NAS_LAYER_REGISTRY.register()
class Conv2d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv2d, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


@NAS_LAYER_REGISTRY.register()
class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_out, C_out, kernel_size=(1, 1), padding=(0, 0),
                      stride=(1, 1), bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


@NAS_LAYER_REGISTRY.register()
class ConvReLuBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvReLuBN, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


@NAS_LAYER_REGISTRY.register()
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_out, bias=False),
            nn.Conv2d(C_out, C_out, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),

            nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=(1, 1), padding=padding, groups=C_out, bias=False),
            nn.Conv2d(C_out, C_out, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


@NAS_LAYER_REGISTRY.register()
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


@NAS_LAYER_REGISTRY.register()
class AdaptiveAvgPool(nn.Module):
    def __init__(self, in_channel_size, spatial_size, affine=True):
        super(AdaptiveAvgPool, self).__init__()
        if isinstance(in_channel_size, tuple):
            if len(in_channel_size) == 2:
                in_channel, out_channel = in_channel_size[0], in_channel_size[1]
            else:
                raise ValueError('Illegal input channel size!')
        else:
            in_channel, out_channel = in_channel_size, in_channel_size
        self.op = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=spatial_size),
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channel, affine=affine),
            nn.ReLU(False)
        )

    def forward(self, x):
        x = F.interpolate(self.op(x), x.size()[2:], mode='bilinear', align_corners=False)
        return x


@NAS_LAYER_REGISTRY.register()
class ConcatLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConcatLayer, self).__init__()
        self.ops = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(False)
        )

    def forward(self, x):
        assert type(x) == list or type(x) == tuple, 'In concat layer input x should be a list or tuple'
        concat_x = torch.cat(x, dim=1)
        x = self.ops(concat_x)
        return x


@NAS_LAYER_REGISTRY.register()
class FPNHead(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FPNHead, self).__init__()
        in_channel = in_channel//3
        self.ops = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(False)
        )
        self.ops2 = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(False)
        )

    def forward(self, x):
        assert type(x) == list or type(x) == tuple, 'In concat layer input x should be a list or tuple'
        assert len(x) == 3, 'FPNHead layer only support three input feature map layer'
        p2, p3, p4 = x
        p23 = torch.cat([p2, p3], dim=1)
        out = self.ops(p23)
        po4 = torch.cat([out, p4], dim=1)
        out = self.ops2(po4)
        return out


@NAS_LAYER_REGISTRY.register()
class ConcatHead(nn.Module):
    def __init__(self, in_channel, spatial_size, seg_nums):
        super(ConcatHead, self).__init__()

        out_channels = in_channel[0] + in_channel[1]
        middle_channels = int(out_channels * 0.5)
        self.ops_feature_merge = nn.Sequential(
            # nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
            #           padding=(1, 1), bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(False),
            nn.Conv2d(out_channels, middle_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(False),
            nn.Conv2d(middle_channels, seg_nums, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False)
        )
        self.out_spatial_size = spatial_size

    def forward(self, f1, f2):
        # x1 = self.ops_feature_0(f1)
        # x2 = self.ops_feature_1(f2)

        output_spatial_2 = max(f1.size(2), f2.size(2))
        output_spatial_3 = max(f1.size(3), f2.size(3))

        if output_spatial_2 != f1.size(2) or output_spatial_3 != f1.size(3):
            f1 = F.interpolate(f1, (output_spatial_2, output_spatial_3),
                               mode='bilinear', align_corners=False)
        if output_spatial_2 != f2.size(2) or output_spatial_3 != f2.size(3):
            f2 = F.interpolate(f2, (output_spatial_2, output_spatial_3),
                               mode='bilinear', align_corners=False)
        concat_x = torch.cat((f1, f2), dim=1)
        out = self.ops_feature_merge(concat_x)
        return out

    def get_sizes(self):
        return {
            "spatial_size": self.out_spatial_size,
            "channels": self.seg_nums
        }

@NAS_LAYER_REGISTRY.register()
class StageGroup(nn.Module):
    def __init__(self, groups, group_conv_dict):
        super(StageGroup, self).__init__()
        self.groups_keys = []
        self.groups = groups
        for k, v in group_conv_dict.items():
            self.add_module(k, v)
            self.groups_keys.append(k)

    def forward(self, x):
        channel_num = x.size()[1]
        feature_attention_list = []
        assert channel_num % self.groups == 0, 'feature channels should be divided by groups'
        split_channel_num = channel_num // self.groups
        for i, k in enumerate(self.groups_keys):
            feature_attention_list.append(getattr(self, k)(x[:, i*split_channel_num:(i+1)*split_channel_num, :, :]))
        return torch.cat(feature_attention_list, dim=1)

    def __str__(self):
        group_op_name_list = []
        for k in self.groups_keys:
            group_op_name_list.append(type(getattr(self, k)).__name__)
        return '     '.join(group_op_name_list)


@NAS_LAYER_REGISTRY.register()
class FeatureMerge(nn.Module):
    def __init__(self,
                 name,
                 in_channels,
                 out_channels,
                 spatial_sizes
                 ):
        """

        Args:
            in_channels: tuple or list, contains the number of channels of features
            out_channels:  tuple or list, contains the output channels of features
            spatial_sizes:  tuple or list, contains the spatial size of input features
        """
        super().__init__()

        self.feature_out = nn.Sequential(
            nn.Conv2d(in_channels[0]+in_channels[1], out_channels[0]+out_channels[1],
                      kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channels[0]+out_channels[1]),
            nn.ReLU(False)
        )

        self.out_spatial_size = max(spatial[0] for spatial in spatial_sizes)
        self.out_channels = out_channels[0] + out_channels[1]
        self.name = name

    def forward(self, f1, f2):
        output_spatial_2 = max(f1.size(2), f2.size(2))
        output_spatial_3 = max(f1.size(3), f2.size(3))
        if output_spatial_2 != f1.size(2) or output_spatial_3 != f1.size(3):
            f1 = F.interpolate(f1, (output_spatial_2, output_spatial_3),
                               mode='bilinear', align_corners=False)
        if output_spatial_2 != f2.size(2) or output_spatial_3 != f2.size(3):
            f2 = F.interpolate(f2, (output_spatial_2, output_spatial_3),
                               mode='bilinear', align_corners=False)
        out = torch.cat([f1, f2], dim=1)
        out = self.feature_out(out)

        return out

    def get_sizes(self):
        return {
            "out_channels": self.out_channels,
            "out_spatial_size": (self.out_spatial_size, self.out_spatial_size),
            "name": self.name
        }