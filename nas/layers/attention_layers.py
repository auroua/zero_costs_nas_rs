# Copyright (c) XiDian University and Xi'an University of Posts&Telecommunication. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.nn import init
from .build import NAS_LAYER_REGISTRY


@NAS_LAYER_REGISTRY.register()
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.w = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self, x):
        b, c, w, h = x.size()
        value = x.view(b, c, -1).permute(0, 2, 1)
        query = x.view(b, c, -1).permute(0, 2, 1)
        key = x.view(b, c, -1)

        sim_map = torch.bmm(query, key) * (c ** -.5)
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.bmm(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(b, c, *x.size()[2:])
        context = self.w(context)
        return context


@NAS_LAYER_REGISTRY.register()
class SEAttention(nn.Module):
    def __init__(self, ch_num, r=8):
        super(SEAttention, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=ch_num, out_features=int(ch_num/r))
        self.fc2 = nn.Linear(in_features=int(ch_num/r), out_features=ch_num)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, x_middle):
        x_c = self.global_pool(x).view(x.size()[:2])
        out = self.fc1(x_c)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out).view(x.size()[0], x.size()[1], 1, 1)
        x_se = x_middle*out
        return x_se + x_middle


@NAS_LAYER_REGISTRY.register()
class SEAttentionStandard(nn.Module):
    def __init__(self, ch_num, r=8):
        super(SEAttentionStandard, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=ch_num, out_features=int(ch_num/r))
        self.fc2 = nn.Linear(in_features=int(ch_num/r), out_features=ch_num)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x_c = self.global_pool(x).view(x.size()[:2])
        out = self.fc1(x_c)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out).view(x.size()[0], x.size()[1], 1, 1)
        x_se = x*out
        return x_se + x


@NAS_LAYER_REGISTRY.register()
class SelfAttentionChannel(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionChannel, self).__init__()
        self.in_channels = in_channels
        # TODO: Does this convolution layer necessary?
        self.w = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self, x, x_middle):
        b, c, w, h = x.size()
        value = x_middle.view(b, c, -1)
        query = x.view(b, c, -1)
        key = x.view(b, c, -1).permute(0, 2, 1)
        sim_map = torch.bmm(query, key) * (c ** -.5)
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.bmm(sim_map, value).contiguous()
        context = context.view(b, c, *x.size()[2:])
        context = self.w(context)
        return context + x_middle


@NAS_LAYER_REGISTRY.register()
class SelfGlobalAttentionChannel(nn.Module):
    def __init__(self, in_channels):
        super(SelfGlobalAttentionChannel, self).__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # TODO: Does this convolution layer necessary?
        self.w = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def forward(self, x, x_middle):
        b, c, w, h = x.size()
        x_global_pool = self.avg_pool(x).view(b, c, 1)
        q = x_global_pool.permute(0, 2, 1)
        sim_map = torch.bmm(x_global_pool, q) * (c ** -.5)
        sim_map = F.softmax(sim_map, dim=-1)
        value = x_middle.view(b, c, -1)
        context = torch.bmm(sim_map, value).contiguous()
        context = context.view(b, c, *x.size()[2:])
        context = self.w(context)
        return context + x_middle


@NAS_LAYER_REGISTRY.register()
class GlobalSE(nn.Module):
    # TODO: This layer only support three input feature map.
    def __init__(self, in_channel, out_channel, r):
        assert r >= 1, 'squeeze ration should larger than 1'
        super(GlobalSE, self).__init__()
        middle_len = int(in_channel/r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.op1 = nn.Sequential(
            nn.Linear(in_channel, middle_len),
            nn.ReLU(False),
            nn.Linear(middle_len, in_channel)
        )
        self.op2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(False)
        )

    def forward(self, x):
        assert type(x) == list or type(x) == tuple, 'In concat layer input x should be a list or tuple'
        assert len(x) == 3, 'FPNHead layer only support three input feature map layer'
        x = torch.cat(x, dim=1)
        b, c, h, w = x.size()
        x_vec = self.pool(x).view(b, c)
        x_vec_op1 = self.op1(x_vec)
        x_middle = torch.sigmoid(x_vec_op1).view(b, c, 1, 1)
        x_e = x_middle * x
        x_out = x_e + x
        out = self.op2(x_out)
        return out


@NAS_LAYER_REGISTRY.register()
class GlobalSEHead(nn.Module):
    def __init__(self, in_channel, spatial_size, seg_nums, r):
        assert r >= 1, 'squeeze ration should larger than 1'
        assert len(in_channel) == 2, "the input channels is the number of channels of the input feature"
        super(GlobalSEHead, self).__init__()

        in_channel = sum(in_channel)
        middle_len = int(in_channel/r)
        self.concat_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(False),
            nn.Conv2d(in_channel, in_channel, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(False)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.op1 = nn.Sequential(
            nn.Linear(in_channel, middle_len),
            nn.ReLU(False),
            nn.Linear(middle_len, in_channel)
        )

        self.out_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(False),
            nn.Conv2d(in_channel, seg_nums, kernel_size=(1, 1), stride=(1, 1),
                      padding=(0, 0), bias=False),
        )

        self.seg_nums = seg_nums
        self.out_spatial_size = spatial_size

    def forward(self, f1, f2):
        output_spatial_2 = max(f1.size(2), f2.size(2))
        output_spatial_3 = max(f1.size(3), f2.size(3))

        if f1.size(2) != output_spatial_2 or f1.size(3) != output_spatial_3:
            f1 = F.interpolate(f1, (output_spatial_2, output_spatial_3),
                               mode='bilinear', align_corners=False)
        if f2.size(2) != output_spatial_2 or f2.size(3) != output_spatial_3:
            f2 = F.interpolate(f2, (output_spatial_2, output_spatial_3),
                               mode='bilinear', align_corners=False)

        x = torch.cat([f1, f2], dim=1)
        x = self.concat_feature(x)

        b, c, h, w = x.size()
        x_vec = self.pool(x).view(b, c)
        x_vec_op1 = self.op1(x_vec)
        x_middle = torch.sigmoid(x_vec_op1).view(b, c, 1, 1)
        x_e = x_middle * x
        x_out = x_e + x
        out = self.out_feature(x_out)
        return out

    def get_sizes(self):
        return {
            "spatial_size": self.out_spatial_size,
            "channels": self.seg_nums
        }

@NAS_LAYER_REGISTRY.register()
class SynthesizerRandomSelfAttention(nn.Module):
    def __init__(self, in_channels, out_ch_num, spatial_size):
        super(SynthesizerRandomSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.w = nn.Conv2d(in_channels, out_ch_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.g = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Add global context attention
        self.weight1 = Parameter(torch.Tensor(self.in_channels, 16))
        self.weight2 = Parameter(torch.Tensor(self.in_channels, 16))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def forward(self, x, x_mid):
        b, c, w, h = x.size()
        g_x = self.g(x_mid)
        value = g_x.view(b, c, -1)
        query = x.view(b, c, -1)
        key = x.view(b, c, -1).permute(0, 2, 1)

        sim_map = torch.bmm(query, key) * (c ** -.5)
        random_map = torch.matmul(self.weight1, self.weight2.T).unsqueeze(dim=0)
        sim_map = F.softmax(sim_map+random_map, dim=-1)
        context = torch.bmm(sim_map, value).contiguous()
        context = context.view(b, c, *x.size()[2:])
        context = self.w(context)
        return context


@NAS_LAYER_REGISTRY.register()
class SynthesizerRandomAttention(nn.Module):
    def __init__(self, in_channels, out_ch_num, spatial_size):
        super(SynthesizerRandomAttention, self).__init__()
        self.in_channels = in_channels
        self.w = nn.Conv2d(in_channels, out_ch_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.g = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Add global context attention
        self.weight1 = Parameter(torch.Tensor(self.in_channels, 16))
        self.weight2 = Parameter(torch.Tensor(self.in_channels, 16))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def forward(self, x, x_mid):
        b, c, w, h = x.size()
        g_x = self.g(x_mid)
        value = g_x.view(b, c, -1)

        random_map = torch.matmul(self.weight1, self.weight2.T).unsqueeze(dim=0)
        sim_map = F.softmax(random_map, dim=-1)
        context = torch.bmm(sim_map, value).contiguous()
        context = context.view(b, c, *x.size()[2:])
        context = self.w(context)
        return context


@NAS_LAYER_REGISTRY.register()
class SynthesizerRandomAttentionMergeFeatureMap(nn.Module):
    def __init__(self, in_channels, out_ch_num, spatial_size):
        super(SynthesizerRandomAttentionMergeFeatureMap, self).__init__()
        self.in_channels = in_channels
        self.w = nn.Conv2d(in_channels, out_ch_num, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.g = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Add global context attention
        self.weight1 = Parameter(torch.Tensor(in_channels, 16))
        self.weight2 = Parameter(torch.Tensor(in_channels, 16))
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

    def forward(self, x):
        x = torch.cat(x, dim=1)
        b, c, w, h = x.size()
        g_x = self.g(x)
        value = g_x.view(b, c, -1)
        query = x.view(b, c, -1)
        key = x.view(b, c, -1).permute(0, 2, 1)

        sim_map = torch.bmm(query, key) * (c ** -.5)
        random_map = torch.matmul(self.weight1, self.weight2.T).unsqueeze(dim=0)
        sim_map = F.softmax(sim_map+random_map, dim=-1)
        context = torch.bmm(sim_map, value).contiguous()
        context = context.view(b, c, *x.size()[2:])
        context = self.w(context)
        return context