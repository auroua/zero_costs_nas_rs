# Copyright (c) Xidian University and Xi'an University of Posts & Telecommunications. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from .build import PREDICTOR_REGISTRY
from nas.utils.search_space import get_input_dim
from nas.layers.gin_conv import GINConv
# from torch_geometric.nn import GINConv
from torch_geometric.nn.glob import global_mean_pool as gmp


@PREDICTOR_REGISTRY.register()
class GINPredictor(nn.Module):
    """
       without using share weights
       three gin layers
       feature concat
    """

    def __init__(self, cfg):
        input_dim, dim1, num_classes = get_input_dim(cfg)
        super(GINPredictor, self).__init__()
        layers = []
        # dim = 32
        # The first two GIN layer is shared by two pred
        nn1 = Sequential(Linear(input_dim, dim1, bias=True), ReLU(), Linear(dim1, dim1, bias=True))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim1)

        nn2_base = Sequential(Linear(dim1, dim1,  bias=True), ReLU(), Linear(dim1, dim1,  bias=True))
        self.conv2_base = GINConv(nn2_base)
        self.bn2_base = torch.nn.BatchNorm1d(dim1)

        nn3_base = Sequential(Linear(dim1, dim1,  bias=True), ReLU(), Linear(dim1, dim1,  bias=True))
        self.conv3_base = GINConv(nn3_base)
        self.bn3_base = torch.nn.BatchNorm1d(dim1)
        # branch1 head
        self.fc = torch.nn.Linear(dim1, num_classes, bias=True)

        self.output_layer = torch.nn.Sigmoid()

        layers.append(self.fc)

        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data_base, edge_index_base, batch_base):
        return self.forward_batch(data_base, edge_index_base, batch_base)

    def forward_batch(self, data_base, edge_index_base, batch_base):
        # Base predictor inference
        x1_base = F.relu(self.conv1(data_base, edge_index_base))
        x1_base = self.bn1(x1_base)

        x2_base = F.relu(self.conv2_base(x1_base, edge_index_base))
        x2_base = self.bn2_base(x2_base)

        x3_base = F.relu(self.conv3_base(x2_base, edge_index_base))
        x3_base = self.bn3_base(x3_base)
        x_embedding_base = gmp(x3_base, batch_base)

        # # branch1 output
        outputs = self.fc(x_embedding_base)

        outputs = self.output_layer(outputs)

        return outputs