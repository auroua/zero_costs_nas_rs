import torch.nn as nn
from ..build import PREDICTOR_REGISTRY


@PREDICTOR_REGISTRY.register()
class GradNorm(nn.Module):
    def __init__(self, neural_arch):
        self.neural_arch = neural_arch


    def forward(self, neural_arch, batch_data):
        pass