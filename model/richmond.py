# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class ChebNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels, 120, K=240)
        self.conv2 = ChebConv(120, 60, K=120)
        self.conv3 = ChebConv(60, 30, K=20)
        self.conv4 = ChebConv(30, out_channels, K=1, bias=False)

    def forward(self, data):
        x, edge_index, edge_weight  = data.x, data.edge_index, data.weight
        x = F.silu(self.conv1(x, edge_index, edge_weight))
        x = F.silu(self.conv2(x, edge_index, edge_weight))
        x = F.silu(self.conv3(x, edge_index, edge_weight))
        x = self.conv4(x, edge_index, edge_weight)
        return torch.sigmoid(x)
