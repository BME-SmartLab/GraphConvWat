# -*- coding: utf-8 -*-
import copy
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader

def build_dataloader(G, set_x, set_y, batch_size, shuffle):
    data    = []
    master_graph    = from_networkx(G)
    for x, y in zip(set_x, set_y):
        graph   = copy.deepcopy(master_graph)
        graph.x = torch.Tensor(x)
        graph.y = torch.Tensor(y)
        data.append(graph)
    loader  = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader
