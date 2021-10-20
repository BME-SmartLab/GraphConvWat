# -*- coding: utf-8 -*-
import numpy as np

def random_deploy(G, sensor_budget, seed=None):
    num_nodes   = len(G.nodes)
    signal_mask = np.ones(shape=(num_nodes,))
    if seed:
        np.random.seed(seed)
    unobserved_nodes    = np.random.choice(
        np.arange(num_nodes),
        size    = num_nodes-sensor_budget,
        replace = False
        )
    signal_mask[unobserved_nodes]   = 0
    return signal_mask
