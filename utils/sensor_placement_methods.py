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

def collect_master_nodes(wds, G):
    master_nodes    = set()

    for tank in wds.tanks:
        node_a  = wds.links[list(tank.links.keys())[0]].from_node.index
        node_b  = wds.links[list(tank.links.keys())[0]].to_node.index
        if node_a in set(G.nodes):
            master_nodes.add(node_a)
        elif node_b in set(G.nodes):
            master_nodes.add(node_b)
        else:
            print('Neither node {} nor {} of tank {} not found in graph.'.format(
                node_a, node_b, tank))
            raise

    for reservoir in wds.reservoirs:
        node_a  = wds.links[list(reservoir.links.keys())[0]].from_node.index
        node_b  = wds.links[list(reservoir.links.keys())[0]].to_node.index
    if node_a in set(G.nodes):
        master_nodes.add(node_a)
    elif node_b in set(G.nodes):
        master_nodes.add(node_b)
    else:
        print('Neither node {} nor {} of reservoir {} not found in graph.'.format(
            node_a, node_b, reservoir))
        raise
    return master_nodes
