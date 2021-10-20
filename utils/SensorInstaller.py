# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np

from graph_utils import get_nx_graph

class SensorInstaller():
    def __init__(self, wds):
        self.wds    = wds
        self.G      = get_nx_graph(wds, mode='weighted')
        self.master_nodes   = self._collect_master_nodes(self.wds, self.G)
        self.sensor_nodes   = set()

    def _collect_master_nodes(self, wds, G):
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

    def deploy_by_random(self, sensor_budget, seed=None):
        num_nodes   = len(self.G.nodes)
        signal_mask = np.ones(shape=(num_nodes,), dtype=np.int8)
        if seed:
            np.random.seed(seed)
        unobserved_nodes    = np.random.choice(
            np.arange(num_nodes),
            size    = num_nodes-sensor_budget,
            replace = False
            )
        signal_mask[unobserved_nodes]   = 0
        self.sensor_nodes   = set(
                np.array(list(self.G.nodes))[np.where(signal_mask)[0]])

    def deploy_by_shortest_path(self, sensor_budget, weight_by=None):
        sensor_nodes    = set()
        for _ in range(sensor_budget):
            path_lengths    = dict()
            for node in self.G.nodes:
                path_lengths[node]  = 0
            for node in self.master_nodes.union(sensor_nodes):
                tempo   = nx.shortest_path_length(
                    self.G,
                    source  = node,
                    weight  = weight_by
                    )
                for key, value in tempo.items():
                    if key not in self.master_nodes.union(sensor_nodes):
                        path_lengths[key]   += value
            sensor_nodes.add(
                    [candidate for candidate, path_length in path_lengths.items()
                        if path_length == np.max(list(path_lengths.values()))][0]
                    )
        self.sensor_nodes   = sensor_nodes
