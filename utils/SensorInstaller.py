# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import random

from utils.graph_utils import get_nx_graph

class SensorInstaller():
    def __init__(self, wds, include_pumps_as_master=False):
        self.wds    = wds
        self.G      = get_nx_graph(wds, mode='weighted')
        self.include_pumps  = include_pumps_as_master
        self.master_nodes   = self._collect_master_nodes(self.wds, self.G)
        self.sensor_nodes   = set()

    def _collect_master_nodes(self, wds, G):
        master_nodes    = set()

        if self.include_pumps:
            for pump in wds.pumps:
                node_a  = pump.to_node.index
                node_b  = pump.from_node.index
                if node_a in set(G.nodes):
                    master_nodes.add(node_a)
                elif node_b in set(G.nodes):
                    master_nodes.add(node_b)
                else:
                    print('Neither node {} nor {} of pump {} not found in graph.'.format(
                        node_a, node_b, pump))
                    raise

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

    def get_shortest_path_lengths(self, nodes):
        lengths = np.zeros(shape=(len(nodes), len(nodes)))
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                lengths[i, j]   = nx.shortest_path_length(self.G, source=source, target=target)
        return lengths

    def get_shortest_paths(self, nodes):
        paths   = []
        for source in list(nodes)[:int(len(nodes)//2+len(nodes)%2)]:
            for target in nodes:
                path    = nx.shortest_path(self.G, source=source, target=target, weight=None)
                paths.append(nx.path_graph(path))
        return paths

    def get_shortest_path_to_node_collection(self, target, node_collection, weight=None):
        closest_node= target
        path_length = np.inf
        for source in node_collection:
            tempo   = nx.shortest_path_length(
                self.G,
                source  = source,
                target  = target,
                weight  = weight
                )
            if tempo < path_length:
                closest_node= source
                path_length = tempo
        assert closest_node is not target
        return set(nx.shortest_path(self.G, source=closest_node, target=target))

    def set_sensor_nodes(self, sensor_nodes):
        self.sensor_nodes   = set(sensor_nodes)

    def deploy_by_random(self, sensor_budget, seed=None):
        num_nodes   = len(self.G.nodes)
        signal_mask = np.zeros(shape=(num_nodes,), dtype=np.int8)
        if seed:
            np.random.seed(seed)
        observed_nodes  = np.random.choice(
            np.arange(num_nodes),
            size    = sensor_budget,
            replace = False
            )
        signal_mask[observed_nodes] = 1
        self.sensor_nodes   = set(self.wds.junctions.index[np.where(signal_mask)[0]])

    def deploy_by_xrandom(self, sensor_budget, seed=None, sensor_nodes=None):
        random.seed(seed)
        if not sensor_nodes:
            sensor_nodes    = set()
        free_nodes  = set(self.G.nodes).difference(sensor_nodes)
        rnd_nodes   = random.sample(free_nodes, sensor_budget)
        self.sensor_nodes   = sensor_nodes.union(rnd_nodes)

    def deploy_by_random_deprecated(self, sensor_budget, seed=None):
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
        self.sensor_nodes   = set(self.wds.junctions.index[np.where(signal_mask)[0]])

    def deploy_by_shortest_path(self, sensor_budget, weight_by=None, sensor_nodes=None):
        if not sensor_nodes:
            sensor_nodes    = set()
        forbidden_nodes = self.master_nodes.union(sensor_nodes)
        for _ in range(sensor_budget):
            path_lengths    = dict()
            for node in forbidden_nodes:
                path_lengths[node]  = 0
            for node in set(self.G.nodes).difference(forbidden_nodes):
                path_lengths[node]  = np.inf

            for node in forbidden_nodes:
                tempo   = nx.shortest_path_length(
                    self.G,
                    source  = node,
                    weight  = weight_by
                    )
                for key, value in tempo.items():
                    if (key not in forbidden_nodes) and (path_lengths[key] > value):
                        path_lengths[key]   = value

            sensor_node = [candidate for candidate, path_length in path_lengths.items()
                        if path_length == np.max(list(path_lengths.values()))][0]
            sensor_nodes.add(sensor_node)
            forbidden_nodes.add(sensor_node)
        self.sensor_nodes   = sensor_nodes

    def deploy_by_shortest_path_v2(self, sensor_budget, weight_by=None):
        """Computationally ineffective implementation."""
        sensor_nodes    = set()
        forbidden_nodes = self.master_nodes
        for _ in range(sensor_budget):
            path_lengths    = dict()
            for node in forbidden_nodes:
                path_lengths[node]  = 0
            for node in set(self.G.nodes).difference(forbidden_nodes):
                path_lengths[node]  = np.inf

            for node in forbidden_nodes:
                tempo   = nx.shortest_path_length(
                    self.G,
                    source  = node,
                    weight  = weight_by
                    )
                for key, value in tempo.items():
                    if (key not in forbidden_nodes) and (path_lengths[key] > value):
                        path_lengths[key]   = value

            sensor_node = [candidate for candidate, path_length in path_lengths.items()
                        if path_length == np.max(list(path_lengths.values()))][0]
            branch  = self.get_shortest_path_to_node_collection(
                sensor_node,
                forbidden_nodes,
                weight  = weight_by
                )
            sensor_nodes.add(sensor_node)
            forbidden_nodes = forbidden_nodes.union(branch)
        self.sensor_nodes   = sensor_nodes

    def deploy_by_shortest_path_with_sensitivity(
            self, sensor_budget, node_weights_arr, weight_by=None, aversion=0, sensor_nodes=None):
        assert aversion >= 0
        if not sensor_nodes:
            sensor_nodes    = set()
        forbidden_nodes = self.master_nodes.union(sensor_nodes)
        node_weights    = dict()
        for i, junc in enumerate(self.wds.junctions):
            node_weights[junc.index]   = node_weights_arr[i]

        for _ in range(sensor_budget):
            path_lengths    = dict()
            for node in forbidden_nodes:
                path_lengths[node]  = 0
            for node in set(self.G.nodes).difference(forbidden_nodes):
                path_lengths[node]  = np.inf

            for node in self.master_nodes.union(sensor_nodes):
                tempo   = nx.shortest_path_length(
                    self.G,
                    source  = node,
                    weight  = weight_by
                    )
                for key, value in tempo.items():
                    if (key not in forbidden_nodes) and (path_lengths[key] > node_weights[key]*value):
                        path_lengths[key]   = node_weights[key]*value
            sensor_node = [candidate for candidate, path_length in path_lengths.items() 
                    if path_length == np.max(list(path_lengths.values()))][0]
            sensor_nodes.add(sensor_node)
            
            forbidden_nodes = forbidden_nodes.union(set(
                nx.algorithms.shortest_paths.single_source_shortest_path(
                    self.G, sensor_node, cutoff=aversion
                    ).keys()
                ))
        self.sensor_nodes   = sensor_nodes

    def deploy_by_shortest_path_with_sensitivity_v2(
            self, sensor_budget, sensitivity_matrix, weight_by=None, aversion=0):
        if aversion > 0:
            print('Aversion is not implemented in this module.')
        sensor_nodes        = set()
        forbidden_nodes     = self.master_nodes
        node_weights   = dict()
        nodal_sensitivities = np.sum(np.abs(sensitivity_matrix), axis=0)
        for i, junc in enumerate(self.wds.junctions):
            node_weights[junc.index]   = nodal_sensitivities[i]

        for _ in range(sensor_budget):
            path_lengths    = dict()
            for node in forbidden_nodes:
                path_lengths[node]  = 0
            for node in set(self.G.nodes).difference(forbidden_nodes):
                path_lengths[node]  = np.inf

            for node in self.master_nodes.union(sensor_nodes):
                tempo   = nx.shortest_path_length(
                    self.G,
                    source  = node,
                    weight  = weight_by
                    )
                for key, value in tempo.items():
                    if (key not in forbidden_nodes) and (path_lengths[key] > node_weights[key]*value):
                        path_lengths[key]   = node_weights[key]*value
            sensor_node = [candidate for candidate, path_length in path_lengths.items() 
                    if path_length == np.max(list(path_lengths.values()))][0]
            branch  = self.get_shortest_path_to_node_collection(
                sensor_node,
                forbidden_nodes,
                weight  = weight_by
                )
            sensor_nodes.add(sensor_node)
            forbidden_nodes = forbidden_nodes.union(branch)
        self.sensor_nodes   = sensor_nodes

    def master_node_mask(self):
        mask = np.zeros(shape=(len(self.wds.junctions),), dtype=np.float32)
        if self.master_nodes:
            for index in list(self.master_nodes):
                mask[np.where(self.wds.junctions.index.values == index)[0][0]]  = 1.
        else:
            print('There are no master nodes in the system.')
        return mask

    def signal_mask(self):
        signal_mask = np.zeros(shape=(len(self.wds.junctions),), dtype=np.float32)
        if self.sensor_nodes:
            for index in list(self.sensor_nodes):
                signal_mask[
                    np.where(self.wds.junctions.index.values == index)[0][0]
                    ]   = 1.
        else:
            print('Sensors are not installed.')
        return signal_mask
