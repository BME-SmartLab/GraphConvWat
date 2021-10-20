# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx

def get_nx_graph(wds, mode='binary'):
    junc_list = []
    for junction in wds.junctions:
        junc_list.append(junction.index)
    G = nx.Graph()
    if mode == 'binary':
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=1., length=pipe.length)
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1., length=0.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1., length=0.)
    elif mode == 'weighted':
        max_weight  = 0
        max_iweight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                weight  = ((pipe.diameter*3.281)**4.871 * pipe.roughness**1.852) / (4.727*pipe.length*3.281)
                iweight = weight**-1
                G.add_edge(pipe.from_node.index, pipe.to_node.index,
                        weight  = weight,
                        iweight = iweight,
                        length  = pipe.length
                        )
                if weight > max_weight:
                    max_weight = weight
                if iweight > max_iweight:
                    max_iweight = iweight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
            d['iweight'] /= max_iweight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1., length=0.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1., length=0.)
    elif mode == 'logarithmic':
        max_weight = 0
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                weight  = np.log10(((pipe.diameter*3.281)**4.871 * pipe.roughness**1.852) / (4.727*pipe.length*3.281))
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=float(weight), length=pipe.length)
                if weight > max_weight:
                    max_weight = weight
        for (_,_,d) in G.edges(data=True):
            d['weight'] /= max_weight
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=1., length=0.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=1., length=0.)
    elif mode == 'pruned':
        for pipe in wds.pipes:
            if (pipe.from_node.index in junc_list) and (pipe.to_node.index in junc_list):
                G.add_edge(pipe.from_node.index, pipe.to_node.index, weight=0., length=pipe.length)
        for pump in wds.pumps:
            if (pump.from_node.index in junc_list) and (pump.to_node.index in junc_list):
                G.add_edge(pump.from_node.index, pump.to_node.index, weight=0., length=0.)
        for valve in wds.valves:
            if (valve.from_node.index in junc_list) and (valve.to_node.index in junc_list):
                G.add_edge(valve.from_node.index, valve.to_node.index, weight=0., length=0.)
    return G

def get_sensitivity_matrix(wds, perturbance):
    wds.solve()
    base_demands    = wds.junctions.basedemand
    base_heads      = wds.junctions.head
    S   = np.empty(shape=(len(wds.junctions), len(wds.junctions)), dtype=np.float64)
    for i, junc in enumerate(wds.junctions):
        wds.junctions.basedemand    = base_demands
        junc.basedemand += perturbance
        wds.solve()
        S[i, :] = (wds.junctions.head-base_heads) / base_heads
    return S
