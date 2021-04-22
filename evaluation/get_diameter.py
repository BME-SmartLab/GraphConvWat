# -*- coding: utf-8 -*-
import os
import argparse
import networkx as nx
from epynet import Network

import sys
sys.path.insert(0, os.path.join('..', 'utils'))
from graph_utils import get_nx_graph

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument(
    '--wds',
    default = 'anytown',
    type    = str
    )
args    = parser.parse_args()

wds_path    = os.path.join('..', 'water_networks', args.wds+'.inp')
wds = Network(wds_path)

G   = get_nx_graph(wds)
d_G = 0
measure = nx.algorithms.distance_measures.diameter
if nx.number_connected_components(G) == 1:
    d_G = measure(G)
else:
    print('Found disconnected components. Check whether EPANET simulation works.')
    for component in nx.connected_components(G):
        d_G += measure(G.subgraph(component))

print('The diameter of {} is {}.'.format(args.wds, d_G))
