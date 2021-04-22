# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from scipy import sparse
import networkx as nx
from epynet import Network
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import torch_geometric as pyg

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
parser.add_argument(
    '--adj',
    default = 'binary',
    choices = ['binary', 'weighted', 'logarithmic', 'pruned'],
    type    = str,
    help    = 'Type of adjacency matrix.'
    )
parser.add_argument(
    '--savepdf',
    action  = 'store_true'
    )
args    = parser.parse_args()

wds_path    = os.path.join('..', 'water_networks', args.wds+'.inp')
wds = Network(wds_path)
wds.solve()

G   = get_nx_graph(wds, mode=args.adj)
A   = np.array(nx.adjacency_matrix(G).toarray())
#D   = np.diag(np.sum(A, axis=0)**-.5)

D   = np.sum(A, axis=0)
D[D != 0]   = D[D != 0]**-.5
D   = np.diag(D)

I   = np.eye(np.shape(A)[0])
L   = I-np.dot(np.dot(D, A), D)
lambda_max  = np.linalg.eigvalsh(L).max()
L_tilde = 2.*L/lambda_max-I

# Sanity check with PyG utils
#pyg_graph   = pyg.utils.from_networkx(G)
#L   = pyg.utils.get_laplacian(
#    pyg_graph.edge_index,
#    edge_weight     = pyg_graph.weight,
#    normalization   = 'sym'
#    )
#L   = pyg.utils.to_dense_adj(L[0], edge_attr=L[1]).squeeze(0).numpy()
#lambda_max  = np.linalg.eigvalsh(L).max()
#L_tilde = 2.*L/lambda_max-I

print('Conditional number of the Laplacian is {:f}.'.format(np.linalg.cond(L_tilde, 2)))
#print('Conditional number of the square of the Laplacian is {:f}.'.format(np.linalg.cond(np.power(L_tilde, 2), 2)))
#print('Conditional number of the cube of the Laplacian is {:f}.'.format(np.linalg.cond(np.power(L_tilde, 3), 2)))

# Coloring only nonzero elements
L_star  = np.abs(L_tilde.copy())
vmin    = L_star[L_star.nonzero()].min()
vmax    = L_star.max()
L_star[L_star == 0] = np.nan

figsize = (8,6)
axlabel = "Node number"
barlabel = "Value of the Laplacian"
fig, ax = plt.subplots(figsize=figsize)
hmap    = sns.heatmap(L_star,
    norm    = LogNorm(vmin = vmin, vmax = vmax),
    cmap    = 'viridis',
    cbar_kws= {'label': barlabel},
    square  = True,
    ax      = ax
    )
hmap.set_xlabel(axlabel)
hmap.set_ylabel(axlabel)

# ----- ----- ----- ----- ----- -----
# Diagram export
# ----- ----- ----- ----- ----- -----
if args.savepdf:
    fmt     = 'pdf'
    fname   = 'adj-'+args.wds+'-'+args.adj+'.'+fmt
    fig.savefig(fname, format=fmt, bbox_inches='tight')
else:
    plt.show()

## Self-importance
#L_tilde=L
#center  = L_tilde.diagonal()
#radius  = np.sum(L_tilde, axis=0)-center
#ratio   = -center/radius
#sns.histplot(ratio, log_scale=True)
##plt.bar(np.arange(len(ratio)), ratio, log=True)
#plt.show()
