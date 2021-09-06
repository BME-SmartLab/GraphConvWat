# -*- coding: utf-8 -*-
import os
import argparse
from csv import writer
import numpy as np
import dask.array as da
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from epynet import Network

from utils.graph_utils import get_nx_graph
from utils.DataReader import DataReader
from utils.Metrics import Metrics
from utils.MeanPredictor import MeanPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds',
                    default = 'anytown',
                    type    = str,
                    help    = "Water distribution system."
                    )
parser.add_argument('--setmet',
                    default = 'fixrnd',
                    choices = ['spc', 'fixrnd', 'allrnd'],
                    type    = str,
                    help    = "How to setup the transducers."
                    )
parser.add_argument('--obsrat',
                    default = .05,
                    type    = float,
                    help    = "Observation ratio."
                    )
parser.add_argument('--batch',
                    default = 80,
                    type    = int,
                    help    = "Batch size."
                    )
parser.add_argument('--adj',
                    default = 'binary',
                    choices = ['binary', 'weighted', 'logarithmic', 'pruned'],
                    type    = str,
                    help    = "Type of adjacency matrix."
                    )
parser.add_argument('--model',
                    default = 'orig',
                    choices = ['orig', 'naive', 'gcn', 'interp'],
                    type    = str,
                    help    = "Model to use."
                    )
parser.add_argument('--metricsdb',
                    default = 'Taylor_metrics',
                    type    = str,
                    help    = "Name of the metrics database."
                    )
parser.add_argument('--tag',
                    default = 'def',
                    type    = str,
                    help    = "Custom tag."
                    )
parser.add_argument('--runid',
                    default = 1,
                    type    = int,
                    help    = "Number of the model."
                    )
parser.add_argument('--db',
                    default = 'doe_pumpfed_1',
                    type    = str,
                    help    = "DB.")
args    = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# Paths
# ----- ----- ----- ----- ----- -----
wds_name    = args.wds
pathToRoot  = os.path.dirname(os.path.realpath(__file__))
pathToExps  = os.path.join(pathToRoot, 'experiments')
pathToLogs  = os.path.join(pathToExps, 'logs')
run_id      = args.runid
run_stamp   = wds_name+'-'+args.setmet+'-'+str(args.obsrat)+'-'+args.adj+'-'+args.tag+'-'
run_stamp   = run_stamp + str(run_id)
pathToDB    = os.path.join(pathToRoot, 'data', 'db_' + wds_name +'_'+ args.db)
pathToModel = os.path.join(pathToExps, 'models', run_stamp+'.pt')
pathToMeta  = os.path.join(pathToExps, 'models', run_stamp+'_meta.csv')
pathToWDS   = os.path.join('water_networks', wds_name+'.inp')
pathToResults   =  os.path.join(pathToRoot, 'experiments', args.metricsdb + '.csv')

# ----- ----- ----- ----- ----- -----
# Functions
# ----- ----- ----- ----- ----- -----
def build_dataloader(G, set_x, set_y, batch_size, shuffle):
    data    = []
    for x, y in zip(set_x, set_y):
        graph   = from_networkx(G)
        graph.x = torch.Tensor(x)
        graph.y = torch.Tensor(y)
        data.append(graph)
    loader  = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader

def restore_real_nodal_p(dta_ldr, num_nodes, num_graphs):
    nodal_pressures = np.empty((num_graphs, num_nodes))
    end_idx = 0
    for i, batch in enumerate(tst_ldr):
        batch.to(device)
        p   = metrics_nrm._rescale(batch.y).reshape(-1, num_nodes).detach().cpu().numpy()
        nodal_pressures[end_idx:end_idx+batch.num_graphs, :]    = p
        end_idx += batch.num_graphs
    return da.array(nodal_pressures)

def predict_nodal_p_gcn(dta_ldr, num_nodes, num_graphs):
    model.load_state_dict(torch.load(pathToModel, map_location=torch.device(device)))
    model.eval()
    nodal_pressures = np.empty((num_graphs, num_nodes))
    end_idx = 0
    for i, batch in enumerate(tst_ldr):
        batch.to(device)
        p   = model(batch)
        p   = metrics_nrm._rescale(p).reshape(-1, num_nodes).detach().cpu().numpy()
        nodal_pressures[end_idx:end_idx+batch.num_graphs, :]    = p
        end_idx += batch.num_graphs
    return da.array(nodal_pressures)

def predict_nodal_p_naive(dta_ldr, num_nodes, num_graphs):
    model   = MeanPredictor(device)
    nodal_pressures = np.empty((num_graphs, num_nodes))
    end_idx = 0
    for i, batch in enumerate(tst_ldr):
        batch.to(device)
        p   = model.pred(batch.y, batch.x[:, -1].type(torch.bool))
        p   = metrics_nrm._rescale(p).reshape(-1, num_nodes).detach().cpu().numpy()
        nodal_pressures[end_idx:end_idx+batch.num_graphs, :]    = p
        end_idx += batch.num_graphs
    return da.array(nodal_pressures)

def predict_nodal_p_interpolation(G, X):
    L   = nx.linalg.laplacianmatrix.laplacian_matrix(G).todense()
    S   = np.array(L)
    F   = X[:,:,0].copy()
    X_h = X[:,:,0].copy()
    flag_observed   = X[:,:,1][0,:]

    k   = 0
    for i, flag in enumerate(flag_observed):
        if flag:
            pivot   = np.copy(S[:,i])
            S[:,i]  = S[:,k]
            S[:,k]  = pivot

            S[i,:]  = S[k,:]
            S[k,:]  = pivot.T

            pivot   = np.copy(F[:,i])
            F[:,i]  = F[:,k]
            F[:,k]  = pivot

            k   += 1

    F   = F[:, :k]
    S_2 = S[:k, k:]
    S_3 = S[k:, k:]
    try:
        S_3_inv = np.linalg.inv(S_3)
    except:
        S_3_inv = np.linalg.pinv(S_3)

    F_h = np.dot(S_3_inv, np.dot(S_2.T, F.T))
    mu  = - np.sum(F_h, axis=0) / np.sum(np.dot(S_3_inv, np.dot(S_2.T, np.ones_like(F.T))), axis=0)
    F_h = np.dot(S_3_inv, np.dot(S_2.T, F.T+mu)).T

    k   = 0
    for i, flag in enumerate(flag_observed):
        if not flag:
            X_h[:,i]    = F_h[:,k]
            k           += 1

    nodal_pressures = da.add(da.multiply(da.array(X_h), scale_std), bias_std)
    return nodal_pressures

def load_model():
    if args.wds == 'anytown':
        from model.anytown import ChebNet as Net
    elif args.wds == 'ctown':
        from model.ctown import ChebNet as Net
    elif args.wds == 'richmond':
        from model.richmond import ChebNet as Net
    else:
        print('Water distribution system is unknown.\n')
        raise
    return Net

def compute_metrics(p, p_hat):
    msec    = da.multiply(p-p.mean(), p_hat-p_hat.mean()).mean()
    sigma   = da.sqrt(da.square(p_hat-p_hat.mean()).mean())
    return msec, sigma

# ----- ----- ----- ----- ----- -----
# Loading datasets
# ----- ----- ----- ----- ----- -----
wds = Network(pathToWDS)
G   = get_nx_graph(wds, mode=args.adj)
seed    = run_id
reader  = DataReader(pathToDB, n_junc=len(wds.junctions.uid), obsrat=args.obsrat, seed=seed)
tst_x, bias_std, scale_std  = reader.read_data(
    dataset = 'tst',
    varname = 'junc_heads',
    rescale = 'standardize',
    cover   = True
    )
tst_y, bias_nrm, scale_nrm  = reader.read_data(
    dataset = 'tst',
    varname = 'junc_heads',
    rescale = 'normalize',
    cover   = False
    )
tst_ldr = build_dataloader(G, tst_x, tst_y, args.batch, shuffle=False)
metrics_nrm = Metrics(bias_nrm, scale_nrm, device)
metrics_std = Metrics(bias_std, scale_std, device)
num_nodes   = len(wds.junctions)
num_graphs  = len(tst_x)

# ----- ----- ----- ----- ----- -----
# Compute metrics
# ----- ----- ----- ----- ----- -----
run_stamp   = run_stamp+'-'+args.model
print(run_stamp)
p   = restore_real_nodal_p(tst_ldr, num_nodes, num_graphs)
if args.model == 'orig':
    p_hat   = p
elif args.model == 'naive':
    p_hat   = predict_nodal_p_naive(tst_ldr, num_nodes, num_graphs)
elif args.model == 'gcn':
    Net = load_model()
    model   = Net(np.shape(tst_x)[-1], np.shape(tst_y)[-1]).to(device)
    p_hat   = predict_nodal_p_gcn(tst_ldr, num_nodes, num_graphs)
elif args.model == 'interp':
    p_hat   = predict_nodal_p_interpolation(G, tst_x)
msec, sigma = compute_metrics(p, p_hat)

# ----- ----- ----- ----- ----- -----
# Write metrics
# ----- ----- ----- ----- ----- -----
results = [run_stamp, msec.compute(), sigma.compute()]
with open(pathToResults, 'a+') as fout:
    csv_writer  = writer(fout)
    csv_writer.writerow(results)
