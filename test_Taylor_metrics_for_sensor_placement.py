# -*- coding: utf-8 -*-
import os
import argparse
import copy
from csv import writer
import numpy as np
import dask.array as da
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from epynet import Network

from utils.graph_utils import get_nx_graph, get_sensitivity_matrix
from utils.DataReader import DataReader
from utils.SensorInstaller import SensorInstaller
from utils.Metrics import Metrics
from utils.MeanPredictor import MeanPredictor
from utils.baselines import interpolated_regularization
from utils.dataloader import build_dataloader

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
parser.add_argument('--deploy',
                    default = 'random',
                    choices = ['random', 'dist', 'hydrodist', 'hds'],
                    type    = str,
                    help    = "Method of sensor deployment.")
parser.add_argument('--obsrat',
                    default = .1,
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
run_stamp   = wds_name+'-'+args.deploy+'-'+str(args.obsrat)+'-'+args.adj+'-'+args.tag+'-'
run_stamp   = run_stamp + str(run_id)
pathToDB    = os.path.join(pathToRoot, 'data', 'db_' + wds_name +'_'+ args.db)
pathToModel = os.path.join(pathToExps, 'models', run_stamp+'.pt')
pathToMeta  = os.path.join(pathToExps, 'models', run_stamp+'_meta.csv')
pathToSens  = os.path.join(pathToExps, 'models', run_stamp+'_sensor_nodes.csv')
pathToWDS   = os.path.join('water_networks', wds_name+'.inp')
pathToResults   =  os.path.join(pathToRoot, 'experiments', args.metricsdb + '.csv')

# ----- ----- ----- ----- ----- -----
# Functions
# ----- ----- ----- ----- ----- -----
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

sensor_shop = SensorInstaller(wds)
sensor_nodes= np.loadtxt(pathToSens, dtype=np.int32)
sensor_shop.set_sensor_nodes(sensor_nodes)

reader  = DataReader(
            pathToDB,
            n_junc  = len(wds.junctions),
            signal_mask = sensor_shop.signal_mask(),
            node_order  = np.array(list(G.nodes))-1
            )
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
num_nodes   = len(wds.junctions)
num_graphs  = len(tst_x)

# ----- ----- ----- ----- ----- -----
# Compute metrics
# ----- ----- ----- ----- ----- -----
run_stamp   = run_stamp+'-'+'gcn'
print(run_stamp)
p   = restore_real_nodal_p(tst_ldr, num_nodes, num_graphs)

Net     = load_model()
model   = Net(np.shape(tst_x)[-1], np.shape(tst_y)[-1]).to(device)
p_hat   = predict_nodal_p_gcn(tst_ldr, num_nodes, num_graphs)

msec, sigma = compute_metrics(p, p_hat)

# ----- ----- ----- ----- ----- -----
# Write metrics
# ----- ----- ----- ----- ----- -----
results = [run_stamp, msec.compute(), sigma.compute()]
with open(pathToResults, 'a+') as fout:
    csv_writer  = writer(fout)
    csv_writer.writerow(results)
