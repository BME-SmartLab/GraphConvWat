# -*- coding: utf-8 -*-
import os
import copy
import argparse
from csv import writer
import numpy as np
import dask.array as da
import pandas as pd
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
parser.add_argument('--batch',
                    default = 80,
                    type    = int,
                    help    = "Batch size."
                    )
parser.add_argument('--setmet',
                    default = 'fixrnd',
                    choices = ['spc', 'fixrnd', 'allrnd'],
                    type    = str,
                    help    = "How to setup the transducers."
                    )
parser.add_argument('--model',
                    default = 'orig',
                    type    = str,
                    help    = "Model to use."
                    )
parser.add_argument('--tag',
                    default = 'def',
                    type    = str,
                    help    = "Custom tag."
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
pathToDB    = os.path.join(pathToRoot, 'data', 'db_' + wds_name +'_'+ args.db)
pathToWDS   = os.path.join('water_networks', wds_name+'.inp')
pathToResults   =  os.path.join(
    pathToRoot, 'experiments', 'relative_error'+'-'+args.wds+'.csv'
    )

# ----- ----- ----- ----- ----- -----
# Functions
# ----- ----- ----- ----- ----- -----
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

def restore_real_nodal_p(dta_ldr, num_nodes, num_graphs):
    nodal_pressures = np.empty((num_graphs, num_nodes))
    end_idx = 0
    for i, batch in enumerate(tst_ldr):
        batch.to(device)
        p   = metrics._rescale(batch.y).reshape(-1, num_nodes).detach().cpu().numpy()
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
        p   = metrics._rescale(p).reshape(-1, num_nodes).detach().cpu().numpy()
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
    diff    = da.subtract(p, p_hat)
    rel_diff= da.divide(diff, p)
    return rel_diff

# ----- ----- ----- ----- ----- -----
# Loading datasets
# ----- ----- ----- ----- ----- -----
wds = Network(pathToWDS)
G   = get_nx_graph(wds, mode='binary')

run_ids = np.arange(20)+1
obsrats = [.05, .1, .2, .4, .8]
df_list = []
for run_id in run_ids:
    print('Processing {}. run...'.format(run_id))
    seed    = run_id
    for obsrat in obsrats:
        run_stamp   = wds_name+'-'+args.setmet+'-'+str(obsrat)+'-binary-'+args.tag+'-'
        run_stamp   = run_stamp + str(run_id)
        pathToModel = os.path.join(pathToExps, 'models', run_stamp+'.pt')

        reader  = DataReader(pathToDB, n_junc=len(wds.junctions.uid), obsrat=obsrat, seed=seed)
        tst_x, _, _ = reader.read_data(
            dataset = 'tst',
            varname = 'junc_heads',
            rescale = 'standardize',
            cover   = True
            )
        tst_y, _, _ = reader.read_data(
            dataset = 'tst',
            varname = 'junc_heads',
            rescale = 'normalize',
            cover   = False
            )
        _, bias_y, scale_y  = reader.read_data(
            dataset = 'trn',
            varname = 'junc_heads',
            rescale = 'normalize',
            cover   = False
            )
        tst_ldr = build_dataloader(G, tst_x, tst_y, args.batch, shuffle=False)
        metrics = Metrics(bias_y, scale_y, device)
        num_nodes   = len(wds.junctions)
        num_graphs  = len(tst_x)

        Net = load_model()
        model   = Net(np.shape(tst_x)[-1], np.shape(tst_y)[-1]).to(device)
        p       = restore_real_nodal_p(tst_ldr, num_nodes, num_graphs)
        p_hat   = predict_nodal_p_gcn(tst_ldr, num_nodes, num_graphs)
        rel_err = compute_metrics(p, p_hat)
        df  = pd.DataFrame(rel_err.compute()).abs()
        df['obsrat']= obsrat
        df['runid'] = run_id
        df_list.append(df)
df  = pd.concat(df_list, ignore_index=True)
df.to_csv(pathToResults)
