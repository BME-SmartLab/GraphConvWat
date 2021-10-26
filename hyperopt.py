# -*- coding: utf-8 -*-
import argparse
import os
import glob
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import ChebConv
from epynet import Network

from utils.graph_utils import get_nx_graph
from utils.DataReader import DataReader
from utils.Metrics import Metrics
from utils.EarlyStopping import EarlyStopping
from utils.dataloader import build_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)
# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--db',
                    default = 'doe_pumpfed_1',
                    type    = str,
                    help    = "DB.")
parser.add_argument('--setmet',
                    default = 'fixrnd',
                    choices = ['spc', 'fixrnd', 'allrnd'],
                    type    = str,
                    help    = "How to setup the transducers.")
parser.add_argument('--obsrat',
                    default = .8,
                    type    = float,
                    help    = "Observation ratio.")
parser.add_argument('--epoch',
                    default = '2000',
                    type    = int,
                    help    = "Number of epochs.")
parser.add_argument('--batch',
                    default = '200',
                    type    = int,
                    help    = "Batch size.")
parser.add_argument('--lr',
                    default = 0.0003,
                    type    = float,
                    help    = "Learning rate.")
args    = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# Paths
# ----- ----- ----- ----- ----- -----
wds_name    = 'anytown'
pathToRoot  = os.path.dirname(os.path.realpath(__file__))
pathToDB    = os.path.join(pathToRoot, 'data', 'db_' + wds_name +'_'+ args.db)
pathToWDS   = os.path.join('water_networks', wds_name+'.inp')
pathToLog   = os.path.join('experiments', 'hyperparams', 'anytown_ho.pkl')

def objective(trial):
    # ----- ----- ----- ----- ----- -----
    # Functions
    # ----- ----- ----- ----- ----- -----
    def train_one_epoch():
        model.train()
        total_loss  = 0
        for batch in trn_ldr:
            batch   = batch.to(device)
            optimizer.zero_grad()
            out     = model(batch)
            loss    = F.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss  += loss.item() * batch.num_graphs
        return total_loss / len(trn_ldr.dataset)

    # ----- ----- ----- ----- ----- -----
    # Loading trn and vld datasets
    # ----- ----- ----- ----- ----- -----
    wds = Network(pathToWDS)
    adj_mode    = trial.suggest_categorical('adjacency', ['binary', 'weighted', 'logarithmic'])
    G   = get_nx_graph(wds, mode=adj_mode)
    
    reader  = DataReader(pathToDB, n_junc=len(wds.junctions.uid), obsrat=args.obsrat, seed=8)
    trn_x, _, _ = reader.read_data(
        dataset = 'trn',
        varname = 'junc_heads',
        rescale = 'standardize',
        cover   = True
        )
    trn_y, bias_y, scale_y  = reader.read_data(
        dataset = 'trn',
        varname = 'junc_heads',
        rescale = 'normalize',
        cover   = False
        )
    vld_x, _, _ = reader.read_data(
        dataset = 'vld',
        varname = 'junc_heads',
        rescale = 'standardize',
        cover   = True
        )
    vld_y, _, _ = reader.read_data(
        dataset = 'vld',
        varname = 'junc_heads',
        rescale = 'normalize',
        cover   = False
        )
    
    # ----- ----- ----- ----- ----- -----
    # Model definition
    # ----- ----- ----- ----- ----- -----
    class Net2(torch.nn.Module):
        def __init__(self, topo):
            super(Net2, self).__init__()
            self.conv1 = ChebConv(np.shape(trn_x)[-1], topo[0][0], K=topo[0][1])
            self.conv2 = ChebConv(topo[0][0], topo[1][0], K=topo[1][1])
            self.conv3 = ChebConv(topo[1][0], np.shape(trn_y)[-1], K=1, bias=False)

        def forward(self, data):
            x, edge_index, edge_weight  = data.x, data.edge_index, data.weight
            x = F.silu(self.conv1(x, edge_index, edge_weight))
            x = F.silu(self.conv2(x, edge_index, edge_weight))
            x = self.conv3(x, edge_index, edge_weight)
            return torch.sigmoid(x)

    class Net3(torch.nn.Module):
        def __init__(self, topo):
            super(Net3, self).__init__()
            self.conv1 = ChebConv(np.shape(trn_x)[-1], topo[0][0], K=topo[0][1])
            self.conv2 = ChebConv(topo[0][0], topo[1][0], K=topo[1][1])
            self.conv3 = ChebConv(topo[1][0], topo[2][0], K=topo[2][1])
            self.conv4 = ChebConv(topo[2][0], np.shape(trn_y)[-1], K=1, bias=False)

        def forward(self, data):
            x, edge_index, edge_weight  = data.x, data.edge_index, data.weight
            x = F.silu(self.conv1(x, edge_index, edge_weight))
            x = F.silu(self.conv2(x, edge_index, edge_weight))
            x = F.silu(self.conv3(x, edge_index, edge_weight))
            x = self.conv4(x, edge_index, edge_weight)
            return torch.sigmoid(x)

    class Net4(torch.nn.Module):
        def __init__(self, topo):
            super(Net4, self).__init__()
            self.conv1 = ChebConv(np.shape(trn_x)[-1], topo[0][0], K=topo[0][1])
            self.conv2 = ChebConv(topo[0][0], topo[1][0], K=topo[1][1])
            self.conv3 = ChebConv(topo[1][0], topo[2][0], K=topo[2][1])
            self.conv4 = ChebConv(topo[2][0], topo[3][0], K=topo[3][1])
            self.conv5 = ChebConv(topo[3][0], np.shape(trn_y)[-1], K=1, bias=False)

        def forward(self, data):
            x, edge_index, edge_weight  = data.x, data.edge_index, data.weight
            x = F.silu(self.conv1(x, edge_index, edge_weight))
            x = F.silu(self.conv2(x, edge_index, edge_weight))
            x = F.silu(self.conv3(x, edge_index, edge_weight))
            x = F.silu(self.conv4(x, edge_index, edge_weight))
            x = self.conv5(x, edge_index, edge_weight)
            return torch.sigmoid(x)

    n_layers= trial.suggest_int('n_layers', 2, 4)
    topo    = []
    for i in range(n_layers):
        topo.append([
            trial.suggest_int('n_channels_{}'.format(i), 5, 50, step=5),
            trial.suggest_int('filter_size_{}'.format(i), 5, 30, step=5)
            ])
    decay   = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    if n_layers == 2:
        model   = Net2(topo).to(device)
        optimizer   = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=decay),
            dict(params=model.conv2.parameters(), weight_decay=decay),
            dict(params=model.conv3.parameters(), weight_decay=0)
            ],
            lr  = args.lr,
            eps = 1e-7
            )
    elif n_layers == 3:
        model   = Net3(topo).to(device)
        optimizer   = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=decay),
            dict(params=model.conv2.parameters(), weight_decay=decay),
            dict(params=model.conv3.parameters(), weight_decay=decay),
            dict(params=model.conv4.parameters(), weight_decay=0)
            ],
            lr  = args.lr,
            eps = 1e-7
            )
    elif n_layers == 4:
        model   = Net4(topo).to(device)
        optimizer   = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=decay),
            dict(params=model.conv2.parameters(), weight_decay=decay),
            dict(params=model.conv3.parameters(), weight_decay=decay),
            dict(params=model.conv4.parameters(), weight_decay=decay),
            dict(params=model.conv5.parameters(), weight_decay=0)
            ],
            lr  = args.lr,
            eps = 1e-7
            )

    # ----- ----- ----- ----- ----- -----
    # Training
    # ----- ----- ----- ----- ----- -----
    trn_ldr = build_dataloader(G, trn_x, trn_y, args.batch, shuffle=True)
    vld_ldr = build_dataloader(G, vld_x, vld_y, len(vld_x), shuffle=False)
    metrics = Metrics(bias_y, scale_y, device)
    estop   = EarlyStopping(min_delta=.000001, patience=50)
    results = pd.DataFrame(columns=['trn_loss', 'vld_loss'])
    header  = ''.join(['{:^15}'.format(colname) for colname in results.columns])
    header  = '{:^5}'.format('epoch') + header
    best_vld_loss   = np.inf
    for epoch in range(0, args.epoch):
        trn_loss    = train_one_epoch()
        model.eval()
        tot_vld_loss    = 0
        for batch in vld_ldr:
            batch   = batch.to(device)
            out     = model(batch)
            vld_loss        = F.mse_loss(out, batch.y)
            tot_vld_loss    += vld_loss.item() * batch.num_graphs
        vld_loss    = tot_vld_loss / len(vld_ldr.dataset)
    
        if estop.step(torch.tensor(vld_loss)):
            print('Early stopping...')
            break
    return estop.best

if __name__ == '__main__':
    sampler = optuna.samplers.TPESampler(n_startup_trials=50, n_ei_candidates=5, multivariate=True)
    study   = optuna.create_study(direction='minimize',
        study_name  = 'v4',
        sampler = sampler,
        storage = 'sqlite:///experiments/hyperparams/anytown_ho-'+str(args.obsrat)+'.db'
        )
    study.optimize(objective, n_trials=300)
