# -*- coding: utf-8 -*-
import argparse
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import ChebConv
from epynet import Network

from utils.graph_utils import get_nx_graph, get_sensitivity_matrix
from utils.DataReader import DataReader
from utils.SensorInstaller import SensorInstaller
from utils.Metrics import Metrics
from utils.EarlyStopping import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----- ----- ----- ----- ----- -----
# Command line arguments
# ----- ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--wds',
                    default = 'anytown',
                    type    = str,
                    help    = "Water distribution system.")
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
                    default = .1,
                    type    = float,
                    help    = "Observation ratio.")
parser.add_argument('--adj',
                    default = 'binary',
                    choices = ['binary', 'weighted', 'logarithmic', 'pruned'],
                    type    = str,
                    help    = "Type of adjacency matrix.")
parser.add_argument('--deploy',
                    default = 'random',
                    choices = ['random', 'dist', 'hydrodist', 'hds'],
                    type    = str,
                    help    = "Method of sensor deployment.")
parser.add_argument('--epoch',
                    default = '1',
                    type    = int,
                    help    = "Number of epochs.")
parser.add_argument('--batch',
                    default = '40',
                    type    = int,
                    help    = "Batch size.")
parser.add_argument('--lr',
                    default = 0.0003,
                    type    = float,
                    help    = "Learning rate.")
parser.add_argument('--decay',
                    default = 0.000006,
                    type    = float,
                    help    = "Weight decay.")
parser.add_argument('--tag',
                    default = 'def',
                    type    = str,
                    help    = "Custom tag.")
parser.add_argument('--deterministic',
                    action  = "store_true",
                    help    = "Setting random seed for sensor placement.")
args    = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# Paths
# ----- ----- ----- ----- ----- -----
wds_name    = args.wds
pathToRoot  = os.path.dirname(os.path.realpath(__file__))
pathToDB    = os.path.join(pathToRoot, 'data', 'db_' + wds_name +'_'+ args.db)
pathToExps  = os.path.join(pathToRoot, 'experiments')
pathToLogs  = os.path.join(pathToExps, 'logs')
run_id  = 1
logs    = [f for f in glob.glob(os.path.join(pathToLogs, '*.csv'))]
run_stamp   = wds_name+'-'+args.setmet+'-'+str(args.obsrat)+'-'+args.adj+'-'+args.tag+'-'
while os.path.join(pathToLogs, run_stamp + str(run_id)+'.csv') in logs:
    run_id  += 1
run_stamp   = run_stamp + str(run_id)
pathToLog   = os.path.join(pathToLogs, run_stamp+'.csv')
pathToModel = os.path.join(pathToExps, 'models', run_stamp+'.pt')
pathToMeta  = os.path.join(pathToExps, 'models', run_stamp+'_meta.csv')
pathToWDS   = os.path.join('water_networks', wds_name+'.inp')

# ----- ----- ----- ----- ----- -----
# Saving hyperparams
# ----- ----- ----- ----- ----- -----
hyperparams = {
        'db': args.db,
        'setmet': args.setmet,
        'obsrat': args.obsrat,
        'adj': args.adj,
        'epoch': args.epoch,
        'batch': args.batch,
        'lr': args.lr,
        }
hyperparams = pd.Series(hyperparams)
hyperparams.to_csv(pathToMeta, header=False)

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

def eval_metrics(dataloader):
    model.eval()
    n   = len(dataloader.dataset)
    tot_loss        = 0
    tot_rel_err     = 0
    tot_rel_err_obs = 0
    tot_rel_err_hid = 0
    for batch in dataloader:
        batch   = batch.to(device)
        out     = model(batch)
        loss    = F.mse_loss(out, batch.y)
        rel_err = metrics.rel_err(out, batch.y)
        rel_err_obs = metrics.rel_err(
            out,
            batch.y,
            batch.x[:, -1].type(torch.bool)
            )
        rel_err_hid = metrics.rel_err(
            out,
            batch.y,
            ~batch.x[:, -1].type(torch.bool)
            )
        tot_loss        += loss.item() * batch.num_graphs
        tot_rel_err     += rel_err.item() * batch.num_graphs
        tot_rel_err_obs += rel_err_obs.item() * batch.num_graphs
        tot_rel_err_hid += rel_err_hid.item() * batch.num_graphs
    loss        = tot_loss / n
    rel_err     = tot_rel_err / n
    rel_err_obs = tot_rel_err_obs / n
    rel_err_hid = tot_rel_err_hid / n
    return loss, rel_err, rel_err_obs, rel_err_hid

# ----- ----- ----- ----- ----- -----
# Loading trn and vld datasets
# ----- ----- ----- ----- ----- -----
wds = Network(pathToWDS)
G   = get_nx_graph(wds, mode=args.adj)

if args.deterministic:
    seed    = run_id
else:
    seed    = None

sensor_budget   = int(len(wds.junctions) * args.obsrat)
print('Deploying {} sensors...\n'.format(sensor_budget))

sensor_shop = SensorInstaller(wds)

if args.deploy == 'random':
    sensor_shop.deploy_by_random(
            sensor_budget   = sensor_budget,
            seed            = seed
            )
elif args.deploy == 'dist':
    sensor_shop.deploy_by_shortest_path(
            sensor_budget   = sensor_budget,
            weight_by       = 'length'
            )
elif args.deploy == 'hydrodist':
    sensor_shop.deploy_by_shortest_path(
            sensor_budget   = sensor_budget,
            weight_by       = 'iweight'
            )
elif args.deploy == 'hds':
    print('Calculating nodal sensitivity to demand change...\n')
    ptb = np.max(wds.junctions.basedemand) / 100
    S   = get_sensitivity_matrix(wds, ptb)
    sensor_shop.deploy_by_shortest_path_with_sensitivity(
            sensor_budget       = sensor_budget,
            sensitivity_matrix  = S,
            weight_by           = 'iweight'
            )
else:
    print('Sensor deployment technique is unknown.\n')
    raise

reader  = DataReader(
            pathToDB,
            n_junc  = len(wds.junctions),
            signal_mask = sensor_shop.signal_mask()
            )
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

if args.wds == 'anytown':
    from model.anytown import ChebNet as Net
elif args.wds == 'ctown':
    from model.ctown import ChebNet as Net
elif args.wds == 'richmond':
    from model.richmond import ChebNet as Net
else:
    print('Water distribution system is unknown.\n')
    raise

model = Net(np.shape(trn_x)[-1], np.shape(trn_y)[-1]).to(device)
optimizer   = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=args.decay),
    dict(params=model.conv2.parameters(), weight_decay=args.decay),
    dict(params=model.conv3.parameters(), weight_decay=args.decay),
    dict(params=model.conv4.parameters(), weight_decay=0)
    ],
    lr  = args.lr,
    eps = 1e-7
    )

# ----- ----- ----- ----- ----- -----
# Training
# ----- ----- ----- ----- ----- -----
trn_ldr = build_dataloader(G, trn_x, trn_y, args.batch, shuffle=True)
vld_ldr = build_dataloader(G, vld_x, vld_y, args.batch, shuffle=False)
metrics = Metrics(bias_y, scale_y, device)
estop   = EarlyStopping(min_delta=.00001, patience=30)
results = pd.DataFrame(columns=[
    'trn_loss', 'vld_loss', 'vld_rel_err', 'vld_rel_err_o', 'vld_rel_err_h'
    ])
header  = ''.join(['{:^15}'.format(colname) for colname in results.columns])
header  = '{:^5}'.format('epoch') + header
best_vld_loss   = np.inf
for epoch in range(0, args.epoch):
    trn_loss    = train_one_epoch()
    vld_loss, vld_rel_err, vld_rel_err_obs, vld_rel_err_hid = eval_metrics(vld_ldr)
    new_results = pd.Series({
        'trn_loss'      : trn_loss,
        'vld_loss'      : vld_loss,
        'vld_rel_err'   : vld_rel_err,
        'vld_rel_err_o' : vld_rel_err_obs,
        'vld_rel_err_h' : vld_rel_err_hid
        })
    results = results.append(new_results, ignore_index=True)
    if epoch % 20 == 0:
        print(header)
    values  = ''.join(['{:^15.6f}'.format(value) for value in new_results.values])
    print('{:^5}'.format(epoch) + values)
    if vld_loss < best_vld_loss:
        best_vld_loss   = vld_loss
        torch.save(model.state_dict(), pathToModel)
    if estop.step(torch.tensor(vld_loss)):
        print('Early stopping...')
        break
results.to_csv(pathToLog)

# ----- ----- ----- ----- ----- -----
# Testing
# ----- ----- ----- ----- ----- -----
if best_vld_loss is not np.inf:
    print('Testing...\n')
    del trn_ldr, vld_ldr, trn_x, trn_y, vld_x, vld_y
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
    tst_ldr = build_dataloader(G, tst_x, tst_y, args.batch, shuffle=False)
    model.load_state_dict(torch.load(pathToModel))
    model.eval()
    tst_loss, tst_rel_err, tst_rel_err_obs, tst_rel_err_hid = eval_metrics(tst_ldr)
    results = pd.Series({
        'tst_loss'      : tst_loss,
        'tst_rel_err'   : tst_rel_err,
        'tst_rel_err_o' : tst_rel_err_obs,
        'tst_rel_err_h' : tst_rel_err_hid
        })
    results.to_csv(pathToLog[:-4]+'_tst.csv')
