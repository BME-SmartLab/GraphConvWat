# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import collections  as mc
import matplotlib.pyplot as plt

from epynet import Network

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
    '--obsrat',
    default = .0,
    type    = float
    )
parser.add_argument(
    '--seed',
    default = None,
    type    = int
    )
parser.add_argument(
    '--savepdf',
    action  = 'store_true'
    )
args    = parser.parse_args()

wds = Network(os.path.join('..', 'water_networks', args.wds+'.inp'))
wds.solve()

def get_node_df(elements, get_head=False):
    data = []
    for junc in elements:
        ser = pd.Series({
            'uid': junc.uid,
            'x': junc.coordinates[0],
            'y': junc.coordinates[1],
        })
        if get_head:
            ser['head'] = junc.head
        data.append(ser)
    data = pd.DataFrame(data)
    if get_head:
        data['head'] = (data['head'] - data['head'].min()) / (data['head'].max()-data['head'].min())
    return data

def get_elem_df(elements, nodes):
    data = []
    for elem in elements:
        ser = pd.Series({
            'uid': elem.uid,
            'x1': nodes.loc[nodes['uid'] == elem.from_node.uid, 'x'].values,
            'y1': nodes.loc[nodes['uid'] == elem.from_node.uid, 'y'].values,
            'x2': nodes.loc[nodes['uid'] == elem.to_node.uid, 'x'].values,
            'y2': nodes.loc[nodes['uid'] == elem.to_node.uid, 'y'].values,
        })
        data.append(ser)
    df = pd.DataFrame(data)
    df['x1'] = df['x1'].str[0]
    df['y1'] = df['y1'].str[0]
    df['x2'] = df['x2'].str[0]
    df['y2'] = df['y2'].str[0]
    df['center_x'] = (df['x1']+df['x2']) / 2
    df['center_y'] = (df['y1']+df['y2']) / 2
    df['orient'] = np.degrees(np.arctan((df['y2']-df['y1'])/(df['x2']-df['x1']))) + 90
    return df

def build_lc_from(df):
    line_collection = []
    for elem_id in df['uid']:
        line_collection.append([
            (df.loc[df['uid'] == elem_id, 'x1'].values[0],
                 df.loc[df['uid'] == elem_id, 'y1'].values[0]),
            (df.loc[df['uid'] == elem_id, 'x2'].values[0],
                 df.loc[df['uid'] == elem_id, 'y2'].values[0])
        ])
    return line_collection
nodes = get_node_df(wds.nodes, get_head=True)
juncs = get_node_df(wds.junctions, get_head=True)
tanks = get_node_df(wds.tanks)
reservoirs = get_node_df(wds.reservoirs)
pipes = get_elem_df(wds.pipes, nodes)
pumps = get_elem_df(wds.pumps, nodes)
valves= get_elem_df(wds.valves, nodes)
pipe_collection = build_lc_from(pipes)
pump_collection = build_lc_from(pumps)
valve_collection = build_lc_from(valves)

fig, ax = plt.subplots()
lc = mc.LineCollection(pipe_collection, linewidths=1, color='k')
ax.add_collection(lc)
lc = mc.LineCollection(pump_collection, linewidths=1, color='k')
ax.add_collection(lc)
lc = mc.LineCollection(valve_collection, linewidths=1, color='k')
ax.add_collection(lc)

cmap    = plt.get_cmap('plasma')
juncs['head']   *= 1.5 # emphasize differences in graphical abstract
colors  = []
signal  = []
for _, junc in juncs.iterrows():
    np.random.seed(args.seed)
    if np.random.rand() < args.obsrat:
        color = cmap(junc['head'])
        signal.append(junc['head'])
    else:
        color = (1.,1.,1.,1.)
        signal.append(np.nan)
    colors.append(color)
    ax.plot(junc['x'], junc['y'], 'ko', mfc=color, mec='k', ms=15)
for _, tank in tanks.iterrows():
    ax.plot(tank['x'], tank['y'], marker=7, mfc='k', mec='k', ms=10)
for _, reservoir in reservoirs.iterrows():
    ax.plot(reservoir['x'], reservoir['y'], marker='o', mfc='k', mec='k', ms=7)
ax.plot(pumps['center_x'], pumps['center_y'], 'ko', ms=13, mfc='white')
for _, pump in pumps.iterrows():
    ax.plot(pump['center_x'], pump['center_y'],
        marker=(3, 0, pump['orient']),
        color='k',
        ms=10
        )
ax.autoscale()
ax.axis('off')
plt.tight_layout()

# ----- ----- ----- ----- ----- -----
# Diagram export
# ----- ----- ----- ----- ----- -----
if args.savepdf:
    fmt     = 'pdf'
    fname   = 'topo-'+args.wds+'.'+fmt
    fig.savefig(fname, format=fmt, bbox_inches='tight')
else:
    plt.show()

signal  = np.expand_dims(np.array(signal).T, axis=1)
ax  = sns.heatmap(
    signal,
    vmin    = 0.,
    vmax    = 1.,
    cmap    = 'plasma',
    cbar    = False,
    square  = True,
    linewidth   = 1,
    linecolor   = 'k',
    xticklabels = False,
    yticklabels = False,
    )

if args.savepdf:
    fmt     = 'pdf'
    fname   = 'signal-'+args.wds+'.'+fmt
    fig.savefig(fname, format=fmt, bbox_inches='tight')
else:
    plt.show()
