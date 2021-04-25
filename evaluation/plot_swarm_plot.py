# -*- coding: utf-8 -*-
import argparse
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser  = argparse.ArgumentParser()
parser.add_argument(
    '--obsrat',
    default = 0.05,
    type    = float
    )
parser.add_argument(
    '--ymin',
    default = 0.,
    type    = float
    )
parser.add_argument(
    '--ymax',
    default = .003,
    type    = float
    )
parser.add_argument(
    '--drop',
    default = 5,
    type    = int
    )
parser.add_argument(
    '--runid',
    default = 'v4',
    type    = str
    )
parser.add_argument(
    '--savepdf',
    action  = 'store_true'
    )
args    = parser.parse_args()

db_path = 'sqlite:///../experiments/hyperparams/anytown_ho-'+str(args.obsrat)+'.db'
study   = optuna.load_study(
    study_name  = args.runid,
    storage = db_path
    )
df  = study.trials_dataframe()
df.drop(index=df.nlargest(args.drop, 'value').index, inplace=True)

palette = {'binary': 'grey', 'weighted': 'lightgrey', 'logarithmic': 'lightgrey'}
sns.set_theme(style='whitegrid')
fig = sns.violinplot(
    data= df,
    x   = 'params_adjacency',
    y   = 'value',
    inner   = None,
    order   = ['binary', 'weighted', 'logarithmic'],
    palette = palette
    )
fig = sns.swarmplot(
    data= df,
    x   = 'params_adjacency',
    y   = 'value',
    hue = 'params_n_layers',
    order   = ['binary', 'weighted', 'logarithmic']
    )
fig.set_xlabel('adjacency matrix')
fig.set_ylabel('loss')
fig.set_ylim([args.ymin, args.ymax])
plt.legend(loc='center left', title='layers')

if args.savepdf:
    fmt     = 'pdf'
    fname   = 'swarm-'+str(args.obsrat)+'.'+fmt
    fig = fig.get_figure()
    fig.savefig(fname, format=fmt, bbox_inches='tight')
else:
    plt.show()
