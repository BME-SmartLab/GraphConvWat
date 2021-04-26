# -*- coding: utf-8 -*-
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    '--savepdf',
    action  = 'store_true'
    )
args    = parser.parse_args()

path_to_csv = os.path.join('..', 'experiments', 'relative_error-'+args.wds+'.csv')
df  = pd.read_csv(path_to_csv, index_col=0)
df  = df[df['runid'] == 8]
colnames= df.columns[:-2]
df_list = []
for colname in colnames:
    small_df                    = df[['obsrat']].copy()
    small_df['runid']           = df['runid']
    small_df['Relative error']  = df[colname]
    small_df['node']            = colname
    df_list.append(small_df)
df  = pd.concat(df_list, ignore_index=True)
df['grouper']   = df['obsrat'].astype(str)+'-'+df['node'].astype(str)
df['node']      = df['node'].astype(int)
dta         = df.groupby('grouper').mean()
dta['node'] = dta['node'].astype(str)

## ___Quick & dirty___
#df  = df.groupby('obsrat').mean()
#df.drop('runid', axis=1, inplace=True)
#dta = df.T
#fig, ax = plt.subplots()
#plot = sns.ecdfplot(data=dta)

sns.set_style('whitegrid')
fig, ax = plt.subplots()
plot = sns.ecdfplot(data=dta, x='Relative error', stat='count', hue='obsrat', palette='colorblind', ax=ax)

# ----- ----- ----- ----- ----- -----
# Diagram export
# ----- ----- ----- ----- ----- -----
if args.savepdf:
    fmt     = 'pdf'
    fname   = 'laplace-'+args.wds+'-'+args.adj+'.'+fmt
    fig.savefig(fname, format=fmt, bbox_inches='tight')
else:
    plt.show()
