# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pandas as pd
import matplotlib.pyplot as plt

from taylorDiagram import TaylorDiagram

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
    '--smin',
    default = 0,
    type    = float
    )
parser.add_argument(
    '--smax',
    default = 1.2,
    type    = float
    )
parser.add_argument(
    '--fill',
    action  = 'store_true'
    )
parser.add_argument(
    '--nocenter',
    action  = 'store_true'
    )
parser.add_argument(
    '--savepdf',
    action  = 'store_true'
    )
args    = parser.parse_args()

# ----- ----- ----- ----- ----- -----
# DB loading
# ----- ----- ----- ----- ----- -----
df  = pd.read_csv(os.path.join('..', 'experiments', 'Taylor_metrics_processed.csv'))

wds = args.wds
std_ref = df.loc[
    (df['wds'] == wds) &
    (df['obs_rat'] == .05) &
    (df['model'] == 'orig'), 'sigma_true'].tolist()[0]

# ----- ----- ----- ----- ----- -----
# Plot assembly
# ----- ----- ----- ----- ----- -----
fig = plt.figure()
dia = TaylorDiagram(std_ref/std_ref, fig=fig, label='reference', extend=False, srange=(args.smin, args.smax))
dia.samplePoints[0].set_color('r')
dia.samplePoints[0].set_marker('P')
cmap    = plt.get_cmap('tab10')

obs_ratios  = [.05, .1, .2, .4, .8]
for i, obs_rat in enumerate(obs_ratios):
    model   = 'naive'
    df_plot = df.loc[
        (df['wds'] == wds) &
        (df['obs_rat'] == obs_rat) &
        (df['model'] == model)]
    naive_sigma = df_plot['sigma_pred'].to_numpy()/std_ref
    naive_rho   = df_plot['corr_coeff'].to_numpy()
    model   = 'gcn'
    df_plot = df.loc[
        (df['wds'] == wds) &
        (df['obs_rat'] == obs_rat) &
        (df['model'] == model)]
    gcn_sigma   = df_plot['sigma_pred'].to_numpy()/std_ref
    gcn_rho     = df_plot['corr_coeff'].to_numpy()

    pt_alpha    = .5
    fill_alpha  = .2
    color   = cmap(i)
    dia.add_sample(gcn_sigma, gcn_rho,
        marker  = 'o',
        ms  = 5,
        ls  = '',
        mfc = color,
        mec = 'none',
        alpha   = pt_alpha,
        #label   = 'ChebConv-'+str(obs_rat)
        )
    dia.add_sample(naive_sigma, naive_rho,
        marker  = 's',
        ms  = 5,
        ls  = '',
        mfc = color,
        mec = 'none',
        alpha   = pt_alpha,
        #label   = 'naive-'+str(obs_rat)
        )
    if args.fill:
        points  = np.array([np.arccos(gcn_rho), gcn_sigma]).T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            dia.ax.plot(points[simplex, 0], points[simplex, 1], '--', alpha=fill_alpha)
        dia.ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=fill_alpha)

        points  = np.array([np.arccos(naive_rho), naive_sigma]).T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            dia.ax.plot(points[simplex, 0], points[simplex, 1], '--', alpha=fill_alpha)
        dia.ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], alpha=fill_alpha)

if not args.nocenter:
    for i, obs_rat in enumerate(obs_ratios):
        model   = 'gcn'
        df_plot = df.loc[
            (df['wds'] == wds) &
            (df['obs_rat'] == obs_rat) &
            (df['model'] == model)]
        gcn_sigma   = df_plot['sigma_pred'].to_numpy()/std_ref
        gcn_rho     = df_plot['corr_coeff'].to_numpy()

        dia.add_sample(gcn_sigma.mean(), gcn_rho.mean(),
            marker  = 'o',
            ms  = 10,
            ls  = '',
            mfc = 'none',
            mec = cmap(i),
            mew = 3,
            label   = 'GraConWat@OR='+str(obs_rat)
            )
    for i, obs_rat in enumerate(obs_ratios):
        model   = 'naive'
        df_plot = df.loc[
            (df['wds'] == wds) &
            (df['obs_rat'] == obs_rat) &
            (df['model'] == model)]
        naive_sigma = df_plot['sigma_pred'].to_numpy()/std_ref
        naive_rho   = df_plot['corr_coeff'].to_numpy()
        dia.add_sample(naive_sigma.mean(), naive_rho.mean(),
            marker  = 's',
            ms  = 10,
            ls  = '',
            mfc = 'none',
            mec = cmap(i),
            mew = 3,
            label   = 'Naive model@OR='+str(obs_rat)
            )

contours = dia.add_contours(levels=6, colors='0.5')
plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

dia.add_grid()
dia._ax.axis[:].major_ticks.set_tick_out(True)
dia._ax.axis['left'].label.set_text('Normalized standard deviation')

fig.legend(dia.samplePoints,
           [p.get_label() for p in dia.samplePoints],
           numpoints=1, prop=dict(size='small'), loc='upper right', framealpha=.5)
fig.tight_layout()

# ----- ----- ----- ----- ----- -----
# Diagram export
# ----- ----- ----- ----- ----- -----
if args.savepdf:
    fmt     = 'pdf'
    fname   = 'Taylor-'+args.wds+'.'+fmt
    fig.savefig(fname, format=fmt)
else:
    plt.show()
