# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx

def linear_regression(L, X):
    idx_on = np.where(X[0,:,1] == 1)[0]
    idx_off = np.array(list(set(np.arange(X.shape[1]))-set(idx_on)), dtype=int)
    idx_off.sort()

    S = L
    F = X[:, idx_on, 0].copy()
    S2 = S[idx_on, :][:, idx_off]
    S3 = S[idx_off, :][:, idx_off]
    try:
        S3_inv = np.linalg.inv(S3)
    except:
        S3_inv = np.linalg.pinv(S3)

    S32 = np.dot(S3_inv, S2.T)
    mu = - np.sum(np.dot(S32, F.T), axis=0) / np.sum(np.dot(S32, np.ones_like(F.T)), axis=0)
    F_tilde = np.dot(S32, F.T+0*mu).T

    X_hat = X[:, :, 0].copy()
    X_hat[:, idx_off] = F_tilde
    return X_hat
