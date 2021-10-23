# -*- coding: utf-8 -*-
import dask.array as da
import numpy as np
import zarr

class DataReader():
    def __init__(self, path_to_db=None, n_junc=None, obsrat=.8, seed=None, signal_mask=None, node_order=None):
        self.path_to_db = path_to_db
        self.obsrat     = obsrat
        self.node_order = node_order
        if seed:
            np.random.seed(seed)
        if signal_mask is None:
            self._set_fixed_random_setup(n_junc)
        else:
            self.obs_ind    = da.from_array(signal_mask)

    def _set_fixed_random_setup(self, n_junc):
        obs_ind = np.ones(shape=(n_junc,))
        obs_len = int(n_junc * (1-self.obsrat))
        assert obs_len > 0
        hid_ind = np.random.choice(
            np.arange(n_junc),
            size    = obs_len,
            replace = False
            )
        obs_ind[hid_ind]= 0
        self.obs_ind    = da.from_array(obs_ind)

    def read_data(self, dataset='trn', varname=None, rescale=None, cover=False):
        store   = zarr.open(self.path_to_db, mode='r')
        arr     = da.from_zarr(url=store[dataset+'/'+varname])
        if rescale == 'normalize':
            bias    = np.float32(store['trn/'+varname].attrs['min'])
            scale   = np.float32(store['trn/'+varname].attrs['range'])
            arr = (arr-bias) / scale
        elif rescale == 'standardize':
            bias    = np.float32(store['trn/'+varname].attrs['avg'])
            scale   = np.float32(store['trn/'+varname].attrs['std'])
            arr = (arr-bias) / scale
        else:
            bias    = np.nan
            scale   = np.nan

        obs_mx  = da.tile(self.obs_ind, (np.shape(arr)[0], 1))
        if cover:
            arr = da.multiply(arr, obs_mx)
            arr = np.expand_dims(arr.compute(), axis=2)
            obs_mx  = np.expand_dims(obs_mx.compute(), axis=2)
            arr = np.concatenate((arr, obs_mx), axis=2)
        else:
            arr = np.expand_dims(arr.compute(), axis=2)

        if self.node_order is not None:
            arr = arr[:, self.node_order, :]
        return arr, bias, scale
