# -*- coding: utf-8 -*-
import os
import argparse
import datetime
import time
import pytz
from psutil import virtual_memory
from tqdm import tqdm
import yaml
import zarr
import numpy as np
import dask.array as da
import dask
import ray
import pyDOE2 as doe
from epynet import Network

# ----- ----- ----- ----- -----
# Parsing command line arguments
# ----- ----- ----- ----- -----
parser  = argparse.ArgumentParser()
parser.add_argument('--params',
                    default = 'db_anytown_doe_pumpfed_1',
                    type    = str,
                    help    = "Name of the hyperparams to use.")
parser.add_argument('--nproc',
                    default = 4,
                    type    = int,
                    help    = "Number of processes to raise.")
parser.add_argument('--batch',
                    default = 50,
                    type    = int,
                    help    = "Batch size.")
args    = parser.parse_args()

# ----- ----- ----- ----- -----
# Paths
# ----- ----- ----- ----- -----
pathToRoot      = os.path.dirname(os.path.realpath(__file__))
pathToExps      = os.path.join(pathToRoot, 'experiments')
pathToParam     = os.path.join(pathToExps, 'hyperparams', 'db', args.params+'.yaml')
with open(pathToParam, 'r') as fin:
    params  = yaml.load(fin, Loader=yaml.Loader)
pathToNetwork   = os.path.join(pathToRoot, 'water_networks', params['wds']+'.inp')
pathToDB        = os.path.join(pathToRoot, 'data', args.params)

class SequenceGenerator():
    """Sequence generator for parametric studies or data generation from experiments.
    Random number generation is not provided by a seed yet, but experiment design can be arbitrarily large.
    Experiment design by LHS has to fit in to the physical RAM.
    400M data points in fp32 costs ~1.5 GB RAM.
    Chunking data to ~40M data points."""
    def __init__(self, store, n_scenes, feat_dict, chunks=None):
        self.store  = store
        self.chunks = chunks
        self.n_scenes   = n_scenes
        self.n_features = 2
        for key, val in feat_dict.items():
            self.n_features += val

    def design_experiments(self, algo):
        if algo == 'rnd':
            return self.nonunique_random()
        elif algo == 'doe':
            return self.latin_hypercube_sampling()

    def nonunique_random(self):
        """Random sequence generation without checking uniqueness."""
        design  = da.random.random(
                    size    = (self.n_scenes, self.n_features),
                    chunks  = self.chunks
                    )
        da.to_zarr(
            design,
            url         = self.store,
            component   = 'raw_design',
            compute     = True
            )

    def latin_hypercube_sampling(self):
        """Latin hypercube sampling - uniqueness guaranteed."""
        design  = doe.lhs(self.n_features, samples=self.n_scenes)
        design  = da.from_array(design, chunks=self.chunks)
        da.to_zarr(
            design,
            url         = self.store,
            component   = 'raw_design',
            compute     = True
            )

    def transform_scenes(self):
        n_junc  = feat_dict['juncs']
        n_group = feat_dict['groups']
        n_pump  = feat_dict['pumps']
        n_tank  = feat_dict['tanks']
        lazy_ops    = []
        raw_design  = da.from_zarr(
                        url         = self.store,
                        component   ='raw_design'
                        )
        junc_demands= da.multiply(
                        orig_dmds,
                        da.add(
                            da.multiply(
                                raw_design[:, :n_junc],
                                dmd_hi - dmd_lo
                                ),
                            dmd_lo
                            )
                        )
        tot_dmds  = da.sum(junc_demands, axis=1, keepdims=True)
        target_tot_dmds = da.multiply(
                            orig_tot_dmd,
                            da.add(
                                da.multiply(
                                    raw_design[:, n_junc],
                                    tot_dmd_hi-tot_dmd_lo
                                    ),
                                tot_dmd_lo
                                )
                            ).reshape((n_scenes, 1))
        junc_demands    = da.multiply(
                            junc_demands,
                            da.divide(
                                target_tot_dmds,
                                tot_dmds
                                )
                            )
        lazy_op = da.to_zarr(
                    junc_demands.astype(np.float32).rechunk(self.chunks),
                    url         = self.store,
                    component   = 'junc_demands',
                    compute     = False,
                    )
        lazy_ops.append(lazy_op)

        group_speeds= da.add(
                        da.multiply(
                            raw_design[:, n_junc+1:n_junc+1+n_group],
                            spd_lmt_hi-spd_lmt_lo
                            ),
                        spd_lmt_lo
                        )
        lazy_op = da.to_zarr(
                    group_speeds.astype(np.float32).rechunk(self.chunks),
                    url         = self.store,
                    component   = 'group_speeds',
                    compute     = False
                    )
        lazy_ops.append(lazy_op)

        tankfed_val = da.less(raw_design[:, n_junc+n_group+1], tankfed_proba)
        tankfed_val = da.reshape(tankfed_val, (-1, 1))
        pump_status = da.less(raw_design[:, n_junc+n_group+2:n_junc+n_group+2+n_pump], pump_off_proba)
        concat_list = []
        for i in range(pump_status.shape[1]):
            concat_list.append(tankfed_val)
        tankfed_val = da.concatenate(concat_list, axis=1)
        pump_status = da.logical_not(da.logical_and(tankfed_val, pump_status))
        del tankfed_val
        lazy_op = da.to_zarr(
                    pump_status.astype(np.float32).rechunk(self.chunks),
                    url         = self.store,
                    component   = 'pump_status',
                    compute     = False
                    )
        lazy_ops.append(lazy_op)

        tank_level  = da.add(
                        da.multiply(
                            raw_design[:, n_junc+n_group+n_pump+2:],
                            da.subtract(
                                wtr_lvl_hi,
                                wtr_lvl_lo
                                ).reshape((1, n_tank))),
                        wtr_lvl_lo
                        )
        lazy_op = da.to_zarr(
                    tank_level.astype(np.float32).rechunk(self.chunks),
                    url         = self.store,
                    component   = 'tank_level',
                    compute     = False
                    )
        lazy_ops.append(lazy_op)

        dask.compute(*lazy_ops)

@ray.remote
class simulator():
    """EPYNET wrappper for one-time initialisation of the water network in a multithreaded environment."""
    def __init__(self):
        """Read network topology from disk."""
        self.wds    = Network(pathToNetwork)
        self.junc_heads = np.empty(shape=(n_batch, n_junc), dtype=np.float32)
        self.pump_flows = np.empty(shape=(n_batch, n_pump), dtype=np.float32)
        self.tank_flows = np.empty(shape=(n_batch, n_tank), dtype=np.float32)

    def evaluate_batch(self, scene_ids, boundaries):
        """Set boundaries and run the simulation. No need to re-set the WDS."""
        for idx, scene_id in enumerate(scene_ids):
            for junc_id, junc in enumerate(self.wds.junctions):
                junc.basedemand = boundaries[0][scene_id, junc_id]
            for gid, grp in enumerate(pump_groups):
                self.wds.pumps[self.wds.pumps.uid.isin(grp)].speed   = boundaries[1][scene_id, gid]
            for pump_id, pump in enumerate(self.wds.pumps):
                pump.status     = boundaries[2][scene_id, pump_id]
            for tank_id, tank in enumerate(self.wds.tanks):
                tank.tanklevel  = boundaries[3][scene_id, tank_id]
            self.wds.solve()
            self.junc_heads[idx,:]  = self.wds.junctions.head.values
            self.pump_flows[idx,:]  = self.wds.pumps.flow.values
            self.tank_flows[idx,:]  = self.wds.tanks.outflow.values-self.wds.tanks.inflow.values
        return [scene_ids, self.junc_heads, self.pump_flows, self.tank_flows]

def read_pump_groups(wds):
    """Returns the list of the pump groups in the WDS.
    Groups are identified by the last characters of the uid.
    The trailing slice after the last 'g' letter should be unique."""
    pump_groups = []
    group       = ['gx']
    for pump in wds.pumps:
        uid = pump.uid
        if uid[uid.rfind('g'):] == group[0][group[0].rfind('g'):]:
            group.append(uid)
        else:
            pump_groups.append(group)
            group   = [uid]
    pump_groups.append(group)
    return pump_groups[1:]

def print_store_stats(store):
    for key in root.keys():
        arr = da.from_zarr(url=store, component=key)
        print(key)
        print('max: {:.2f}'.format(arr.max().compute()))
        print('min: {:.2f}'.format(arr.min().compute()))
        print('avg: {:.2f}'.format(arr.mean().compute()))
        print('std: {:.2f}'.format(arr.std().compute()))
        print('')

def chunk_computation(boundaries):
    junc_heads  = np.empty_like(boundaries[0], dtype=np.float32)
    pump_flows  = np.empty_like(boundaries[2], dtype=np.float32)
    tank_flows  = np.empty_like(boundaries[3], dtype=np.float32)
    boundary_id = ray.put(boundaries)

    junc_dmds_id    = ray.put(junc_demands)
    group_speeds_id = ray.put(group_speeds)
    pump_status_id  = ray.put(pump_status)
    tank_level_id   = ray.put(tank_level)
    workers = [simulator.remote() for i in range(n_proc)]
    results = {}

    scene_id_batches    = []
    new_batch   = []
    for idx in range(junc_heads.shape[0]):
        if (idx % n_batch) == 0:
            scene_id_batches.append(new_batch)
            new_batch   = []
        new_batch.append(idx)
    scene_id_batches.append(new_batch)
    scene_id_batches    = scene_id_batches[1:]
    
    progressbar = tqdm(total=len(scene_id_batches))
    for worker in workers:
        results[worker.evaluate_batch.remote(scene_id_batches.pop(), boundary_id)] = worker
    
    ready_ids, _ = ray.wait(list(results))
    while ready_ids:
        ready_id= ready_ids[0]
        result  = ray.get(ready_id)
        worker  = results.pop(ready_id)
        if scene_id_batches:
            results[worker.evaluate_batch.remote(scene_id_batches.pop(), boundary_id)] = worker
    
        for idx in range(len(result[0])):
            junc_heads[result[0][idx], :]    = result[1][idx,:]
            pump_flows[result[0][idx], :]    = result[2][idx,:] 
            tank_flows[result[0][idx], :]    = result[3][idx,:] 
    
        ready_ids, _    = ray.wait(list(results))
        progressbar.update(1)
    progressbar.close()
    return junc_heads, pump_flows, tank_flows

# ----- ----- ----- ----- -----
# Loading WDS
# ----- ----- ----- ----- -----
wds         = Network(pathToNetwork)
dmd_lo      = params['demand']['nodalLo'] 
dmd_hi      = params['demand']['nodalHi'] 
tot_dmd_lo  = params['demand']['totalLo'] 
tot_dmd_hi  = params['demand']['totalHi'] 
spd_lmt_lo  = params['pumpSpeed']['limitLo'] 
spd_lmt_hi  = params['pumpSpeed']['limitHi'] 
wtr_lvl_lo  = np.array(wds.tanks.minlevel * 1.01, dtype=np.float32)
wtr_lvl_hi  = np.array(wds.tanks.maxlevel * .99, dtype=np.float32)
tankfed_proba   = params['feed']['gravityFedProba']
pump_off_proba  = params['feed']['pumpOffProba']
pump_groups = read_pump_groups(wds)

# ----- ----- ----- ----- -----
# Initialization
# ----- ----- ----- ----- -----
n_scenes    = params['nScenes']
feat_dict   = { 'juncs' : len(wds.junctions.uid),
                'groups': len(pump_groups),
                'pumps' : len(wds.pumps.uid),
                'tanks' : len(wds.tanks.uid)
                }
n_proc  = args.nproc
n_batch = args.batch
orig_dmds = np.array(wds.junctions.basedemand, dtype=np.float32)
orig_dmds = orig_dmds.reshape(1, -1)
orig_tot_dmd    = np.sum(wds.junctions.basedemand)

assert n_proc <= n_scenes
assert n_batch <= n_scenes
assert np.ceil(n_scenes/n_batch) >= n_proc

store   = zarr.DirectoryStore(pathToDB)
root    = zarr.group(
            store       = store,
            overwrite   = True,
            synchronizer= zarr.ThreadSynchronizer()
            )
now     = datetime.datetime.now(pytz.UTC)
root.attrs['creation_date']   = str(now)
root.attrs['gmt_timestap']    = int(now.strftime('%s'))
root.attrs['description']     = 'WDS digitwin experiment design'
scene_generator = SequenceGenerator(
                    store, n_scenes, feat_dict,
                    chunks  = ( params['chunks']['height'],
                                params['chunks']['width']
                                ),
                    )

print(
    'Writing unscaled random experiment design to data store... ',
    end     = "",
    flush   = True
    )
scenes  = scene_generator.design_experiments(params['genAlgo'])
print('OK')

print(
    'Splitting and scaling raw experiment design... ',
    end     = "",
    flush   = True
    )
scene_generator.transform_scenes()
del root['raw_design']
print('OK')
print_store_stats(store)

# ----- ----- ----- ----- -----
# Scene evaluation
# ----- ----- ----- ----- -----
junc_demands_store  = da.from_zarr(
                url         = store,
                component   ='junc_demands',
                )
group_speeds_store = da.from_zarr(
                url         = store,
                component   ='group_speeds',
                )
pump_status_store = da.from_zarr(
                url         = store,
                component   ='pump_status',
                )
tank_level_store  = da.from_zarr(
                url         = store,
                component   ='tank_level',
                )
now = datetime.datetime.now(pytz.UTC)
root.attrs['creation_date']   = str(now)
root.attrs['gmt_timestap']    = int(now.strftime('%s'))
root.attrs['description']     = 'WDS digitwin experiment results'
junc_heads_store    = root.empty(
                'junc_heads',
                shape   = junc_demands_store.shape,
                chunks  = ( params['chunks']['height'],
                            params['chunks']['width']
                            ),
                dtype   = 'f4'
                )
pump_flows_store  = root.empty(
                'pump_flows',
                shape   = pump_status_store.shape,
                chunks  = ( params['chunks']['height'],
                            params['chunks']['width']
                            ),
                dtype   = 'f4'
                )
tank_flows_store  = root.empty(
                'tank_flows',
                shape   = tank_level_store.shape,
                chunks  = ( params['chunks']['height'],
                            params['chunks']['width']
                            ),
                dtype   = 'f4'
                )

n_junc  = feat_dict['juncs']
n_group = feat_dict['groups']
n_pump  = feat_dict['pumps']
n_tank  = feat_dict['tanks']

n_experiment= junc_demands_store.shape[0]
chunk_len   = root['junc_demands'].chunks[0]
n_full_batch= n_experiment // chunk_len
print('Computing {} full batch...\n'.format(n_full_batch))

mem = virtual_memory()
ray.init()
time.sleep(10)
workers     = [simulator.remote() for i in range(n_proc)]
scene_ids   = list(np.arange(chunk_len))
for batch_id in range(n_full_batch):
    beg_idx = batch_id*chunk_len
    end_idx = beg_idx + chunk_len
    junc_demands= np.array(junc_demands_store[beg_idx:end_idx, :])
    group_speeds= np.array(group_speeds_store[beg_idx:end_idx, :])
    pump_status = np.array(pump_status_store[beg_idx:end_idx, :])
    tank_level  = np.array(tank_level_store[beg_idx:end_idx, :])
    boundaries  = [junc_demands, group_speeds, pump_status, tank_level]

    junc_heads, pump_flows, tank_flows  = chunk_computation(boundaries)

    junc_heads_store[beg_idx:end_idx, :]   = junc_heads
    pump_flows_store[beg_idx:end_idx, :]   = pump_flows
    tank_flows_store[beg_idx:end_idx, :]   = tank_flows
if n_experiment % chunk_len:
    beg_idx = end_idx
    junc_demands= np.array(junc_demands_store[beg_idx:, :])
    group_speeds= np.array(group_speeds_store[beg_idx:, :])
    pump_status = np.array(pump_status_store[beg_idx:, :])
    tank_level  = np.array(tank_level_store[beg_idx:, :])
    boundaries  = [junc_demands, group_speeds, pump_status, tank_level]

    junc_heads, pump_flows, tank_flows  = chunk_computation(boundaries)

    junc_heads_store[beg_idx:, :]   = junc_heads
    pump_flows_store[beg_idx:, :]   = pump_flows
    tank_flows_store[beg_idx:, :]   = tank_flows
    pass
ray.shutdown()

pump_speeds_store   = root.empty(
    'pump_speeds',
    shape   = pump_status_store.shape,
    chunks  = (params['chunks']['height'],params['chunks']['width']),
    dtype   = 'f4'
    )
beg_idx = 0
for gid in range(len(pump_groups)):
    for pid in range(len(pump_groups[gid])):
        pump_speeds_store[:, beg_idx+pid] = group_speeds_store[:, gid]
    beg_idx = beg_idx+pid+1
del root['group_speeds']
del root['pump_status']

head_treshold   = 0
junc_heads  = da.from_zarr(
    url = store,
    component   ='junc_heads'
    )
min_heads   = junc_heads.min(axis=1).compute()
idx_ok      = np.where(min_heads > head_treshold)[0]
if len(idx_ok) < len(min_heads):
    for key in root.keys():
        arr = da.from_zarr(root[key])
        arr.to_zarr(
            url         = store,
            component   = key+'-tmp',
            overwrite   = True,
            compute     = True
            )
        arr = da.from_zarr(root[key+'-tmp'])
        arr = arr[idx_ok, :].rechunk(scene_generator.chunks).to_zarr(
            url         = store,
            component   = key,
            overwrite   = True,
            compute     = True
            )
        del root[key+'-tmp']

print('-----')
print_store_stats(store)

# ----- ----- ----- ----- -----
# Splitting
# ----- ----- ----- ----- -----
vld_split   = params['data']['vldSplit']
tst_split   = params['data']['tstSplit']
idx_trn = int(np.floor(len(idx_ok) * (1-tst_split-vld_split)))
idx_vld = int(np.floor(len(idx_ok) * (1-tst_split)))

unsplit_keys= list(root.keys())
root_trn    = zarr.hierarchy.group(
    store       = store,
    overwrite   = True,
    synchronizer= zarr.ThreadSynchronizer(),
    path        = 'trn'
    )
root_vld    = zarr.hierarchy.group(
    store       = store,
    overwrite   = True,
    synchronizer= zarr.ThreadSynchronizer(),
    path        = 'vld'
    )
root_tst    = zarr.hierarchy.group(
    store       = store,
    overwrite   = True,
    synchronizer= zarr.ThreadSynchronizer(),
    path        = 'tst'
    )

for key in unsplit_keys:
    arr = da.from_zarr(root[key])
    arr_avg = da.mean(arr[:idx_trn, :]).compute()
    arr_std = da.std(arr[:idx_trn, :]).compute()
    arr_min = da.min(arr[:idx_trn, :]).compute()
    arr_max = da.max(arr[:idx_trn, :]).compute()
    arr_range   = arr_max - arr_min

    arr[:idx_trn, :].to_zarr(
        url         = store,
        component   = 'trn/'+key,
        overwrite   = True,
        compute     = True
        )
    arr[idx_trn:idx_vld, :].rechunk(scene_generator.chunks).to_zarr(
        url         = store,
        component   = 'vld/'+key,
        overwrite   = True,
        compute     = True
        )
    arr[idx_vld:, :].rechunk(scene_generator.chunks).to_zarr(
        url         = store,
        component   = 'tst/'+key,
        overwrite   = True,
        compute     = True
        )

    root_trn[key].attrs['avg'] = float(arr_avg)
    root_trn[key].attrs['std'] = float(arr_std)
    root_trn[key].attrs['min'] = float(arr_min)
    root_trn[key].attrs['range'] = float(arr_range)
    del root[key]
print(root.tree())
