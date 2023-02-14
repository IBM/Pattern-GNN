"""
This script provides functionality for setting up datasets for training based on config file settings.
"""
import math, os, pickle, sys, re, glob, logging, itertools
import torch
import numpy as np
import pandas as pd
from datatable import dt
# from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split, StratifiedKFold
from pathlib import Path
from train_helpers.simulator import apply_simulator

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.gcn_utils import GraphData

########## Splitting Functions ##########

def daily_total_split(daily_totals, split_per, dt_max):
    daily_totals = np.array(daily_totals)
    # total = daily_totals.sum()
    d_ts = daily_totals[dt_max:]
    I = list(range(len(d_ts)-dt_max))
    split_scores = dict()
    for i,j in itertools.combinations(I, 2):
        if j >= i+dt_max:
            split_totals = [d_ts[:i].sum(), d_ts[i+dt_max:j].sum(), d_ts[j+dt_max:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v/split_totals_sum for v in split_totals]
            split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
            score = max(split_error) #- (split_totals_sum/total) + 1
            split_scores[(i,j)] = score
        else:
            continue
    i,j = min(split_scores, key=split_scores.get)
    i,j = i+dt_max, j+dt_max
    split = [list(range(dt_max, i)), list(range(i+dt_max, j)), list(range(j+dt_max, len(daily_totals)))]
    return split

def randomsplit(num_samples, train_split=0.6, val_split=0.2, test_split=0.2, seed=1):
    assert train_split + val_split + test_split == 1, "The splits do not add up to 1"
    tr_inds, te_inds = train_test_split(np.arange(num_samples), test_size=(1 - train_split), random_state=seed)  # 42
    te_inds, val_inds = train_test_split(te_inds, test_size=test_split / (1 - train_split), random_state=seed)  # 42
    tr_val_inds = np.concatenate([tr_inds, val_inds])
    val_folds = [(tr_inds, val_inds)]
    return tr_val_inds, te_inds, val_folds 

def temporal_daily_split(labels, timestamps, train_split, val_split, te_split, data_dir):
    """
    Performs splitting by considering transactions chronologically and assigning desired proportions of data to each
    fold
    :param labels: Label data
    :param timestamps: Timestamps corresponding to labels
    :param train_split: Percentage of data to allocate to train fold
    :param val_split: Percentage of data to allocate to validation fold
    :param te_split: Percentage of data to allocate to test fold
    :return: Train, validation, test indices
    """
    assert train_split + val_split + te_split == 1, "split percentages do not add up to 1"
    # indexing magic because edges are not necessarily stored chronologically

    # if there is a config file from the manual features extraction, then we extract the maximum time window (dt_max) from this file to use for data splitting, o/w dt_max=0 
    dir_filenames = os.listdir(data_dir)
    config_files = [i for i in dir_filenames if 'config' in i]
    if len(config_files) == 1:
        with open(os.path.join(data_dir, config_files[0]), 'r') as f:
            data_config = f.read()
        # extract all feature time windows (dts)
        dts = re.findall("(?<=_dt:)\d*", data_config); dts = [int(i) for i in dts]; dt_max = max(dts)
        logging.info(f"MF time windows are {dts}")
    else:
        dt_max = 0

    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], []
    n_days = int(timestamps.max() / 3600 / 24 + 1)
    logging.info(f"N_DAYS = {n_days}, {timestamps.max()}")
    n_samples = labels.shape[0]
    for day in range(n_days):
        l = day * 24 * 3600; r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(labels[day_inds].float().mean())
        weighted_daily_irs.append(labels[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds); daily_trans.append(day_inds.shape[0])

    logging.debug(f"n_samples = {n_samples}, dt_max = {dt_max}, daily_trans = {daily_trans}")
    split_inds = {k: [] for k in range(3)}
    dt_max_days = math.ceil((dt_max - 0.001)/24)
    if '100m' in data_dir:
        dt_max_days = max(dt_max_days, 0)
    elif 'sim' in data_dir:
        dt_max_days = max(dt_max_days, 0)
    # We are slightly more generous with the training data for aml-e_medium and aml-e_large, because otherwise we throw away too much data - 1 day less is removed each time
    else:
        dt_max_days = max(dt_max_days-1, 0)

    # Find the best possible split given daily transaction numbers and desired split proportions - brute force search
    split_days = daily_total_split(daily_trans, [train_split, val_split, te_split], dt_max_days)
    for i in range(3):
        for day in split_days[i]:
            split_inds[i].append(daily_inds[day])

    tr_inds = torch.cat(split_inds[0]); val_inds = torch.cat(split_inds[1]); te_inds = torch.cat(split_inds[2])

    actual_total = tr_inds.shape[0] + val_inds.shape[0] + te_inds.shape[0]
    logging.info(f"Total train samples: {tr_inds.shape[0] / labels.shape[0] * 100 :.2f}% || {tr_inds.shape[0] / actual_total * 100 :.2f}% || IR: "f"{labels[tr_inds].float().mean() * 100 :.2f}% || Train days: {split_days[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / labels.shape[0] * 100 :.2f}% || {val_inds.shape[0] / actual_total * 100 :.2f}% || IR: "f"{labels[val_inds].float().mean() * 100:.2f}% || Val days: {split_days[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / labels.shape[0] * 100 :.2f}% || {te_inds.shape[0] / actual_total * 100 :.2f}% || IR: "f"{labels[te_inds].float().mean() * 100:.2f}% || Test days: {split_days[2][:5]}")
    logging.info(f"Total Thrown Out: { (labels.shape[0] - actual_total) / labels.shape[0] * 100 :.2f}%")

    # assertions - check for disjointness
    assert set(tr_inds.numpy()).isdisjoint(set(te_inds.numpy())) and set(tr_inds.numpy()).isdisjoint(set(val_inds.numpy())) and set(val_inds.numpy()).isdisjoint(set(te_inds.numpy())), "indices are not disjoint"

    return tr_inds, val_inds, te_inds

def get_min_times(df_nodes, df_edges, simulator='eth'):
    src_col, dst_col, time_col = ('from_id', 'to_id', 'Timestamp') if simulator == 'aml-e' else ('Source Node', 'Destination Node', 'Timestamp')
    max_time = df_edges['Timestamp'].max().to_numpy()[0][0] + 10
    min_node_times = np.ones(df_nodes.shape[0], dtype=int) * int(max_time)
    edges = df_edges[:, [src_col, dst_col, time_col]].to_numpy()
    for u,v,t in edges:
        u,v,t = int(u),int(v),int(t)
        for i in [u,v]:
            min_node_times[i] = min(min_node_times[i], t)
    return min_node_times

def get_temporal_split_times(timestamps, split=(0.65,0.15,0.2)):
    total = timestamps.shape[0]
    max_time = np.max(timestamps) + 1
    sorted_timestamps = np.sort(timestamps)
    cum_split = [sum(split[:i]) for i in range(len(split))]
    s = [int(i*total) for i in cum_split] 
    times = [sorted_timestamps[i] for i in s] + [max_time]
    return times


def get_val_folds(config, args, y, timestamps=None):
    """
    Creates train/ validation/ test folds depending on split method provided by config
    :param config: Configuration file
    :param args: Auxiliary command line arguments
    :param y: Data labels
    :param timestamps: Timestamps in seconds
    :return: Train inds, list containing tuples of validation folds (corresponding to train data), and test inds
    """
    if config.split_method == "random":
        tr_inds, te_inds, val_folds = randomsplit(y.shape[0])
    elif config.split_method == "temporal_node":
        times = get_temporal_split_times(timestamps) # [0, 0.65, 0.8, 1.0]
        tmp_tr_inds = np.where(timestamps<times[1])[0]
        tmp_val_inds = np.where((timestamps>=times[1]) & (timestamps<times[2]))[0]
        te_inds = np.where(times[2]<=timestamps)[0]
        tr_inds = np.concatenate([tmp_tr_inds, tmp_val_inds])
        val_folds = [(tmp_tr_inds, tmp_val_inds)]
        return tr_inds, te_inds, val_folds, times
    else:
        y, timestamps = torch.tensor(y), torch.tensor(timestamps)
        if config.split_method == "temporal_daily":
            tmp_tr, tmp_val, te_inds = temporal_daily_split(y, timestamps, *config.split_vals, config.data_dir)
        elif config.split_method == "temporal_edge":
            times = get_temporal_split_times(timestamps.numpy(), split=(0.6,0.2,0.2)) # [0, 0.65, 0.8, 1.0]
            tmp_tr = torch.where(timestamps<times[1])[0]
            tmp_val = torch.where((timestamps>=times[1]) & (timestamps<times[2]))[0]
            te_inds = torch.where(times[2]<=timestamps)[0]
        
        if config.simulator == "aml-e":
            tr_inds = np.concatenate([tmp_tr, tmp_val])
            e_tr = tmp_tr.numpy(); e_val = tr_inds; te_inds = te_inds.numpy()
            tr_cutoff_ind, val_cutoff_ind = tmp_tr.shape[0], tmp_val.shape[0]
            val_folds = [(np.arange(0, tr_cutoff_ind), np.arange(tr_cutoff_ind, tr_cutoff_ind + val_cutoff_ind))]
            return e_tr, e_val, tr_inds, te_inds, val_folds
        else:
            tr_inds = np.concatenate([tmp_tr, tmp_val]); te_inds = te_inds.numpy()
            tr_cutoff_ind, val_cutoff_ind = tmp_tr.shape[0], tmp_val.shape[0]
            val_folds = [(np.arange(0, tr_cutoff_ind), np.arange(tr_cutoff_ind, tr_cutoff_ind + val_cutoff_ind))]
        
    return tr_inds, te_inds, val_folds

def get_aml_data_benchmark_edge_filenames(args, config):
    if 'edge_file' in config:
        if config.edge_file in os.listdir(config.data_dir):
            csv_filename = config.edge_file
    jay_filename = f"{csv_filename.strip('.csv')}.jay"
    return csv_filename, jay_filename

def get_aml_data_benchmark_full(args, config):
    csv_filename, jay_filename = get_aml_data_benchmark_edge_filenames(args, config)
    jay_file = f"{config.data_dir}{jay_filename}"
    if jay_filename not in os.listdir(config.data_dir):
        csv_file = f"{config.data_dir}{csv_filename}"
        # label_file = "labels.txt"
        old_cols = ['EdgeID', 'Timestamp','timestamp','From Bank','Account','To Bank','Account.1','Amount Received','Receiving Currency', 'Amount Paid','Payment Currency',   'Payment Format','Is Laundering']
        new_cols = ['EdgeID', 'Timestamp','Timestamp','from_bank','from_id','to_bank','to_id',    'Amount Received','Received Currency',  'Amount Sent','Sent Currency',      'Payment Format','Is Laundering']
        old_cols = old_cols + ['from_bank','from_id','to_bank','to_id','Received Currency','Amount Sent','Sent Currency']
        new_cols = new_cols + ['from_bank','from_id','to_bank','to_id','Received Currency','Amount Sent','Sent Currency']
        old_cols = old_cols + ['SourceAccountId',   'TargetAccountId', ' Is Laundering']
        new_cols = new_cols + ['from_id',           'to_id'          , 'Is Laundering' ]
        old_cols = old_cols + ['SourceVertexID',    'DestinationVertexID']
        new_cols = new_cols + ['from_id',           'to_id'              ]
        cols_dict = dict(zip(old_cols, new_cols))
        label_name = 'Is Laundering'

        df_edges = pd.read_csv(csv_file)
        ### rename columns:
        col_dict = {col:cols_dict[col] for col in df_edges.columns if col in cols_dict}
        df_edges = df_edges.rename(columns=col_dict)

        time_cols = ['Timestamp']
        for col in time_cols:
            # df_edges[col] = pd.to_datetime(df_edges[col]).astype(int)/ 10**9
            df_edges[col] = df_edges[col] - df_edges[col].min()

        ################# REMOVE SELF LOOPS #################
        if "remove_self_loops" in config:
            if config.remove_self_loops:
                logging.info(f"Removing self loops"); df_edges = df_edges[df_edges['from_id'] != df_edges['to_id']]

        dt_edges = dt.Frame(df_edges)
    elif jay_filename in os.listdir(config.data_dir):
        logging.info(f"Loading from {jay_filename}"); dt_edges = dt.fread(jay_file)
    
    max_id = dt_edges[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = dt.Frame({'NodeID': np.arange(max_id), 'DegIn [9:10)': np.ones(max_id)})

    timestamps = torch.Tensor(dt_edges['Timestamp'].to_numpy()[:,0])

    labels = dt_edges['Is Laundering'].to_numpy()
    logging.info(f"Labels sum/len = {sum(labels)} / {len(labels)}"); logging.info(f"df_nodes.shape = {df_nodes.shape}"); logging.info(f"df_edges.shape = {df_edges.shape}")
    y = torch.LongTensor(labels)

    return df_nodes, dt_edges, timestamps, y, df_edges

def get_aml_datasplits(config, args, data, edge_timestamps, node_timestamps, node_features, data_dir):
    edge_index = data.edge_index; edge_attr = data.edge_attr
    logging.info(f'Max edge index = {edge_index.max()}')
    x = data.x; y = data.y

    if config.split_method == 'random':
        tr_inds, te_inds, val_folds = get_val_folds(config, args, y, edge_timestamps)
    elif config.split_method == 'temporal_node': 
        tr_inds, te_inds, val_folds, times = get_val_folds(config, args, y, node_timestamps)
    else:    
        e_tr, e_val, tr_inds, te_inds, val_folds = get_val_folds(config, args, y, edge_timestamps)

    node_folder = data_dir
    tr_file =  f"{node_folder}/af_nodeid_static_train.csv"; val_file = f"{node_folder}/af_nodeid_static_trval.csv"; te_file =  f"{node_folder}/af_nodeid_static_all.csv"

    if not config.node_feats:
        tr_x = torch.ones_like(x); val_x = torch.ones_like(x); te_x = torch.ones_like(x)
    else:
        tr_nodes = dt.fread(f"{tr_file}"); val_nodes = dt.fread(f"{val_file}"); te_nodes = dt.fread(f"{te_file}")

        tr_x = torch.Tensor(tr_nodes[:, node_features].to_numpy()); val_x = torch.Tensor(val_nodes[:, node_features].to_numpy()); te_x = torch.Tensor(te_nodes[:, node_features].to_numpy())
        
        if not config.node_feats:
            tr_x = torch.ones_like(torch.unsqueeze(tr_x[:,0], 1)); val_x = torch.ones_like(torch.unsqueeze(val_x[:,0], 1)); te_x = torch.ones_like(torch.unsqueeze(te_x[:,0], 1))
    
    if args.edges == "none":
        tr_edge_index = torch.ones((2,0)).long(); tr_edge_attr = torch.ones((0,1)).long(); val_edge_index = torch.ones((2,0)).long()
        val_edge_attr = torch.ones((0,1)).long(); te_edge_index = torch.ones((2,0)).long(); te_edge_attr = torch.ones((0,1)).long()
    elif config.split_method == 'random':
        te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,           edge_timestamps
    elif config.split_method == 'temporal_node':
        e_tr = edge_attr[:,0] < times[1]
        e_val = edge_attr[:,0] < times[2]
        tr_edge_index, tr_edge_attr, tr_edge_times = edge_index[:,e_tr], edge_attr[e_tr], edge_timestamps[e_tr]
        val_edge_index, val_edge_attr, val_edge_times = edge_index[:,e_val], edge_attr[e_val], edge_timestamps[e_val]
        te_edge_index, te_edge_attr, te_edge_times = edge_index, edge_attr, edge_timestamps
    else:
        tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:,e_tr],  edge_attr[e_tr],  y[e_tr, :],  edge_timestamps[e_tr]
        val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val, :], edge_timestamps[e_val]
        te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,           edge_timestamps
    if config.split_method == 'random':
        tr_data = GraphData (x=te_x, y=te_y, edge_index=te_edge_index, edge_attr=te_edge_attr, timestamps=edge_timestamps)
        val_data = GraphData(x=te_x, y=te_y, edge_index=te_edge_index, edge_attr=te_edge_attr, timestamps=edge_timestamps)
        te_data = GraphData (x=te_x, y=te_y, edge_index=te_edge_index, edge_attr=te_edge_attr, timestamps=edge_timestamps)
    elif config.split_method == 'temporal_node':
        tr_data = GraphData( x=tr_x,  y=y, edge_index=tr_edge_index,  edge_attr=tr_edge_attr , timestamps=tr_edge_times , node_timestamps=node_timestamps)
        val_data = GraphData(x=val_x, y=y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times, )
        te_data = GraphData( x=te_x,  y=y, edge_index=te_edge_index,  edge_attr=te_edge_attr , timestamps=te_edge_times , )
    else:
        tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times )
        val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
        te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times )
    return tr_data, val_data, te_data, tr_inds, te_inds, val_folds  



def get_aml_data(config, args):
    """
    Loads AML data directly from the feature files, takes a bit longer but a lot more natural, understandable and error-proof.
    :param config: Configuration file
    :param args: Auxiliary command line arguments
    :return: Train, validation and test data as well as val_folds and te_inds. Only homogenous implementation so far.
    """
    df_nodes, df_edges, edge_timestamps, y, pd_edges = get_aml_data_benchmark_full(args, config)
    if args.y_from_file: 
        if args.readout == 'node' or args.readout == 'graph': # TODO: graph could be either node, edge, or even graph classification... Default to node classification for now.
            y, y_names = get_y_from_file(config, args)
        elif args.readout == 'edge':
            y, y_names = get_edge_y_from_file(pd_edges, config, args)
        else:
            raise NotImplementedError(f"No aml pretraining implemented for {args.readout} readout!")
    if args.simple_efeats:
        edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    elif args.no_efeats:
        edge_features = ['Payment Format']
    else:
        assert(args.features == "raw" or args.features == "mf"), "Transaction feats can only be 'raw' or 'mf'"
        if args.features == "raw":
            edge_features = ['Timestamp', 'from_id', 'to_id', 'Amount Received', 'Received Currency', 'Payment Format']
        else:
            edge_feat_names = list(df_edges.names); edge_feat_names.remove('EdgeID'); edge_feat_names.remove('Is Laundering'); edge_features = edge_feat_names
    if args.noID:
        id_cols = ['from_id', 'to_id']
        edge_features = [i for i in edge_features if i not in id_cols]
    
    node_feat_names = list(df_nodes.names)
    if 'NodeID' in node_feat_names:
        node_feat_names.remove('NodeID')

    if config.node_feats:
        if args.y_from_file and args.features == "mf": 
            node_features = [feat for feat in node_feat_names if feat not in y_names]
        else:
            node_features = node_feat_names
    else:
        #use one node feature as placeholder for now
        node_features = ['DegIn [9:10)']

    logging.debug(f"edge_features = {edge_features}")
    logging.debug(f"node_features = {node_features}")
    
    x = torch.tensor(df_nodes[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges[:, edge_features].to_numpy()).float()
    data = GraphData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    if args.readout == 'node' or args.readout == 'graph': # TODO: graph could be either node, edge, or even graph classification... Default to node classification for now.
        if "min_time" not in df_nodes.names:
            min_node_times = get_min_times(df_nodes, df_edges, config.simulator)
            df_nodes['min_time'] = min_node_times
        node_timestamps = df_nodes['min_time'].to_numpy()[:,0]
        logging.debug(f"node_timestamps.shape = {node_timestamps.shape}")
    else:
        node_timestamps = None

    tr_data, val_data, te_data, tr_inds, te_inds, val_folds = get_aml_datasplits(config, args, data, edge_timestamps, node_timestamps, node_features, config.data_dir)
    if args.y_from_sim:
        load_dataset_name = f"y_sim_{args.y_pretrain}.pickle"
        load_dataset_dir = f"{config.data_dir}"
        if load_dataset_name in os.listdir(load_dataset_dir):
            logging.info(f"Loading sim labels")
            with open(f"{load_dataset_dir}{load_dataset_name}", 'rb') as f:
                data = pickle.load(f)
                tr_data, val_data, te_data = data
        else:
            logging.info(f"Calculating sim labels")
            tr_data, val_data, te_data = apply_simulator([tr_data, val_data, te_data], args)
            data = (tr_data, val_data, te_data)
            if load_dataset_name not in os.listdir(load_dataset_dir):
                with open(f"{load_dataset_dir}{load_dataset_name}", 'wb') as f:
                    pickle.dump(data, f)

    return tr_data, val_data, te_data, val_folds, te_inds

########### ETH data loading ###########
def get_eth_data_full():
    edge_file = "/dccstor/aml-e/datasets/ETH_transNets/ETH_fullintersection_transNet_wlabels_shortened.csv"
    node_file = f"/dccstor/aml-e/datasets/ETH_node_classif/ETH_fullintersection_nodes_enhanced_v3.csv"
    node_filepath_save = f"dccstor/aml-e/datasets/ETH/ETH_nodes_enhanced_min_time.csv"
    if node_file_save in os.listdir(save_dir): node_file = node_filepath_save

    df_nodes = dt.fread(f"{node_file}"); df_edges = dt.fread(f"{edge_file}")

    if 'Timestamp' in df_edges.names:
        edge_timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy()[:,0])
    else:
        edge_timestamps = None

    if "min_time" not in df_nodes.names:
        min_node_times = get_min_times(df_nodes, df_edges); df_nodes['min_time'] = min_node_times; df_nodes.to_csv(node_filepath_save)
    node_timestamps = df_nodes['min_time'].to_numpy()[:,0]

    labels = df_nodes['Is Phishing'].to_numpy()
    logging.info(f"Labels sum/len = {sum(labels)} / {len(labels)}"); logging.info(f"df_nodes.shape = {df_nodes.shape}"); logging.info(f"df_edges.shape = {df_edges.shape}")
    y = torch.LongTensor(labels)
    return df_nodes, df_edges, edge_timestamps, node_timestamps, y

def fill_in_data(x, num_nodes):
    x_dict = {}
    for row in x:
        x_dict[int(row[0].item())] = row.reshape(1,-1)
    x_full = torch.cat([x_dict[i] if i in x_dict else torch.zeros((x[:1].shape)) for i in range(num_nodes)])
    return x_full

def get_eth_datasplits(config, args, data, edge_timestamps, node_timestamps, node_features):
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    x = data.x
    x = x if config.node_feats else torch.ones_like(torch.unsqueeze(data.x[:,0], 1))
    y = data.y
    tr_inds, te_inds, val_folds, times = get_val_folds(config, args, y, timestamps=node_timestamps)
    if args.save_splits:
        save_dir = '/dccstor/aml-e/datasets/clusterino/indices'
        with open(f'{save_dir}/val_folds.pickle', 'wb') as tr_handle:
            pickle.dump(val_folds, tr_handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{save_dir}/te_inds.pickle', 'wb') as handle:
            pickle.dump(te_inds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()

    node_folder = "/dccstor/aml-e/datasets/ETH_node_classif"
    tr_file = f"{node_folder}/ETH_fullintersection_nodes_tr.csv"; val_file = f"{node_folder}/ETH_fullintersection_nodes_trval.csv"; te_file = f"{node_folder}/ETH_fullintersection_nodes_all.csv"
    tr_nodes = dt.fread(f"{tr_file}"); val_nodes = dt.fread(f"{val_file}"); te_nodes = dt.fread(f"{te_file}")
    # Add temporary node IDs
    ids_plus_node_features = ['Acc ID'] + node_features
    tr_x = torch.Tensor(tr_nodes[:, ids_plus_node_features].to_numpy()); val_x = torch.Tensor(val_nodes[:, ids_plus_node_features].to_numpy()); te_x = torch.Tensor(te_nodes[:, ids_plus_node_features].to_numpy())
    num_nodes = x.shape[0]
    tr_x = fill_in_data(tr_x, num_nodes); val_x = fill_in_data(val_x, num_nodes); te_x = fill_in_data(te_x, num_nodes)
    # Remove temporary node IDs
    tr_x = tr_x[:, 1:]; val_x = val_x[:, 1:]; te_x = te_x[:, 1:]
    if not config.node_feats:
        tr_x = torch.ones_like(torch.unsqueeze(tr_x[:,0], 1)); val_x = torch.ones_like(torch.unsqueeze(val_x[:,0], 1)); te_x = torch.ones_like(torch.unsqueeze(te_x[:,0], 1))

    if args.edges == "none":
        tr_edge_index = torch.ones((2,0)).long(); tr_edge_attr = torch.ones((0,1)).long(); tr_edge_times = None
        val_edge_index = torch.ones((2,0)).long(); val_edge_attr = torch.ones((0,1)).long(); val_edge_times = None
        te_edge_index = torch.ones((2,0)).long(); te_edge_attr = torch.ones((0,1)).long(); te_edge_times = None
    else:
        e_tr = edge_attr[:,0] < times[1]
        e_val = edge_attr[:,0] < times[2]
        tr_edge_index, tr_edge_attr, tr_edge_times = edge_index[:,e_tr], edge_attr[e_tr], edge_timestamps[e_tr]
        val_edge_index, val_edge_attr, val_edge_times = edge_index[:,e_val], edge_attr[e_val], edge_timestamps[e_val]
        te_edge_index, te_edge_attr, te_edge_times = edge_index, edge_attr, edge_timestamps
        logging.info(f'tr_edge_index.shape = {tr_edge_index.shape}, tr_edge_attr.shape = {tr_edge_attr.shape}, tr_edge_times.shape = {tr_edge_times.shape}, edge_timestamps.shape = {edge_timestamps.shape}')
    
    tr_data = GraphData( x=tr_x,  y=y, edge_index=tr_edge_index,  edge_attr=tr_edge_attr , timestamps=tr_edge_times , node_timestamps=node_timestamps)
    val_data = GraphData(x=val_x, y=y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times, )
    te_data = GraphData( x=te_x,  y=y, edge_index=te_edge_index,  edge_attr=te_edge_attr , timestamps=te_edge_times , )
    return tr_data, val_data, te_data, tr_inds, te_inds, val_folds


def get_eth_data(config, args):
    """
    Loads ethereum phishing dataset
    :param config: Configuration file
    :param args: Auxiliary command line arguments
    :return: Train and test data (as either multi_relational or homogeneous), validation folds (list of tuples)
    """
    df_nodes, df_edges, edge_timestamps, node_timestamps, y = get_eth_data_full()

    # ['Transaction ID', 'Timestamp', 'Source Node', 'Destination Node', 'Value', 'Nonce', 'Block Nr', 'Gas', 'Gas Price', 'Transaction Type', 'Is Phishing']
    if args.simple_efeats:
        edge_features = ['Timestamp', 'Value', 'Nonce', 'Gas', 'Gas Price']
    elif args.no_efeats:
        edge_features = ['Value']
    else:
        edge_features = ['Timestamp', 'Value', 'Nonce', 'Block Nr', 'Gas', 'Gas Price', 'Transaction Type']
    # ['Acc ID', 'Out Degree', 'In Degree', 'Fan In', 'Fan Out', 'Max Nonce', 'Avg Amount Out', 'Avg Amount In', ... ]
    node_features = ['Max Nonce']

    if args.features == "raw":
        node_features = ['Max Nonce']
    elif args.features == 'mf':
        node_features = list(df_nodes.names[1:-2])

    if config.multi_relational:
        raise ValueError('config.multi_relational not implemented') # TODO
    else:
        x = torch.tensor(df_nodes[:, node_features].to_numpy()).float()
        if 'node_feats' in config:
            if not config.node_feats: x = torch.ones_like(torch.unsqueeze(x[:,0], 1))
        edge_index = torch.LongTensor(df_edges[:, ['Source Node', 'Destination Node']].to_numpy().T)
        edge_attr = torch.tensor(df_edges[:, edge_features].to_numpy()).float()
        data = GraphData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    if config.multi_relational:
        raise ValueError('config.multi_relational not implemented') # TODO
    else:
        if config.split_method == "temporal_node":
            tr_data, val_data, te_data, tr_inds, te_inds, val_folds = get_eth_datasplits(config, args, data, edge_timestamps, node_timestamps, node_features)
            if args.y_from_sim:
                tr_data, val_data, te_data = apply_simulator([tr_data, val_data, te_data], args)
        elif config.split_method == "random":
            tr_inds, te_inds, val_folds = get_val_folds(config, args, y)
            if args.y_from_file: 
                y, y_names = get_y_from_file(config, args)
                if args.features == "mf": node_features = [feat for feat in node_features if feat not in y_names]
                x = torch.tensor(df_nodes[:, node_features].to_numpy()).float()
                data = GraphData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            if args.y_from_sim:
                data = GraphData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
                data = apply_simulator([data], args)
                data = data[0]
            tr_data = data
            val_data = data
            te_data = data

    # return tr_data, te_data, val_folds, te_inds
    return tr_data, val_data, te_data, val_folds, te_inds

def get_y_from_file(config, args):
    node_files = [filename for filename in os.listdir(config.data_dir) if "pattern" in filename]
    sortd = False
    if sum(['sorted' in f for f in node_files]) > 0: node_files, sortd = [f for f in node_files if 'sorted' in f], True
    assert len(node_files) == 1, f"Found {len(node_files)} node files with pattern in file name. Exactly 1 required."
    node_file = f"{config.data_dir}/{node_files[0]}"
    df_tti = dt.fread(f"{node_file}")
    if not sortd:
        df_tti = df_tti.sort('NodeID')
        # Save SORTED
        sorted_outfile = f"{node_file.strip('.csv')}_sorted.csv"
        df_tti.to_csv(sorted_outfile)
    if args.y_list is not None: 
        y_names = args.y_list
    else: 
        y_names = list(set(list(df_tti.names)) - {'NodeID'})
        args.y_list = y_names
    y = torch.Tensor(df_tti[:,y_names].to_numpy()).long()
    y = (y >= 1).long()
    return y, y_names

def get_edge_y_from_file(df_edges, config, args):
    if args.y_list is not None: 
        y_names = args.y_list
    else: 
        y_names = list(set(df_edges.columns.to_list()) - {'EdgeID'})
        args.y_list = y_names
    y = torch.Tensor(df_edges[y_names].to_numpy())
    y = (y > 0).long()
    logging.debug('ok, got edge_y')
    return y, y_names