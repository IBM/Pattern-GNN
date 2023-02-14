"""
This script provides utility functions for GNN data preprocessing, loss functions, and creating induced subgraphs based
on a chosen list of node indices
"""
from typing import Union
import logging, torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.utils import subgraph
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.transforms import BaseTransform


def to_adj_lists(edge_index, num_nodes):
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v in edge_index.T:
        u,v = int(u), int(v)
        adj_list_out[u] += [v]
        adj_list_in[v] += [u]
    return adj_list_in, adj_list_out

def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def to_adj_edges_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    # calculate adjacent edges with times per node
    adj_edges_out = dict([(i, []) for i in range(num_nodes)])
    adj_edges_in = dict([(i, []) for i in range(num_nodes)])
    for i, (u,v,t) in enumerate(edges):
        u,v,t = int(u), int(v), int(t)
        adj_edges_out[u] += [(i, v, t)]
        adj_edges_in[v] += [(i, u, t)]
    return adj_edges_in, adj_edges_out

def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

def time_deltas(data, adj_edges_list):
    time_deltas = torch.zeros(data.edge_index.shape[1], 1)
    if data.timestamps is None:
        return time_deltas
    for v, edges in adj_edges_list.items():
        if len(edges) < 1: continue
        a = np.array(edges)
        a = a[a[:, -1].argsort()]
        a_tds = [0] + [a[i+1,-1] - a[i,-1] for i in range(a.shape[0]-1)]
        tds = np.hstack((a[:,0].reshape(-1,1), np.array(a_tds).reshape(-1,1)))
        for i,td in tds:
            time_deltas[i] = td
    return time_deltas

def ports_old(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        nbs_unique = set(nbs)
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

def add_reverse(data_):
    data = data_.clone()
    num_edges = data.edge_index.shape[1]
    edge_type = torch.cat([torch.zeros((num_edges, 1)), torch.ones((num_edges, 1))], dim=0)
    edge_index = torch.cat([data.edge_index, data.edge_index.flipud()], dim=1)
    edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    edge_attr = torch.cat([edge_attr, edge_type], dim=1)
    if data.readout == 'edge':
        y = torch.cat([data.y, data.y], dim=0)
        data.y = y
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    return data

def remove_reverse(data):
    edge_attr = data.edge_attr[:,:-1]
    edge_type = data.edge_attr[:,-1].long()
    out_edges = torch.where(edge_type == 1)[0]
    data.edge_index[:, out_edges] = data.edge_index[:, out_edges].flipud()
    data.edge_attr = edge_attr
    return data

class GraphData(Data):
    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, 
        readout: str = 'node', 
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs
        ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = 'ce'
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps  
        elif edge_attr is not None:
            self.timestamps = edge_attr[:,0].clone()
        else:
            self.timestamps = None

    def add_ports(self):
        reverse_ports = True
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self.edge_index, adj_list_in)
        out_ports = [ports(self.edge_index.flipud(), adj_list_out)] if reverse_ports else []
        self.edge_attr = torch.cat([self.edge_attr, in_ports] + out_ports, dim=1)
        return self

    def add_time_deltas(self):
        reverse_tds = True
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = [time_deltas(self, adj_list_out)] if reverse_tds else []
        self.edge_attr = torch.cat([self.edge_attr, in_tds] + out_tds, dim=1)
        return self

    def set_y(self, function, threshholds=None):
        if isinstance(function, list):
            if threshholds is None: threshholds = [None]*len(function)
            ys = []
            for i, (f,thresh) in enumerate(zip(function, threshholds)):
                if f is not None:
                    logging.info(f"Calculating feat {i}: {f.__name__}")
                    ys.append(f(self, thresh))
            y = torch.cat(ys, dim=1)
        else:
            y = function(self, threshholds)
        self.y = y
        if len(y.unique()) > 2:
            self.loss_fn = 'mse'
        elif y.shape[1] > 1:
            self.loss_fn = 'bce_multi'
        else:
            self.loss_fn = 'ce'
        logging.info(f"y.shape = {y.shape}")
        logging.info(f"self.loss_fn = {self.loss_fn}")
        return y


def z_normalize(data, device, feature_normalize=True):
    """
    Z-score data normalization
    :param data: Torch tensor containing feature data
    :param device: GPU or CPU
    :param feature_normalize: Normalize feature-wise or sample-wise
    :return: Z-score normalized data
    """
    if feature_normalize:
        std = data.std(0).unsqueeze(0)
        std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
        return (data - data.mean(0).unsqueeze(0)) / std
    else:
        std = data.std(1).unsqueeze(-1)
        std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
        return (data - data.mean(1).unsqueeze(-1)) / std


def l2_normalize(data, device, feature_normalize=True):
    """
    L2 normalization
    :param data: Torch tensor containing feature data
    :param device: GPU or CPU
    :param feature_normalize: Normalize feature-wise or sample-wise
    :return: L2-normalized data
    """
    if feature_normalize:
        norm = data.square().sum(0).unsqueeze(0).sqrt()
        norm = torch.where(norm == 0, torch.tensor(1, dtype=torch.float32).cpu(), norm)
        return data / norm
    else:
        norm = data.square().sum(1).unsqueeze(-1).sqrt()
        norm = torch.where(norm == 0, torch.tensor(1, dtype=torch.float32).cpu(), norm)
        return data / norm


def generate_filtered_graph(node_indices, node_features=None, labels=None, edge_index=None, edge_attr=None, relabel_nodes=True, data=None):
    """
    Generate induced homogeneous subgraph based on selected nodes and create new feature matrix. Edges containing a node
    within the indexed nodes are filtered node labels are re-indexed them s.t. they correspond to the new (smaller) node
    matrix.
    :param node_indices: Indices of nodes to be included in induced subgraph (can be a list of lists e.g. for train/
    val/ test folds)
    :param node_features: Node feature tensor [n_nodes, num_features]
    :param labels: Data labels
    :param edge_index: Edge index [2, n_edges]
    :param edge_attr: Edge attributes [n_edges, n_edge_attr]
    :param relabel_nodes: Relabel nodes once the new induced subgraph is generated
    :return: Subgraphs induced by train/ validation/ test node indices
    """
    assert (data is not None or node_features is not None), f"generate_filtered_graph needs either a data object or node features as input"
    if data is not None:
        assert (isinstance(data, Data) or isinstance(data, GraphData)), f"input data is not of pyg type Data"
        node_features = data.x.detach().clone()
        labels = data.y.detach().clone()
        edge_index = data.edge_index.detach().clone()
        edge_attr = data.edge_attr.detach().clone() if data.edge_attr is not None else None
    if isinstance(node_indices, list):
        datasets = []
        for i, inds in enumerate(node_indices):
            node_index = torch.tensor(node_indices[i]) if not isinstance(node_indices[i], torch.Tensor) else \
                node_indices[i].clone().detach()
            x = node_features[node_index]
            _edge_index, _edge_attr = subgraph(node_index, edge_index, edge_attr, relabel_nodes=relabel_nodes,
                                               num_nodes=labels.shape[0])
            y = labels[node_index]
            edge_index = _edge_index
            edge_attr = _edge_attr if edge_attr is not None else None
            data = GraphData(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
            datasets.append(data)
    return datasets


class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x
        device=x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        ids[:data.batch_size] = 1
        data.x = torch.cat([x, ids], dim=1)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def normalize_data(tr_data, val_data, _te_data, fn_norm, device, node_name=None):
    if isinstance(tr_data, Data):
        tr_data.x = fn_norm(tr_data.x, device)
        val_data.x = fn_norm(val_data.x, device)
        te_data = _te_data.detach().clone()
        te_data.x = fn_norm(_te_data.x, device)
        if tr_data.edge_attr is not None:
            tr_data.edge_attr = fn_norm(tr_data.edge_attr, device)
            val_data.edge_attr = fn_norm(val_data.edge_attr, device)
            te_data.edge_attr = fn_norm(_te_data.edge_attr, device)
    elif isinstance(tr_data, HeteroData):
        tr_data[node_name].x = fn_norm(tr_data[node_name].x, device)
        val_data[node_name].x = fn_norm(val_data[node_name].x, device)
        te_data = _te_data.detach().clone()
        te_data[node_name].x = fn_norm(_te_data[node_name].x, device)
    else:
        logging.warning("Nothing to normalize!")
    return tr_data, val_data, te_data

def get_batch_size(config, args, batch):
    if args.readout == 'edge':
        _batch_size = batch.edge_label.shape[0]
    else: 
        _batch_size = batch[args.node_name].batch_size if config.multi_relational else batch.batch_size
    return _batch_size