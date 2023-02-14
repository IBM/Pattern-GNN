"""
This file provides different type2 GNN architectures. Currently only 'NetSAGE' and 'HeteroNetSAGE' have been properly
implemented for training (corresponding to 'type2_gnn_mlp' and 'type2_hetero_sage')
"""
from audioop import reverse
import os, sys, logging
from re import X

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm, Linear, GATv2Conv, HeteroConv, MetaLayer, PNAConv
from torch_geometric.nn import GINConv, GINEConv, global_add_pool
from torch_geometric.nn import RGCNConv, FastRGCNConv, RGATConv

# from inits import glorot, zeros

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.mlp import MLP


class PNA(torch.nn.Module):
	def __init__(self, num_features, num_gnn_layers, n_classes=2,n_hidden=100, embedding=False, edge_features=False, edge_updates=True,readout='node', residual=False, edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None, reverse=False):
		super().__init__()
		n_hidden = int((n_hidden // 5) * 5)
		self.n_hidden = n_hidden
		self.readout = readout
		self.num_gnn_layers = num_gnn_layers
		self.reverse = reverse
		self.edge_updates = edge_updates
		self.final_dropout = final_dropout
		
		aggregators = ['mean', 'min', 'max', 'std']
		scalers = ['identity', 'amplification', 'attenuation']
		
		self.node_emb = nn.Linear(num_features, n_hidden)
		self.edge_emb = nn.Linear(edge_dim, n_hidden)
		
		self.convs = nn.ModuleList()
		self.emlps = nn.ModuleList()
		self.batch_norms = nn.ModuleList()
		self.convs_r = nn.ModuleList()
		self.emlps_r = nn.ModuleList()
		self.batch_norms_r = nn.ModuleList()
		for _ in range(self.num_gnn_layers):
			conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,aggregators=aggregators, scalers=scalers, deg=deg,edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,divide_input=False)
			if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
			self.convs.append(conv)
			self.batch_norms.append(BatchNorm(n_hidden))
			if self.reverse:
				conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,aggregators=aggregators, scalers=scalers, deg=deg,edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,divide_input=False)
				if self.edge_updates: self.emlps_r.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
				self.convs_r.append(conv)
				self.batch_norms_r.append(BatchNorm(n_hidden))
				
		if self.readout == 'edge':
			self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
		elif self.readout == 'node':
			self.mlp = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
		elif self.readout == 'graph':
			self.mlp = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), Linear(50, 25), nn.ReLU(),Linear(25, n_classes))
			
	def forward(self, data):
		x = data.x #.round().long()
		edge_index = data.edge_index
		edge_attr = data.edge_attr #.round().long()
		src, dst = edge_index
		
		# x = self.node_emb(x.squeeze())
		x = self.node_emb(x)
		edge_attr = self.edge_emb(edge_attr)
		
		for i in range(self.num_gnn_layers):
			if self.reverse:
				x_f = F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))
				x_r = F.relu(self.batch_norms_r[i](self.convs_r[i](x, edge_index.flipud(), edge_attr)))
				x = (x + x_f + x_r) / 3
				if self.edge_updates:
					edge_attr_f = self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1))
					edge_attr_r = self.emlps_r[i](torch.cat([x[dst], x[src], edge_attr], dim=-1))
					edge_attr = (edge_attr + edge_attr_f + edge_attr_r) / 3
			else:
				x = x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr))) / 2
				if self.edge_updates:
					edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
					
					
		if self.readout == 'graph':
		# logging.info(f"data.batch: {data.batch}") # TODO: batch not implemented for NeighborLoader, see: https://github.com/pyg-team/pytorch_geometric/discussions/5660
			x = global_add_pool(x, data.batch)
		elif self.readout == 'edge':
			logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
			x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
			logging.debug(f"x.shape = {x.shape}")
			x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
			logging.debug(f"x.shape = {x.shape}")
		out = x
		return self.mlp(out), None
		
class GINe(torch.nn.Module):
	def __init__(self, num_features, num_gnn_layers, n_classes=2,n_hidden=100, embedding=False, edge_features=False, edge_updates=False,readout='node', residual=True, edge_dim=None, dropout=0.0, final_dropout=0.5, reverse=True):
		super().__init__()
		self.n_hidden = n_hidden
		self.readout = readout
		self.num_gnn_layers = num_gnn_layers
		self.reverse = reverse
		self.edge_updates = edge_updates
		self.final_dropout = final_dropout
		
		self.node_emb = nn.Linear(num_features, n_hidden)
		self.edge_emb = nn.Linear(edge_dim, n_hidden)
		
		self.convs = nn.ModuleList()
		self.emlps = nn.ModuleList()
		self.batch_norms = nn.ModuleList()
		self.convs_r = nn.ModuleList()
		self.emlps_r = nn.ModuleList()
		self.batch_norms_r = nn.ModuleList()
		for _ in range(self.num_gnn_layers):
			conv = GINEConv(nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden)), edge_dim=self.n_hidden)
			if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
			self.convs.append(conv)
			self.batch_norms.append(BatchNorm(n_hidden))
			if self.reverse:
				conv = GINEConv(nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden)), edge_dim=self.n_hidden)
				if self.edge_updates: self.emlps_r.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
				self.convs_r.append(conv)
				self.batch_norms_r.append(BatchNorm(n_hidden))
				
		if self.readout == 'edge':
			self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
		elif self.readout == 'node':
			self.mlp = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
		elif self.readout == 'graph':
			self.mlp = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), Linear(50, 25), nn.ReLU(),Linear(25, n_classes))
			
	def forward(self, data):
		x = data.x #.round().long()
		edge_index = data.edge_index
		edge_attr = data.edge_attr #.round().long()
		src, dst = edge_index
		
		# x = self.node_emb(x.squeeze())
		x = self.node_emb(x)
		edge_attr = self.edge_emb(edge_attr)
		
		for i in range(self.num_gnn_layers):
			if self.reverse:
				x_f = F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))
				x_r = F.relu(self.batch_norms_r[i](self.convs_r[i](x, edge_index.flipud(), edge_attr)))
				x = (x + x_f + x_r) / 3
				if self.edge_updates:
					edge_attr_f = self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1))
					edge_attr_r = self.emlps_r[i](torch.cat([x[dst], x[src], edge_attr], dim=-1))
					edge_attr = (edge_attr + edge_attr_f + edge_attr_r) / 3
			else:
				x = x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr))) / 2
				if self.edge_updates:
					edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
					
					
		if self.readout == 'graph':
			x = global_add_pool(x, data.batch)
		elif self.readout == 'edge':
			logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
			x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
			logging.debug(f"x.shape = {x.shape}")
			x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
			logging.debug(f"x.shape = {x.shape}")
		out = x
		return self.mlp(out), None
		
		
# New MLP model that is better aligned with the GNNs (in terms of no. layers, batch norm, dropout, etc.)
class newMLPe(torch.nn.Module):
	def __init__(self, num_features, num_gnn_layers, n_classes=2,n_hidden=100, embedding=False, edge_features=False, edge_updates=False,readout='node', residual=True, edge_dim=None, dropout=0.0, final_dropout=0.5, reverse=True):
		super().__init__()
		self.n_hidden = n_hidden
		self.readout = readout
		self.n_layers = num_gnn_layers
		self.reverse = reverse
		self.edge_updates = edge_updates
		self.final_dropout = final_dropout
		
		self.node_emb = nn.Linear(num_features, n_hidden)
		self.edge_emb = nn.Linear(edge_dim, n_hidden)
		
		if self.readout == 'edge':
			self.mlp = MLP(n_in=3 * self.n_hidden, n_out=self.n_hidden, n_hidden=self.n_hidden, n_layers=self.n_layers)
		else:
			self.mlp = MLP(n_in=self.n_hidden, n_out=self.n_hidden, n_hidden=self.n_hidden, n_layers=self.n_layers)
			
		self.mlp_out = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))
		
	def forward(self, data):
		x = data.x #.round().long()
		edge_index = data.edge_index
		edge_attr = data.edge_attr #.round().long()
		src, dst = edge_index
		
		# x = self.node_emb(x.squeeze())
		x = self.node_emb(x)
		edge_attr = self.edge_emb(edge_attr)
		
		if self.readout == 'graph':
			x = global_add_pool(x, data.batch)
		elif self.readout == 'edge':
			x = self.mlp(torch.cat([x[src], x[dst], edge_attr], dim=-1))
		else:
			x = self.mlp(x)
		out = x
		return self.mlp_out(out), None
		
		
# Used when loading a pretrained model for finetuning on continuing training
def adapt_model(model, args, node_dim, edge_dim, out_dim, n_classes=2):
	if args.freeze:
		for param in model.parameters():
			param.requires_grad = False
	# Replace the fully-connected layers
	# Parameters of newly constructed modules have requires_grad=True by default
	if args.swap_in:
		if hasattr(model, 'edge_emb'):
			model.edge_emb = torch.nn.Linear(edge_dim, model.edge_emb.out_features)
		if hasattr(model, 'node_emb'):
			model.node_emb = torch.nn.Linear(node_dim, model.node_emb.out_features)
	if args.swap_out:
		if hasattr(model, 'mlp'):
			model.mlp[-1] = torch.nn.Linear(model.mlp[-1].weight.shape[1], out_dim)
	return model