"""
This file provides training/ tuning functionality for homogeneous/ multirelational GNNs.
"""
import pathlib
from xml.etree.ElementTree import TreeBuilder
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from munch import munchify
from torch_geometric.loader import NeighborLoader, DataLoader, ImbalancedSampler, LinkNeighborLoader
from torch_geometric.data import Data
from torch_geometric.utils import degree
from datetime import datetime
import os, psutil, sys, gc, time, pickle, logging, socket, json, optuna, torch

from tqdm import tqdm
import matplotlib.pyplot as plt

from models.gnn_type2 import adapt_model, PNA, GINe, newMLPe #, MLP2
from utils.evaluate import evaluate
from utils.gcn_utils import generate_filtered_graph, z_normalize, l2_normalize, GraphData, AddEgoIds, normalize_data, add_reverse, remove_reverse, get_batch_size
from utils.util import open_csv, write_csv, set_seed

from torch.utils.tensorboard import SummaryWriter
# from torch.profiler import profile, record_function, ProfilerActivity

script_start = time.time()

def get_model(config, args, params, sample_data, n_classes, readout, residual=False, num_relations=None, node_name=None):
#set seed
	if args.torch_seed is not None:
		set_seed(args.torch_seed)
		logging.info(f'Seed set to {args.torch_seed}')
		# Get feature dimensions
	if not config.model == "type2_hetero_sage":
		num_features = sample_data.x.shape[1]
		edge_dim = sample_data.edge_attr.shape[1] if config.network_type == 'type1' else None # need if??
		# Load model if requested (e.g., for finetuning or continuing training)
	if args.load_model:
		logging.info(f"Loading model from {args.model_path}")
		# model = torch.load(args.model_path)
		checkpoint = torch.load(f"{args.model_path}/model.pt")
		model = checkpoint['model']
		model = adapt_model(model, args, num_features, edge_dim, out_dim=n_classes)
		return model
		# Select the model
	if config.model == "gin":
		edge_features = True
		edge_updates = True if config.edge_updates2 else False
		config.generate_embedding = False # TODO: implement if needed
		model = GINe(num_features=num_features, num_gnn_layers=round(params['n_gnn_layers']), n_classes=n_classes,n_hidden=round(params['n_hidden']), embedding=config.generate_embedding, readout=readout,residual=residual, edge_features=edge_features, edge_updates=edge_updates, edge_dim=edge_dim, dropout=params['dropout'],reverse=config.reverse_mp, final_dropout=params['final_dropout'])
	elif config.model == "mlp":
		edge_features = True
		edge_updates = True if config.edge_updates2 else False
		config.generate_embedding = False # TODO: implement if needed
		model = newMLPe(num_features=num_features, num_gnn_layers=round(params['n_gnn_layers']), n_classes=n_classes,n_hidden=round(params['n_hidden']), embedding=config.generate_embedding, readout=readout,residual=residual, edge_features=edge_features, edge_updates=edge_updates, edge_dim=edge_dim, dropout=params['dropout'],reverse=config.reverse_mp, final_dropout=params['final_dropout'])
	elif config.model == "pna":
		edge_features = True
		edge_updates = True if config.edge_updates2 else False
		d = degree(sample_data.edge_index[1], num_nodes=sample_data.num_nodes, dtype=torch.long)
		deg = torch.bincount(d, minlength=1)
		model = PNA(num_features=num_features, num_gnn_layers=round(params['n_gnn_layers']), n_classes=n_classes,n_hidden=round(params['n_hidden']), embedding=config.generate_embedding, readout=readout,residual=residual, edge_features=edge_features, edge_updates=edge_updates, edge_dim=edge_dim, dropout=params['dropout'], deg=deg,reverse=config.reverse_mp, final_dropout=params['final_dropout'])
	elif config.model == "rgcn":
		raise ValueError('RGCN not currently implemented')
	return model
	
	
def train_gnn_type2(_tr_data, _te_data, _val_data, val_folds, te_inds, config, model_settings, args, csv, log_dir,save_run_embed=False, tb_logging=False, trial=None, **params):
	"""
    Trains type2 GNN and saves models/ logs performance metrics.
    :param _tr_data: Train data
    :param _val_data: Val data (optional)
    :param _te_data: Test data
    :param val_folds: List of validation folds (list of tuples)
    :param te_inds: edge indices used for testing
    :param config: Config file
    :param model_settings: Model settings file
    :param args: Auxiliary command line arguments
    :param csv: CSV file to which performance metrics and hyperparameters are being logged
    :param log_dir: Path to log directory
    :param params: Dictionary containing hyperparameter values
    :return: Returns torch model
    """
	
	#set seed
	if args.torch_seed is not None:
		set_seed(args.torch_seed)
		logging.info(f'Seed set to {args.torch_seed}')
		
	device = args.device
	imbalanced_sampling = False
	# if imbalanced_sampling:
	#     params['w_ce1'] = 1.0
	#     params['w_ce2'] = 1.0
	clipping = False
	
	if args.y_list is not None:
		acc_csv = open_csv(log_dir / f"accuracies_per_class.csv", header='run,epoch,'+','.join(args.y_list)+'\n')
		
	count = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
				count += 1
		except:
			pass
			
	node_name = None
	args.node_name = node_name
	
	# Get loss function
	if args.graph_simulator:
		params['loss'] = _tr_data.loss_fn
	else:
		y = _tr_data.y
		if len(y.unique()) > 2:
			params['loss'] = 'mse'
		elif y.shape[1] > 1:
			params['loss'] = 'bce_multi'
		else:
			loss_functions = {0:"ce", 1:"bce", 2:"mse", 3:"bce_multi"} #, 3:"fl"}
			try:
				if round(params['loss']) in loss_functions:
					params['loss'] = loss_functions[round(params['loss'])]
			except:
				assert params['loss'] in loss_functions.values(), f"only {list(loss_functions.values())} losses are currently implemented, {params['loss']} is not implemented"
				
				# Get readout
	if args.readout:
		readout = args.readout
	elif args.graph_simulator:
		readout = _tr_data.readout
	elif config.simulator == 'eth':
		readout = 'node'
	elif config.network_type == 'type1':
		readout = 'edge'
	else:
		readout = 'node'
	args.readout = readout
	
	if config.batching:
		if args.optuna:
			batch_size = params['batch_size']
		else:
			batch_size = config.batch_size
			
	best_tr_embed, best_val_embed, best_te_embed = torch.Tensor([0.]), torch.Tensor([0.]), torch.Tensor([0.])
	
	logging.info("Using Cuda GPU" if torch.cuda.is_available() else "No CUDA GPU available")
	
	## Set up TensorBoard logging
	if config.neighbor_list[0] > 0:
		model_name = f"{config.model}_sampled"
	else:
		model_name = f"{config.model}"
	if tb_logging:
		if config.simulator == 'eth':
			comment = f"_{model_name}_feats_{args.features}_nodefeats_{config.node_feats}_batchsize_{batch_size}_data_eth_{args.unique_name}"
		else:
		# comment = f"_{model_name}_{config.network_type}_loss_{params['loss']}_dropout_{params['dropout']}_feats_{args.features}_clip_{clipping}"
			if "medium" in config.data_dir:
				comment = f"_{model_name}_{config.network_type}_y_{args.y_pretrain}_feats_{args.features}_nodefeats_{config.node_feats}_eu2_{config.edge_updates2}_L2_{args.L2}_batchsize_{batch_size}_reversemp{config.reverse_mp}_data_medium_{args.unique_name}"
			elif "large" in config.data_dir:
				comment = f"_{model_name}_{config.network_type}_y_{args.y_pretrain}_feats_{args.features}_nodefeats_{config.node_feats}_eu2_{config.edge_updates2}_L2_{args.L2}_batchsize_{batch_size}_reversemp{config.reverse_mp}_data_large_{args.unique_name}"
			elif "100m" in config.data_dir:
				comment = f"_{model_name}_{config.network_type}_y_{args.y_pretrain}_feats_{args.features}_nodefeats_{config.node_feats}_eu2_{config.edge_updates2}_L2_{args.L2}_batchsize_{batch_size}_reversemp{config.reverse_mp}_data_100m_{args.unique_name}"
			else:
				comment = f"_{model_name}_{config.network_type}_y_{args.y_pretrain}_feats_{args.features}_nodefeats_{config.node_feats}_eu2_{config.edge_updates2}_L2_{args.L2}_batchsize_{batch_size}_reversemp{config.reverse_mp}_data_{args.unique_name}"
		current_time = datetime.now().strftime("%b%d_%H-%M-%S")
		writer = SummaryWriter(comment=comment)
		##### DONE #####
	else:
		current_time = datetime.now().strftime("%b%d_%H-%M-%S")
		
		## Set parameters that might need rounding or a dictionary lookup -- normalization_method, aggregation, loss_function
		# e.g. if categorical parameter, but random float is used in bayesian opt for the parameter
		# normalization method
	normalization_methods = {"z_normalize": z_normalize, "l2_normalize": l2_normalize}
	if 'norm_method' in params:
		if params['norm_method'] in normalization_methods:
			fn_norm = normalization_methods[params['norm_method']]
		else:
			normalization_methods = [z_normalize, l2_normalize]
			fn_norm = normalization_methods[round(params['norm_method'])]
			# node aggregation function
	aggr = ['sum', 'mean', 'max']
	if 'aggr' in params:
		if params['aggr'] not in aggr:
			params['aggr'] = aggr[round(params['aggr'])]
			# loss function and n_classes
	y_classes = _tr_data.y.shape[1] if 'y' in _tr_data else _tr_data[node_name].y.shape[1]
	if y_classes > 1:
		n_classes = y_classes
	else:
		n_classes = 1 if params['loss'] in ["bce", "mse"] else 2 #### TODO: automate this! and generalise code to more than 2 classes
		##### DONE #####
		
		# Initialize lists for some metrics we want to collect
	tr_metric_list, val_metric_list, te_metric_list = [], [], []
	tr_acc_list, val_acc_list, te_acc_list = [], [], []
	epoch_save = 0
	
	## Iterate over validation folds. In each iteration, split train/valid data, create data loaders, TRAIN model, VALIDATE model and TEST model
	for tr_inds, val_inds in val_folds[:1]:
	
	## Split train and val data and normalize
		if not config.multi_relational:
			if _val_data is not None:
				logging.info(f"using tr_, val_, te_ data")
				tr_data, val_data = _tr_data, _val_data
			elif config.simulator == "eth":
				tr_data, val_data = generate_filtered_graph(node_indices=[tr_inds, val_inds], data=_tr_data, relabel_nodes=True)
			elif readout == 'edge' and _val_data is not None:
				tr_data = GraphData(x=_tr_data.x.detach().clone(), y=_tr_data.y.detach().clone(),edge_index=_tr_data.edge_index.detach().clone(),edge_attr=_tr_data.edge_attr.detach().clone())
				val_data = GraphData(x=_val_data.x.detach().clone(), y=_val_data.y.detach().clone(),edge_index=_val_data.edge_index.detach().clone(),edge_attr=_val_data.edge_attr.detach().clone())
			else:
				tr_data, val_data = generate_filtered_graph(node_indices=[tr_inds, val_inds], data=_tr_data, relabel_nodes=True)
			if not args.edges == 'none':
				tr_data, val_data, te_data = normalize_data(tr_data, val_data, _te_data, fn_norm, device)
		else:
			raise ValueError('config.multi_relational not implemented')
			
		if config.batching:
			if args.n_gnn_layers is not None:
				logging.info(f"setting no. gnn layers to {args.n_gnn_layers}")
				params['n_gnn_layers'] = args.n_gnn_layers
			num_neighbors = config.neighbor_list
			num_neighbors = [num_neighbors[i] for i in range(round(params['n_gnn_layers']))]
			if config.multi_relational:
				raise ValueError('config.multi_relational not implemented')
				
			elif not config.multi_relational:
				tr_input_nodes = torch.arange(tr_data.x.shape[0])
				val_input_nodes = torch.arange(val_data.x.shape[0])
				te_input_nodes = torch.arange(te_data.x.shape[0])
				
				tr_data.node_indices = tr_input_nodes
				val_data.node_indices = val_input_nodes
				te_data.node_indices = te_input_nodes
				
			if imbalanced_sampling:
				logging.info(f"Using imbalanced sampler")
				sampler = ImbalancedSampler(tr_data, input_nodes=tr_input_nodes)
				shuffle = False
			else:
				sampler = None
				shuffle = True
				
			num_workers = 0
			
			if args.y_from_file or config.simulator == "eth":
				tr_inds, val_inds = val_folds[0]
				tr_input_nodes = tr_inds
				val_input_nodes = val_inds
				te_input_nodes = te_inds
				tr_data.node_indices = tr_input_nodes
				val_data.node_indices = val_input_nodes
				te_data.node_indices = te_input_nodes
				
			for data in [tr_data, val_data, te_data]:
				data.readout = args.readout
				
			for name, data in zip(['train', 'validation', 'test'], [tr_data, val_data, te_data]):
				logging.info(f"{name} shapes: x = {data.x.shape}, y = {data.y.shape}, edge_index = {data.edge_index.shape}, edge_attr = {data.edge_attr.shape}")
				
			if config.reverse_mp:
				tr_data = add_reverse(tr_data)
				val_data = add_reverse(val_data)
				te_data = add_reverse(te_data)
				
			if args.ego:
				transform = AddEgoIds()
			else:
				transform = None
				
			logging.info(f"num_neighbors = {num_neighbors}")
			if readout == "node" or readout == "graph":
				train_loader = NeighborLoader(tr_data, input_nodes=tr_input_nodes,num_neighbors=num_neighbors, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, transform=transform,disjoint=args.disjoint)
				val_loader = NeighborLoader(val_data, input_nodes=val_input_nodes,num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False, num_workers=num_workers, transform=transform,disjoint=args.disjoint)
				test_loader = NeighborLoader(te_data, input_nodes=te_input_nodes,num_neighbors=num_neighbors, batch_size=batch_size, shuffle=False, num_workers=num_workers, transform=transform,disjoint=args.disjoint)
			elif readout == "edge":
				train_loader = LinkNeighborLoader(tr_data,num_neighbors=num_neighbors, # edge_label_index=tr_data.edge_index[:, tr_inds], edge_label=tr_data.y[tr_inds],batch_size=batch_size, shuffle=shuffle, sampler=sampler)
				val_loader = LinkNeighborLoader(val_data,num_neighbors=num_neighbors, edge_label_index=val_data.edge_index[:, val_inds], edge_label=val_data.y[val_inds],batch_size=batch_size, shuffle=False)
				test_loader = LinkNeighborLoader(te_data,num_neighbors=num_neighbors, edge_label_index=te_data.edge_index[:, te_inds], edge_label=te_data.y[te_inds],batch_size=batch_size, shuffle=False)
			else:
				train_loader = DataLoader(tr_data, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
				val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
				test_loader = DataLoader(te_data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
				
				# tr_data, val_data, te_data = tr_data.to(device), val_data.to(device), te_data.to(device)
				
		if not config.multi_relational:
			logging.info(f"test_loader = {test_loader}")
			sample_batch = next(iter(test_loader))
			if config.reverse_mp: sample_batch = remove_reverse(sample_batch)
			_tr_data = sample_batch
			sample_data = sample_batch
			
		residual = False if readout == 'edge' else True
		if config.model == "rgcn":
			num_relations = _tr_data.edge_attr[:, config.edge_type_col_ind].unique().shape[0] #TODO: use full train data, or even ALL data, but not sample batch
		else:
			num_relations = None
			
		if args.repeat is not None and os.path.isfile(f'/dccstor/aml-e/datasets/model_checkpoints/checkpoint_{args.unique_name}.tar'):
			logging.info('Loading Model...')
			checkpoint = torch.load(f'/dccstor/aml-e/datasets/model_checkpoints/checkpoint_{args.unique_name}.tar')
			model = checkpoint['model']
		else:
			model = get_model(config, args, params, sample_data, n_classes, readout, residual=residual, num_relations=num_relations, node_name=node_name)
			
		model.to(device)
		
		if args.L2:
			optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=6.898837451757206e-5)
		elif args.optuna:
			optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
		else:
			optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
			if args.repeat is not None and os.path.isfile(f'/dccstor/aml-e/datasets/model_checkpoints/checkpoint_{args.unique_name}.tar'):
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
				
		scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
		if params['loss'] == "ce":
			logging.info('using CE loss')
			loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([params['w_ce1'], params['w_ce2']]).to(device))
			y_type = 'binary'
		elif params['loss'] == "bce":
			logging.info('using BCE loss')
			num_pos_labels = torch.sum(_tr_data.y) if 'y' in _tr_data else torch.sum(_tr_data[node_name].y)
			total_labels = _tr_data.y.shape[0] if 'y' in _tr_data else _tr_data[node_name].y.shape[0]
			num_neg_labels = total_labels - num_pos_labels
			pos_weight = torch.tensor([torch.div(num_neg_labels, num_pos_labels, rounding_mode='floor')], device=device)
			loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
			y_type = 'binary'
		elif params['loss'] == "bce_multi":
			logging.info('using BCE loss with multiple classes, no weights')
			loss_fn = torch.nn.BCEWithLogitsLoss()
			y_type = 'multiclass'
		elif params['loss'] == "mse":
			logging.info('using MSE loss')
			loss_fn = torch.nn.MSELoss()
			y_type = 'continuous'
			
			# Main training loop
		epoch, n_waiting_rounds, best_val_loss, best_val_score = 0, 0, 1e9, 0
		if args.repeat is not None and os.path.isfile(f'/dccstor/aml-e/datasets/model_checkpoints/checkpoint_{args.unique_name}.tar'):
			epoch = checkpoint['epoch']
		t_max_epoch = 0
		t_epochs = []
		train_losses, val_losses = [], []
		if args.load_opt:
			checkpoint = torch.load(f"{args.model_path}/model.pt")
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			# epoch = checkpoint['epoch']
			tr_loss = checkpoint['loss']
		while True:
			epoch_start = time.time()
			logging.debug(f"epoch = {epoch}")
			epoch += 1
			model.train()
			optimizer.zero_grad()
			if not config.batching:
				raise ValueError('NOT config.batching is not implemented')
			else:
				tr_batch_losses = []
				n_hidden = round(params['n_hidden'])
				n_rows_tr_embed = tr_data[node_name].x.shape[0] if config.multi_relational else tr_data.x.shape[0]
				_tr_embed = torch.empty(n_rows_tr_embed, 2 * n_hidden, device='cpu') if config.model in ['type2_hetero_sage', 'type2_gnn_mlp'] else torch.empty(n_rows_tr_embed, n_hidden, device='cpu')
				for batch in tqdm(train_loader, disable=not args.tqdm):
					try:
						if config.reverse_mp: batch = remove_reverse(batch)
						batch.to(device)
						_batch_size = get_batch_size(config, args, batch)
						node_inds = batch[node_name].node_indices[:_batch_size] if config.multi_relational else batch.node_indices[:_batch_size]
						optimizer.zero_grad()
						
						out, _tr_batch_embed = model(batch)
						out = out[:_batch_size]
						if config.generate_embedding: _tr_embed[node_inds] = _tr_batch_embed[:_batch_size].detach().cpu()
						tr_y = batch[node_name].y[:_batch_size] if config.model == "type2_hetero_sage" else batch.y[:_batch_size]
						if n_classes > 1:
							tr_y = tr_y.view(-1).long() if params['loss'] == 'ce' else tr_y.float()
							tr_loss = loss_fn(out, tr_y)
						else:
							tr_y = tr_y.view(-1)
							tr_loss = loss_fn(out.squeeze(), tr_y.float())
							
						tr_batch_losses.append(tr_loss.detach())
						tr_loss.backward()
						if clipping:
							torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
						optimizer.step()
						# scheduler.step()
					except RuntimeError as e:
						raise
							
				tr_loss_avg = torch.mean(torch.tensor(tr_batch_losses))
				
				model.eval()
				with torch.no_grad():
					val_batch_losses = []
					n_rows_val_embed = val_data[node_name].x.shape[0] if config.multi_relational else val_data.x.shape[0]
					_val_embed = torch.empty(n_rows_val_embed, 2 * n_hidden, device='cpu') if config.model in ['type2_hetero_sage', 'type2_gnn_mlp'] else torch.empty(n_rows_val_embed, n_hidden, device='cpu')
					for batch in val_loader:
						if config.reverse_mp: batch = remove_reverse(batch)
						batch.to(device)
						_batch_size = get_batch_size(config, args, batch)
						node_inds = batch[node_name].node_indices[:_batch_size] if config.multi_relational else batch.node_indices[:_batch_size]
						out, _val_batch_embed = model(batch)
						out = out[:_batch_size]
						if config.generate_embedding: _val_embed[node_inds] = _val_batch_embed[:_batch_size].detach().cpu()
						val_y = batch[node_name].y[:_batch_size] if config.model == "type2_hetero_sage" else batch.y[:_batch_size]
						if n_classes > 1:
							val_y = val_y.view(-1).long() if params['loss'] == 'ce' else val_y.float()
							val_loss = loss_fn(out, val_y)
						else:
							val_y = val_y.view(-1)
							val_loss = loss_fn(out.squeeze(), val_y.float())
							
						val_batch_losses.append(val_loss.detach())
						
					val_loss = torch.mean(torch.tensor(val_batch_losses))
					
			scheduler.step()
			train_losses.append(tr_loss_avg.item())
			val_losses.append(val_loss.item())
			if tb_logging:
				writer.add_scalar('Loss/train', tr_loss_avg.item(), epoch)
				writer.add_scalar('Loss/val', val_loss.item(), epoch)
				
			val_score = evaluate(val_loader, model, config, args, y_type=y_type, only_f1=True)
			logging.debug(f"Val_score = {val_score}")
			update = False
			if val_score > best_val_score or epoch == 1:
				update = True
				# best_val_loss = val_loss
				best_val_score = val_score
				n_waiting_rounds = 0
				epoch_save = epoch
				if config.batching:
					if args.save_preds:
						tr_metrics, best_tr_preds = evaluate(train_loader, model, config, args, y_type=y_type, return_preds=True)
						val_metrics, best_val_preds = evaluate(val_loader, model, config, args, y_type=y_type, return_preds=True)
						te_metrics, best_te_preds = evaluate(test_loader, model, config, args, y_type=y_type, return_preds=True)
					else:
						tr_metrics = evaluate(train_loader, model, config, args, y_type=y_type)
						val_metrics = evaluate(val_loader, model, config, args, y_type=y_type)
						te_metrics = evaluate(test_loader, model, config, args, y_type=y_type)
				else:
					tr_metrics = evaluate(tr_data, model, config, args, y_type=y_type)
					val_metrics = evaluate(val_data, model, config, args, y_type=y_type)
					te_metrics = evaluate(te_data, model, config, args, y_type=y_type)
				best_tr_metrics = tr_metrics
				best_val_metrics = val_metrics
				best_te_metrics = te_metrics
			elif tb_logging:
				if config.batching:
					k_eval = 50
					tr_metrics = evaluate(train_loader, model, config, args, topk=k_eval, y_type=y_type, only_f1=args.only_f1)
					val_metrics = val_score if args.only_f1 else evaluate(val_loader, model, config, args, y_type=y_type)
					te_metrics = evaluate(test_loader, model, config, args, topk=k_eval, y_type=y_type, only_f1=args.only_f1)
				else:
					tr_metrics = evaluate(tr_data, model, config, args, y_type=y_type, only_f1=args.only_f1)
					val_metrics = val_score if args.only_f1 else evaluate(val_loader, model, config, args, y_type=y_type)
					te_metrics = evaluate(te_data, model, config, args, y_type=y_type, only_f1=args.only_f1)
			if tb_logging:
				if args.only_f1:
					if type(tr_metrics) is tuple:
						writer.add_scalar(f"f1/train", tr_metrics[3], epoch)
						writer.add_scalar(f"f1/val", val_metrics[3], epoch)
						writer.add_scalar(f"f1/test", te_metrics[3], epoch)
					else:
						writer.add_scalar(f"f1/train", tr_metrics, epoch)
						writer.add_scalar(f"f1/val", val_metrics, epoch)
						writer.add_scalar(f"f1/test", te_metrics, epoch)
				else:
				# accs, pres, recs, f1s, aucs
					for i, metric in enumerate(['acc', 'prec', 'rec', 'f1', 'auc', 'ap']):
						writer.add_scalar(f"{metric}/train", tr_metrics[i], epoch)
						writer.add_scalar(f"{metric}/val", val_metrics[i], epoch)
						writer.add_scalar(f"{metric}/test", te_metrics[i], epoch)
					if args.y_list is not None:
						if args.tb: write_csv(acc_csv, [f"run_x", epoch, *te_metrics[-1]])
						
			if update:
				if config.generate_embedding:
					if config.batching:
						n_rows_te_embed = te_data[node_name].x.shape[0] if config.multi_relational else te_data.x.shape[0]
						_te_embed = torch.empty(n_rows_te_embed, 2 * n_hidden, device='cpu') if config.model in ['type2_hetero_sage', 'type2_gnn_mlp'] else torch.empty(n_rows_te_embed, n_hidden, device='cpu')
						for batch in test_loader:
							if config.reverse_mp: batch = remove_reverse(batch)
							batch.to(device)
							_batch_size = get_batch_size(config, args, batch)
							node_inds = batch[node_name].node_indices[:_batch_size] if config.multi_relational else batch.node_indices[:_batch_size]
							out, _te_batch_embed = model(batch)
							_te_embed[node_inds] = _te_batch_embed[:_batch_size].detach().cpu()
					else:
						_, _te_embed = model(te_data)
						
					best_tr_embed = _tr_embed.cpu()
					best_val_embed = _val_embed.cpu()
					best_te_embed = _te_embed.cpu()
					
			epoch_end = time.time()
			t_epoch = epoch_end - epoch_start
			t_epochs.append(t_epoch)
			logging.debug(f"AVG epoch time = {sum(t_epochs)/len(t_epochs)}")
			t_elapsed = epoch_end - script_start
			logging.debug(f"Time elapsed = {t_elapsed}")
			if t_epoch > t_max_epoch:
				t_max_epoch = t_epoch
			logging.debug(f"Current epoch time = {t_max_epoch} (max={t_max_epoch})")
			t_left = 84600 - t_max_epoch - t_elapsed
			logging.debug(f"Time left = {t_left}")
			
			if t_left < 0 and args.repeat is not None:
				logging.info(f"Saving model and submitting new job")
				torch.save({'epoch': epoch,'model': model,'optimizer_state_dict': optimizer.state_dict()}, f'/dccstor/aml-e/datasets/model_checkpoints/checkpoint_{args.unique_name}.tar')
				# if int(command_list[idx]) > 0:
				# 	re_run(mem, command)
					
			if n_waiting_rounds > config.early_stopping_rounds or epoch >= config.n_epochs or t_left < 0:
				tr_metric_list.append([epoch_save, *best_tr_metrics[:5]])
				val_metric_list.append(best_val_metrics[:5])
				te_metric_list.append(best_te_metrics[:5])
				tr_acc_list.append([epoch_save, *best_tr_metrics[-1]])
				val_acc_list.append([epoch_save, *best_val_metrics[-1]])
				te_acc_list.append([epoch_save, *best_te_metrics[-1]])
				curves = dict(zip(['roc_curve', 'p_r_curve'], best_te_metrics[6:8]))
				curves_labels = dict(zip(['roc_curve', 'p_r_curve'], [('fpr','tpr'), ('precision', 'recall')]))
				logging.info(f"Breaking current training config after {n_waiting_rounds} waiting rounds and {epoch} total epochs")
				break
			else:
				n_waiting_rounds += 1
				
				
			if args.optuna:
			# Add prune mechanism
				trial.report(best_val_score, epoch)
				
				if trial.should_prune():
					raise optuna.exceptions.TrialPruned()
					
					
	run_file = log_dir / "runs"
	current_runs = [x for x in pathlib.Path(run_file).glob("*") if 'run' in str(x)]
	if len(current_runs) == 0:
		run_id = 1
	else:
		run_id = max([int(str(x).split("_")[-1]) for x in current_runs]) + 1
	pathlib.Path(run_file / f"run_{run_id}").mkdir(parents=True, exist_ok=True)
	
	# saving embeddings from current run
	if save_run_embed:
		torch.save(best_tr_embed.detach().cpu(), run_file / f"run_{run_id}/best_tr_embed.pt")
		torch.save(best_val_embed.detach().cpu(), run_file / f"run_{run_id}/best_val_embed.pt")
		torch.save(best_te_embed.detach().cpu(), run_file / f"run_{run_id}/best_te_embed.pt")
		
		# saving train losses
	df = pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses})
	df.to_csv(run_file / f"run_{run_id}/train_losses.csv")
	
	# saving roc and p_r curves as plots
	# with open(run_file / f"run_{run_id}/curves.pickle", 'wb') as handle:
	#     pickle.dump(curves, handle, protocol=pickle.HIGHEST_PROTOCOL)
	if args.save_preds:
		for suff, pred in zip(['tr', 'val', 'te'], [best_tr_preds, best_val_preds, best_te_preds]):
			with open(run_file / f"run_{run_id}/{suff}_preds.pickle", 'wb') as handle:
				pickle.dump(pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
	for name, values in curves.items():
		x,y = values
		x_label, y_label = curves_labels[name]
		# df = pd.DataFrame({x_label: x, y_label: y})
		# df.to_csv(run_file / f"run_{run_id}/{name}.csv")
		plt.plot(x, y)
		plt.title(f"{name}")
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.savefig(run_file / f"run_{run_id}/{name}.png")
		plt.close()
		
		# saving model
	if args.save_model:
		save_path = run_file / f"run_{run_id}/model.pt"
		# torch.save(model, save_path)
		torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'model': model,'optimizer_state_dict': optimizer.state_dict(),'loss': tr_loss,}, save_path)
		
		# logging train, val, test metrics in general performance file
	tr_metric_list = np.array(tr_metric_list).mean(0)
	val_metric_list = np.array(val_metric_list).mean(0)
	te_metric_list = np.array(te_metric_list).mean(0)
	tr_acc_list = np.array(tr_acc_list).mean(0)
	val_acc_list = np.array(val_acc_list).mean(0)
	te_acc_list = np.array(te_acc_list).mean(0)
	paramss_reordered = [params[x] for x in model_settings.params.keys()]
	if tb_logging:
		tb_log = current_time + "_" + socket.gethostname() + comment
	else:
		tb_log = current_time + "_" + socket.gethostname()
	logs = [f"run_{run_id}", tb_log, *paramss_reordered, *tr_metric_list, *val_metric_list, *te_metric_list]
	write_csv(csv, logs)
	if args.y_list is not None: write_csv(acc_csv, [f"run_{run_id}", *te_acc_list])
	
	if tb_logging:
	# logging.info(f"Closing Tensorboard.")
		writer.close()
		
		# process = psutil.Process(os.getpid())
		# logging.debug(f"mem use: {process.memory_info().rss / 1024 ** 2:.2f}")
	model.cuda()
	model.cpu()
	del model
	gc.collect()
	del optimizer
	torch.cuda.empty_cache()
	model = None
	
	if args.optuna:
		return best_val_score
	else:
		return val_metric_list[3], best_val_loss, model, best_tr_embed, best_val_embed, best_te_embed, run_id
		
		
def bayes_opt_tune_type2_gnn(_tr_data, _te_data, _val_data, val_folds, te_inds, config, model_settings, args, csv, log_dir):

	val_f1s, te_f1s = [], []
	# tr_embeddings, val_embeddings, te_embeddings = [], [], []
	global meta_best_val_f1
	global meta_best_tr_embed
	global meta_best_val_embed
	global meta_best_te_embed
	meta_best_val_f1 = 0.0
	meta_best_tr_embed = torch.Tensor([0.])
	meta_best_val_embed = torch.Tensor([0.])
	meta_best_te_embed = torch.Tensor([0.])
	# if args.load_model is not None: config.n_epochs = config.n_epochs * 2
	
	def model_function(**params):
		val_metric, best_val_loss, model, best_tr_embed, best_val_embed, best_te_embed, run_id = train_gnn_type2(_tr_data, _te_data, _val_data, val_folds, te_inds, config, model_settings, args, csv, log_dir, tb_logging=args.tb, **params)
		global meta_best_val_f1
		global meta_best_tr_embed
		global meta_best_val_embed
		global meta_best_te_embed
		if config.generate_embedding:
			if val_metric > meta_best_val_f1:
				meta_best_val_f1 = val_metric
				logging.info(f"meta val metric update = {meta_best_val_f1}")
				meta_best_tr_embed = best_tr_embed
				meta_best_val_embed = best_val_embed
				meta_best_te_embed = best_te_embed
				
				torch.save(meta_best_tr_embed, log_dir / "best_tr_embed.pt")
				torch.save(meta_best_val_embed, log_dir / "best_val_embed.pt")
				torch.save(meta_best_te_embed, log_dir / "best_te_embed.pt")
		return val_metric
		
	param_space = model_settings.bayes_opt_params
	bayes_optimizer = BayesianOptimization(model_function, param_space, random_state=config.bayes_seed)
	bayes_optimizer.maximize(n_iter=config.n_bayes_opt_iters,init_points=config.n_init_points_bayes_opt,acq='ei',alpha=1e-1)  # what about kappa decay?...
	
	if config.generate_embedding:
		torch.save(meta_best_tr_embed, log_dir / "best_tr_embed.pt")
		torch.save(meta_best_val_embed, log_dir / "best_val_embed.pt")
		torch.save(meta_best_te_embed, log_dir / "best_te_embed.pt")
		
	params = munchify(bayes_optimizer.max['params'])
	
	return params