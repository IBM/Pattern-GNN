import optuna
import pathlib
from train_helpers.train_gnn_type2 import train_gnn_type2
from train_helpers.train_setup import get_gnn_data, get_eth_data, get_gnn_data_from_file
from train_helpers.simulator import get_gnn_data_from_simulator
from utils.util import open_csv, get_settings, get_log_model_name
import sys

config, model_settings, args = get_settings()
out_dir = f"/u/lucv/FC/gnn_logs/optuna"
if args.features in ["raw", "mf"]:
	out_dir = f"{out_dir}_{args.features}"
if args.y:
	out_dir = f"{out_dir}_{args.y}"
if config.simulator == "eth":
	dataset_name = 'eth'
	log_model_name = get_log_model_name(config, args, sim_num_nodes=False)
	log_dir = pathlib.Path(f"{out_dir}/{dataset_name}/{config.split_method}/{log_model_name}")
	# log_dir = pathlib.Path(f"{config.data_dir}/logs/{config.split_method}/{config.model}")
	pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
	pathlib.Path(log_dir / "runs").mkdir(parents=True, exist_ok=True)
	csv = open_csv(log_dir / f"performance_metrics.csv", header=model_settings.header)
	
	# tr_data, te_data, val_folds, te_inds = get_eth_data(config, args)
	# val_data = None
	tr_data, val_data, te_data, val_folds, te_inds = get_eth_data(config, args)
	
	print('adding ports')
	for data in [tr_data, val_data, te_data]:
		print(type(tr_data))
		print(tr_data.edge_attr.shape)
		# data.set_y(functions[args.y])
		if args.ports:
			data.add_ports()
	print('done')
	
	
	
elif config.model_type == "gnn":
	dataset_name = config.data_dir.strip('/').split('/')[-1]
	if args.graph_simulator: dataset_name = args.y
	log_model_name = get_log_model_name(config, args)
	log_dir = pathlib.Path(f"{out_dir}/{dataset_name}/{config.split_method}/{log_model_name}")
	# log_dir = pathlib.Path(f"{config.data_dir}/logs/{config.split_method}/{config.model}")
	pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
	pathlib.Path(log_dir / "runs").mkdir(parents=True, exist_ok=True)
	csv = open_csv(log_dir / f"performance_metrics.csv", header=model_settings.header)
	if args.graph_simulator:
		tr_data, val_data, te_data = get_gnn_data_from_simulator(config, args,num_nodes = args.sim_num_nodes,avg_degree = args.sim_avg_degree,delta = args.sim_delta)
		te_inds = None
		val_folds = [(None, None)]
	else:
		if args.y:
			tr_data, te_data, val_folds, te_inds = get_gnn_data_from_file(config, args)
		else:
			if config.network_type == "type1":
				tr_data, val_data, te_data, val_folds, te_inds = get_gnn_data(config, args)
				for data in [tr_data, val_data, te_data]:
					if args.ports:
						data.add_ports()
			else:
				val_data = None
				tr_data, te_data, val_folds, te_inds = get_gnn_data(config, args)
				for data in [tr_data, te_data]:
					if args.ports:
						data.add_ports()
						
def objective(trial):

	params = {'lr': trial.suggest_float('lr', 5e-5, 1e-2),'n_hidden': trial.suggest_int('n_hidden', 16, 64),'n_mlp_layers': 1,'n_gnn_layers': trial.suggest_int('n_gnn_layers', 2, 4),'loss': 'ce','w_ce1': 1,'w_ce2': trial.suggest_float('w_ce2', 1, 2),'norm_method': 'z_normalize','dropout': trial.suggest_float('dropout', 0.05, 0.2),'final_dropout': trial.suggest_float('final_dropout', 0.05, 0.2),'batch_size': 2048,'weight_decay': 0}
	
	f1 = train_gnn_type2(tr_data, te_data, val_data, val_folds, te_inds, config, model_settings, args, csv, log_dir,save_run_embed=config.generate_embedding, tb_logging=args.tb, trial=trial, **params)
	
	return f1
	
storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage("/dccstor/aml-e/datasets/optuna/eth_1.log"),)

study = optuna.create_study(direction="maximize", study_name='eth_1', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_startup_trials=32, n_warmup_steps=40, interval_steps=5), storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=10)