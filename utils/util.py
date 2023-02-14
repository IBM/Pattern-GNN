"""
This script provides a number of utility functions for parsing command line arguments and logging performance metrics
"""
import os, sys, json, argparse, copy, random, datetime, logging, torch
import numpy as np
import pandas as pd
from pathlib import Path
from munch import munchify

pretrain_feats_sim = {"all_simulated": ['deg_in', 'deg_out', 'fan_in', 'fan_out', 'ratio_in', 'ratio_out', 'deg_in_timespan', 'timevar_in', 'timevar_out', 'max_deg_in', 'C2_check', 'C3_check', 'C4_check', 'C5_check', 'C6_check',  'C2_count', 'C3_count', 'C4_count', 'K3_check', 'K4_check', 'K5_check', 'K6_check', 'K7_check', 'K3_count', 'K4_count', 'K5_count', 'K6_count', 'K7_count', 'SG1_check', 'SG2_check', 'SG3_check', 'SG4_check', 'BP1_check', 'BP2_check', 'BP3_check', 'BP4_check', 'SG_max'],"binary_2hop": ['deg_in_check', 'deg_out_check', 'fan_in_check', 'fan_out_check', 'ratio_in_check', 'ratio_out_check', 'deg_in_timespan_check', 'timevar_in_check', 'timevar_out_check', 'max_deg_in_check', 'C2_check', 'C3_check', 'C4_check', 'K3_check', 'SG2_check', 'BP2_check'],"binary_all": ['deg_in_check', 'deg_out_check', 'fan_in_check', 'fan_out_check', 'ratio_in_check', 'ratio_out_check', 'deg_in_timespan_check', 'timevar_in_check', 'timevar_out_check', 'max_deg_in_check', 'C2_check', 'C3_check', 'C4_check', 'C5_check', 'C6_check', 'K3_check', 'K4_check','SG1_check', 'SG2_check', 'SG_max_check','BP1_check', 'BP2_check'],"binary_full": ['deg_in_check', 'deg_out_check', 'fan_in_check', 'fan_out_check', 'ratio_in_check', 'ratio_out_check', 'deg_in_timespan_check', 'timevar_in_check', 'timevar_out_check', 'max_deg_in_check', 'C2_check', 'C3_check', 'C4_check', 'C5_check', 'C6_check', 'K3_check', 'K4_check', 'K5_check', 'SG1_check', 'SG2_check', 'SG3_check', 'SG4_check', 'SG_max_check', 'BP1_check', 'BP2_check', 'BP3_check', 'BP4_check', 'gather_cc_check'],"continuous_2hop": ['deg_in', 'deg_out', 'fan_in', 'fan_out', 'ratio_in', 'ratio_out', 'deg_in_timespan', 'timevar_in', 'timevar_out', 'max_deg_in', 'C2_count', 'C3_count', 'C4_count', 'K3_count', 'SG2_count', 'BP2_count'],"continuous_all": ['deg_in', 'deg_out', 'fan_in', 'fan_out', 'ratio_in', 'ratio_out', 'deg_in_timespan', 'timevar_in', 'timevar_out', 'max_deg_in','C2_count', 'C3_count', 'C4_count','K3_count', 'SG_max'],"edge_all": ['e_time_mod7'],"edge_binary": ['e_time_mod7_check'],"test_all": ['max_deg_in'],"test_binary": ['deg_in_check', 'deg_out_check', 'fan_in_check', 'fan_out_check']}
pretrain_feats_mf = {"all_mf": None,"aml-e_medium": ['FanIn [2:3)', 'FanIn [3:4)', 'FanIn [4:5)','FanOut [2:3)', 'FanOut [3:4)', 'FanOut [4:5)','DegIn [2:3)', 'DegIn [30:inf)','DegOut [2:3)', 'DegOut [30:inf)','ScatGat [2:3)','LCCycle [2:3)'],"eth": ['FanIn [2:3)', 'FanIn [3:4)','FanOut [2:3)', 'FanOut [3:4)', 'DegIn [2:3)', 'DegIn [3:4)', 'DegIn [30:inf)','DegOut [2:3)', 'DegOut [3:4)', 'DegOut [30:inf)','ScatGat [2:3)', 'ScatGat [3:4)','LCCycle [2:3)', 'LCCycle [3:4)', 'LCCycle [4:5)', 'LCCycle [5:6)', 'LCCycle [6:7)', 'LCCycle [7:8)']}
pretrain_choices = list(pretrain_feats_sim.keys()) + list(pretrain_feats_mf.keys())
model_choices = ["gcn", "mlp", "type1", "type2", "type2_hetero_sage","type2_hetero_gat", "type2_gnn_mlp", "pc_gnn", "type2_homo_gat", "gin", "gat", "gine", "pna", "gcnn", "old_gin"]
gnn_model_choices = ["type2_hetero_sage", "type2_hetero_gat", "type2_gnn_mlp", "pc_gnn", "type2_homo_gat", "gin", "gat", "rgcn","type2_hetero_sage_sampled", "type2_hetero_gat_sampled", "type2_gnn_mlp_sampled", "pc_gnn_sampled", "type2_homo_gat_sampled", "gin_sampled", "gat_sampled", "gine", "custom"]

def open_csv(file_name, header):
	"""
    Open CSV to prepare for writing
    :param file_name: Name of CSV
    :param header: Header to be written
    :return: CSV file object
    """
	if Path(file_name).is_file():
		csv = open(file_name, 'a')
	else:
		csv = open(file_name, 'w')
		csv.write(header)
		csv.flush()
	return csv
	
	
def write_csv(csv, arr):
	"""
    Write data to CSV file object
    :param csv: CSV file object
    :param arr: List containing values to write to CSV
    """
	arr = [str(v) for v in arr]
	csv.write(','.join(arr))
	csv.write('\n')
	csv.flush()
	
	
def load_options():
	"""
    Parses command line arguments. Most of these are actually covered by the config files but still useful for e.g.
    selecting feature set to use for LGBM
    :return: Argparse object containing all the arguments
    """
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_path", default=None, type=str, help="Select the path")
	parser.add_argument("--log", default='info', choices=['debug', 'info', 'warning', 'error', 'critical'], help="Level of logging")
	parser.add_argument("--log_folder_name", default='logs_test', type=str, help="parent folder for storing logs")
	
	parser.add_argument("--seed", default=1, type=int, help="Select the random seed for stratified splitting")
	parser.add_argument("--torch_seed", default=None, type=int, help="Select the random seed for GNN training")
	parser.add_argument("--n_gnn_layers", default=None, type=int, help="Hard code the number of gnn layers")
	parser.add_argument("--n_splits", default=3, type=int, help="Select the number stratified k-fold splits")
	
	parser.add_argument("--features", default="raw", type=str, choices=["raw", "mf", "emb", "mf+emb", "raw+emb"])
	parser.add_argument("--noID", action='store_true', help="Remove all possible IDs (from node feats and from edge attributes)")
	parser.add_argument("--edges", default="type2", type=str, choices=["none"])
	parser.add_argument("--edge_file", default=None, type=str, help="Path to message passing edges, if different from normal graph edges")
	
	parser.add_argument("--to_undirected", action='store_true', help="Choose if graph should be undirected")
	parser.add_argument("--embedding", action='store_true', help="Choose if GNN should return embedding") # not currently implemented for most recent gnn models
	parser.add_argument("--save_model", action='store_true', help="Choose whether to save the pytorch model")
	parser.add_argument("--load_model", action='store_true', help="Choose whether to load a pytorch model (or just the settings of a model if set to False but finetune set to True)")
	parser.add_argument("--model_path", default=None, type=str, help="path of model file to load")
	parser.add_argument("--readout", default=None, type=str, choices=["node", "edge", "graph"])
	parser.add_argument("--emb_file", default=None, type=str)
	parser.add_argument("--tb", action='store_true', help="Use tensorboard logging")
	parser.add_argument("--ports", action='store_true', help="Add unique neighbour port numbers to edge attributes")
	parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP. If True, then this overpowers config file.")
	parser.add_argument("--edge_updates2", action='store_true', help="Use edge updates. If True, then this overpowers config file.")
	parser.add_argument("-tds", "--time_deltas", action='store_true', help="Add in and out time deltas to edge attributes")
	parser.add_argument("--ego", action='store_true', help="Add an ID to the center node of a NeighborLoader subgraph (Requires --disjoint to work properly)")
	parser.add_argument("--disjoint", action='store_true', help="Use disjoint neighbor sampling (requires lots of memory)")
	
	parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
	parser.add_argument("--y_from_file", action='store_true', help="Use pre-calculated y labels from a file.")
	parser.add_argument("--y_from_sim", action='store_true', help="Calculate y labels from simulator.")
	parser.add_argument('--y_list', nargs='+', default=None, type=str, help='List of features to use as labels')
	parser.add_argument("--y_pretrain", default=None, type=str, choices=pretrain_choices)
	parser.add_argument("--graph_simulator", action='store_true', help="Use graph simulator to generate data")
	parser.add_argument("--bidirectional_simulator", action='store_true', help="Use bidirectional graph simulator to generate data")
	parser.add_argument("--sim_num_nodes", default=8192, type=int, help="Size of simulated graphs")
	parser.add_argument("--sim_num_graphs", default=1, type=int, help="Number of disconnected graphs per simulated dataset")
	parser.add_argument("--sim_generator", default='chordal', type=str, choices=["chordal", "barabasi", "random", "erdos"])
	parser.add_argument("--sim_avg_degree", default=None, type=int, help="Average degree in simulated graphs")
	parser.add_argument("--sim_delta", default=None, type=int, help="Mean delta for sampling destination nodes. Small delta means very localized connectivity.")
	parser.add_argument("--sim_num_patterns", default=100, type=int, help="Number of manual patterns to add")
	parser.add_argument("--tf_type", default='dynamic', type=str, choices=['static', 'dynamic'], help="type of transaction features")
	parser.add_argument("--L2", action='store_true', help="Use L2 regularization")
	parser.add_argument("--optuna", action='store_true', help="Only for Optuna HP optimization")
	parser.add_argument("--model_settings", default='model_settings.json', type=str, help="Select the path")
	parser.add_argument("--only_f1", action='store_true', help="Calculate and tb log only the f1 scores")
	parser.add_argument("--repeat", default=None, type=int, help="Days of training to be done (for v. large datasets = v. slow training)")
	parser.add_argument("--unique_name", default='', type=str, help="unique name of model/run, mainly used for multiday training and tensorboard logs")
	parser.add_argument("--simple_efeats", action='store_true', help="Only use t, amount, nonce and gas edge features")
	parser.add_argument("--no_efeats", action='store_true', help="Only use amount")
	
	parser.add_argument("--dynamic_vertex", action='store_true', help="To load the AML data directly from the feature files")
	
	parser.add_argument("--node_name", default=None, type=str, choices=["acc", "tx"])
	parser.add_argument("--freeze", action='store_true', help="When chosen, existing layers of the loaded model are frozen during training/fine-tuning.")
	parser.add_argument("--swap_in", action='store_true', help="New input embedding layers.")
	parser.add_argument("--swap_out", action='store_true', help="New output layer.")
	parser.add_argument("--load_opt", action='store_true', help="Load optimizer state.")
	parser.add_argument("--finetune", action='store_true', help="Flag for finetuning.")
	parser.add_argument("--ft_use_config", action='store_true', help="Use pretrained model config.")
	parser.add_argument("--ft_use_model_settings", action='store_true', help="Use pretrained model settings.")
	parser.add_argument("--ft_use_args", action='store_true', help="Use pretrained model args.")
	parser.add_argument("--save_preds", action='store_true', help="Save model predictions.")
	# parser.add_argument("--rerun", action='store_true', help="Decide whether or not to rerun in case of cuda OOM.")
	parser.add_argument("--save_splits", action='store_true', help="Save val_folds and te_inds for eth.")
	parser.add_argument("--num_seeds", default=1, type=int, help="Number of different seeds to run")
	
	return parser.parse_args()
	
	
def get_best_params(path):
	"""
    Obtain best parameter set from performance metrics CSV
    :param path: Path to metrics CSV
    :return: Dictionary containing best parameters corresponding to best validation F1
    """
	df = pd.read_csv(path)
	best_params = df.sort_values(by=['val_f1', 'te_f1'], ascending=False).iloc[0].to_dict()
	
	metric_headers = ['tr_', 'val_', 'te_']
	best_params = {k: v for k, v in best_params.items() if not any(_ in k for _ in metric_headers)}
	
	return best_params
	
	
def get_directories(base_path):
	"""
    Return all non-empty directories contained in root directory
    :param base_path: Path to root directory
    :return: List of non-hidden directories
    """
	return [x for x in Path(base_path).glob("*") if not str(x).split("/")[-1].startswith(".") and x.is_dir()]
	
default_config_args = {"node_feats": 1,"edge_updates2": 0,"reverse_mp": 0,"remove_self_loops": 1,"bayes_seed": 1}

duplicate_config_args = ["reverse_mp","edge_updates2"]

def get_logging_location(args, config):
# logging location
	out_dir = f"/dccstor/aml-e/logs/{args.log_folder_name}"
	
	if config.simulator == "eth":
		dataset_name = 'eth'
	elif args.graph_simulator:
		dataset_name = 'sim'
	else:
		dataset_name = config.data_dir.strip('/').split('/')[-1]
		
	log_model_name = get_log_model_name(config, args)
	
	timestamp_ = datetime.datetime.now()
	timestamp_str = '{:%b%d_%H%M%S%f}'.format(timestamp_)
	
	log_dir = Path(f"{out_dir}/{dataset_name}/{log_model_name}/{timestamp_str}")
	Path(log_dir).mkdir(parents=True, exist_ok=True)
	Path(log_dir / "runs").mkdir(parents=True, exist_ok=True)
	return log_dir
	
def autocomplete(config, args):
	for k,v in default_config_args.items():
		if k not in config:
			config[k] = v
	for k in default_config_args:
		if k in args:
			if args[k]:
				config[k] = 1
	return config
	
finetune_args_list = ['finetune', 'freeze', 'swap_in', 'swap_out', 'load_opt', 'unique_name', 'y_from_file', 'graph_simulator', 'y_pretrain', 'unique_name']
def merge_with_finetune_args(args, args_ft):
	args_tmp = copy.deepcopy(args)
	for k in finetune_args_list:
		args_tmp[k] = args_ft[k]
	return args_tmp
	
pretrain_args_list = ['n_gnn_layers', 'features', 'ports', 'time_deltas', 'ego', 'disjoint', 'dynamic_vertex']
def merge_with_pretrained_args(args, args_ft):
	args_ft_tmp = copy.deepcopy(args_ft)
	for k in pretrain_args_list:
		args_ft_tmp[k] = args[k]
	return args_ft_tmp
	
def get_finetune_settings(args_ft):
	main_model_dir = Path(args_ft.model_path).parent.parent.absolute()
	# Fetch args (from pretrained model args path and merge with new finetuning args_ft OR just use args_ft)
	args_path = f"{main_model_dir}/args.json"
	if args_ft.ft_use_args:
		with open(args_path) as f_args:
			args = json.load(f_args)
		args = munchify(args)
		args = merge_with_pretrained_args(args, args_ft)
	else:
		args = copy.deepcopy(args_ft)
		# Fetch configs
	config_path = f"{main_model_dir}/config.json" if args_ft.ft_use_config else f"{args.config_path}"
	with open(f"{config_path}") as f_config:
		config = json.load(f_config)
	config = munchify(config)
	config = autocomplete(config, args)
	# Fetch model settings
	if args_ft.ft_use_model_settings:
		model_settings_path = f"{main_model_dir}/model_settings.json"
		with open(model_settings_path) as f_model_settings:
			model_settings = json.load(f_model_settings)
	else:
		with open(args.model_settings) as f_model_settings:
			model_settings = json.load(f_model_settings)
		model_settings = model_settings[config.model]
	model_settings = munchify(model_settings)
	# Get logging location
	log_model_name = get_log_model_name(config, args)
	log_dir = f"{args_ft.model_path}/finetuning/{log_model_name}"
	Path(log_dir).mkdir(parents=True, exist_ok=True)
	# Save settings
	with open(f'{log_dir}/args.json', 'w') as outfile:
		json.dump(vars(args), outfile)
	with open(f'{log_dir}/config.json', 'w') as outfile:
		json.dump(vars(config), outfile)
	with open(f'{log_dir}/model_settings.json', 'w') as outfile:
		json.dump(vars(model_settings), outfile)
		# Calculate some settings
	args = calculate_args(args)
	args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	return config, model_settings, args, log_dir
	
def get_settings():
	"""
    Gather setting and config files
    :return: Config munch dictionary, model_settings (containing either tuning/ training parameters), auxiliary command
    line arguments
    """
	args = load_options()
	args = munchify(vars(args))
	if args.finetune:
		config, model_settings, args, log_dir = get_finetune_settings(args)
	else:
	# Fetch configs from config path
		with open(f"{args.config_path}") as f_config:
			config = json.load(f_config)
		config = munchify(config)
		config = autocomplete(config, args)
		# Fetch model settings from model settings path
		with open(args.model_settings) as f_model_settings:
			model_settings = json.load(f_model_settings)
		model_settings = munchify(model_settings[config.model])
		# Get logging location
		log_dir = get_logging_location(args, config)
		# Save settings
		with open(f'{log_dir}/args.json', 'w') as outfile:
			json.dump(vars(args), outfile)
		with open(f'{log_dir}/config.json', 'w') as outfile:
			json.dump(vars(config), outfile)
		with open(f'{log_dir}/model_settings.json', 'w') as outfile:
			json.dump(vars(model_settings), outfile)
			# Calculate some settings
		args = calculate_args(args)
		args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		# Set up logs file
	print(f"Logs can also be found here: {log_dir}")
	numeric_level = getattr(logging, args.log.upper(), None)
	if not isinstance(numeric_level, int):
		raise ValueError('Invalid log level: %s' % args.log)
	logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)-5.5s] %(message)s", handlers=[logging.FileHandler(f"{log_dir}/logs.log"),logging.StreamHandler(sys.stdout)])
	args.log_dir = log_dir
	# Log args, config, and model_settings
	logging.info('MODEL SETTINGS:')
	# Open performance.csv for saving results
	csv = open_csv(f"{log_dir}/performance_metrics.csv", header=model_settings.header)
	return config, model_settings, args, log_dir, csv
	
def get_log_model_name(config, args, sampled=True, bidirec=True, ports=True, time_deltas=True, ego=True, disjoint=True, readout=True, emlp=True, n_gnn_layers=True):
	log_model_name = f"{config.model}"
	if sampled and config.neighbor_list[0] > 0: log_model_name = f"{log_model_name}_sampled"
	if bidirec and config.reverse_mp: log_model_name = f"{log_model_name}_bidirec"
	if ports and args.ports: log_model_name = f"{log_model_name}_ports"
	if ego and args.ego: log_model_name = f"{log_model_name}_ego"
	if disjoint and args.disjoint: log_model_name = f"{log_model_name}_disjoint"
	if readout and args.readout is not None: log_model_name = f"{log_model_name}_{args.readout}"
	if emlp and config.edge_updates2: log_model_name = f"{log_model_name}_emlp"
	if n_gnn_layers and args.n_gnn_layers is not None: log_model_name = f"{log_model_name}_{args.n_gnn_layers}"
	log_model_name = f"{log_model_name}_{config.batch_size}"
	if time_deltas and args.time_deltas: log_model_name = f"{log_model_name}_timedeltas"
	if args.finetune:
		log_model_name = f"{log_model_name}_FT"
		if args.freeze: log_model_name = f"{log_model_name}_freeze"
		if args.swap_in: log_model_name = f"{log_model_name}_swap_in"
		if args.swap_out: log_model_name = f"{log_model_name}_swap_out"
	if args.graph_simulator:
		log_model_name = f"{log_model_name}_SIM"
		if args.sim_generator is not None: log_model_name = f"{log_model_name}_{args.sim_generator}"
		if args.sim_num_graphs is not None: log_model_name = f"{log_model_name}_{args.sim_num_graphs}"
		if args.sim_num_nodes is not None: log_model_name = f"{log_model_name}_{args.sim_num_nodes}"
		if args.sim_avg_degree is not None: log_model_name = f"{log_model_name}_{args.sim_avg_degree}"
		if args.sim_delta is not None: log_model_name = f"{log_model_name}_{args.sim_delta}"
		if args.bidirectional_simulator: log_model_name = f"{log_model_name}_bi"
	if args.y_pretrain:
		log_model_name = f"{log_model_name}_{args.y_pretrain}"
	return log_model_name
	
def calculate_args(args):
	if args.y_pretrain:
		if args.graph_simulator:
			args.y_list = pretrain_feats_sim[args.y_pretrain]
		elif args.y_from_file:
			args.y_list = pretrain_feats_mf[args.y_pretrain]
		elif args.y_from_sim:
			args.y_list = pretrain_feats_sim[args.y_pretrain]
		else:
			args.y_list = args.y_list
	return args
	
def set_seed(seed: int = 0) -> None:
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# When running on the CuDNN backend, two further options must be set
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# Set a fixed value for the hash seed
	os.environ["PYTHONHASHSEED"] = str(seed)
	logging.info(f"Random seed set as {seed}")