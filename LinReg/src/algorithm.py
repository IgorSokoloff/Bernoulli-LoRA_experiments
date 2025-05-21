"""
p-LoRA project
"""
from src.experiment import *

##################################################################################################################################################
####################################################### Ray parallel computation functions #######################################################
##################################################################################################################################################
# Functions below are not the part of the class Algorithm on purpose

def stopping_criterion(stop_criteruim_values_dict, stop_criteria_conditions_dict, collectable_metrics_dict):
    list_ans = [stop_criteria_conditions_dict[metric](collectable_metrics_dict[metric][-1], max_value) for metric, max_value in stop_criteruim_values_dict.items()]
    return all(list_ans)

def save_data(collectable_metrics_dict, logs_path, experiment_str):
    # we need to pass here only logs_path and proper extension
    print("Data saving...")
    for metric, value in collectable_metrics_dict.items():
        print(f"Saving {metric}...")
        np.save(logs_path + metric + '_' + experiment_str, np.array(value, dtype=np.float64))
    print("Data saved.")

def print_last_point_metrics(collectable_metrics_dict):
    print("---------------------------------------------")
    print("Last point metrics:")
    for metric, value in collectable_metrics_dict.items():
        if value is not None:
            out_value = value[-1] if len(value)>0 else "N/A"
            print(f"{metric}: {out_value}")
    print("---------------------------------------------")

def run_single_launch(seed, states_dict, data_dict, stop_criteruim_values_dict, stop_criteria_conditions_dict, alg_params_dict, collectable_metrics_dict, oracle_dict, update, init_collectable_metrics_dict, update_collectable_metrics_dict, init_states_dict, init_oracles_dict, fill_alg_params_dict):
    # Sample iid batch of indices uniformly at random
    alg_params_dict["rs_bernoulli"] = RandomState(seed)
    alg_params_dict["rs_sketch"] = RandomState(alg_params_dict["NUM_LAUNCHES"]+seed)
    alg_params_dict["rs_noise"] = RandomState(2*alg_params_dict["NUM_LAUNCHES"]+seed)
    alg_params_dict["rs_sample"] = RandomState(3*alg_params_dict["NUM_LAUNCHES"]+seed)
    alg_params_dict["rs_init"] = RandomState(4*alg_params_dict["NUM_LAUNCHES"]+seed)
    alg_params_dict["rs_page"] = RandomState(5*alg_params_dict["NUM_LAUNCHES"]+seed)
    
    oracle_dict = init_oracles_dict(states_dict, d_copy(oracle_dict), data_dict, alg_params_dict)
    
    alg_params_dict = fill_alg_params_dict(states_dict, oracle_dict, data_dict, d_copy(alg_params_dict))
    states_dict, alg_params_dict = init_states_dict(d_copy(states_dict), oracle_dict, data_dict, d_copy(alg_params_dict))
    
    collectable_metrics_dict = init_collectable_metrics_dict(states_dict, d_copy(collectable_metrics_dict), alg_params_dict, oracle_dict, data_dict) 
    
    while stopping_criterion(stop_criteruim_values_dict, stop_criteria_conditions_dict, collectable_metrics_dict):

        states_dict, collectable_metrics_dict, alg_params_dict = update(d_copy(states_dict), data_dict, d_copy(collectable_metrics_dict), d_copy(alg_params_dict), oracle_dict, update_collectable_metrics_dict)
        # Below, we assume that "iters" are being collected
        if collectable_metrics_dict["iters"][-1]%alg_params_dict["PRINT_EVERY"] ==0:
            display.clear_output(wait=True)
            print_last_point_metrics(collectable_metrics_dict)
        if collectable_metrics_dict["iters"][-1]%alg_params_dict["SAVE_EVERY"] ==0 and alg_params_dict["NUM_LAUNCHES"]==1:
            save_data(collectable_metrics_dict, alg_params_dict["logs_path"], alg_params_dict["experiment_str"])

    collectable_metrics_dict["solution"] = states_dict["x"]
    return d_copy(collectable_metrics_dict)

def run_algorithm(states_dict, data_dict, stop_criteruim_values_dict, stop_criteria_conditions_dict, alg_params_dict, collectable_metrics_dict, oracle_dict, update, init_collectable_metrics_dict, update_collectable_metrics_dict, init_states_dict, init_oracles_dict, fill_alg_params_dict):              
    from src.algorithm import run_single_launch
    collectable_metrics_dicts_ar = []
    
    if alg_params_dict["use_ray"]:
        ray.init()
        run_single_launch = ray.remote(run_single_launch)
        futures = [run_single_launch.remote(seed, d_copy(states_dict), data_dict, stop_criteruim_values_dict, stop_criteria_conditions_dict, d_copy(alg_params_dict), d_copy(collectable_metrics_dict), oracle_dict, update, init_collectable_metrics_dict, update_collectable_metrics_dict, init_states_dict, init_oracles_dict, fill_alg_params_dict) for seed in alg_params_dict["seeds"]]
        collectable_metrics_dicts_ar = ray.get(futures)
    else:
        for seed in range(alg_params_dict["NUM_LAUNCHES"]):
            metrics_dict = run_single_launch(seed + alg_params_dict["seed"], d_copy(states_dict), data_dict, stop_criteruim_values_dict, stop_criteria_conditions_dict, d_copy(alg_params_dict), d_copy(collectable_metrics_dict), oracle_dict, update, init_collectable_metrics_dict, update_collectable_metrics_dict, init_states_dict, init_oracles_dict, fill_alg_params_dict)
            collectable_metrics_dicts_ar.append(metrics_dict)
    
    avg_keys = set(ALLOWABLE_AVG_KEYS) & set(collectable_metrics_dict.keys())
    for key in avg_keys:
        values = [d[key] for d in collectable_metrics_dicts_ar]
        min_len = min(len(arr) for arr in values) #Determine the minimum length among all arrays
        truncated_values = [arr[:min_len] for arr in values] # Truncate each array to min_len
        # Store truncated arrays individually
        for i in range(alg_params_dict["NUM_LAUNCHES"]):
            collectable_metrics_dict[f"{key}_{i}"] = truncated_values[i]
        # Compute the statistics on the truncated arrays
        collectable_metrics_dict[f"{key}_mean"]   = np.mean(truncated_values, axis=0)
        collectable_metrics_dict[f"{key}_median"] = np.median(truncated_values, axis=0)
        collectable_metrics_dict[f"{key}_std"]    = np.std(truncated_values, axis=0)
        
        # for i in range(alg_params_dict["NUM_LAUNCHES"]):
        #     collectable_metrics_dict[key+f"_{i}"] = values[i]
        # collectable_metrics_dict[key+"_mean"] = np.mean(values, axis=0) # mean over launches
        # collectable_metrics_dict[key+"_median"] = np.median(values, axis=0)
        # collectable_metrics_dict[key+"_std"] = np.std(values, axis=0) # std over launches
    copy_keys = set(ALLOWABLE_NON_AVG_KEYS) & set(collectable_metrics_dict.keys())
    # Copy the metrics that are not averaged
    for key in copy_keys:
        values = [d[key] for d in collectable_metrics_dicts_ar]
        collectable_metrics_dict[key] = values[0]
    
    # Revove redundant keys
    for key in avg_keys:
        collectable_metrics_dict.pop(key, None)
                    
    save_data(collectable_metrics_dict, alg_params_dict["logs_path"], alg_params_dict["experiment_str"])
    print("End-point:")
    print_last_point_metrics(collectable_metrics_dict)

##################################################################################################################################################################
####################################################### Class Algortihm related ##################################################################################
##################################################################################################################################################################

#TODO: investigate what else can be shifted to the class "Experiment"
class Algorithm(Experiment):
    def __init__(self, args):
        self.arg_values = vars(self.parse_args())
        # Print the input arguments
        print("Input arguments:")
        for key, value in self.arg_values.items():
            print(f"{key}: {value}")
        
        self.general_arg_asserts()
        self.algorithm_launching_arg_asserts()
        
        self.project_specific_asserts()
        
        print_time(self.arg_values["print_status"])
        
        self.init_alg_launch_dicts()
        self.init_data_dict()
        self.init_alg_params_dict()
        
        self.init_exp_param_extension()
        self.init_exp_data_extension()
        self.init_w_init_extension()
        self.init_exp_name_extension()
        
        self.init_paths_and_folders()
        self.load_prepared_datasets()
        self.load_w_init()
        self.load_parameters()
        self.init_comp_params_dict()
        self.compute_params()
        self.init_oracles()
        self.fill_alg_params_dict_and_states()
        

    #Project dependend function
    def parse_args(self):
        parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')

        # At this point we assume that the dataset is already downloaded and is in the data folder
        parser.add_argument('--alg_name', action='store', dest='alg_name', type=str, default="", help='The name of the algoritm we run') 
        parser.add_argument('--exp_name', action='store', dest='exp_name', type=str, default="", help='The name of the experiment') #it can be lets say AuxPAGE with zero grad init
        parser.add_argument('--sampling', action='store', dest='sampling', type=str, default="", help='The sampling of the experiment')            
        parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms', help='The name of the dataset')
        parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default="log-reg", help='log-reg or l1_norm')
        parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
        parser.add_argument('--regularizer_type', action='store', dest='regularizer_type', type=str, default=0, help='str-cvx or non-cvx')
        parser.add_argument('--la', action='store', dest='la_init', type=str, default="0", help='lambda')
        
        # parser.add_argument('--cond_number', action='store', dest='cond_number', type=str, default="100", help='Desired condition number')
        parser.add_argument('--is_sparse_dataset', action='store', dest='is_sparse_dataset', type=int, default=0, help='sparse dataset or not')
        parser.add_argument('--is_replacement', action='store', dest='is_replacement', type=int, default=0, help='Batch sampling with replacement or without')
        parser.add_argument('--is_grad_comp_init', action='store', dest='is_grad_comp_init', type=int, default=1, help='Whether to init with full grad or not')
        # Stopping criterium params
        parser.add_argument('--max_epochs', action='store', dest='max_epochs', type=float, default=10, help='Maximum number of epochs')
        parser.add_argument('--max_bits', action='store', dest='max_bits', type=int, default=100, help='Maximum number of bits transmitted from worker to server')
        parser.add_argument('--max_comms', action='store', dest='max_comms', type=int, default=100, help='Maximum number of commumnication rounds')
        parser.add_argument('--max_iters', action='store', dest='max_iters', type=int, default=100, help='Maximum number of iterations')
        parser.add_argument('--stop_criteruim_params', action='store', dest='stop_criteruim_params', type=str, default="['max_iters']", help='List of stopping criteriums')
        parser.add_argument('--computable_params', action='store', dest='comp_params_str', type=str, default="['L0', 'Li', 'L_pm']", help='list of computable params')

        parser.add_argument('--collectable_metrics', action='store', dest='collectable_metrics', type=str, default="['iters']", help='List of metrics to be collected during optimization procedure')
        parser.add_argument('--loadable_params', action='store', dest='loadable_params', type=str, default="['L0', 'Li', 'L_pm']", help='list of params to be loaded from disk')
        parser.add_argument('--loadable_datasets', action='store', dest='loadable_datasets', type=str, default="[]", help='list of loadable data arrays')        
        parser.add_argument('--factor', action='store', dest='factor', type=float, default=1.0, help='Stepsize factor')
        parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-7, help='tolerance')
        parser.add_argument('--step_size', action='store', dest='step_size_init', type=str, default="thl", help='Stepsize type or value')
        
        # Params for p_LoRA project
        parser.add_argument('--dim', action='store', dest='dim', type=int, default=100, help='dimentionality')
        parser.add_argument('--prob', action='store', dest='prob', type=float, default=0.1, help='probability')
        parser.add_argument('--num_samples', action='store', dest='num_samples', type=int, default=100, help='Number of datasamples for each worker')
        parser.add_argument('--rank', action='store', dest='rank', type=int, default=2, help='rank')
        parser.add_argument('--noise_scale', action='store', dest='noise_scale', type=float, default=0.05, help='noise scale')
        parser.add_argument('--batchsize', action='store', dest='batchsize', type=int, default=1, help='batchsize')
        parser.add_argument('--momentum', action='store', dest='momentum', type=float, default=0.1, help='momentum')
        
        # Params for specific l1_norm generataion routine
        parser.add_argument('--compressor', action='store', dest='compressor', type=str, default="TopK", help='Compressor type')
        parser.add_argument('--qC', action='store', dest='qC', type=float, default=0.1, help='fraction of contractive compressor for 3PCv2')
        
        # System level params
        parser.add_argument('--NUM_LAUNCHES', action='store', dest='NUM_LAUNCHES', type=int, default=10, help='Number of launches of stochastic algorithm')
        parser.add_argument('--PRINT_EVERY', action='store', dest='PRINT_EVERY', type=int, default=1000000000, help='How often to print metrics')
        parser.add_argument('--SAVE_EVERY', action='store', dest='SAVE_EVERY', type=int, default=1000000000, help='How often to save metrics')
        parser.add_argument('--print_status', action='store', dest='print_status', type=int, default=0, help='print_status')
        parser.add_argument('--use_ray', action='store', dest='use_ray', type=int, default=0, help='use_ray or not')
        parser.add_argument('--seed', action='store', dest='seed', type=int, default=1, help='seed')
        parser.add_argument('--seeds', action='store', dest='seeds', nargs='+', type=int, default=[1], help='List of seed values')

        return parser.parse_args()

    #Project dependend function
    def project_specific_asserts(self):
        #p_LoRA specific asserts
        assert(self.arg_values['dim']>=0)
        assert(self.arg_values['rank']>0)
        assert(self.arg_values['momentum']>0)
        assert(self.arg_values['num_samples']>0)
        assert(self.arg_values['num_workers']>0)
        assert(self.arg_values['prob']>0 and self.arg_values['prob']<=1)
        assert self.arg_values["dataset"] in LINREG_DATASETS
        assert self.arg_values["noise_scale"]>=0    
        assert(self.arg_values['batchsize']>0)
        
        # Legacy asserts
        # 
        # assert(self.arg_values['qC']>0 and self.arg_values['qC']<=1)
        
        # if self.arg_values['alg_name'] == "EF21-P":
        #     assert self.arg_values['compressor'] == "TopK"
        # if self.arg_values['alg_name'] == "MARINA-P":
        #     assert self.arg_values['compressor'] in ["sameRandK", "indRandK", "PermK"]
        # if self.arg_values['alg_name'] == "GD":
        #     assert self.arg_values['compressor'] == "I"
        # assert self.arg_values['sampling'] in ALLOWABLE_SAMPLINGS
        # assert self.arg_values['compressor'] in ALLOWABLE_COMPRESSORS
    
    def algorithm_launching_arg_asserts(self):
        assert self.arg_values["is_replacement"] in [0,1]
        assert self.arg_values["is_grad_comp_init"] in [0,1]
        assert isinstance(self.arg_values['tol'], float) and self.arg_values['tol'] > 0
        assert isinstance(self.arg_values['PRINT_EVERY'], int) and self.arg_values['PRINT_EVERY'] > 0
        assert isinstance(self.arg_values['SAVE_EVERY'], int) and self.arg_values['SAVE_EVERY'] > 0
        assert isinstance(self.arg_values['NUM_LAUNCHES'], int) and self.arg_values['NUM_LAUNCHES'] > 0
        assert isinstance(self.arg_values['seed'], int) and self.arg_values['seed'] > 0
        assert all(isinstance(seed, int) and seed > 0 for seed in self.arg_values['seeds']), "All elements in 'seeds' must be integers greater than 0"
        #assert len(self.arg_values['seeds'])>= self.arg_values['NUM_LAUNCHES']
        assert self.arg_values["use_ray"] in [0, 1]
        
        if 'max_epochs' in self.arg_values['stop_criteruim_params']:
            assert(isinstance(self.arg_values['max_epochs'], int) and self.arg_values['max_epochs'] > 0)
        if 'max_bits' in self.arg_values['stop_criteruim_params']:
            assert(isinstance(self.arg_values['max_bits'], int) and self.arg_values['max_bits'] > 0)
        if 'max_comms' in self.arg_values['stop_criteruim_params']:
            assert(isinstance(self.arg_values['max_comms'], int) and self.arg_values['max_comms'] > 0)
        if 'max_iters' in self.arg_values['stop_criteruim_params']:
            assert(isinstance(self.arg_values['max_iters'], int) and self.arg_values['max_iters'] > 0)
    
    def general_arg_asserts(self):
        print("Checking input arguments...")
        assert isinstance(self.arg_values['exp_name'], str) and self.arg_values['exp_name'] != ""
        assert isinstance(self.arg_values['alg_name'], str) and self.arg_values['alg_name'] != ""
        
        #assert isinstance(self.arg_values['sampling'], str) and self.arg_values['sampling'] != ""
        assert self.arg_values['exp_name'] in ALLOWABLE_EXPERIMENTS
        assert self.arg_values['alg_name'] in ALLOWABLE_ALGORITHMS

        
        assert isinstance(self.arg_values['dataset'], str) and self.arg_values['dataset'] != ""
        assert self.arg_values['loss_func'] in SUPPORTED_LOSS_FUNCS
        assert self.arg_values['regularizer_type'] in SUPPORTED_REGULARIZERS
        assert isinstance(self.arg_values['num_workers'], int) and self.arg_values['num_workers'] > 0
        assert isinstance(self.arg_values['factor'], float) and self.arg_values['factor'] > 0
        assert self.arg_values["is_sparse_dataset"] in [0,1]
        
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['stop_criteruim_params'])), "Not all elements are strings."
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['loadable_params'])), "Not all elements are strings."
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['collectable_metrics'])), "Not all elements are strings."
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['loadable_datasets'])), "Not all elements are strings." 
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['comp_params_str'])), "Not all elements are strings."

    def init_data_dict(self):   
        self.data_dict = parse_params_to_dict(self.arg_values['loadable_datasets'], ALLOWABLE_DATASETS)
        assert set(self.data_dict.keys()).issubset(ALLOWABLE_DATASETS)
        
    def init_alg_launch_dicts(self):
        self.stop_criteria_values_fed = dict(zip(ALLOWABLE_STOP_CRITERIA, [self.arg_values['max_epochs'], self.arg_values['max_bits'], self.arg_values['max_comms'], self.arg_values['max_iters'], self.arg_values['tol'], self.arg_values['tol'], self.arg_values['tol']]))
        self.stop_criteria_values_dict = parse_params_to_dict(self.arg_values['stop_criteruim_params'], ALLOWABLE_STOP_CRITERIA)
        self.stop_criteria_conditions_dict = {key: ALLOWABLE_STOP_CRITERIA_CONDITIONS[key] for key in self.stop_criteria_values_dict.keys()}
        self.collectable_metrics_dict = parse_params_to_dict(self.arg_values['collectable_metrics'], ALLOWABLE_COLLECTABLE_METRICS)
        assert set(self.stop_criteria_values_dict.keys()).issubset(ALLOWABLE_STOP_CRITERIA)
        assert set(self.collectable_metrics_dict.keys()).issubset(ALLOWABLE_COLLECTABLE_METRICS)
        for stop_crit in self.stop_criteria_values_dict.keys():
            self.stop_criteria_values_dict[stop_crit] = self.stop_criteria_values_fed[stop_crit]
    
    def init_paths_and_folders(self):
        # init paths:
        script_directory = self.script_directory()
        self.project_path = script_directory + "/"
        self.data_path = self.project_path + "data_{0}/".format(self.arg_values['dataset'])
        self.dataset_path = self.data_path + 'data' + self.exp_data_extension + "/"
        
        # init fodlers:
        self.experiment_str = self.arg_values['exp_name'] + self.exp_name_extension
        self.logs_path = self.project_path + "logs/logs_{0}/".format(self.experiment_str)
        
        if not os.path.exists(self.project_path + "logs/"):
            os.makedirs(self.project_path + "logs/")
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)    
        
    #Project dependend function
    def compute_params(self):
        if self.comp_params_set:
            self.alg_params_dict.update(self.comp_params_dict)
    
    def fill_alg_params_dict_and_states(self):
        #Legacy
        # self.alg_params_dict["qC"] = self.arg_values["qC"]
        # self.alg_params_dict["sampling"] = self.arg_values["sampling"]
        # self.alg_params_dict["is_grad_comp_init"] = self.arg_values["is_grad_comp_init"]
        # self.alg_params_dict["batchsize"] = self.arg_values["batchsize"]
        # self.alg_params_dict["compressor"] = self.arg_values["compressor"]
        #self.alg_params_dict["la"] = self.arg_values["la"]
         
        #p_LoRA specific
        self.alg_params_dict["prob"] = self.arg_values["prob"]
        self.alg_params_dict["step_size_init"] = self.arg_values["step_size_init"]
        self.alg_params_dict["max_iters"] = int(self.arg_values["max_iters"])
        self.alg_params_dict["tol"] = self.arg_values["tol"]
        self.alg_params_dict["rank"] = int(self.arg_values["rank"])
        self.alg_params_dict["momentum"] = self.arg_values["momentum"]
        
        if "num_samples_ft" in self.alg_params_dict:
            self.alg_params_dict["num_samples_ft"] = int(self.alg_params_dict["num_samples_ft"])  
        if "num_samples_pre" in self.alg_params_dict:
            self.alg_params_dict["num_samples_pre"] = int(self.alg_params_dict["num_samples_pre"])
            
        if "batchsize" in self.arg_values:
            self.alg_params_dict["batchsize"] = int(self.arg_values["batchsize"])
        
        self.alg_params_dict["oracle_params"] = {"la": self.alg_params_dict["la_ft"],
                                                 "n_samples": self.alg_params_dict["num_samples_ft"],
                                                 "batchsize": self.alg_params_dict["batchsize"]}
        self.alg_params_dict["noise_scale"] = self.arg_values["noise_scale"]
        
        self.alg_params_dict["dim"] = self.x_0.shape[0]
        self.alg_params_dict["dim_in"] = int(np.sqrt(self.alg_params_dict["dim"]))
        self.alg_params_dict["dim_out"] = int(np.sqrt(self.alg_params_dict["dim"]))
        
        #lines below are for every project
        self.alg_params_dict["exp_name"] = self.arg_values["exp_name"]
        self.alg_params_dict["alg_name"] = self.arg_values["alg_name"]
        self.alg_params_dict["factor"] = self.arg_values["factor"]
        self.alg_params_dict["num_workers"] = int(self.arg_values['num_workers'])
        self.alg_params_dict["num_samples"] = int(self.arg_values['num_samples'])
        self.alg_params_dict['print_status'] = self.arg_values['print_status']
        self.alg_params_dict["NUM_LAUNCHES"] = self.arg_values['NUM_LAUNCHES']
        self.alg_params_dict["PRINT_EVERY"] = self.arg_values['PRINT_EVERY']
        self.alg_params_dict["SAVE_EVERY"] = self.arg_values['SAVE_EVERY']
        self.alg_params_dict["use_ray"] = self.arg_values['use_ray']
        self.alg_params_dict["seed"] = self.arg_values['seed']
        self.alg_params_dict["seeds"] = self.arg_values['seeds']
        self.alg_params_dict["logs_path"] = self.logs_path
        self.alg_params_dict["experiment_str"] = self.experiment_str
        self.alg_params_dict["data_path"] = self.data_path
        self.alg_params_dict["dataset_path"] = self.dataset_path
        self.alg_params_dict["project_path"] = self.project_path
        self.states_dict = {"x": self.x_0.copy(), "x_prev": None, "g": None}
                
    def run(self):
        print('------------------------------------------------------------------------------------------------')
        start_time = time.time()
        print("Running algorithm...")
        
        run_algorithm(self.states_dict, self.data_dict, self.stop_criteria_values_dict, self.stop_criteria_conditions_dict, self.alg_params_dict, self.collectable_metrics_dict, self.oracle_dict, self.update, self.init_collectable_metrics_dict, self.update_collectable_metrics_dict, self.init_states_dict, self.init_oracles_dict, self.fill_alg_params_dict)
        print("Experiment finished.")
        time_diff = time.time() - start_time
        print(f"Computation time: {time_diff} sec")
        peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"Peak Memory Usage: {peak_memory_usage / 1024} MB")
        print('------------------------------------------------------------------------------------------------')
    

    


    
        
    
        
    
        