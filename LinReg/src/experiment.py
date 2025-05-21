from src.utils import *
from src.oracle_functions import *

###################################################
######### Global project-dependend params #########
###################################################



ALLOWABLE_EXPERIMENTS = ['GD', 'RAC-LoRA_A', 'RAC-LoRA_B', 'RC-LoRA', #GD type experimenents
                         'stoch_RC-LoRA-SGD', 'stoch_RC-LoRA-MVR', 'stoch_RAC-LoRA_A-SGD', 'stoch_RAC-LoRA_B-SGD', 
                         'finite_RC-LoRA-SGD', 'finite_RC-LoRA-PAGE', 'finite_RAC-LoRA_A-SGD', 'finite_RAC-LoRA_B-SGD']

ALLOWABLE_ALGORITHMS = ['GD', 'RAC-LoRA_A', 'RAC-LoRA_B', 'RC-LoRA'] #GD, stoch and finite sum settings 
                        #'RC-LoRA-SGD', 'RC-LoRA-MVR', 'RC-LoRA-PAGE' 'RAC-LoRA-A-SGD', 'RAC-LoRA-B-SGD']
ALLOWABLE_SAMPLINGS = ['full']
ALLOWABLE_COMPRESSORS = ['sameRandK', 'indRandK', 'PermK', 'TopK', 'I', 'sameRandK-TopK', 'indRandK-TopK', 'PermRandK-TopK']
ALLOWABLE_STOP_CRITERIA = ['epochs', 'bits', 'comms', 'iters', 'arg_res', 'func_diff', 'sqnorm']

ALLOWABLE_COLLECTABLE_METRICS = ['epochs', 'bits', 'comms', 'iters', 'arg_res', 'func_diff', 'sqnorm', 
                                 #'r-la_min-H_A', 'r-la_min-H_B', 'r-la_max-H_A', 'r-la_max-H_B', 'r-la_min-H_p', 'r-la_max-H_p' (I understood that it is not ineresting to keep track them)
                                 ]


#ALLOWABLE_COLLECTABLE_METRICS += [item+"_grad_comp" for item in ALLOWABLE_COLLECTABLE_METRICS] # when full grad is computed  
ALLOWABLE_AVG_KEYS = ['func_diff', 'arg_res', 'sqnorm']


#ALLOWABLE_AVG_KEYS += [item+"_grad_comp" for item in ALLOWABLE_AVG_KEYS] # when full grad is computed
ALLOWABLE_NON_AVG_KEYS = list(set(ALLOWABLE_COLLECTABLE_METRICS) - set(ALLOWABLE_AVG_KEYS))

ALLOWABLE_DATASETS = set(['X', 'y', 'X_mean', 'y_mean', 'X_pre', 'y_pre', 'c_pre', 'X_ft', 'y_ft', 'c_ft', 'A_pre', 'b_pre', 'A_ft', 'b_ft'])


ALLOWABLE_STOP_CRITERIA_CONDITIONS = {'epochs': lambda cur_epoch_number, max_epochs: cur_epoch_number <= max_epochs,
                                    'bits': lambda cur_bits_number, max_bits: cur_bits_number <= max_bits,
                                    'comms': lambda cur_comms_number, max_comms: cur_comms_number <= max_comms,
                                    'iters': lambda cur_iters_number, max_iters: cur_iters_number <= max_iters,
                                    'arg_res': lambda cur_arg_res, tol: cur_arg_res >= tol,
                                    'func_diff': lambda cur_func_diff, tol: cur_func_diff >= tol,
                                    'sqnorm': lambda cur_sqnorm, tol: cur_sqnorm >= tol
                                    }
#'L_0,i', 'wtL_0', 'L_pm', 'L_0,pm', 

ALLOWABLE_PARAMS = set(['L_0_pre', 'L_0_ft', 'momentum',
                        'num_samples_pre','num_samples_ft',
                        'la_pre','la_ft', 
                        'x_star_pre', 'f_star_pre', 'x_star_ft', 'f_star_ft'])

# r-L_0,pm -  "real" L_0,pm - relates to the experimentaly observed value

LIBSVM_CLASSIFICATION_DATASETS = set([
    'a9a', 'w8a', 'mushrooms', 'ijcnn1', 'covtype', 'phishing', 'rcv1',
    'real-sim', 'news20.binary', 'cod-rna', 'dna', 'svmguide3',
    'svmguide1', 'svmguide2', 'splice', 'madelon', 'gisette', 'dexter',
    'dorothea', 'colon-cancer', 'leukemia', 'lung-cancer', 'rcv1.binary',
    'sector', 'usps', 'mnist'
])

SUPPORTED_LOSS_FUNCS = set([
    'log-reg',
    'quadratic',
    'l1_norm',
    'lin-reg'
])

SUPPORTED_REGULARIZERS = ['str-cvx', 'cvx', 'non-cvx']

# Specific datasets for the auxpage project
QUADRATIC_DATASETS = set(['synthetic_dense', 'synthetic_sparse', 'synthetic_sparse_zero'])
L1_NORM_DATASETS = set(['synthetic_dense', 'synthetic_sparse', 'synthetic_sparse_zero'])
LINREG_DATASETS = set(['synthetic_dense'])

ALLOWABLE_PLOT_FAMILIES = ['ALL', 'SINGLE_RELEASE']


# Ray parameter
NUM_CORES = 48

##########################
# Auxiliary biased stuff #
##########################
#legacy code from hfh project
cost_function_biased = lambda c,k,p: (p + (1 - p)*c)*(1 + ((1 + np.sqrt(1 - p)) / p - 1) * k) # cost function in the biased case
def get_stepsize_biased(L_0, delta, p): 
    return 1/(L_0 + delta*np.sqrt((1-p)/((1-np.sqrt(1-p))**2)))
def get_optimal_params_biased(c, kappa):
    p = smp.symbols('p')
    expr = 0.5*(c*(-kappa*(-2*p**2 + smp.sqrt(1-p)*p + 2*smp.sqrt(1-p) + 2)/p**2 - 2) + kappa*(-1/smp.sqrt(1-p) - 2) + 2)
    sol = np.array(list(map(complex, smp.solve(expr, p, domain=smp.Interval.Lopen(0, 1)))))
    sol_real = np.real(sol)
    if sol_real.shape[0]==3:
        cost_1 = cost_function_biased(c, kappa, sol_real[1])
        cost_2 = cost_function_biased(c, kappa, sol_real[2])
        if cost_1 <= cost_2:
            p_opt = sol_real[1]
            cost_opt = cost_1
        else:
            p_opt = sol_real[2]
            cost_opt = cost_2
        if cost_opt>1:
            p_opt = 1
    elif sol_real.shape[0]==2:
        cost_opt = cost_function_biased(c, kappa, sol_real[1])
        p_opt = sol_real[1]
    else:
        print(c, kappa, sol_real)
        raise ValueError("")
    return p_opt, cost_opt
##########################

############################
# Auxiliary unbiased stuff #
############################
#legacy code from hfh project
def cost_prime_unbiased (c,k,p):
    term1 = (-((1 - p) / p**2) - p**(-1)) * (c * (1 - p) + p) * k
    term2 = 2 * np.sqrt((1 - p) / p)
    term3 = (1 - c) * (1 + np.sqrt((1 - p) / p) * k)
    return sign(term1 / term2 + term3)

p_hat_unbiased = lambda c: (3*c)/(3*c + 1)

def cost_prime_grid_unbiased(c_vals,k_vals):
    P = np.array([p_hat_unbiased(c) for c in c_vals], dtype=np.float64)
    #dim = P.shape[0]
    z = np.zeros((k_vals.shape[0], c_vals.shape[0]))
    for i,k in enumerate(k_vals):
        for j,c in enumerate(c_vals):
            z[i,j] = cost_prime_unbiased(c,k,P[j])
    return z

cost_function_unbiased = lambda c,k,p: (p + (1 - p) * c) * (1 + k * np.sqrt((1 - p) / p))
def get_stepsize_unbiased(L_0, delta, p): 
    return 1/(L_0 + delta*np.sqrt((1-p)/(p)))

def get_optimal_params_unbiased(c, kappa):
    p = smp.symbols('p')
    expr = (-((1 - p) / p**2) - p**(-1)) * (c * (1 - p) + p) * kappa / (2 * smp.sqrt((1 - p) / p)) + (1 - c) * (1 + smp.sqrt((1 - p) / p) * kappa)
    
    sol = np.array(list(map(complex, smp.solve(expr, p, domain=smp.Interval.Lopen(0, 1)))))
    sol_real = np.real(sol)
    
    sol_attached = np.append(sol_real, 1.0)
    costs = np.array([cost_function_unbiased (c, kappa, sol) for sol in  sol_attached], dtype=np.float64)
    p_opt = sol_attached[np.argmin (costs)]
    return p_opt, cost_function_unbiased(c, kappa, p_opt)
####################################################################################

####################################################################################
class Experiment():
    def __init__(self):
        pass
    
    def init_regularizers(self):
        self.regularizer = {"str-cvx":regularizer_scvx,
                            "cvx":regularizer_cvx,
                            "non-cvx":regularizer_noncvx}[self.arg_values["regularizer_type"]]
        self.regularizer_grad = {"str-cvx":regularizer_scvx_grad,
                                 "cvx":regularizer_cvx_grad,
                                 "non-cvx":regularizer_noncvx_grad}[self.arg_values["regularizer_type"]]
        
        #TODO: add regularizer_hess
        self.regularizer_hess = {"str-cvx":regularizer_scvx_hess,
                                 "cvx":regularizer_cvx_hess,
                                 "non-cvx":regularizer_noncvx_hess}[self.arg_values["regularizer_type"]]
        
        self.regularizer_hess_bound = {"str-cvx":regularizer_scvx_hess_bound,
                                       "cvx":regularizer_cvx_hess_bound,
                                       "non-cvx":regularizer_noncvx_hess_bound}[self.arg_values["regularizer_type"]]
    
    def init_oracles(self):
        my_print("Defining oracles...", self.arg_values["print_status"])
        
        self.init_regularizers()
        
        self.oracle_loss = {#"log-reg":logreg_loss_distributed, 
                            #"quadratic":quad_loss_ij,
                            "lin-reg":linreg_loss_ij,
                            #"l1_norm":l1_norm_loss_i_distributed
                            }[self.arg_values['loss_func']]
        
        self.oracle_grad = {#"log-reg":logreg_grad_distributed,
                            #"quadratic":quad_grad_ij,
                            "lin-reg":linreg_grad_ij,
                            #"l1_norm":l1_norm_grad_i_distributed
                            }[self.arg_values['loss_func']]
        
        self.oracle_minibatch_grad = {#"log-reg":logreg_grad_distributed,
                            #"quadratic":quad_grad_ij,
                            "lin-reg":linreg_minibatch_grad_ij,
                            #"l1_norm":l1_norm_grad_i_distributed
                            }[self.arg_values['loss_func']]
        
        self.oracle_hess = {#"log-reg":logreg_hess_distributed,
                            #"quadratic":quad_hess_ij,
                            "lin-reg":linreg_hess_ij,
                            #"l1_norm": None
                            }[self.arg_values['loss_func']]
        
        self.oracle_hess_bound = {#"log-reg":logreg_hess_bound_distributed,
                                  #"quadratic":quad_hess_ij,
                                  "lin-reg":linreg_hess_ij_bound,
                                  #"l1_norm": None
                                  }[self.arg_values['loss_func']]
        
        # per-worker losses computed at different points, each corresponding to the same worker
        self.local_losses = {"log-reg":None, 
                            "quadratic":None,
                            "lin-reg":None,
                            "l1_norm": l1_norm_local_losses_i_distributed}[self.arg_values['loss_func']]
        
        # per-worker grads computed at different points, each corresponding to the same worker
        self.local_grads = {"log-reg":None, 
                            "quadratic":quad_local_grads,
                            "lin-reg":None,
                            "l1_norm": l1_norm_local_grads_i_distributed}[self.arg_values['loss_func']]
        
        # per-worker grads computed at the same point
        self.non_local_grads = {"log-reg":None, 
                            "quadratic":None,
                            "lin-reg":None,
                            "l1_norm": l1_norm_non_local_grads_i_distributed}[self.arg_values['loss_func']]
         
        self.oracle_dict = {"f": lambda w, X, y, c, params: self.oracle_loss(w, X, y, c, params, self.regularizer),
                            "grad": lambda w, X, y, params: self.oracle_grad(w, X, y, params, self.regularizer_grad),
                            "minibatch_grad": lambda w, A, b, params: self.oracle_minibatch_grad(w, A, b, params, self.regularizer_grad),
                            "hess": lambda w, X, y, params: self.oracle_hess(w, X, y, params, self.regularizer_hess),
                            "hess_bound": lambda w, X, y, params: self.oracle_hess_bound(w, X, y, params, self.regularizer_hess_bound),
                            "local_losses": lambda W, X, Y, params: self.local_losses(W, X, Y, params, self.regularizer),
                            "local_grads": lambda W, X, Y, params: self.local_grads(W, X, Y, params, self.regularizer_grad),
                            "non_local_grads": lambda w, X, Y, params: self.non_local_grads(w, X, Y, params, self.regularizer_grad),
                            }

        
        
    def load_prepared_datasets(self):
        # If there are bugs, see implementation in the source code for the hfh project
        for data_part_name in self.data_dict.keys():
            path = self.dataset_path + data_part_name + self.exp_data_extension
            self.data_dict[data_part_name] = load_param(path, data_part_name, self.arg_values["print_status"])
    
    def get_part_dataset(self, data_part_name, inds):
        # If there are bugs, see implementation in the source code for the hfh project
        path = self.dataset_path + data_part_name + self.exp_data_extension
        return load_selected_sparse_matrices(path, data_part_name, inds, self.arg_values["print_status"])
        
    def load_w_init(self):
        self.x_0 = np.array(np.load(self.data_path + 'w_init' + self.w_init_extension + '.npy'), dtype=np.float64)

    def load_parameters(self):
        for param in self.alg_params_dict.keys():
            self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
            param_path = self.comp_params_path + param + self.exp_params_extension
            self.alg_params_dict[param] = load_param(param_path, param, self.arg_values["print_status"])
    
    def init_comp_params_dict(self):
        try: 
            comp_params_list = ast.literal_eval(self.arg_values["comp_params_str"])
        except ValueError:
            print("The string is not a valid list representation.")
        
        if isinstance(comp_params_list, list) and all(isinstance(item, str) for item in comp_params_list):
            self.comp_params_dict = {key: None for key in comp_params_list}
        else:
            print("The list does not contain only string elements.")

        self.comp_params_set = set(comp_params_list)
        assert(self.comp_params_set.issubset(ALLOWABLE_PARAMS))        
    
    #Project dependend functions    
    def init_exp_param_extension(self):
        if self.arg_values["loss_func"]=="lin-reg":
            #self.arg_values["la"] = float(self.arg_values["la_init"])            
            self.exp_params_extension = '_{0}_{1}_{2}_d{3}_nw{4}_ns{5}'.format(
                                        self.arg_values["loss_func"],
                                        self.arg_values["regularizer_type"],
                                        self.arg_values["dataset"], 
                                        self.arg_values["dim"], 
                                        self.arg_values["num_workers"], 
                                        self.arg_values["num_samples"]
                                        )
        else: 
            raise ValueError("other options are not supported")
    
    #Project dependend functions
    def init_exp_data_extension(self):   
        self.exp_data_extension = {"lin-reg":'_{0}_{1}_d{2}_nw{3}_ns{4}'.format(
                                    self.arg_values["loss_func"],
                                    self.arg_values["dataset"], 
                                    self.arg_values["dim"], 
                                    self.arg_values["num_workers"], 
                                    self.arg_values["num_samples"])
                                  }[self.arg_values["loss_func"]]
    
    #Project dependend functions
    def init_w_init_extension(self):
        self.w_init_extension = {"lin-reg":'_{0}_{1}_{2}_d{3}_nw{4}_ns{5}'.format(
                                  self.arg_values["loss_func"],
                                  self.arg_values["regularizer_type"],
                                  self.arg_values["dataset"], 
                                  self.arg_values["dim"], 
                                  self.arg_values["num_workers"], 
                                  self.arg_values["num_samples"])
                                }[self.arg_values["loss_func"]]
    
    #Project dependend functions
    def init_exp_name_extension(self):
        if self.arg_values['exp_name'] in ALLOWABLE_EXPERIMENTS:
            if self.arg_values['exp_name'] == "GD":
                self.exp_name_extension = self.exp_params_extension + "f{0}".format(myrepr(self.arg_values['factor']))
            elif self.arg_values['exp_name'] in ["RAC-LoRA_A","RAC-LoRA_B"]:
                self.exp_name_extension = self.exp_params_extension + "r{0}_f{1}".format(self.arg_values['rank'], myrepr(self.arg_values['factor']))
            elif self.arg_values['exp_name'] == "RC-LoRA":
                self.exp_name_extension = self.exp_params_extension + "r{0}_p{1}_f{2}".format(self.arg_values['rank'],myrepr(self.arg_values['prob']),
                                                                                              myrepr(self.arg_values['factor']))
            elif self.arg_values['exp_name'] == 'stoch_RC-LoRA-SGD':
                self.exp_name_extension = self.exp_params_extension + "r{0}_ns{1}_p{2}_f{3}".format(self.arg_values['rank'], myrepr(self.arg_values['noise_scale']), 
                                                                                                    myrepr(self.arg_values['prob']), myrepr(self.arg_values['factor']))
            elif self.arg_values['exp_name'] == 'stoch_RC-LoRA-MVR':
                self.exp_name_extension = self.exp_params_extension + "r{0}_ns{1}_m{2}_p{3}_f{4}".format(self.arg_values['rank'], 
                                                                                                         myrepr(self.arg_values['noise_scale']), myrepr(self.arg_values['momentum']),
                                                                                                         myrepr(self.arg_values['prob']), myrepr(self.arg_values['factor']))
            elif self.arg_values['exp_name'] in ['stoch_RAC-LoRA_A-SGD', 'stoch_RAC-LoRA_B-SGD']:
                self.exp_name_extension = self.exp_params_extension + "r{0}_ns{1}_f{2}".format(self.arg_values['rank'], myrepr(self.arg_values['noise_scale']), myrepr(self.arg_values['factor']))
            elif self.arg_values['exp_name'] == 'finite_RC-LoRA-SGD':
                self.exp_name_extension = self.exp_params_extension + "r{0}_b{1}_p{2}_f{3}".format(self.arg_values['rank'], int(self.arg_values['batchsize']), 
                                                                                                    myrepr(self.arg_values['prob']), myrepr(self.arg_values['factor']))
            elif self.arg_values['exp_name'] == 'finite_RC-LoRA-PAGE':
                self.exp_name_extension = self.exp_params_extension + "r{0}_b{1}_p{2}_f{3}".format(self.arg_values['rank'], int(self.arg_values['batchsize']),
                                                                                                    myrepr(self.arg_values['prob']), myrepr(self.arg_values['factor']))
            elif self.arg_values['exp_name'] in ['finite_RAC-LoRA_A-SGD', 'finite_RAC-LoRA_B-SGD']:
                self.exp_name_extension = self.exp_params_extension + "r{0}_b{1}_f{2}".format(self.arg_values['rank'], int(self.arg_values['batchsize']), myrepr(self.arg_values['factor']))
        else:
            raise ValueError("other options are not supported")
    # stopped here
    #TODO: breakpoint here
    
    #Project dependend functions
    def init_dataset_path(self):
        self.dataset_path = {"lin-reg": self.data_path + 'data' + self.exp_data_extension + "/",
                             #"quadratic": self.data_path + 'data' + self.exp_data_extension + "/",
                             #"l1_norm": self.data_path + 'data' + self.exp_data_extension + "/",
                             #"log-reg": self.data_path
                             }[self.arg_values["loss_func"]]
    
    #Project dependend functions
    #legacy code from previous projects
    def extract_str_from_param(self, str):
        str_list = str_filter(extract_str_multiple(self.alg_params_dict.keys(), [str, "_"+self.arg_values['exp_name']]), "_func_opt")
        assert len(str_list)>0
        
        if len(str_list)==1:
            extracted_str = str_list[0]
        else:
            if self.arg_values['sampling'] == "NICE":
                str_list = str_filter(str_list, "imp")
            elif "imp" in self.arg_values['sampling']:
                str_list = str_filter(str_list, "NICE")
            extracted_str = str_list[0]
        assert len(str_list)==1
        return extracted_str
    
    def init_alg_params_dict(self):
        self.alg_params_dict = parse_params_to_dict(self.arg_values['loadable_params'], ALLOWABLE_PARAMS)
        assert set(self.alg_params_dict.keys()).issubset(ALLOWABLE_PARAMS)
        
    def init_load_params_dict(self):
        try: 
            load_params_list = ast.literal_eval(self.arg_values["loadable_params"])
        except ValueError:
            print("The string is not a valid list representation.")
        
        if isinstance(load_params_list, list) and all(isinstance(item, str) for item in load_params_list):
            self.load_params_dict = {key: None for key in load_params_list}
        else:
            print("The list does not contain only string elements.")
        
        self.loadable_params_list = load_params_list
        self.loadable_params_set = set(load_params_list)
        assert(self.loadable_params_set.issubset(ALLOWABLE_PARAMS))
        
    def save_comp_params(self):
        self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
        if not os.path.exists(self.comp_params_path):
            os.mkdir(self.comp_params_path)
        for param in self.comp_params_dict.keys():
            param_path = self.comp_params_path + param + self.exp_params_extension
            save_param(param_path, param, self.comp_params_dict[param], self.arg_values["print_status"])
            
    def log_peak_memory_usage(self):
        peak_memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        my_print(f"Peak Memory Usage: {peak_memory_usage / 1024} MB", self.arg_values["print_status"])