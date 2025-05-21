"""
p-LoRA project
"""

"""
A script for the data preprocessing.
It takes the given dataset and outomes the partition.

Adaptation for project hfh.
Note for me: If I need a removed commmented code, I can find in similar script for the MSPPM projects 
"""

from src.experiment import * 

#TODO: 
# - remove the unnessesary functions defined in the class Experiment;

class DataPreprocessing(Experiment):
    def __init__(self, args):
        # Create a dictionary of arguments and their values
        self.arg_values = vars(self.argument_parser())
        # Print the input arguments
        print("-"*80)
        my_print("Input arguments:", self.arg_values["print_args"])
        for key, value in self.arg_values.items():
            my_print(f"{key}: {value}", self.arg_values["print_args"])

        self.general_asserts()
        if self.arg_values["loss_func"]=="l1_norm":
            self.l1_norm_asserts()
        elif self.arg_values["loss_func"]=="lin-reg":
            self.linreg_asserts()
        elif self.arg_values["loss_func"]=="log-reg":
            self.logreg_asserts()
        else:
            raise ValueError("loss_func is not supported")
        
        print_time(self.arg_values["print_status"])

        self.path_initialisation()
        self.init_comp_params_dict()
        self.init_load_params_dict()
        self.init_alg_params_dict()
        #self.load_params_dict = parse_params_to_dict(self.arg_values['loadable_params'], ALLOWABLE_PARAMS)
        
        self.init_regularizers()
        self.init_exp_param_extension()
        self.init_exp_data_extension()
        self.init_dataset_path()
        self.init_w_init_extension()
        self.init_oracles()
        
        # a flag indicating that dataset is loaded
        # self.is_dataset_loaded = 0
        
    @staticmethod
    def argument_parser() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')
        #at this point we assume that the dataset is already downloaded and is in the data folder
        parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms', help='The name of the dataset')
        parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default="log-reg", help='log-reg or l1_norm')
        parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
        parser.add_argument('--hetero', action='store', dest='hetero', type=int, default=0, help='hetero setting')
        parser.add_argument('--is_minimize', action='store', dest='is_minimize', type=int, default=1, help='minimize or not')
        parser.add_argument('--regularizer_type', action='store', dest='regularizer_type', type=str, default=0, help='str-cvx or non-cvx')
        parser.add_argument('--la', action='store', dest='la_init', type=str, default="0", help='lambda')
        parser.add_argument('--generate_dataset', action='store', dest='generate_dataset', type=int, default=0, help='generate_dataset')
        parser.add_argument('--load_raw_dataset', action='store', dest='load_raw_dataset', type=int, default=0, help='load_raw_dataset')
        parser.add_argument('--load_prepared_dataset', action='store', dest='load_prepared_dataset', type=int, default=0, help='load_prepared_dataset')
        parser.add_argument('--is_sparse_dataset', action='store', dest='is_sparse_dataset', type=int, default=0, help='sparse dataset or not')
        parser.add_argument('--print_args', action='store', dest='print_args', type=int, default=1, help='print_args')
        parser.add_argument('--print_status', action='store', dest='print_status', type=int, default=0, help='print_status')
        parser.add_argument('--computable_params', action='store', dest='comp_params_str', type=str, default="['L0', 'Li', 'L_pm']", help='list of computable params')
        parser.add_argument('--loadable_params', action='store', dest='loadable_params', type=str, default="[]", help='list of loadable params')
        parser.add_argument('--loadable_datasets', action='store', dest='loadable_datasets', type=str, default="[]", help='list of loadable data arrays')
        parser.add_argument('--batchsize', action='store', dest='batchsize', type=int, default=1, help='batchsize')
        parser.add_argument('--use_ray', action='store', dest='use_ray', type=int, default=1, help='use ray or not')
        
        #params for specific l1_norm generataion routine
        parser.add_argument('--noise_scale', action='store', dest='noise_scale', type=float, default=0.05, help='noise scale')
        parser.add_argument('--dim', action='store', dest='dim', type=int, default=100, help='dimentionality of generated l1_norm function')
        parser.add_argument('--num_samples', action='store', dest='num_samples', type=int, default=100, help='Number of datasamples for each worker')
        return parser.parse_args()

    def general_asserts(self):
        assert self.arg_values["loss_func"] in SUPPORTED_LOSS_FUNCS, f"loss_func={self.arg_values['loss_func']} is not supported"
        #assert set(self.arg_values["loadable_params"]).issubset(ALLOWABLE_PARAMS), f"loadable_params={self.arg_values['loadable_params']} contains not allowable params"
        assert self.arg_values["num_workers"] > 0
        assert self.arg_values["hetero"] in [0,1]
        assert self.arg_values["use_ray"] in [0,1]
        assert self.arg_values["is_minimize"] in [0,1] 
        assert self.arg_values["regularizer_type"] in SUPPORTED_REGULARIZERS
        assert self.arg_values["generate_dataset"] in [0,1]
        assert self.arg_values["load_prepared_dataset"] in [0,1]
        assert self.arg_values["load_raw_dataset"] in [0,1]
        assert self.arg_values["load_prepared_dataset"]*self.arg_values["generate_dataset"]==0 #only one of them can be 1
        assert self.arg_values["load_prepared_dataset"]*self.arg_values["load_raw_dataset"]==0 #only one of them can be 1
        assert self.arg_values["load_raw_dataset"]*self.arg_values["generate_dataset"]==0 #only one of them can be 1
        assert self.arg_values["is_sparse_dataset"] in [0,1]
        assert self.arg_values["print_args"] in [0,1]
        assert self.arg_values["print_status"] in [0,1]
        assert is_float(self.arg_values["la_init"]) or self.arg_values["la_init"]=="auto"
                
        # if self.arg_values['la_init'] == 'auto' and self.arg_values['regularizer_type'] == 'non-cvx':
        #     raise ValueError("la=auto is not supported for non-cvx regularizer")
        
        # if self.arg_values['is_minimize'] == 1 and self.arg_values['regularizer_type'] == 'non-cvx':
        #     raise ValueError("non-cvx regularizer is not supported for minimize")
        
        if self.arg_values['regularizer_type'] == 'non-cvx':
            if any(substring in self.arg_values['comp_params_str'] for substring in ["mu_0", "mui", "muii", "muii_min"]):
                raise ValueError("mu_0 is not supported for non-cvx setting")
        assert all(isinstance(item, str) for item in ast.literal_eval(self.arg_values['comp_params_str'])), "Not all elements are strings."

    def l1_norm_asserts(self):
        assert self.arg_values["loss_func"]=="l1_norm"
        assert self.arg_values["dataset"] in L1_NORM_DATASETS
        #assert self.arg_values["generate_dataset"]==1
        assert self.arg_values["dim"]>=0
        assert self.arg_values["num_samples"]>0
        assert self.arg_values["noise_scale"]>=0
        assert self.arg_values["batchsize"] > 0
        
    def linreg_asserts(self):
        assert self.arg_values["loss_func"]=="lin-reg"
        assert self.arg_values["dataset"] in LINREG_DATASETS
        #assert self.arg_values["generate_dataset"]==1
        assert self.arg_values["dim"]>0
        assert self.arg_values["num_samples"]>0    
    
    def logreg_asserts(self):
        if self.arg_values["loss_func"]=="log-reg" and self.arg_values["generate_dataset"]==1:
            raise ValueError("log-reg loss_func temporarily is not supported for generate_dataset")

    # consider mooving to the class Experiment
    def path_initialisation(self):
        self.data_name = self.arg_values["dataset"] + ".txt"
        # Path to the directory of the script that is running
        script_directory = os.path.dirname(os.path.abspath(__file__))
        self.raw_data_path = script_directory +'/data/'
        self.project_path = script_directory + "/"
        self.data_path = self.project_path + "data_{0}/".format(self.arg_values["dataset"])
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

    def generation(self, seed=42):
        if self.arg_values['loss_func'] == 'l1_norm' and self.arg_values['dataset'] == 'synthetic_dense':
            self.generate_l1_norm_synthetic_dense(seed)
        elif self.arg_values['loss_func'] == 'l1_norm' and self.arg_values['dataset'] == 'synthetic_sparse':        
            self.generate_l1_norm_synthetic_sparse(seed)
        elif self.arg_values['loss_func'] == 'l1_norm' and self.arg_values['dataset'] == 'synthetic_sparse_zero':        
            self.generate_l1_norm_synthetic_sparse_zero(seed)
        elif self.arg_values['loss_func'] == 'lin-reg' and self.arg_values['dataset'] == 'synthetic_dense':
            self.generate_linreg_synthetic_dense(seed)
    
    def generate_linreg_synthetic_dense(self, seed):
        num_samples = self.arg_values["num_samples"]
        num_workers = self.arg_values["num_workers"]
        dim = self.arg_values["dim"]
        #temporary restriction
        assert(num_workers==1)
        
        pretrained_fraction = 0.9
        num_samples_pre = int(num_samples*pretrained_fraction)
        num_samples_ft = num_samples - num_samples_pre
        self.comp_params_dict["num_samples_pre"] = num_samples_pre
        self.comp_params_dict["num_samples_ft"] = num_samples_ft
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        #version 1
        # A, b = make_regression(n_samples=num_samples, n_features=dim, n_informative=dim, bias=2.0, tail_strength=0.5, effective_rank=10, random_state=42)
        # A_pre = A[0:num_samples_pre].copy()
        # b_pre = b[0:num_samples_pre].copy()
        # A_ft = A[num_samples_pre:].copy()
        # b_ft = b[num_samples_pre:].copy()   
        # A_pre_scaled, b_pre_scaled, A_ft_scaled, b_ft_scaled = scale_arrays(A_pre, b_pre, A_ft, b_ft)
        
        # self.X_pre, self.y_pre, self.c_pre = transform_to_quadratic_form(A_pre_scaled, b_pre_scaled)
        # self.X_ft, self.y_ft, self.c_ft = transform_to_quadratic_form(A_ft_scaled, b_ft_scaled)
        # print (f"frobenius_diff: {frobenius_diff(self.X_pre, self.X_ft)}")

        #version 2
        A_pre, b_pre = make_regression(n_samples=num_samples_pre, n_features=dim, n_informative=dim,      noise=20.0, bias=0.0, tail_strength=0.8, effective_rank=64, random_state=42)
        A_ft, b_ft =   make_regression(n_samples=num_samples_ft,  n_features=dim, n_informative=dim // 2, noise=50.0, bias=10.0, tail_strength=0.9, effective_rank=32, random_state=84)
        
        #A_pre_rot = apply_random_rotation(A_pre, seed=123)
        #A_ft_rot  = apply_random_rotation(A_ft,  seed=999)
        #A_pre_scaled, b_pre_scaled, A_ft_scaled, b_ft_scaled = scale_arrays(A_pre_rot, b_pre, A_ft_rot, b_ft)
        
        A_pre_scaled, b_pre_scaled = scale_arrays(A_pre, b_pre)
        A_ft_scaled, b_ft_scaled = scale_arrays_custom(A_ft, b_ft, target_mean=1.0, target_std=2.)
        
        self.A_pre, self.b_pre = A_pre_scaled.copy(), b_pre_scaled.copy()
        self.X_pre, self.y_pre, self.c_pre = transform_to_quadratic_form(A_pre_scaled, b_pre_scaled)
        
        self.A_ft, self.b_ft = A_ft_scaled.copy(), b_ft_scaled.copy()
        self.X_ft, self.y_ft, self.c_ft = transform_to_quadratic_form(A_ft_scaled, b_ft_scaled)
        
        #print (f"frobenius_diff: {frobenius_diff(self.X_pre, self.X_ft)}")
        
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        
        data_to_save = {'X_pre': self.X_pre, 'y_pre': self.y_pre, 'c_pre': self.c_pre, 
                        'X_ft': self.X_ft,  'y_ft': self.y_ft,   'c_ft': self.c_ft,
                        'A_pre': self.A_pre, 'b_pre': self.b_pre, 'A_ft': self.A_ft, 'b_ft': self.b_ft}
        self.save_dataset(data_to_save)
        
        # Setting lambda and x_0 depending on the setup
        if self.arg_values["loss_func"] == "lin-reg" and self.arg_values["regularizer_type"] == "str-cvx":
            raise NotImplementedError("str-cvx regularizer is not yet implemented")
            #example from Q-RR:
            # desired_cond_number = 1e+4
            # L_non_reg = np.float64(linalg.eigh(a=hess_f_non_reg, eigvals_only=True, turbo=True, type=1, eigvals=(d_0-1, d_0-1))[0])
            # la = L_non_reg/(2*(desired_cond_number + 1))
        elif self.arg_values["loss_func"] == "lin-reg" and self.arg_values["regularizer_type"] == "cvx":
            raise NotImplementedError("cvx regularizer option is not yet implemented")
        
        elif self.arg_values["loss_func"] == "lin-reg" and self.arg_values["regularizer_type"] == "non-cvx":
            #L_pre setting
            L_pre_non_reg = second_matrix_norm(self.oracle_dict["hess_bound"](np.zeros(dim), self.X_pre, self.y_pre,
                                                                              {"la":0, "n_samples":self.comp_params_dict["num_samples_pre"]}))
            self.comp_params_dict["la_pre"] = L_pre_non_reg
            self.comp_params_dict['L_0_pre'] = second_matrix_norm(self.oracle_dict["hess_bound"](np.zeros(dim), self.X_pre, self.y_pre, 
                                                                                    {"la":self.comp_params_dict["la_pre"], "n_samples":self.comp_params_dict["num_samples_pre"]}))
            
            #L_ft setting
            L_ft_non_reg = second_matrix_norm(self.oracle_dict["hess_bound"](np.zeros(dim), self.X_ft, self.y_ft, 
                                                                             {"la":0, "n_samples":self.comp_params_dict["num_samples_ft"]}))
            self.comp_params_dict["la_ft"] = L_ft_non_reg
            self.comp_params_dict['L_0_ft'] = second_matrix_norm(self.oracle_dict["hess_bound"](np.zeros(dim), self.X_ft, self.y_ft,
                                                                                                {"la":self.comp_params_dict["la_ft"], "n_samples":self.comp_params_dict["num_samples_ft"]}))
            
            #minimize for pretraining
            x_star, f_star = self.minimize(self.X_pre, self.y_pre, self.c_pre, 
                                           {"la":self.comp_params_dict["la_pre"], "n_samples":self.comp_params_dict["num_samples_pre"], "stepsize":1/self.comp_params_dict['L_0_pre']}, seed)
            self.comp_params_dict['x_star_pre'] = x_star.copy()
            self.comp_params_dict['f_star_pre'] = f_star
            
        else:
            raise ValueError("regularizer_type is not supported")
             
        self.x_0 = self.comp_params_dict['x_star_pre'].copy()
        self.save_w_init()
            
        final_memory = process.memory_info().rss
        total_memory_used = final_memory - initial_memory
        # Sum nbytes for all NumPy arrays in the dictionary
        arrays_memory = 0
        for value in data_to_save.values():
            # If value is a NumPy array, sum up nbytes
            if isinstance(value, np.ndarray):
                arrays_memory += value.nbytes
            # If value is a scalar (float, int, etc.), you can skip or handle separately

        print(f"Memory used by NumPy arrays: {arrays_memory / (1024 ** 3):.2f} GB")
        print(f"Total memory used during the generation: {total_memory_used / (1024 ** 3):.2f} GB")

    def generate_l1_norm_synthetic_sparse_zero(self, seed):
        #legacy code from previous projects
        num_samples = self.arg_values["num_samples"]
        num_workers = self.arg_values["num_workers"]
        batchsize = self.arg_values["batchsize"]
        la = self.arg_values["la"]
        assert(num_samples==1)
        dim = self.arg_values["dim"]
        rs1 = RandomState(seed+1)
        rs2 = RandomState(seed+2)
        rs3 = RandomState(seed+3)
        rs4 = RandomState(seed+4)
        rs5 = RandomState(seed+5)
        
        my_print("Sparse dataset generation...", self.arg_values["print_status"])
        self.dataset_path = self.data_path + 'data' + self.exp_data_extension + "/"

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)

        if not self.arg_values["use_ray"]:
            self.X = []
            self.y = np.zeros((num_workers, dim), dtype=np.float64)
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            Li = np.zeros(num_workers)
    
            la_quad = 1e-6
            #init matrices
            for i in tqdm(range(self.arg_values["num_workers"]), desc='Samples Progress', leave=False):
                xi_s = rs1.normal(loc=0, scale=1, size=1)[0]
                nu_s = 1 + self.arg_values["noise_scale"]*xi_s
                H = (nu_s/4)*((np.tri(self.arg_values["dim"], self.arg_values["dim"], 1, dtype=int) - np.tri(self.arg_values["dim"], self.arg_values["dim"], -2, dtype=int))*(-1) + 3*np.eye(self.arg_values["dim"]))
                self.X.append(csr_matrix(H))
                
                self.y[i] = np.zeros(self.arg_values["dim"], dtype=np.float64)
            
            X_avg = sum(self.X)/self.arg_values["num_workers"]
            X_avg_dense = X_avg.toarray()
            la_min = min_eigval(X_avg_dense)
            
            #update matrices and compute Li
            add_X = csr_matrix((la_quad-la_min)*np.eye(self.arg_values["dim"]))
            
            for i in range(self.arg_values["num_workers"]):
                self.X[i] = self.X[i] + add_X
                X_i_dense = self.X[i].toarray()
                Li[i] = second_matrix_norm(X_i_dense)
            
            data_to_save = {'X': self.X, 'y': self.y}
            self.save_dataset(data_to_save)
            
            self.x_0 = rs5.randn(dim).astype(np.float64)
            self.save_w_init()
            
            self.comp_params_dict["L_0,i"] = Li
            self.comp_params_dict["L_0"] = np.mean(Li)
            self.comp_params_dict["L_0,v"] = second_matrix_norm(X_avg_dense)
            self.comp_params_dict["wtL_0"] = np.sqrt(np.mean(Li**2))
            self.comp_params_dict["L_pm"] = np.sqrt(self.comp_params_dict["wtL_0"]**2 - self.comp_params_dict["L_0"]**2)
            self.comp_params_dict["L_pm,v"] = np.sqrt(self.comp_params_dict["wtL_0"]**2 - self.comp_params_dict["L_0,v"]**2)
            
            # Calculate total memory usage
            final_memory = process.memory_info().rss
            total_memory_used = final_memory - initial_memory
            X_matrices_memory = sum(mat.data.nbytes + mat.indptr.nbytes + mat.indices.nbytes for mat in self.X)
            print(f"Memory used by X matrices: {X_matrices_memory / (1024 ** 3):.2f} GB")
            print(f"Total memory used during the generation: {total_memory_used / (1024 ** 3):.2f} GB")
        else:
            raise ValueError("paralell version is not supported at this point")
    
    def save_w_init(self):
        np.save(self.data_path + 'w_init' + self.w_init_extension, self.x_0)
    
    def save_dataset(self, data_to_save):
        for key, value in data_to_save.items():
            if key=="X":
                if self.arg_values["is_sparse_dataset"]:
                    file_path = self.dataset_path + 'X' + self.exp_data_extension + '.h5'
                    with h5py.File(file_path, 'w', track_order=True) as f:
                        for i in range(len(self.X)):
                            matrix = self.X[i]
                            grp = f.create_group(f'matrix_{i}')
                            grp.create_dataset('data', data=matrix.data)
                            grp.create_dataset('indices', data=matrix.indices)
                            grp.create_dataset('indptr', data=matrix.indptr)
                            grp.create_dataset('shape', data=matrix.shape)
                    my_print("Sparse datasets are saved.", self.arg_values["print_status"])
                else:
                    np.save(self.dataset_path + 'X' + self.exp_data_extension, self.X)
                    my_print("Dense datasets are saved.", self.arg_values["print_status"])
            else:
                np.save(self.dataset_path + key + self.exp_data_extension, value)
                my_print(f"{key} dataset is saved.", self.arg_values["print_status"])
        
    def load_raw_dataset(self):
        if self.arg_values["loss_func"]=="l1_norm":
            raise ValueError("l1_norm loss_func temporarily is not supported for load_raw_dataset")
        elif self.arg_values["loss_func"]=="log-reg":
            self.load_raw_log_reg()
        else:
            raise ValueError("loss_func is not supported")
            
    #consider mooving to the class Experiment    
    def load_raw_log_reg(self, seed=42):
        #this function is in the progress
        if self.arg_values["is_sparse_dataset"]:
            raise NotImplementedError("sparse datasets are not supported at this point")
        else:
            self.dataset_path = self.data_path
            if not os.path.exists(self.dataset_path):
                os.mkdir(self.dataset_path)
                
            data, enc_labels = load_svmlight_dataset(self.RAW_DATA_PATH + self.data_name, self.arg_values["is_sparse_dataset"])  
            assert (type(enc_labels) == np.ndarray)
            if np.sum(np.isnan(enc_labels)) > 0:
                raise ValueError("nan values of labels")
            if np.sum(np.isnan(data)) > 0:
                raise ValueError("nan values in data matrix")
            my_print (f"Data shape: {data.shape}", self.arg_values["print_status"])
            self.X_0 = np.float64(data)
            self.y_0 = enc_labels
            assert len(self.X_0.shape) == 2
            assert len(self.y_0.shape) == 1
            nan_check([self.X_0, self.y_0])
            dim = self.X_0.shape[1]
            
            self.any_vector = np.zeros(dim)
            rs_w_init = RandomState(seed)
            self.x_0 = rs_w_init.normal(loc=0.0, scale=1.0, size=dim)

            data_to_save = {'X_0': self.X_0, 'y_0': self.y_0}
            self.save_dataset(data_to_save)
            self.save_w_init()
            
            # pass num_samples based on the dataset itself 
            self.arg_values["num_samples"] = self.X_0.shape[0]     
    
    def compute_params(self):
        #self.comp_params_dict - dicitionary of computable parameters we want to save later
        #self.alg_params_dict - dicitionary of loaded params we do not want so save
 
        num_samples = self.arg_values["num_samples"]
        num_workers = self.arg_values["num_workers"]
        dim = self.arg_values["dim"]
       
        if len(self.loadable_params_list) > 0:
            for param in self.loadable_params_list:
                self.comp_params_path = self.data_path + 'comp_params' + self.exp_params_extension + "/"
                param_path = self.comp_params_path + param + self.exp_params_extension
                self.alg_params_dict[param] = load_param(param_path, param)
        
        
    
    def gd_minimize(self, f_d, grad_d, x_0, gamma=1e-3, max_iter=10000, tol=1e-8):
        """
        A simple gradient descent minimization routine with a stopping criterion.

        Parameters
        ----------
        f_d : callable
            A function that returns the objective value f(w).
        grad_d : callable
            A function that returns the gradient of the objective at w.
        x_0 : np.ndarray
            The initial guess for the minimization.
        gamma : float, optional
            Step size (learning rate) for the gradient descent.
        max_iter : int, optional
            Maximum number of iterations.
        tol : float, optional
            Tolerance for the squared norm of the gradient. Stop when sqnorm(grad_x_t) < tol.

        Returns
        -------
        x_star : np.ndarray
            The approximate minimizer found by gradient descent.
        f_star : float
            The objective value at x_star.
        """
        x_t = np.copy(x_0)
        t = 0

        # Compute initial gradient and its squared norm
        grad_x_t = grad_d(x_t).copy()
        sqn_grad_x_t = np.dot(grad_x_t, grad_x_t)

        # Use tqdm manually in a while loop
        with tqdm(total=max_iter) as pbar:
            # Repeat while t < max_iter and the squared norm of the gradient is larger than tol
            while t < max_iter and sqn_grad_x_t > tol:
                pbar.set_description(f"Iter {t}, sqnorm(grad_x_t)={sqn_grad_x_t:e}, f(x_t)={f_d(x_t):e}")
                # Gradient descent update: x_{t+1} = x_t - gamma * grad(f)(x_t)
                x_t -= gamma * grad_x_t
                # Recompute gradient and its squared norm
                grad_x_t = grad_d(x_t).copy()
                sqn_grad_x_t = np.dot(grad_x_t, grad_x_t)
                # Increment iteration counter
                t += 1
                pbar.update(1)

        # Final point and its objective value
        x_star = x_t
        f_star = f_d(x_star)

        tqdm.write(f"Stopped at iteration {t}, sqnorm(grad_x_t)={sqn_grad_x_t:e}")

        return x_star, f_star

    
    def minimize(self, X, y, c, params, seed=45):
        rs2 = RandomState(seed)
        num_attempts = {"str-cvx":1, "non-cvx":1}[self.arg_values["regularizer_type"]]
        max_iter_opt = {"str-cvx":1_000_00, "non-cvx":10000}[self.arg_values["regularizer_type"]]

        f_d = lambda w: self.oracle_dict["f"](w, X, y, c, params)
        grad_d = lambda w: self.oracle_dict["grad"](w, X, y, params) 
        hess_d = lambda w: self.oracle_dict["hess"](w, X, y, params)
        
        #L_0 = self.comp_params_dict['L_0']
        my_print("Computing numerical solution...", self.arg_values["print_status"])
        if self.arg_values["dataset"] == "synthetic_sparse_zero":
            f_star_min = 0.0
            x_star_min = np.zeros(self.arg_values["dim"], dtype=np.float64)
        else:
            f_star_min = np.inf
            for j in tqdm(range(num_attempts)):
                x_j = rs2.normal(loc=0.0, scale=1.0, size=self.arg_values["dim"])
                
                x_star, f_star = self.gd_minimize(f_d, grad_d, x_j, gamma=params["stepsize"], max_iter=max_iter_opt)
                #minimize_result = minimize(fun=f_d, jac=grad_d, hess=hess_d, x0=x_j, method="Newton-CG", tol=1e-10, options={"maxiter": max_iter_opt})
                if f_star < f_star_min:
                    f_star_min = f_star
                    x_star_min = x_star
                    
        return x_star_min.copy(), f_star_min
    
        #self.comp_params_dict["f_star"] = 
        #self.comp_params_dict["x_star"] = 

if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(sys.argv[1:])
    if data_preprocessing.arg_values["load_prepared_dataset"]==1:
        data_preprocessing.load_prepared_datasets()
    if data_preprocessing.arg_values["load_raw_dataset"]==1:
        data_preprocessing.load_raw_dataset()
    if data_preprocessing.arg_values["generate_dataset"]==1:
        data_preprocessing.generation()
    data_preprocessing.compute_params()
    
    if data_preprocessing.arg_values["is_minimize"]==1:
        #TODO: revise for the strongly convex setting
        raise NotImplementedError("minimize is not supported at this point")
        data_preprocessing.minimize()
    data_preprocessing.save_comp_params()
    data_preprocessing.log_peak_memory_usage()


def generate_l1_norm_synthetic_dense(self, seed):
    raise NotImplementedError("generate_l1_norm_synthetic_dense is not supported at this point")
    
