"""
RC-LORA

RAC-LoRA, A and B derived as a special case of that script
"""

# state - short name for states_dict
## state["G"] - relates to the full gradient estimate in a matric form
# collectable_metric - short name for collectable_metrics_dict
# alg_param - short name for alg_params_dict
# oracle - short name for oracle_dict
# data - short name for data_dict

from src.algorithm import *

class RC_LoRA(Algorithm):
    def __init__(self, args=None):
        super().__init__(args)
            
    def script_directory(self):
        return os.path.dirname(os.path.abspath(__file__))
    
    def gd_update(self, state, oracle, data, alg_param):
        if alg_param["exp_name"] in ['RC-LoRA', 'stoch_RC-LoRA-SGD', 'stoch_RC-LoRA-MVR', 'finite_RC-LoRA-SGD', 'finite_RC-LoRA-PAGE']:
            c_t = alg_param["rs_bernoulli"].binomial(1, alg_param["prob"], 1)[0] #sample bernoully random varinable
        elif alg_param["exp_name"] in ['RAC-LoRA_A', 'stoch_RAC-LoRA_A-SGD', 'finite_RAC-LoRA_A-SGD']:
            c_t = 0 
        elif alg_param["exp_name"] in ['RAC-LoRA_B', 'stoch_RAC-LoRA_B-SGD', 'finite_RAC-LoRA_B-SGD']:
            c_t = 1
        else:
            raise NotImplementedError("Unknown experiment name")
            
        if c_t == 1:
            B_S = alg_param["rs_sketch"].randn(alg_param["dim_out"],alg_param["rank"])
            H_B = left_sketch_B(B_S).copy()
            state["gd_update"] = np.reshape(H_B@state["G"],(alg_param["dim"],))
        else:
            A_S = alg_param["rs_sketch"].randn(alg_param["rank"], alg_param["dim_in"])
            H_A = right_sketch_A(A_S).copy()
            state["gd_update"] = np.reshape(state["G"]@H_A,(alg_param["dim"],))
        return d_copy(state), d_copy(alg_param)
    
    def grad_estimates (self, state, oracle, data, alg_param):
        state["FG_prev"] = state["FG"].copy()
        state["FG"] = oracle["FG"](state["x"], alg_param).copy() # full gradient at the new point
        state["G_prev"] = state["G"].copy()
        if alg_param["exp_name"] in ['RC-LoRA', 'RAC-LoRA_A', 'RAC-LoRA_B']:
            state["G"] = state["FG"].copy()
            alg_param['epochs_single_iter']=1
            
        elif alg_param["exp_name"] in ['stoch_RC-LoRA-SGD', 'stoch_RAC-LoRA_A-SGD', 'stoch_RAC-LoRA_B-SGD']:
            # TODO: Check that I sample different noise for each iteration
            state["G"] = state["FG"] + alg_param["rs_noise"].normal(loc=0, scale=alg_param["noise_scale"], size=(alg_param["dim_out"],alg_param["dim_in"]))
            alg_param['epochs_single_iter']=1
            
        elif alg_param["exp_name"] == 'stoch_RC-LoRA-MVR':
            # TODO: Check that I sample different noise for each iteration
            Noise = alg_param["rs_noise"].normal(loc=0, scale=alg_param["noise_scale"], size=(alg_param["dim_out"],alg_param["dim_in"])).copy()
            state["SG_prev"] = state["FG_prev"] + Noise
            state["SG"] = state["FG"] + Noise
            state["G"] = state["SG"] + (1 - alg_param["momentum"]) * (state["G_prev"] - state["SG_prev"])
            alg_param['epochs_single_iter'] = 1
            
        elif alg_param["exp_name"] in ['finite_RC-LoRA-SGD', 'finite_RAC-LoRA_A-SGD', 'finite_RAC-LoRA_B-SGD']:
            alg_param['oracle_params']['inds']=alg_param['rs_sample'].choice(a=np.arange(alg_param["num_samples_ft"]), size=alg_param["batchsize"], replace=False, p=alg_param["probs"]).copy()
            state['MG'] = oracle['MG'](state["x"], alg_param).copy()
            state["G"] = state['MG'].copy()
            alg_param['epochs_single_iter'] = alg_param["batchsize"]/alg_param["num_samples_ft"]
            
        elif alg_param["exp_name"] == 'finite_RC-LoRA-PAGE':
            
            c_p = alg_param["rs_page"].binomial(1, alg_param["prob_page"], 1)[0] #sample bernoully random varinable
            if c_p==1:
                state["G"] = state["FG"].copy()
                alg_param['epochs_single_iter'] = 1
            else:
                alg_param['oracle_params']['inds']=alg_param['rs_sample'].choice(a=np.arange(alg_param["num_samples_ft"]), size=alg_param["batchsize"], replace=False, p=alg_param["probs"]).copy()
                state["MG_prev"] = oracle['MG'](state["x_prev"], alg_param).copy()
                state["MG"] = oracle['MG'](state["x"], alg_param).copy()
                state["G"] = state["G_prev"] + state["MG"] - state["MG_prev"]
                alg_param['epochs_single_iter'] = 2*alg_param["batchsize"]/alg_param["num_samples_ft"]
        else:
            raise NotImplementedError("Unknown experiment name")
        
        
        state, alg_param = self.gd_update(d_copy(state), oracle, data, d_copy(alg_param))
        
        return d_copy(state), d_copy(alg_param)
    
    # Algorithm dependendent function
    def fill_alg_params_dict(self, state, oracle, data, alg_param):
        alg_param["step_size"] = np.float64(1/alg_param["L_0_ft"])*alg_param['factor']
        alg_param["probs"] = np.ones(alg_param["num_samples_ft"], dtype=np.float64)/alg_param["num_samples_ft"] # for minibatch methods; we assume uniform sampling
        if alg_param["exp_name"] in ['finite_RC-LoRA-PAGE']:
            alg_param["prob_page"] = alg_param["batchsize"]/(alg_param["batchsize"] + alg_param["num_samples_ft"])
        
        return d_copy(alg_param)
        
    # Algorithm dependendent function
    def init_oracles_dict(self, state, oracle, data, alg_param):
        oracle["FG"] = lambda x, alg_param: oracle["grad"](x, data["X_ft"], data["y_ft"], alg_param["oracle_params"]).reshape((alg_param["dim_out"], alg_param["dim_in"])).copy() # full gradient
        oracle["MG"] = lambda x, alg_param: oracle["minibatch_grad"](x, data["A_ft"], data["b_ft"], alg_param["oracle_params"]).reshape((alg_param["dim_out"], alg_param["dim_in"])).copy() # minibatched gradient 
        return d_copy(oracle)
    
    # Algorithm dependendent function   
    def init_states_dict(self, state, oracle, data, alg_param):
        state["FG"] = oracle["FG"](state["x"], alg_param).copy()
        if alg_param["exp_name"] in ['RC-LoRA', 'RAC-LoRA_A', 'RAC-LoRA_B']:
            state["G"] = state["FG"].copy()
            alg_param['epochs_single_iter']=1
            
        elif alg_param["exp_name"] in ['stoch_RC-LoRA-SGD', 'stoch_RC-LoRA-MVR', 'stoch_RAC-LoRA_A-SGD', 'stoch_RAC-LoRA_B-SGD']: # TODO: Check that I sample the same noise for each launch
            state["G"] = state["FG"] + alg_param["rs_noise"].normal(loc=0, scale=alg_param["noise_scale"], size=(alg_param["dim_out"],alg_param["dim_in"]))
            alg_param['epochs_single_iter']=1
            
        elif alg_param["exp_name"] in ['finite_RC-LoRA-SGD', 'finite_RC-LoRA-PAGE', 'finite_RAC-LoRA_A-SGD', 'finite_RAC-LoRA_B-SGD']:
            alg_param['oracle_params']['inds']=alg_param['rs_sample'].choice(a=np.arange(alg_param["num_samples_ft"]), size=alg_param["batchsize"], replace=False, p=alg_param["probs"]).copy()
            state["MG"] = oracle['MG'](state["x"], alg_param).copy()
            state["G"] = state['MG'].copy()
            alg_param['epochs_single_iter'] = alg_param["batchsize"]/alg_param["num_samples_ft"]
        else:
            raise NotImplementedError("Unknown experiment name")
        
        state, alg_param = self.gd_update(d_copy(state), oracle, data, d_copy(alg_param))
        return d_copy(state), d_copy(alg_param)
    
    # Algorithm dependendent function
    def init_collectable_metrics_dict(self, state, collectable_metric, alg_param, oracle, data):
        if "iters" in collectable_metric.keys():
            collectable_metric["iters"] = [0]
        if "epochs" in collectable_metric.keys():
            collectable_metric["epochs"] = [alg_param['epochs_single_iter']]
        if "total_cost" in collectable_metric.keys():
            collectable_metric["total_cost"] = [0]
        if "sqnorm" in collectable_metric.keys():
            collectable_metric["sqnorm"] = [sq_fro_norm(state["FG"])] #full gradient norm
        return d_copy(collectable_metric)
    
    # Algorithm dependendent function
    def update_collectable_metrics_dict(self, state, collectable_metric, oracle, data, epochs_single_iter):
        if "iters" in collectable_metric.keys():
            collectable_metric["iters"].append(collectable_metric["iters"][-1]+1)
        if "epochs" in collectable_metric.keys():
            collectable_metric["epochs"].append(collectable_metric["epochs"][-1]+epochs_single_iter)
        if "total_cost" in collectable_metric.keys():
            collectable_metric["total_cost"].append(collectable_metric["total_cost"][-1]+epochs_single_iter) #epochs and total cost are the same
        if "sqnorm" in collectable_metric.keys():
            collectable_metric["sqnorm"].append(sq_fro_norm(state["FG"])) #full gradient norm
        return d_copy(collectable_metric)
        
    # Algorithm dependendent function
    def update(self, state, data, collectable_metric, alg_param, oracle, update_collectable_metrics_dict):
        state["x_prev"] = state["x"].copy()
        state["x"] = state["x_prev"] - alg_param["step_size"]*state["gd_update"] # gradient-like step
        state, alg_param = self.grad_estimates(d_copy(state), oracle, data, d_copy(alg_param))
        
        collectable_metric = update_collectable_metrics_dict(state, d_copy(collectable_metric), oracle, data, alg_param['epochs_single_iter'])
        return d_copy(state), d_copy(collectable_metric), d_copy(alg_param)
        
if __name__ == "__main__":
    RC_LoRA().run()
    
    
    