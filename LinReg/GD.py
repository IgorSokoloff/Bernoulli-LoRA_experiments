"""
GD
"""
# state - short name for states_dict
## state["g"] - relates to the full gradient estimate
# collectable_metric - short name for collectable_metrics_dict
# alg_param - short name for alg_params_dict
# oracle - short name for oracle_dict
# data - short name for data_dict

from src.algorithm import *

class GD(Algorithm):
    def __init__(self, args=None):
        super().__init__(args)

    def script_directory(self):
        return os.path.dirname(os.path.abspath(__file__))

    # Algorithm dependendent function
    def fill_alg_params_dict(self, state, oracle, data, alg_param):
        alg_param["step_size"] = np.float64(1/alg_param["L_0_ft"])*alg_param['factor']
        return d_copy(alg_param)    
    
    #we initialise g according to the fact that g^0 is computed to be full grad
    def init_states_dict(self, state, oracle, data, alg_param):
        state["g"] = oracle["grad"](state["x"], data["X_ft"], data["y_ft"], alg_param["oracle_params"]).copy()
        return d_copy(state), d_copy(alg_param)
    
    # Algorithm dependendent function
    def init_collectable_metrics_dict(self, state, collectable_metric, alg_param, oracle, data):
        # Initialization of collectable metrics
        if "iters" in collectable_metric.keys():
            collectable_metric["iters"] = [0]
        if "epochs" in collectable_metric.keys():
            #we initialise metrics in the begininngin according to the fact that g^0 is computed to be full grad
            collectable_metric["epochs"] = [1]
        if "total_cost" in collectable_metric.keys():
            collectable_metric["total_cost"] = [1]
        if "sqnorm" in collectable_metric.keys():
            collectable_metric["sqnorm"] = [sq_two_norm(state["g"])]
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
            collectable_metric["sqnorm"].append(sq_two_norm(state["g"]))
            #collectable_metric["sqnorm"].append(sqnorm(oracle["grad"](state["x"], data["X_ft"], data["y_ft"])))
        return d_copy(collectable_metric)
        
    # Algorithm dependendent function
    def update(self, state, data, collectable_metric, alg_param, oracle, update_collectable_metrics_dict):
        # gradient-like step:
        state["x_prev"] = state["x"].copy() #x_prev = x.copy()
        x_prev = state["x_prev"].copy()
        state["x"] = x_prev - alg_param["step_size"]*state["g"] #x = x - step_size*grad
        
        # updating gradient estimate:
        state["g"] = oracle["grad"](state["x"], data["X_ft"], data["y_ft"], alg_param["oracle_params"]).copy() # g_tpo = grad_tpo(x)
        epochs_single_iter = 1
        
        collectable_metric = update_collectable_metrics_dict(state, d_copy(collectable_metric), oracle, data, epochs_single_iter)
        return d_copy(state), d_copy(collectable_metric), d_copy(alg_param)
        
if __name__ == "__main__":
    GD().run()
    
    
    