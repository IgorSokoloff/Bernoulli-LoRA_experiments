from src.experiment import *

from matplotlib import pyplot as plt

# x_metric_str - title of the x axis; example: "iters"
# y_metric_str - title of the y axis; example: "norms"

class Plotter(Experiment):
    def __init__(self, settings_dict, parameters_dict):
        # for key, value in settings_dict.items():
        #     setattr(self, key, value)
        # for key, value in parameters_dict.items():
        #     setattr(self, key, value)
        
        self.arg_values = settings_dict|parameters_dict
        self.settings_asserts()
        self.init_paths_and_folders()
        self.x_label = {'epochs':'Data passes','comms':'Communications','iters':'Iterations','bits':'bits/n', 'gammas':r"$\gamma$", 'n':'n'}[self.arg_values["x_metric_str"]]
        self.y_label = {'arg_res':r"$\| x^t - x_{\star} \|^2$", 'sqnorm':r"$\|\nabla f(x^t)\|^2$", "func_diff":r"$f(x^t) - f(x_{\star})$", 'noise':r"$\frac{\gamma \sigma_{n, \star}^2}{\gamma \mu^2+2 \mu}$", 'sigma':'noise' }[self.arg_values["y_metric_str"]]
        
        if not self.arg_values["show_all_trajectories"]:
            self.NUM_LAUNCHES = 1
            
    #Project dependend function
    # TODO: consider shifting some of the functions to the parent class
    def init_paths_and_folders(self):
        self.notebook_path = os.path.abspath(get_ipython().starting_dir)
        self.RAW_DATA_PATH = self.notebook_path +'/data/'
        self.project_path= self.notebook_path + "/"
        self.plot_path_extension = self.arg_values["loss_func"] + "_" + self.arg_values["dataset"]
        self.data_path = self.project_path + "data_{0}/".format(self.arg_values["dataset"])
        
        assert isinstance(self.notebook_path, str) and self.notebook_path != "", "notebook_path must be a non-empty string"
        assert isinstance(self.RAW_DATA_PATH, str) and self.RAW_DATA_PATH != "", "RAW_DATA_PATH must be a non-empty string"
        assert isinstance(self.project_path, str) and self.project_path != "", "project_path must be a non-empty string"
        assert isinstance(self.data_path, str) and self.data_path != "", "data_path must be a non-empty string"
        
        if self.arg_values['save_separately']:
            self.plot_path = self.project_path + "plots_{0}/".format(self.plot_path_extension)
        else:
            self.plot_path = self.project_path + "plots_test/"
        
    #Project dependend function
    # TODO: consider shifting some of the functions to the parent class
    def settings_asserts(self):
        assert self.arg_values['y_avg_option'] in ["mean", "median"]
        assert self.arg_values['x_metric_str'] in ALLOWABLE_COLLECTABLE_METRICS
        assert self.arg_values['y_metric_str'] in ALLOWABLE_COLLECTABLE_METRICS
        assert self.arg_values['minimize_over'] in ["x_metric", "y_metric"]
        assert self.arg_values['xscale'] in ["log", "linear"], "xscale must be either 'log' or 'linear'"
        assert self.arg_values['yscale'] in ["log", "linear"], "yscale must be either 'log' or 'linear'"
        assert isinstance(self.arg_values['ub_x_logs_upload'], (int, float)) and self.arg_values['ub_x_logs_upload'] > 0, "ub_x_logs_upload must be a positive number"
        assert isinstance(self.arg_values['freq'], int) and self.arg_values['freq'] > 0, "freq must be a positive integer"
        assert isinstance(self.arg_values['mooving_avg_window'], int) and self.arg_values['mooving_avg_window'] > 0, "mooving_avg_window must be a positive integer"
        assert isinstance(self.arg_values["save_plot"], int) and self.arg_values["save_plot"] in [0, 1], "save_plot must be an integer either 0 or 1"
        assert isinstance(self.arg_values['save_separately'], int) and self.arg_values['save_separately'] in [0, 1], "save_separately must be an integer either 0 or 1"
        assert isinstance(self.arg_values["show_std"], int) and self.arg_values["show_std"] in [0, 1], "draw_std must be an integer either 0 or 1"
        assert isinstance(self.arg_values['show_all_trajectories'], int) and self.arg_values['show_all_trajectories'] in [0, 1], "show_all_trajectories must be an integer either 0 or 1"
        assert isinstance(self.arg_values['print_status'], int) and self.arg_values['print_status'] in [0, 1], "print_status must be an integer either 0 or 1"
        assert isinstance(self.arg_values["show_plot"], int) and self.arg_values["show_plot"] in [0, 1], "draw must be an integer either 0 or 1"
        assert isinstance(self.arg_values['grid'], int) and self.arg_values['grid'] in [0, 1], "grid must be an integer either 0 or 1"
        assert isinstance(self.arg_values['mooving_avg'], int) and self.arg_values['mooving_avg'] in [0, 1], "mooving_avg must be an integer either 0 or 1"
        assert isinstance(self.arg_values['common_legend'], int) and self.arg_values['common_legend'] in [0, 1], "common_legend must be an integer either 0 or 1"
        assert isinstance(self.arg_values['NUM_LAUNCHES'], int) and self.arg_values['NUM_LAUNCHES'] >= 1, "NUM_LAUNCHES must be an integer greater than or equal to 1"
        assert isinstance(self.arg_values['tol_for_stopping'], (int, float)) and self.arg_values['tol_for_stopping'] > 0, "tol_for_stopping must be a positive number"
        assert isinstance(self.arg_values['tol'], (int, float)) and self.arg_values['tol'] > 0, "tol must be a positive number"
        assert isinstance(self.arg_values['show_title'], int) and self.arg_values['show_title'] in [0, 1], "show_title must be an integer either 0 or 1"
        assert isinstance(self.arg_values['show_noise_level'], int) and self.arg_values['show_noise_level'] in [0, 1], "show_noise_level must be an integer either 0 or 1"
        
    #Project dependend function
    def load_one_exp_dict_logs (self, exp_dict):
        number_points = None
        if os.path.isfile(exp_dict["logs_path_x"]) and os.path.isfile(exp_dict["logs_path_y"]): # if main logs exists
            x_metric = fix_shape(load_np_array(exp_dict["logs_path_x"]))          
            y_metric = fix_shape(load_np_array(exp_dict["logs_path_y"]))
            
            number_points1 = len(x_metric[x_metric < self.arg_values["ub_x_logs_upload"]])
            number_points2 = len(y_metric[y_metric > self.arg_values["tol_for_stopping"]])
            number_points = min(number_points1, number_points2)
            exp_dict["x_metric"] = x_metric[:number_points:self.arg_values["freq"]]
            if self.arg_values["mooving_avg"]:
                exp_dict["y_metric"] = moving_average_with_padding(y_metric[:number_points:self.arg_values["freq"]], self.arg_values["mooving_avg_window"])
            else: 
                exp_dict["y_metric"] = y_metric[:number_points:self.arg_values["freq"]]

            # TODO: in future, one can generalise it to any additional metrics
            if "logs_path_x_grad_comp" in exp_dict.keys() and "logs_path_y_grad_comp" in exp_dict.keys(): # if full grad logs exists
                if os.path.isfile(exp_dict["logs_path_x_grad_comp"]) and os.path.isfile(exp_dict["logs_path_y_grad_comp"]):
                    x_metric_grad_comp = fix_shape(load_np_array(exp_dict["logs_path_x_grad_comp"]))          
                    y_metric_grad_comp = fix_shape(load_np_array(exp_dict["logs_path_y_grad_comp"]))
                    number_points = len(x_metric_grad_comp[x_metric_grad_comp < self.arg_values["ub_x_logs_upload"]])
                    exp_dict["x_metric_grad_comp"] = x_metric_grad_comp[:number_points]
                    exp_dict["y_metric_grad_comp"] = y_metric_grad_comp[:number_points]
            
        else:
            print("One of the files is not found: %s, %s"%(exp_dict["logs_path_x"], exp_dict["logs_path_y"]))
            #print("Loading status:")
            #print(exp_dict["logs_path_x"], ": ", os.path.isfile(exp_dict["logs_path_x"]))
            #print(exp_dict["logs_path_y"], ": ", os.path.isfile(exp_dict["logs_path_y"]))

            exp_dict["x_metric"] = np.array([-1]) #emplhasising the error
            exp_dict["y_metric"] = np.array([-1])
                
        if self.arg_values["show_std"]:    
            if os.path.isfile(exp_dict["logs_path_y_std"]) is not None:
                y_metric_std = fix_shape(load_np_array(exp_dict["logs_path_y_std"]))
                if number_points is not None:              
                    exp_dict["y_metric_std"] = y_metric_std[:number_points]
                else:
                    print ("number_points is None: x_metric or y_metric are not loaded")
            else:
                print(exp_dict["logs_path_y_std"], " is not found")
                exp_dict["y_metric_std"] = np.array([-1])
        else:
            exp_dict["y_metric_std"] = np.array([-1])
                
    #Project dependend function
    def init_colors_and_markers(self):
        self.color_ar = ['red','orange', 'aqua', 'violet','cornflowerblue','coral', 'lime','goldenrod', 'maroon',
                                'black', 'brown', 'yellowgreen', "purple", "violet", "magenta", "green","chocolate","crimson",'blue', 'darkgreen','darkorange']
        all_colors = mcolors.CSS4_COLORS
        remaining_colors = list(set(all_colors.keys()) - set(self.color_ar))
        self.color_ar.extend(remaining_colors)
        self.marker_ar = ["*", "v", ">", "s", "p", "P", "h", "H", "+", "x", "X", "D", "d", "|", "_",1,2,3,4,5,6,7,8,9,"^", "<", "o"]
        all_markers = list(mmarkers.MarkerStyle.markers.keys())
        remaining_markers = list(set(all_markers) - set(self.marker_ar))
        self.marker_ar.extend(remaining_markers)

    #Project depended function
    def prepare_one_exp_data_dict(self, exp_data_dict):
        exp_data_dict["exp_name"] = self.arg_values['exp_name'] 
        exp_data_dict["exp_str"] = self.arg_values['exp_name'] + self.exp_name_extension
        exp_data_dict["logs_path"] = self.project_path + self.arg_values['logs_folder'] +"logs_{0}/".format(exp_data_dict["exp_str"])
        
        exp_data_dict["step_size_init"] = self.arg_values["step_size_init"]
        
        if self.arg_values["plot_family"] == "ALL":
            #TODO: exp_data_dict or arg_values
            if exp_data_dict["exp_name"] == "GD":
                exp_data_dict["label"] = f"{exp_data_dict['exp_name']}_f{self.arg_values['factor']}"
            elif exp_data_dict["exp_name"] in ["RAC-LoRA_A", "RAC-LoRA_B", 'stoch_RAC-LoRA_A-SGD', 'stoch_RAC-LoRA_B-SGD', 'finite_RAC-LoRA_A-SGD', 'finite_RAC-LoRA_B-SGD']:
                exp_data_dict["label"] = f"{exp_data_dict['exp_name']}_f{self.arg_values['factor']}"
            elif exp_data_dict["exp_name"] in ["RC-LoRA", 'stoch_RC-LoRA-SGD', 'stoch_RC-LoRA-MVR', 'finite_RC-LoRA-SGD', 'finite_RC-LoRA-PAGE']:
                exp_data_dict["label"] = f"{exp_data_dict['exp_name']}_p{self.arg_values['prob']}_f{self.arg_values['factor']}"
            else:
                raise NotImplementedError("Wrong method name")
                exp_data_dict["label"] = exp_data_dict['exp_name']    
            
            exp_data_dict["marker"] = self.marker_ar[self.it]
            exp_data_dict["color"] = self.color_ar[self.it]
            exp_data_dict["linestyle"] = "solid"
            
        elif self.arg_values["plot_family"] == "SINGLE_RELEASE":
            exp_data_dict["marker"] = self.arg_values["marker"]
            exp_data_dict["color"] = self.arg_values["color"]
            exp_data_dict["label"] = self.arg_values["label"]
            exp_data_dict["linestyle"] = self.arg_values["linestyle"]
            if exp_data_dict["exp_name"] == "RAC-LoRA_A":
                exp_data_dict["label"] = f"RAC-LoRA-GD(A)"
                
            elif exp_data_dict["exp_name"] in ['stoch_RAC-LoRA_A-SGD', 'finite_RAC-LoRA_A-SGD']:
                exp_data_dict["label"] = f"RAC-LoRA-SGD(A)"
                
            elif exp_data_dict["exp_name"] == "RAC-LoRA_B":
                exp_data_dict["label"] = f"RAC-LoRA-GD(B)"
                
            elif exp_data_dict["exp_name"] in ['stoch_RAC-LoRA_B-SGD', 'finite_RAC-LoRA_B-SGD']:
                exp_data_dict["label"] = f"RAC-LoRA-SGD(B)"
                
            elif exp_data_dict["exp_name"] in ["RC-LoRA", 'stoch_RC-LoRA-SGD', 'stoch_RC-LoRA-MVR', 'finite_RC-LoRA-SGD', 'finite_RC-LoRA-PAGE']:
                if exp_data_dict["exp_name"] == "RC-LoRA":
                    exp_data_dict["label"] = f"Bernoulli-LoRA-GD(p={self.arg_values['prob']})"
                    
                elif exp_data_dict["exp_name"] in ['stoch_RC-LoRA-SGD', 'finite_RC-LoRA-SGD']:
                    exp_data_dict["label"] = f"Bernoulli-LoRA-SGD(p={self.arg_values['prob']})"
                    
                elif exp_data_dict["exp_name"] == "stoch_RC-LoRA-MVR":
                    exp_data_dict["label"] = f"Bernoulli-LoRA-MVR(p={self.arg_values['prob']})"
                
                elif exp_data_dict["exp_name"] == "finite_RC-LoRA-PAGE":
                    exp_data_dict["label"] = f"Bernoulli-LoRA-PAGE(p={self.arg_values['prob']})"
                
                exp_data_dict["marker"] = self.marker_ar[self.it]
                exp_data_dict["color"] = self.color_ar[self.it]
            else:
                raise NotImplementedError("Wrong method name")
            
        
        exp_data_dict["show_label"] = 1
        exp_data_dict["logs_path_x"] = exp_data_dict["logs_path"] + self.arg_values["x_metric_str"] + '_' + exp_data_dict["exp_str"] + ".npy"
        exp_data_dict["logs_path_y"] = exp_data_dict["logs_path"] + self.arg_values["y_metric_str"] + "_" + self.arg_values["y_avg_option"] + '_' + exp_data_dict["exp_str"] + '.npy'

        if self.arg_values["show_std"]:
            exp_data_dict["logs_path_y_std"] = exp_data_dict["logs_path"] + self.arg_values["y_metric_str"] + "_std" + '_' + exp_data_dict["exp_str"] + '.npy'
        
        if self.arg_values["show_grad_comp"]:
            exp_data_dict["logs_path_x_grad_comp"] = exp_data_dict["logs_path"] + self.arg_values["x_metric_str"] + "_grad_comp" + '_' + exp_data_dict["exp_str"] + ".npy"
            exp_data_dict["logs_path_y_grad_comp"] = exp_data_dict["logs_path"] + self.arg_values["y_metric_str"] + "_grad_comp" + "_" + self.arg_values["y_avg_option"] + '_' + exp_data_dict["exp_str"] + '.npy'
            exp_data_dict["marker_grad_comp"] = 'D'
            exp_data_dict["color_grad_comp"] = 'gold'
            exp_data_dict["marker_size_grad_comp"] = self.arg_values["marker_size"] - 5
            if exp_data_dict["exp_name"]== "AuxPAGE":    
                exp_data_dict["label_grad_comp"] = "Full grad computation"
            else:
                exp_data_dict["label_grad_comp"] = None

        self.it += 1
        
        if self.arg_values["show_all_trajectories"]:
            # Upload all the trajectories
            for i in range(self.NUM_LAUNCHES):
                self.exp_data_dict[self.arg_values['exp_name']].append({})
                exp_data_dict["exp_str"] = self.arg_values['exp_name'] + self.exp_name_extension
                exp_data_dict["logs_path"] = self.project_path + "logs/logs_{1}/".format(exp_data_dict["exp_str"])
                exp_data_dict["label"] = None
                exp_data_dict["show_label"] = 0
                exp_data_dict["logs_path_x"] = exp_data_dict["logs_path"] + self.arg_values["x_metric_str"] + '_' + exp_data_dict["exp_str"] + ".npy"
                exp_data_dict["logs_path_y"] = exp_data_dict["logs_path"] + self.arg_values["y_metric_str"] + f"_{i}" + '_' + exp_data_dict["exp_str"] + '.npy'
                exp_data_dict["logs_path_y_std"] = None # We don't draw std around mean each launch
                exp_data_dict["marker"] = None
                exp_data_dict["color"] = self.color_ar[self.it]
                if self.show_noise_level:
                    raise NotImplementedError("show_noise_level is not implemented at this point, see MSPPM project for implementation example") 
                self.it += 1
        
    #Project dependend function
    def plt_settings(self):
        plt.rcParams.update({
        'lines.linewidth': 2,
        'xtick.labelsize': self.arg_values["label_fontsize"],
        'ytick.labelsize': self.arg_values["label_fontsize"],
        'legend.fontsize': self.arg_values["label_fontsize"],
        'axes.titlesize': self.arg_values["label_fontsize"],
        'axes.labelsize': self.arg_values["label_fontsize"],
        'figure.figsize': [10, 8],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'text.usetex': True,
        'font.family': 'serif',
        })
        self.ax.set_xscale(self.arg_values['xscale'])
        self.ax.set_yscale(self.arg_values['yscale'])
        
        self.ax.grid()
        if self.arg_values["plot_family"] == "SINGLE_RELEASE":
            # shift = max (0.05*(self.arg_values["ub_x_axis"] - self.x_left_lim), 0)
            # self.ax.set_xlim(left=self.x_left_lim-shift, right=self.arg_values["ub_x_axis"])
            
            if self.arg_values["fixed_x_ticks"] is not None:
                # Define custom tick positions and labels
                if self.arg_values["x_metric_str"] == "epochs":
                    xticks = self.arg_values["fixed_x_ticks"]
                    # xticks = [1.00000, 1.0002, 1.0004, 1.0006, 1.0008, 1.001]
                    self.ax.xaxis.set_major_locator(FixedLocator(xticks))
                    self.ax.xaxis.set_major_formatter(FixedFormatter([f"{tick:.5f}" for tick in xticks]))
                else:
                    raise NotImplementedError("fixed_x_ticks is not implemented for x_metric_str other than 'epochs'")
        
        
        self.ax.locator_params(axis='x', nbins=8)
        
        # handles, labels = [], []
        # h, l = self.ax.get_legend_handles_labels()
        # # handles.extend(h)
        # # labels.extend(l)
        # if not self.arg_values["common_legend"]:
        #     self.ax.legend(loc='lower right')
        # plt.tight_layout(pad=1.0, h_pad=1.0, w_pad=1.0)
        # self.fig.subplots_adjust(wspace=0.2, hspace=0.2)  # Adjust these values as needed
        # if self.arg_values["common_legend"]:
        #     by_label = dict(zip(labels, handles))
        #     self.fig.legend(by_label.values(), by_label.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(by_label))
        
        #self.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs='all'))
        #self.ax.yaxis.set_major_formatter(NullFormatter())  # Hide major tick labels

        #self.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='all'))
        #self.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='all'))
        
        #self.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='all', numticks=7))
        
        # Set major ticks manually
        tick_locs = [10**exp for exp in [3, 0, -3, -6, -9, -12, -15]]  # 10^5 to 10^-16
        self.ax.yaxis.set_major_locator(FixedLocator(tick_locs))
        self.ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'$10^{{{int(np.log10(val))}}}$'))
        
        #self.ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f'{x:.0e}' if x in [10**i for i in range(-6, 1)] else ''))

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)
            
    #Project dependend function
    def plot_one_exp_dict(self, ax, exp_dict):
        exp_dict["x_metric"], exp_dict["y_metric"] = shapes_alignment(exp_dict["x_metric"].copy(), exp_dict["y_metric"].copy())
        if exp_dict["label"] is not None: #draw the line corresponding to the method (is usully has the legend) 
            # markers_on = inds[inds % (int(len(inds[:-(1 + 2 * self.it)]) / 5)) == 0].astype(int)
            
            #TODO: made a dependence on "ub_x_axis" 
            
            inds1 = np.arange (start=0, stop=exp_dict["x_metric"].shape[0],step=1)
            inds = np.arange (start=0, stop=exp_dict["x_metric"][exp_dict["x_metric"] <= self.arg_values["ub_x_axis"]].shape[0],step=1)
            num_inds = len(inds)
            marker_frequency = max(1, num_inds // 10)
            shift = int(marker_frequency/len(self.exp_data_dict.keys()))*self.it
            # inds = np.arange (start=0, stop=exp_dict["x_metric"].shape[0],step=marker_frequency)    
            # offset = (self.it+5) % marker_frequency
            markers_on = inds[shift::marker_frequency]
            
            ax.plot(exp_dict["x_metric"][:num_inds], exp_dict["y_metric"][:num_inds], label=exp_dict["label"], color=exp_dict["color"], 
                        marker=exp_dict["marker"], markevery=list(markers_on), markersize=self.arg_values["marker_size"], markerfacecolor=exp_dict["color"], 
                        markeredgecolor = 'black',
                        linestyle=exp_dict["linestyle"])
            if self.arg_values["show_noise_level"]:
                ax.axhline(y=exp_dict["noise_level"], color=exp_dict["color"], linestyle='--')
            if "x_metric_grad_comp" in exp_dict.keys():
                ax.plot(exp_dict["x_metric_grad_comp"], exp_dict["y_metric_grad_comp"], label=exp_dict["label_grad_comp"], color=exp_dict["color_grad_comp"], 
                        marker=exp_dict["marker_grad_comp"], markersize=exp_dict["marker_size_grad_comp"], markerfacecolor=exp_dict["color_grad_comp"], 
                        markeredgecolor = 'black',
                        linestyle='')
            
        
        if exp_dict["label"] is None and self.arg_values["show_all_trajectories"]:
            ax.plot(exp_dict["x_metric"], exp_dict["y_metric"], 'r', color=exp_dict["color"], alpha=0.5)
            
        if self.arg_values["show_std"] and exp_dict["logs_path_y_std"]:
            if not np.array_equal(exp_dict["y_metric_std"] , np.array([-1])): # if std_ar is provided
                # Ensure all values are positive and non-zerost
                exp_dict["y_metric"] = np.maximum(exp_dict["y_metric"], 1e-10)
                exp_dict["y_metric_std"] = np.maximum(exp_dict["y_metric_std"], 1e-10)
                upper_bound = exp_dict["y_metric"] + exp_dict["y_metric_std"]
                lower_bound = exp_dict["y_metric"] - exp_dict["y_metric_std"]
                lower_bound = np.maximum(lower_bound, 1e-10)
                ax.fill_between(exp_dict["x_metric"], lower_bound, upper_bound, color=exp_dict["color"], alpha=0.2)
            else:
                my_print("std_ar is not provided", self.arg_values["print_status"])            
    
    #Project dependend function
    def init_plot_title(self):
        if self.arg_values["plot_family"] == "SINGLE_RELEASE":
            pass  
            #some legacy lines of code
            #dim_latex = {10:r"$10$", 100:r"$100$", 1000:r"$1000$"}[self.arg_values['dim']]
            #num_workers_latex = {10:r"$10$", 100:r"$100$",1000:r"$1000$"}[self.arg_values['num_workers']]
            #noise_scale_latex = {0.01:r"$0.01$", 0.1:r"$0.1$", 1.0:r"$1$", 10.0:r"$10$", 100.0:r"$100$"}[self.arg_values['noise_scale']]
            #self.sup_title = f"loss: {self.arg_values['loss_func']};    dataset: synthetic;    d= {dim_latex};   n= {num_workers_latex};    s= {noise_scale_latex}"
            #self.sup_title = f"rank: {self.arg_values['rank']}"
            
        if self.arg_values["plot_family"] == "ALL":
            self.sup_title = f"rank: {self.arg_values['rank']}"

        if self.arg_values["show_title"]:
            self.fig.suptitle(self.sup_title, fontsize=self.arg_values["label_fontsize"])
    
    #Project dependend function
    def draw_plots(self):
        if not self.arg_values["grid"]:
            self.plot_single()
        else:
            self.plot_grid()
    
    #Project dependend function
    def save_plot(self):      
        if self.arg_values["save_plot"]:
            if not os.path.exists(self.plot_path):
                os.mkdir(self.plot_path)
            #self.file_title = self.arg_values["plot_file_title"]
            if self.arg_values["plot_family"] == "SINGLE_RELEASE":
                self.file_title =  f"{self.arg_values['plot_family']}_x-{self.arg_values['x_metric_str']}_y-{self.arg_values['y_metric_str']}_"+\
                                self.arg_values['exp_name'] + self.exp_params_extension +f"_r{self.arg_values['rank']}.pdf"
            else:
                raise NotImplementedError("save_plot is not implemented for plot_family other than 'SINGLE_RELEASE'")
            # self.file_title = f"{self.arg_values['plot_family']};\
            #                     x_label:{self.arg_values['x_metric_str']};\
            #                     y_label:{self.arg_values['y_metric_str']};\
            #                     loss:{self.arg_values['loss_func']};\
            #                     dataset:{self.arg_values['dataset']};\
            #                     d:"+ myrepr(self.arg_values['dim']) + ";nw="".pdf"
            self.fig.savefig(self.plot_path + self.file_title, bbox_inches='tight')
    
    #######################################   
    ### Functions to draw a single plot ###
    #######################################
    def draw_single_plot(self):
        # draw multiple curves (defined by self.exp_data_dict) on a single plot
        if self.arg_values["show_plot"]:
            self.init_plot_title()
            self.it = 0
            x_left_lim_ar = []
            for exp_id in self.exp_data_dict.keys():
                #self.it = len(self.exp_data_dict[exp_name])-1
                self.plot_one_exp_dict(ax=self.ax, exp_dict=self.exp_data_dict[exp_id])
                x_left_lim_ar.append(self.exp_data_dict[exp_id]["x_metric"][0])
                self.it =+1
            self.x_left_lim = min(x_left_lim_ar)
            self.plt_settings()
            if not self.arg_values["common_legend"]:
                #self.ax.legend(loc='upper right', fontsize=self.arg_values["label_fontsize"] - 4, frameon=True)
                self.ax.legend(loc='center right', fontsize=self.arg_values["label_fontsize"] - 4, frameon=True)
            plt.tight_layout()
            plt.show()
            self.save_plot()
    
    def inits_and_load_for_single_plot(self):
        self.init_exp_data_extension()
        self.init_w_init_extension()
        self.init_dataset_path()
        self.init_exp_param_extension()
        self.init_alg_params_dict()
        self.load_parameters()
        
    #Project dependend functions
    def plot_single(self):
        for rank in self.arg_values["rank_ar"]:
            fig, ax = plt.subplots(figsize=(15, 10), squeeze=False)
            self.fig = fig
            self.ax = ax[0, 0]
            self.init_colors_and_markers() 
            self.exp_data_dict = {}
            self.it = 0
            self.arg_values['rank'] = rank
            for j in range(len(self.arg_values["exps"])):
                self.arg_values.update(self.arg_values["exp_dicts"][self.arg_values["exps"][j]])
                self.inits_and_load_for_single_plot()
                self.exp_dict_id = None
                if self.arg_values["exp_name"] == "GD":
                    if self.arg_values["factor_dict"] is not None:
                        self.arg_values["factor_ar"] = self.arg_values["factor_dict"]
                    for factor in self.arg_values["factor_ar"]:
                        self.arg_values['factor'] = factor
                        self.exp_dict_id = create_str_id([self.arg_values["exp_name"], self.arg_values["factor"]])
                        self.exp_data_dict[self.exp_dict_id]={}
                        self.init_exp_name_extension()
                        self.prepare_one_exp_data_dict(self.exp_data_dict[self.exp_dict_id])
                    continue
                elif self.arg_values["exp_name"] in ["RAC-LoRA_A", "RAC-LoRA_B", 'stoch_RAC-LoRA_A-SGD', 'stoch_RAC-LoRA_B-SGD', 'finite_RAC-LoRA_A-SGD', 'finite_RAC-LoRA_B-SGD']:
                    if self.arg_values["factor_dict"] is not None:
                        self.arg_values["factor_ar"] = self.arg_values["factor_dict"][self.arg_values['rank']]
                    for factor in self.arg_values["factor_ar"]:
                        self.arg_values['factor'] = factor
                        self.exp_dict_id = create_str_id([self.arg_values["exp_name"], self.arg_values['rank'], self.arg_values["factor"]])
                        self.exp_data_dict[self.exp_dict_id]={}
                        self.init_exp_name_extension()
                        self.prepare_one_exp_data_dict(self.exp_data_dict[self.exp_dict_id])
                    continue
                else:
                    assert self.arg_values["exp_name"] in ["RC-LoRA", 'stoch_RC-LoRA-SGD', 'stoch_RC-LoRA-MVR', 'finite_RC-LoRA-SGD', 'finite_RC-LoRA-PAGE']
                    if self.arg_values["prob_dict"] is not None:
                        self.arg_values["prob_ar"] = self.arg_values["prob_dict"][self.arg_values['rank']]
                    for prob in self.arg_values["prob_ar"]:
                        self.arg_values["prob"] = prob
                        if self.arg_values["factor_dict"] is not None:
                            self.arg_values["factor_ar"] = self.arg_values["factor_dict"][self.arg_values['rank']][self.arg_values["prob"]]
                        for factor in self.arg_values["factor_ar"]:
                            self.arg_values['factor'] = factor
                            self.exp_dict_id = create_str_id([self.arg_values["exp_name"], self.arg_values['rank'], self.arg_values["prob"], self.arg_values["factor"]])
                            self.exp_data_dict[self.exp_dict_id]={}
                            self.init_exp_name_extension()
                            self.prepare_one_exp_data_dict(self.exp_data_dict[self.exp_dict_id])   
                assert self.exp_dict_id is not None, "exp_dict_id is not defined"
                
            for exp_id in self.exp_data_dict.keys():
                self.load_one_exp_dict_logs(self.exp_data_dict[exp_id])
            self.draw_single_plot()
    
    #Project dependend function
    def plot_grid(self):
        # Determine the number of rows and columns
        nrows = len(self.arg_values['num_workers_ar'])
        ncols = len(self.arg_values['noise_scale_ar'])
        # Create a grid of subplots with shared y-axis per row
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False, sharey='row')
        self.fig = fig
        self.init_colors_and_markers()
        # Iterate over num_workers and noise_scale combinations
        for i, num_workers in enumerate(self.arg_values['num_workers_ar']):
            self.arg_values['ub_x_axis'] = self.arg_values['ub_x_axis_per_n'][num_workers]
            for j, noise_scale in enumerate(self.arg_values['noise_scale_ar']):
                self.ax = axes[i, j]
                self.exp_data_dict = {}
                self.it = 0
                self.arg_values['num_workers'] = num_workers 
                self.arg_values['noise_scale'] = noise_scale
                self.arg_values['prob'] = 1/self.arg_values['num_workers'] # according to our specific setting
                
                
                
                # Prepare the latex strings for the subplot title
                num_workers_latex = f"${num_workers}$"
                noise_scale_latex = {0.01:r"$0.01$", 0.1:r"$0.1$", 1.0:r"$1$", 10.0:r"$10$", 100.0:r"$100$"}.get(noise_scale, f"${noise_scale}$")
                # This loop is project-dependent  
                for k in range(len(self.arg_values["exps"])):
                    self.arg_values.update(self.arg_values["exp_dicts"][self.arg_values["exps"][k]])
                    self.inits_and_load_for_single_plot()
                    
                    if self.arg_values["factor_dict"] is not None:
                        self.arg_values["factor_ar"] = self.arg_values["factor_dict"][self.arg_values["step_size_init"]][num_workers][noise_scale]
                    
                    for factor in self.arg_values["factor_ar"]:
                        self.arg_values['factor'] = factor
                        self.exp_dict_id = create_str_id([
                            self.arg_values['exp_name'], self.arg_values['num_workers'], self.arg_values['noise_scale'],
                            self.arg_values['prob'], self.arg_values['factor'], self.arg_values["compressor"], self.arg_values["step_size_init"]
                        ])
                        self.init_exp_name_extension()
                        self.exp_data_dict[self.exp_dict_id] = {}
                        self.prepare_one_exp_data_dict(self.exp_data_dict[self.exp_dict_id])
                for exp_id in self.exp_data_dict.keys():
                    self.load_one_exp_dict_logs(self.exp_data_dict[exp_id])
                # Plot the data on the subplot axis
                x_left_lim_ar = []
                self.it = 0
                for exp_id in self.exp_data_dict.keys():
                    self.plot_one_exp_dict(ax=self.ax, exp_dict=self.exp_data_dict[exp_id])
                    x_left_lim_ar.append(self.exp_data_dict[exp_id]["x_metric"][0])
                    self.it += 1
                if x_left_lim_ar:
                    self.x_left_lim = min(x_left_lim_ar)
                else:
                    self.x_left_lim = 0
                # Apply settings to the subplot axis
                self.plt_settings()
                # Set y-axis to log scale with ticks at powers of ten
                self.ax.set_yscale('log')

                self.ax.yaxis.set_major_locator(LogLocator(base=10.0))
                self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: r'$10^{{{:.0f}}}$'.format(np.log10(y))))
                self.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))

                # Set the subplot title
                #self.ax.set_title(f"n= {num_workers_latex};    s= {noise_scale_latex};", fontsize=self.arg_values["label_fontsize"])
                self.ax.set_title(rf"n= {num_workers_latex};    $\sigma_A$= ${self.alg_params_dict['L_pm']:.2f}$", fontsize=self.arg_values["label_fontsize"])
                
                # Set x label only on bottom row
                if i == nrows - 1:
                    self.ax.set_xlabel(self.x_label)
                else:
                    self.ax.set_xlabel('')
                # Set y label only on leftmost column
                if j == 0:
                    self.ax.set_ylabel(self.y_label)
                else:
                    self.ax.set_ylabel('')
                # Adjust tick label sizes
                self.ax.tick_params(axis='both', which='major', labelsize=self.arg_values["label_fontsize"] - 5)
                self.ax.tick_params(axis='both', which='minor', labelsize=self.arg_values["label_fontsize"] - 5)
                # Set legend location for this subplot if not using common legend
                if not self.arg_values["common_legend"]:
                    self.ax.legend(loc='upper right', fontsize=self.arg_values["label_fontsize"] - 4, frameon=False)
        # Set the overall figure title
        dim_latex = {10: r"$10$", 100: r"$100$", 1000: r"$1000$"}.get(self.arg_values['dim'], f"${self.arg_values['dim']}$")
        #stepsize = {"const": "constant", "polyak": "polyak", }.get(self.arg_values["step_size_init"], self.arg_values["step_size_init"])
        self.sup_title = f"loss: {self.arg_values['loss_func']};    dataset: synthetic;    d= {dim_latex}"
        if self.arg_values["show_title"]:
            self.fig.suptitle(self.sup_title, fontsize=self.arg_values["label_fontsize"])
        # Collect handles and labels for legend
        handles, labels = [], []
        for ax_row in axes:
            for ax in ax_row:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
        by_label = dict(zip(labels, handles))
        if self.arg_values["common_legend"]:
            # Determine the number of columns for the legend
            if self.arg_values["plot_family"] == "ALL":
                legend_columns = 3  # Adjust this number as needed
            else:
                legend_columns = 2
            self.fig.legend(
                by_label.values(), by_label.keys(),
                loc='lower center',
                bbox_to_anchor=(0.5, -0.20),  # Adjust y-coordinate to make space for the larger legend
                ncol=legend_columns,
                fontsize=self.arg_values["label_fontsize"] - 2,
                frameon=False
            )
        plt.tight_layout()
        # Adjust bottom margin to make space for the legend
        if self.arg_values["plot_family"] == "ALL":
            self.fig.subplots_adjust(bottom=0.3)  # Increase bottom margin as needed
        plt.show()
        self.save_plot()

        
