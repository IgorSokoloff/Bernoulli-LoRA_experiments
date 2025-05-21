from src.plotting import *

from matplotlib import pyplot as plt    

class ColorBar(Plotter):
    def __init__(self, general_settings_dict, plot_settings_dict, colorbar_settings_dict):
        super().__init__(general_settings_dict, plot_settings_dict)
        self.arg_values.update(colorbar_settings_dict)
        
    def draw_colorbars(self):
        for setting in self.arg_values["settings"]:
            self.arg_values["setting"] = setting
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            self.fig = fig
            self.ax1 = ax1
            self.ax2 = ax2
            
            if self.arg_values["setting"] == "unbiased":
                self.load_unbiased_colorbar_data()
                self.fix_unbiased_colorbar_bug()
            elif self.arg_values["setting"] == "biased":
                self.load_biased_colorbar_data()

            self.plot_colorbar_pair()
            self.plot_methods_dots()
            
            if self.arg_values["show_colorbar"]:
                plt.show()
        
        if self.arg_values["save"]:
            filename = f"colorbars_test/contourf-{self.arg_values['dataset']}_{self.arg_values['setting']}"
            self.fig.savefig(filename + ".pdf", bbox_inches='tight')
        return self.c_vals, self.k_vals, self.Z_cost, self.Z_p_opt 
    
    def load_unbiased_colorbar_data(self):
        self.min_cost_level = 5*10**(-7)
        self.min_kappa_level = 10**(-6)
        
        num_levels = 200
        c_levels = np.zeros(num_levels+1, dtype=np.float64)
        k_levels = np.zeros(num_levels+1, dtype=np.float64)
        c_levels[0] = self.min_cost_level
        k_levels[0] = self.min_kappa_level
        c_levels[1:num_levels+1] = np.logspace(start=-20, stop=0, num=num_levels, endpoint=True, base=2, dtype=np.float64)
        k_levels[1:num_levels+1] = np.concatenate((np.logspace(start=-19, stop=-5, num=int(num_levels/2), endpoint=False, base=2, dtype=np.float64),
                        np.logspace(start=-5, stop=10, num=int(num_levels/2), endpoint=True, base=2, dtype=np.float64) ))

        self.c_vals = c_levels.copy()
        self.k_vals = k_levels.copy()
        
        self.Z_cost = np.load(f'colorbars_data/Z_cost_unb_exponential_nl-{num_levels}.npy')
        self.Z_p_opt = np.load(f'colorbars_data/Z_p_opt_unb_exponential_nl-{num_levels}.npy')
        
        self.axis_x_locator = tck.FixedLocator(np.logspace(-9, 3, num=13, base=10))
        self.axis_y_locator = tck.FixedLocator(np.logspace(-9, 3, num=13, base=10))
        self.cost_colorbar_locator = tck.FixedLocator(np.logspace(-7, 0, num=8, base=10))
        self.p_colorbar_locator = tck.FixedLocator(np.logspace(-9, 3, num=13, base=10))
        
        
    def load_biased_colorbar_data(self):
        num_c_levels,num_kappa_levels = 21, 27
        base = 2
        self.min_cost_level, self.min_kappa_level = 5 * 10**(-7), 10**(-8)
        
        c_levels = np.array([base**(-i) for i in range(num_c_levels)] + [self.min_cost_level], dtype=np.float64)
        kappa_levels = np.array([base**(-i) for i in range(num_kappa_levels)] + [self.min_kappa_level], dtype=np.float64)
        self.c_vals = c_levels.copy()
        self.k_vals = kappa_levels.copy()
        self.Z_cost = np.load(f'colorbars_data/Z_cost_b_exponential_ncl-{num_c_levels}_nkl-{num_kappa_levels}_b-{base}.npy')
        self.Z_p_opt = np.load(f'colorbars_data/Z_p_opt_b_exponential_ncl-{num_c_levels}_nkl-{num_kappa_levels}_b-{base}.npy')
        #filename = f"colorbars_test/contourf-{dataset}_{self.arg_values["setting"]}_ncl-{num_c_levels}_nkl-{num_kappa_levels}_b-{base}"
        self.axis_x_locator = tck.FixedLocator(np.logspace(-7, 0, num=8, base=10))
        self.axis_y_locator = tck.FixedLocator(np.logspace(-8, 0, num=9, base=10))
        self.cost_colorbar_locator = tck.FixedLocator(np.logspace(-7, 0, num=8, base=10))
        self.p_colorbar_locator = tck.FixedLocator(np.logspace(-7, 0, num=8, base=10))
    
    def colorbar_settings(self):
        ax1_title = {"unbiased": r'$\operatorname{min}_p \left\{ \Phi_u(p)=(p + (1-p)c)\left( 1 + \kappa \sqrt{\frac{1-p}{p}} \right) \right\}$',
                    "biased": r'$\operatorname{min}_p \left\{ \Phi(p)=(p+(1-p) c)\left(1+\left(\frac{1+\sqrt{1-p}}{p}-1\right) \kappa\right) \right\}$'
                }[self.arg_values["setting"]]
        ax2_title = {"unbiased": r'$\operatorname{argmin}_p \left\{ \Phi_u(p)=(p + (1-p)c)\left( 1 + \kappa \sqrt{\frac{1-p}{p}} \right) \right\}$',
                    "biased": r'$\operatorname{argmin}_p \left\{ \Phi(p)=(p+(1-p) c)\left(1+\left(\frac{1+\sqrt{1-p}}{p}-1\right) \kappa\right) \right\}$'
                }[self.arg_values["setting"]]
        #ax2_title = r'$p_{opt}$'
        self.ax1.set_title(ax1_title, pad=self.arg_values["title_pad"])
        self.ax1.set_xlabel(r'$c$', fontsize=self.arg_values["size"])
        self.ax1.set_ylabel(r'$\kappa$', fontsize=self.arg_values["size"])
        self.ax2.set_title(ax2_title, pad=self.arg_values["title_pad"])
        self.ax2.set_xlabel(r'$c$', fontsize=self.arg_values["size"])
        self.ax2.set_ylabel(r'$\kappa$', fontsize=self.arg_values["size"])
        
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax2.set_xscale('log')
        self.ax2.set_yscale('log')
        
        self.ax1.xaxis.set_major_locator(self.axis_x_locator)
        self.ax1.yaxis.set_major_locator(self.axis_y_locator)
        self.ax2.xaxis.set_major_locator(self.axis_x_locator)
        self.ax2.yaxis.set_major_locator(self.axis_y_locator)
        
        self.fig.canvas.header_visible = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'FreeSerif'
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = self.arg_values["marker_size"]
        plt.rcParams['xtick.labelsize'] = self.arg_values["size"]
        plt.rcParams['ytick.labelsize'] = self.arg_values["size"]
        plt.rcParams['legend.fontsize'] = self.arg_values["size"]
        plt.rcParams['axes.titlesize'] = self.arg_values["size"]+2
        plt.rcParams['axes.labelsize'] = self.arg_values["size"]+2
        
        #plt.rcParams["figure.figsize"] = [13, 9]
        
        self.formatter = LogFormatterSciNotation(base=10, labelOnlyBase=True)
        plt.colorbar(self.c1, ax=self.ax1, ticks=self.cost_colorbar_locator, format=self.formatter)
        plt.colorbar(self.c2, ax=self.ax2, ticks=self.p_colorbar_locator, format=self.formatter)
        
    def plot_colorbar_pair(self):
        exponent_p = np.floor(np.log10(np.min(self.Z_p_opt)))
        exponent_cost = np.floor(np.log10(np.min(self.Z_cost)))
        result_min_p = 10 ** exponent_p
        result_min_cost = 10 ** exponent_cost
        levels_ranges_p_opt = [result_min_p, 1.0]
        levels_ranges_cost =  [result_min_cost, 1.0]
        levels_cost = np.logspace(np.log10(levels_ranges_cost[0]), np.log10(levels_ranges_cost[1]), self.arg_values["smooth_n_levels"])
        levels_p_opt = np.logspace(np.log10(levels_ranges_p_opt[0]), np.log10(levels_ranges_p_opt[1]), self.arg_values["smooth_n_levels"])
        self.c1 = self.ax1.contourf(self.c_vals, self.k_vals, self.Z_cost, cmap="jet", levels=levels_cost, norm=mcolors.LogNorm(vmin=levels_ranges_cost[0], vmax=levels_ranges_cost[1]))
        self.c2 = self.ax2.contourf(self.c_vals, self.k_vals, self.Z_p_opt, cmap="jet", levels=levels_p_opt, norm=mcolors.LogNorm(vmin=levels_ranges_p_opt[0], vmax=levels_ranges_p_opt[1]))
        self.colorbar_settings()
        
    def plot_single_point_dict(self, ax1, ax2, point_dict):
        if point_dict["label"] is not None: #draw the line corresponding to the method (is usully has the legend) 
            for ax in [ax1, ax2]:  
                ax.plot(point_dict["x_metric"], point_dict["y_metric"], marker= point_dict["marker"], color=point_dict["color"], markersize=point_dict["markersize"], markeredgecolor=point_dict["markeredgecolor"], zorder=3)
                if point_dict["show_label"]:
                    text = ax.text(point_dict["x_metric"]*point_dict["x_offset"], point_dict["y_metric"]*point_dict["y_offset"], point_dict["label"], fontsize=point_dict["label_fontsize"], verticalalignment=point_dict["verticalalignment"], horizontalalignment=point_dict["horizontalalignment"], color=point_dict["label_text_color"])              
                    text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])

    def prepare_single_points_dict(self, point_dict):
        point_dict["exp_name"] = self.arg_values['exp_name']         
        point_dict["show_label"] = 1
        point_dict["x_metric"] = np.array(self.alg_params_dict["cost_iter"])
        point_dict["y_metric"] = np.array(self.alg_params_dict["kappa"])
        # if self.arg_values['exp_name'] == "GD":
        #     point_dict["label"] = self.arg_values['exp_name']
        # else:
        #     point_dict["label"] = self.arg_values['exp_name'] + "-" + self.arg_values['sampling']
        point_dict["label"] = self.arg_values['exp_name']
        point_dict["label_fontsize"] = 20
        point_dict["label_text_color"] = "white"
        point_dict["marker"] = '*'
        point_dict["color"] = "black"
        point_dict["markeredgecolor"] = "white"
        point_dict["markersize"] = 20
        point_dict["verticalalignment"] = 'bottom'
        point_dict["horizontalalignment"] = 'left'
        point_dict["x_offset"] = self.arg_values["x_offset_ar"][self.j]
        point_dict["y_offset"] = self.arg_values["y_offset_ar"][self.j]
        
        self.it += 1
        
    #Project dependend functions
    def plot_methods_dots(self):
        self.init_colors_and_markers()
        
        self.points_dict = {}
        self.it = 0
        # this loop is project dependend  
        for j in range(len(self.arg_values["exps"])):
            self.j = j # we need to keep track the index of the current experiment
            self.arg_values['sampling'] = self.arg_values["sampling_ar"][j]
            self.arg_values['batchsize'] = self.arg_values["batchsize_ar"][j]
            self.arg_values['exp_name'] = self.arg_values["exps"][j]
            
            if self.arg_values['settings_ar'][j] != self.arg_values["setting"]:
                continue
            
            self.init_exp_data_extension()
            self.init_w_init_extension() 
            self.init_dataset_path()
            self.init_exp_param_extension()
            self.init_alg_params_dict()
            
            if self.arg_values['exp_name']=="GD":# we do not load them since I did not compute them explicitly beforehead
                self.alg_params_dict["cost_iter"] = 1
                self.alg_params_dict["kappa"] = self.min_kappa_level
            else:
                self.load_parameters()
                self.cost_str = self.extract_str_from_param("cost")
                self.kappa_str = self.extract_str_from_param("kappa")
                self.alg_params_dict["cost_iter"] = self.alg_params_dict[self.cost_str]
                self.alg_params_dict["kappa"] = self.alg_params_dict[self.kappa_str]
                
            self.point_dict_id = create_str_id([self.arg_values['exp_name'], self.arg_values['sampling'], self.arg_values['batchsize']])
            #self.points_dict[self.point_dict_id]=[]
            self.init_exp_name_extension()
            self.points_dict[self.point_dict_id]={}
            self.prepare_single_points_dict(self.points_dict[self.point_dict_id])
        
        if self.arg_values["show_points"]:
            self.it = 0
            for point_dict_id in self.points_dict.keys():
                #self.it = len(self.points_dict[exp_name])-1
                self.plot_single_point_dict(ax1=self.ax1, ax2=self.ax2, point_dict=self.points_dict[point_dict_id])
                # self.it =-1
                self.it =+1
    
    def fix_unbiased_colorbar_bug(self):
        c_vals_inds = np.argwhere((self.c_vals > 10**(-2)) & (self.c_vals < 5*10**(-2)))
        k_vals_inds = np.argwhere((self.k_vals > 5*10**(-6)) & (self.k_vals < 10**(-5)))

        # Create a grid of indices
        c_grid, k_grid = np.meshgrid(c_vals_inds, k_vals_inds)

        # Slice the Z_p_opt array using the grid of indices
        Z_p_opt_slice = self.Z_p_opt[k_grid, c_grid]

        target_idx = np.argwhere(Z_p_opt_slice > 0.9)

        if len(target_idx) != 1:
            raise ValueError("There should be exactly one element in Z_p_opt_slice greater than 0.9.")

        row_idx, col_idx = target_idx[0]

        neighboring_indices = [
            (row_idx - 1, col_idx), (row_idx + 1, col_idx),  # Vertical neighbors
            (row_idx, col_idx - 1), (row_idx, col_idx + 1),  # Horizontal neighbors
            (row_idx - 1, col_idx - 1), (row_idx - 1, col_idx + 1),  # Diagonal neighbors
            (row_idx + 1, col_idx - 1), (row_idx + 1, col_idx + 1)   # Diagonal neighbors
        ]

        # Filter out invalid neighbors (out of bounds)
        valid_neighbors = [
            (r, c) for r, c in neighboring_indices
            if 0 <= r < Z_p_opt_slice.shape[0] and 0 <= c < Z_p_opt_slice.shape[1]
        ]
        neighbor_values = [Z_p_opt_slice[r, c] for r, c in valid_neighbors]
        p_avg = np.mean(neighbor_values)

        # Retrieve the original indices in Z_p_opt
        original_c_idx = c_grid[0, col_idx]
        original_k_idx = k_grid[row_idx, 0]

        # Update the corresponding value in the original Z_p_opt array
        self.Z_p_opt[original_k_idx, original_c_idx] = p_avg
        
        outlier_c_val = self.c_vals[c_grid[0, col_idx]]
        outlier_k_val = self.k_vals[k_grid[row_idx, 0]]
        
        self.Z_cost[original_k_idx, original_c_idx] = cost_function_unbiased (outlier_c_val, outlier_k_val, p_avg)
        
        # #new upd
        # Z_prime = cost_prime_grid_unbiased(self.c_vals, self.k_vals).copy()
        
        # # self.Z_p_opt[sign_matrix(Z_prime)<0] = 1 #area for wchich p = 1 is optimal
        # # self.Z_cost[sign_matrix(Z_prime)<0] = 1
        
        # self.Z_p_opt[Z_prime< NUM_ZERO] = 1 #area for wchich p = 1 is optimal
        # self.Z_cost[Z_prime< NUM_ZERO] = 1
        