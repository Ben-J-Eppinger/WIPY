#########################
### solver parameters ###
#########################

solver: str = "specfem2d"       # options: specfem2d
material: str = "elastic"      # options: acoustic, elastic
n_events: int = 10              
n_proc: int = -1                # -1 means that the miximum number of processors will be used  

#############################
### preprocess parameters ###
#############################

# filtering
# options: "bandpass", "lowpass", "highpass", or None
filter: str = "lowpass"    
# freq_min: float = 1
freq_max: float = 20
filter_order: int = 10

# muting
# options: "mute_far_offsets", "mute_short_offsets", "mute_above_func", "mute_below_func"
mute: list[str] = ["mute_short_offsets"]
max_offset: float = 180
min_offset: float = 5
mute_above_func = lambda offset: 0.2 + 0.00666*offset
mute_below_func = lambda offset: 0.0 + 0.00666*offset
t_taper: float = 0.01

# normalization
# options: "trace_normalize", "event_normalize"
normalize = ["trace_normalize", "event_normalize"]

############################
### inversion parameters ###
############################
# optimizatoin method
# options: 
#   GD (gradient descent)
#   LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
#   CG (Conjugate Gradient)
optimize: str = "CG"                        

max_iter: int = 50

# misfit function
# options: L2_norm, NC_norm
misfit: str = "NC_norm"                     
additional_misfit_parameters: list = []

smooth_v: float = 20.0
smooth_h: float = 20.0

precond: str =  "approx_hessian"    # options: None, approx_hessian, from_file ***note gradients are devided by the precond in both cases

invert_params: list[str] = ["vp", "vs"]    # options: vp, vs, rho
invert_params_weights: dict = {"vp": 1.0, "vs": 1.0}

# bounds fro listed parameters
vp_bounds: list[float] = [299, 450]
vs_bounds: list[float] = [150, 300]
rho_bounds: list[float] = [250, 300]
vp_vs_ratio_bounds: list[float] = [1.1, 3.0]
scale_vs_from_vp: bool = True

# max/min update (e.g., 0.1 means that the max/min update is 10% of the model being updated)
max_update: float = 0.2
min_update: float = 0.01

#######################
### misc parameters ###
#######################
# options: ["x", "z"] for elastic material, ["p"] for acoustic material 
components: list[str] = ["x", "z"]       
save_traces = True 

