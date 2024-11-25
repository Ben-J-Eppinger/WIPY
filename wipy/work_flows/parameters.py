#########################
### solver parameters ###
#########################

solver: str = "specfem2d"       # options: specfem2d
material: str = "acoustic"      # options: acoustic, elastic
n_events: int = 26              
n_proc: int = -1                # -1 means that the minumum number of processors will be used  

#############################
### preprocess parameters ###
#############################

# filtering
# options: "bandpass", "lowpass", "highpass", or None
filter: str = "lowpass"
# freq_min: float = 1
freq_max: float = 2.0
filter_order: int = 10

# muting
# options: "mute_far_offsets", "mute_short_offsets", "mute_above_func", "mute_below_func". []
mute: list[str] = []
max_offset: float = 6000.0
min_offset: float = 500.0
mute_above_func = lambda offset: 1.0 + (9/13000)*offset
mute_below_func = lambda offset: 1.0 + (9/13000)*offset
t_taper: float = 0.5

# normalization
# options: "trace_normalize", "event_normalize", []
normalize = ["trace_normalize"]

############################
### inversion parameters ###
############################
# optimizatoin method
# options: 
#   GD (gradient descent)
#   LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
#   CG (Conjugate Gradient)
optimize: str = "GD"                         

max_iter: int = 100

# misfit function
# options: L2_norm, NC_norm, disperion, Wasserstein, GSOT, backproject
misfit: str = "WaveCo"
max_freq: float = 5.0
min_freq: float = 0.5
gamma: float = 5.0
additional_misfit_parameters: list[str] = [max_freq, min_freq, gamma]                     

smooth_h: float = 100.0
smooth_v: float = 100.0

precond: str =  "approx_hessian"    # options: None, approx_hessian, from_file ***note gradients are devided by the precond in both cases

invert_params: list[str] = ["vp"]    # options: vp, vs, rho
invert_params_weights: dict = {"vp": 1.0}

# bounds fro listed parameters
vp_bounds: list[float] = [1000.0, 4700.0]
vs_bounds: list[float] = []
rho_bounds: list[float] = [0.9, 1.1]
vp_vs_ratio_bounds: list[float] = []
scale_vs_from_vp: bool = True

# max/min update (e.g., 0.1 means that the max/min update is 10% of the model being updated)
max_update: float = 0.1
min_update: float = 0.01

#######################
### misc parameters ###
#######################
# options: ["x", "z"] for elastic material, ["p"] for acoustic material 
components: list[str] = ["p"]       
save_traces: bool = False 
# options "d": displacement, "v": velocity, "a": acceleration: "p": preasure
seismotype: str = "p"
save_traces: bool = False 

