#########################
### solver parameters ###
#########################

solver: str = "specfem2d"       # options: specfem2d
material: str = "acoustic"      # options: acoustic, elastic
n_events: int = 40              
n_proc: int = -1                # -1 means that the maximum number of processors will be used  

#############################
### preprocess parameters ###
#############################

# filtering
# options: "bandpass", "lowpass", "highpass", or None
filter: str = "bandpass" 
freq_max: float = 2.5 
freq_min: float = 0.5 
filter_order: int = 10

# muting
# options: "mute_far_offsets", "mute_short_offsets", "mute_above_func", "mute_below_func". []
mute: list[str] = []
max_offset: float = 10000.0
min_offset: float = 1000.0
mute_above_func = lambda offset: 1.0 + (9/13000)*offset
mute_below_func = lambda offset: 1.0 + (9/13000)*offset
t_taper: float = 0.5

# normalization
# options: "trace_normalize", "event_normalize", []
normalize = ["trace_normalize"]
# normalize: list[str] = ["event_normalize"]

############################
### inversion parameters ###
############################
# optimizatoin method
# options: 
#   GD (gradient descent)
#   LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
#   CG (Conjugate Gradient)
optimize: str = "LBFGS"                         

max_iter: int = 1000

# misfit function
# options: L2_norm, NC_norm, disperion, Wasserstein, GSOT, WavePhase, WaveAmp, backproject
# misfit: str =  "WaveLog"
# max_freq: float = 1.0*freq_max
# min_freq: float = 1.0*freq_min
# additional_misfit_parameters: list[str] = [max_freq, min_freq]    
# misfit: str = "L2_norm" 
# additional_misfit_parameters: list[str] = []                   
# misfit: str = "ddWavePhase"
# min_freq: float = 1.0*freq_min
# max_freq: float = 1.0*freq_max
# eps = 15.0
# eta = 0.5
# max_dist = 1000.0
# additional_misfit_parameters: list[str] = [max_freq, min_freq, eps, eta, max_dist]   
misfit: str = "WavePhase"
min_freq: float = 1.0*freq_min
max_freq: float = 1.0*freq_max
eps = 15.0
eta = 0.5
max_dist = 1000.0
additional_misfit_parameters: list[str] = [max_freq, min_freq, eps, eta]   
smooth_v: float = 100.0 
smooth_h: float = 100.0 

#preconditioner
# options: None, approx_hessian, from_file 
precond: list[str] =  ["approx_hessian", "from_file"]    

invert_params: list[str] = ["vp"]    # options: vp, vs, rho
invert_params_weights: dict = {"vp": 1.0}

# bounds fro listed parameters
vp_bounds: list[float] = [1500.0, 4700.0]
vs_bounds: list[float] = []
rho_bounds: list[float] = [0.9, 1.1]
vp_vs_ratio_bounds: list[float] = []
scale_vs_from_vp: bool = True

# max/min update (e.g., 0.1 means that the max/min update is 10% of the model being updated)
max_update: float = 0.05
min_update: float = 0.01

#######################
### misc parameters ###
#######################
# options: ["x", "z"] for elastic material, ["p"] for acoustic material 
components: list[str] = ["p"]       
save_traces: bool = False 
# options "d": displacement, "v": velocity, "a": acceleration: "p": preasure, "x": potential
seismotype: str = "x"
save_traces: bool = False 

