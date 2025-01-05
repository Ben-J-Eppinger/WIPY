#########################
### solver parameters ###
#########################

solver: str = "specfem2d"       # options: specfem2d
material: str = "elastic"      # options: acoustic, elastic
n_events: int = 41              
n_proc: int = -1                # -1 means that the maximum number of processors will be used  

#############################
### preprocess parameters ###
#############################

# filterings
# options: "bandpass", "lowpass", "highpass", or None
filter: str = "bandpass"
freq_min: float = 5.0
freq_max: float = 65.0
filter_order: int = 10

# muting
# options: "mute_far_offsets", "mute_short_offsets", "mute_above_func", "mute_below_func", []
mute: list[str] = ["mute_far_offsets", "mute_short_offsets", "mute_above_func", "mute_below_func"]
max_offset: float = 150.0 #120.0   
min_offset: float = 30.0 #5.0 
mute_below_func = lambda offset: 0.125 + (0.22/200)*offset
mute_above_func = lambda offset: 0.075 + (0.05/200)*offset
t_taper: float = 0.01 #0.05

# normalization
# options: "trace_normalize", "event_normalize", []
normalize: list[str] = ["trace_normalize"]

############################
### inversion parameters ###
############################
# optimizatoin method
# options: 
#   GD (gradient descent)
#   LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)
#   CG (Conjugate Gradient)
optimize: str = "GD" 

max_iter: int = 250

# misfit function
# options: L2_norm, NC_norm, backproject, FATT, Wasserstein, dispersion, GSOT
# misfit: str = "backproject"
# additional_misfit_parameters: list = []
# misfit: str = "NC_norm"
# additional_misfit_parameters: list = []
# misfit: str = "WaveCorr"
# max_freq: float = 20.0
# min_freq: float = 5.0
# additional_misfit_parameters: list[str] = [max_freq, min_freq]  
misfit: str = "WavePhase"
min_freq: float = 5.0
max_freq: float = 65.0
additional_misfit_parameters: list[str] = [max_freq, min_freq]  
             
smooth_h: float = 5.0 #5.0 #10.0 #20.0 
smooth_v: float = 5.0 #5.0 #10.0 #20.0 

precond: str =  "approx_hessian"    # options: None, approx_hessian, from_file ***note gradients are devided by the precond in both cases

invert_params: list[str] = ["vp", "vs"]    # options: vp, vs, rho
invert_params_weights: dict = {"vp": 1.0, "vs": 1.0}

# bounds fro listed parameters
vp_bounds: list[float] = [450.0, 5000.0]
vs_bounds: list[float] = [250.0, 3000.0]
# rho_bounds: list[float] = [0.9, 1.1]
vp_vs_ratio_bounds: list[float] = [1.5, 2.5] #[1.10, 3.00]
scale_vs_from_vp: bool = False   

# max/min update (e.g., 0.1 means that the max/min update is 10% of the model being updated)
max_update: float = 0.05
min_update: float = 0.01

#######################
### misc parameters ###
#######################
# options: ["x", "z"] for elastic material, ["p"] for acoustic material 
components: list[str] = ["x", "z"] 
# optionsL "d": displacement, "v": velocity, "a": acceleration: "p": preasure
seismotype: str = "d"
save_traces: bool = False 

