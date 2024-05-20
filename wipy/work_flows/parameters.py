#########################
### solver parameters ###
#########################

solver: str = "specfem2d"       # options: specfem2d
material: str = "acoustic"      # options: acoustic, elastic
n_events: int = 10              
n_proc: int = -1                # -1 means that the miximum number of processors will be used  

#############################
### preprocess parameters ###
#############################

filter: str = "lowpass"    # options: bandpass(freq_min, freq_max), lowpass(freq_max), highpass(freq_min)
# freq_min: float = 1
freq_max: float = 5.0
filter_order: int = 10

mute: list[str] = []
# options: "mute_far_offsets", "mute_short_offsets", "mute_above_func", "mute_below_func"
max_offset: float = 180
min_offset: float = 20
mute_above_func = lambda offset: 0.1 + 0.005*offset
mute_below_func = lambda offset: 0.4 + 0.000*offset
t_taper: float = 0.1

normalize = ["trace_normalize", "event_normalize"]

############################
### inversion parameters ###
############################
optimize: str = "LBFGS"                        # options: GD (gradient descent), LBFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno)

max_iter: int = 50

misfit: str = "NC_norm"                     # options: L2_norm, NC_norm

smooth_v: float = 20.0
smooth_h: float = 20.0

precond: str =  "approx_hessian"    # options: None, approx_hessian, from_file

invert_params: list[str] = ["vp"]    # options: vp, vs, rho

# bounds fro listed parameters
vp_bounds: list[float] = [299, 450]
rho_bounds: list[float] = [250, 300]

# max/min update (e.g., 0.1 means that the max/min update is 10% of the model being updated)
max_update: float = 0.1
min_update: float = 0.01

# misc parameters
# options: ["x", "z"] for elastic material,, ["p"] for acoustic material
components: list[str] = ["p"]       
save_traces = True 

