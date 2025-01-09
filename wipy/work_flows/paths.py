wipy_root_path = "/home/beneppinger/WIPY_Projects/marmousi2"
solver_exec_path = "/home/beneppinger/WIPY_Projects/marmousi2/specfem2d/bin"
solver_data_path = "/home/beneppinger/WIPY_Projects/marmousi2/specfem2d/DATA"
obs_data_path = "/home/beneppinger/WIPY_Projects/marmousi2/OBS_data"
# model_init_path = "/home/beneppinger/WIPY_Projects/marmousi2/TFP_3.0Hz_results/model_final"
model_init_path = "/home/beneppinger/WIPY_Projects/marmousi2/model_init"
# model_init_path = "/home/beneppinger/WIPY_Projects/marmousi2/model_true"

# **note that the precond path is a path to a file rather than to a directory
precond_path = "/home/beneppinger/WIPY_Projects/marmousi2/Precond/proc000000_precond.bin"

# check to see if all the paths are valid
if __name__ == "__main__":
    import subprocess as sp
    for key in dir():
        if "_path" in key:
            path = vars()[key]
            s = sp.run(["ls", path], capture_output=True, text=True)
            if s.stderr: 
                print("\n" + path)
                print(s.stderr)