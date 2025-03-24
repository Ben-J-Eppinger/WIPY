wipy_root_path = "/home/beneppinger/WIPY_Projects/marmousi3"
solver_exec_path = "/home/beneppinger/WIPY_Projects/marmousi3/specfem2d/bin"
solver_data_path = "/home/beneppinger/WIPY_Projects/marmousi3/specfem2d/DATA"
obs_data_path = "/home/beneppinger/WIPY_Projects/marmousi3/OBS_data"
model_init_path = "/home/beneppinger/WIPY_Projects/marmousi3/0.5-4.5Hz_amp_results/model_final"
# model_init_path = "/home/beneppinger/WIPY_Projects/marmousi3/model_init"
# model_init_path = "/home/beneppinger/WIPY_Projects/marmousi3/model_true"

# **note that the precond path is a path to a file rather than to a directory
precond_path = "/home/beneppinger/WIPY_Projects/marmousi3/Precond/proc000000_precond.bin"

# check to see if all the paths are valid
if __name__ == "__main__":
    import subprocess as sp
    for key in dir():
        if "_path" in key:
            path = vars()[key]
            s = sp.run(["ls", path], capture_output=True, text=True)
            print(key + " --- " + path)
            if s.stderr: 
                print("\n" + path)
                print(s.stderr)