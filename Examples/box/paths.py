wipy_root_path = "/home/beneppinger/WIPY_Projects/box"
solver_exec_path = "/home/beneppinger/WIPY_Projects/box/specfem2d/bin"
solver_data_path = "/home/beneppinger/WIPY_Projects/box/specfem2d/DATA"
obs_data_path = "/home/beneppinger/WIPY_Projects/box/OBS_data"
# model_init_path = "/home/beneppinger/WIPY_Projects/box/model_init"
model_init_path = "/home/beneppinger/WIPY/Examples/box/model_true"

# **note that the precond path is a parth to a file rather than to a directory
# precond_path = "/home/beneppinger/WIPY_tests/box_model_precond/precond.bin"

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