import subprocess as sp
import numpy as np
from typing import Callable
from joblib import Parallel, delayed
from wipy.base import paths, params # import the paths and params classes so that we can use them in our constrcutor function {__init__()}
from wipy.wipy_utils import utils


class solver_base:

    def __init__(self, PATHS: paths, PARAMS: params) -> None:
        self.PATHS: paths = PATHS
        self.PARAMS: params = PARAMS

        # make dictinaries mapping parameters names to the corresponding kerenel name output by scecfem2d
        if self.PARAMS.material == "elastic":
            params_to_kernel_names: dict = {"vp": "alpha_kernel", "vs": "beta_kernel", "rho": "rho_kernel"}
        elif self.PARAMS.material == "acoustic":
            params_to_kernel_names: dict = {"vp": "c_acoustic_kernel", "rho": "rho_acoustic_kernel"}

        # make a list of the kernel names we are actually going to used based on the type of material and parameters we invert for
        kernels_used: list[str] = []
        for param in self.PARAMS.invert_params: 
            kernels_used.append(params_to_kernel_names[param])

        # if using the approx hessian precond, append it to the end of teh kernels_used list
        if self.PARAMS.precond == "approx_hessian":
            if self.PARAMS.material == "elastic":
                kernels_used.append("Hessian1_kernel")
            elif self.PARAMS.material == "acoustic":
                kernels_used.append("Hessian1_acoustic_kernel")

        self.PARAMS.params_to_kernel_names = params_to_kernel_names
        self.PARAMS.kernels_used = kernels_used

    
    def setpar(self, path, par, new_par) -> None:
        """
        Reads a parameter file and updates it with a new par value.
        inputs:
            path: The path the parameter file
            par: The parameter to be updated
            new_par: the value for the parameter 
        """

        with open(path, 'r') as fid: 
            lines: list[str] = fid.readlines()

            for i in range(len(lines)):
                s0: str = lines[i].split('=')[0]

                if par in s0 and '#' not in s0:
                    lines[i]  = par + " = " + new_par + '\n'

        with open(path, 'w') as fid:
            fid.writelines(lines)


    def call_solver(self, func: Callable[[str], None]) -> None:
        """
        Calls the solver for each event in paralell. Used to call forwards/backwards solvers and to smooth kernels.
        inputs:
            func: the solver function being called, e.g., forwards, backwards, or smooth. 
        """
        path_names: list[str] = ["/".join([self.PATHS.scratch_solver_path, "{:06d}".format(x)]) for x in range(self.PARAMS.n_events)]
        Parallel(n_jobs=self.PARAMS.n_proc)(delayed(func)(path) for path in path_names)

    
    def export_traces(self) -> None:
        """
        copies traces from scratch/solver folders (000000 through 00000N) to scratch/traces
        """
        src_paths: list[str] = ["/".join([self.PATHS.scratch_solver_path, "{:06d}".format(x), "OUTPUT_FILES"]) for x in range(self.PARAMS.n_events)]
        dest_paths: list[str] = ["/".join([self.PATHS.scratch_traces_path, "syn", "{:06d}".format(x)]) for x in range(self.PARAMS.n_events)]

        for path_num in range(self.PARAMS.n_events):
            sp.run(
                ["cp *.su " + dest_paths[path_num]],
                shell=True,
                cwd=src_paths[path_num],
                capture_output=True
            )


    def save_traces(self) -> None: 
        """
        copies traces from scratch to output direcotry 
        """
        sp.run(
            ["cp", "-r", self.PATHS.scratch_traces_path, self.PATHS.OUTPUT],
            cwd=self.PATHS.wipy_root_path,
            capture_output=True,
        )

    
    def export_kernels(self) -> None: 
        """
        copies kernels from scratch/solver/<event_num>/OUTPUT_FILES to scratch/eval_grad_kernels/<event_num>
        """

        for event in range(self.PARAMS.n_events):
            event_str = "{:06d}".format(event)
            sp.run(
                ["cp *kernel.bin " + "/".join([self.PATHS.scratch_eval_grad_path, "kernels", event_str])],
                cwd="/".join([self.PATHS.scratch_solver_path, event_str, "OUTPUT_FILES"]),
                shell=True,
                capture_output=True
            )


    def combine_kernels(self) -> None:
        """
        sum the kernels for all events together by runing xcomine from the scratch/solver/000000/
        """

        # get all the kernel folder names
        kernel_names: list[str] = ["/".join([self.PATHS.scratch_eval_grad_path, "kernels", "{:06d}".format(event)+"\n"]) for event in range(self.PARAMS.n_events)]

        # write the kernels names in a text file
        file_path = "/".join([self.PATHS.scratch_solver_path, "000000", "kernel_names.txt"])
        with open(file_path, "w") as fid:
            fid.writelines(kernel_names)

        command = ["./bin/xcombine_sem", 
                   ",".join(self.PARAMS.kernels_used), 
                   "kernel_names.txt", 
                   "/".join([self.PATHS.scratch_eval_grad_path, "sum"])]

        # call the spcefem combine function
        sp.run(
            command,
            cwd="/".join([self.PATHS.scratch_solver_path, "000000"]),
            capture_output=True,
        )

        # copy the cordinate, NPEC_ibool, and jacobian binary files the the sum folder so
        # the smoother can be used later

        names = ["proc000000_NSPEC_ibool.bin", 
                 "proc000000_jacobian.bin", 
                 "proc000000_x.bin",
                 "proc000000_z.bin"]
        
        for name in names:
            sp.run(
                ["cp", name, "/".join([self.PATHS.scratch_eval_grad_path, "sum"])],
                cwd="/".join([self.PATHS.scratch_solver_path, "000000", "DATA"])
            )

    def smooth_kernels(self) -> None:
        """"
        smooth the kerenels in the scratch/eval_grad/sum folder and output them in the sctratch/eval_grad/smooth folder
        by call the smoother from the sctratch/solver/000000 folder
        """

        for param in self.PARAMS.kernels_used:

            command = ["./bin/xsmooth_sem", 
                       "{:.2f}".format(self.PARAMS.smooth_h/np.sqrt(8)), 
                       "{:.2f}".format(self.PARAMS.smooth_v/np.sqrt(8)),
                       param, 
                       "/".join([self.PATHS.scratch_eval_grad_path, "sum"]),
                       "/".join([self.PATHS.scratch_eval_grad_path, "sum_smooth"]),
                       "true"]
            
            sp.run(
                command,
                cwd="/".join([self.PATHS.scratch_solver_path, "000000"]),
                capture_output=True
            )


    def export_smoothed_kernels(self) -> None:
        """"
        Copies the needed kernels from the scratch/eval_grad/sum_smooth folder to the scratch/eval_grad/gradient folder
        """

        for param in self.PARAMS.invert_params:

                src_path = "/".join([self.PATHS.scratch_eval_grad_path, "sum_smooth", "proc000000_"+self.PARAMS.params_to_kernel_names[param]+"_smooth.bin"])
                des_path = "/".join([self.PATHS.scratch_eval_grad_path, "gradient", "proc000000_grad_"+param+".bin"])

                sp.run(
                    ["cp", src_path, des_path],
                    capture_output=True
                )


    def apply_precond(self, h: dict[str: np.ndarray]) -> dict[str: np.ndarray]: 
        """
        Applies the preconditioner to a dictionary of representaion of the descent direction
        inputs: 
            h: a dictionary representation of  the descent direction
        outputs:
            h: a dictionary representation of the preconditioned descent direction
        """

        print("\nApplying Preconditioner \n")

        if self.PARAMS.precond == "approx_hessian":
            
            hess_path = "/".join([self.PATHS.scratch_eval_grad_path, "sum_smooth", "proc000000_"+self.PARAMS.kernels_used[-1]+"_smooth.bin"])
            hess = utils.read_fortran_binary(hess_path)

            for key in h.keys():
                h[key] = h[key]/hess

        return h

        # setup case where preconditioner can be loaded from file



        










