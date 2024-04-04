import subprocess as sp
from joblib import Parallel, delayed


class solver:

    def setup(PATHS, PARAMS):
        """
        Function prepares system to run the solver and other capbilites by initializing the needed directories
        inputs:
            PATHS: WIPY class containing all important directory locations
            PARAMS: WIPY class containing all input parameters
        """

        sp.run(
            ["rm -r scratch"],
            cwd=PATHS.wipy_root_path,
            shell=True,
            capture_output=True,
            text=True
        )

        sp.run(
            ['mkdir OUTPUT'],
            cwd=PATHS.wipy_root_path,
            shell=True,
            capture_output=True,
            text=True
        )
        
        sp.run(
            ['mkdir scratch'],
            cwd=PATHS.wipy_root_path,
            shell=True,
            capture_output=True,
            text=True
        )

        sp.run(
            ['mkdir scratch/solver'],
            cwd=PATHS.wipy_root_path,
            shell=True,
            capture_output=True,
            text=True
        )

        for i in range(PARAMS.n_events):

            dir_name = "{:06d}".format(i)

            SP = sp.run(
                ["mkdir " + dir_name],
                cwd=PATHS.scratch_solver_path,
                shell=True,
                capture_output=True,
                text=True
            )
            
            SP = sp.run(
                ["cp -r " + PATHS.solver_exec_path + " " + "/".join([PATHS.scratch_solver_path, dir_name])], 
                cwd=PATHS.scratch_solver_path,
                shell=True,
                capture_output=True,
                text=True
            )
            
            SP = sp.run(
                ["cp -r " + PATHS.solver_data_path + " " + "/".join([PATHS.scratch_solver_path, dir_name])], 
                cwd=PATHS.scratch_solver_path,
                shell=True,
                capture_output=True,
                text=True
            )
            
            SP = sp.run(
                ["mkdir OUTPUT_FILES"],
                cwd="/".join([PATHS.scratch_solver_path, dir_name]),
                shell=True,
                capture_output=True,
                text=True
            )

            SP = sp.run(
                ["mkdir SEM"],
                cwd="/".join([PATHS.scratch_solver_path, dir_name]),
                shell=True,
                capture_output=True,
                text=True
            )

    def call_solver(PATHS, PARAMS, func):
        """
        Calls the solver for each event in paralell. Used to call forwards/backwards solvers and to smooth kernels.
        inputs:
            PATHS: WIPY class containing all important directory locations
            PARAMS: WIPY class containing all input parameters
        """
        path_names = ["/".join([PATHS.scratch_solver_path,"{:06d}".format(x)]) for x in range(PARAMS.n_events)]
        PP = Parallel(n_jobs=PARAMS.n_proc)(delayed(func)(path) for path in path_names)


class specfem2d(solver):

    def forward(path):
        """
        Calls specfem2d forward solver
        inputs:
            path [string]: Will be the root the directory of the specfem call. It should contain the DATA, OUTPUT_FILES, bin, and SEM directories. 
        """
        
        #
        # Add line here to make sure that the simulation_type = 1
        #

        SP = sp.run(
            ["./bin/xmeshfem2D"],
            cwd=path,
            shell=True,
            capture_output=True,
            text=True
        )
        
        SP = sp.run(
            ["./bin/xspecfem2D"],
            cwd=path,
            shell=True,
            capture_output=True,
            text=True
        )

    def adjoint(path):
        """
        Calls specfem2d adjoint solver
        inputs:
            path [string]: Will be the root the directory of the specfem call. It should contain the DATA, OUTPUT_FILES, bin, and SEM directories. 
        """
        
        #
        # Add line here to make sure that the simulation_type = 3
        #
        
        SP = sp.run(
            ["./bin/xspecfem2D"],
            cwd=path,
            shell=True,
            capture_output=True,
            text=True
        )
    
    def smooth(path):
        return None
    
    def sum_kernels(path_list):
        return None







