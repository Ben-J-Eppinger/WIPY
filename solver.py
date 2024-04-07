import subprocess as sp
from joblib import Parallel, delayed


class solver:

    def __init__(self, PATHS, PARAMS):
        self.PATHS = PATHS
        self.PARAMS = PARAMS


    def setup(self):
        """
        Function prepares system to run the solver and other capbilites by initializing the needed directories
        inputs:
            self: 
        """
      
        PATHS = self.PATHS
        PARAMS = self.PARAMS

        # make basic directory structure
        sp.run(
            ["rm", "-r", "scratch"],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        sp.run(
            ['mkdir', 'OUTPUT'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )
        
        sp.run(
            ['mkdir', 'scratch'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        sp.run(
            ['mkdir', 'scratch/solver'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        # make a solver directry for each event with solver executables and data 
        for i in range(PARAMS.n_events):

            dir_num = "{:06d}".format(i)

            SP = sp.run(
                ["mkdir", dir_num],
                cwd=PATHS.scratch_solver_path,
                capture_output=True,
            )
            
            SP = sp.run(
                ["cp", "-r", PATHS.solver_exec_path, "/".join([PATHS.scratch_solver_path, dir_num])], 
                cwd=PATHS.scratch_solver_path,
                capture_output=True,
            )
            
            SP = sp.run(
                ["cp", "-r", PATHS.solver_data_path, "/".join([PATHS.scratch_solver_path, dir_num])], 
                cwd=PATHS.scratch_solver_path,
                capture_output=True,
            )

            SP = sp.run(
                ["cp", "SOURCE_"+dir_num, "SOURCE"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num, 'DATA']),
                capture_output=True,
            )
            
            SP = sp.run(
                ["mkdir", "OUTPUT_FILES"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num]),
                capture_output=True,
            )

            SP = sp.run(
                ["mkdir", "SEM"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num]),
                capture_output=True,
            )

    
    def setpar(self, path, par, new_par):
        """
        Reads a parameter file and updates it with a new par value.
        inputs:
            path: The path the parameter file
            par: The parameter to be updated
            new_par: the value for the parameter 
        """

        with open(path, 'r') as fid: 
            lines = fid.readlines()

            for i in range(len(lines)):
                s0 = lines[i].split('=')[0]

                if par in s0 and '#' not in s0:
                    lines[i]  = par + " = " + new_par + '\n'

        with open(path, 'w') as fid:
            fid.writelines(lines)


    def call_solver(self, func):
        """
        Calls the solver for each event in paralell. Used to call forwards/backwards solvers and to smooth kernels.
        inputs:
            func: the solver function being called, e.g., forwards, backwards, or smooth. 
        """
        path_names = ["/".join([self.PATHS.scratch_solver_path,"{:06d}".format(x)]) for x in range(self.PARAMS.n_events)]
        PP = Parallel(n_jobs=self.PARAMS.n_proc)(delayed(func)(path) for path in path_names)


class specfem2d(solver):

    def forward(self, path):
        """
        Calls specfem2d forward solver
        inputs:
            path [string]: Will be the root the directory of the specfem call. It should contain the DATA, OUTPUT_FILES, bin, and SEM directories. 
        """
        super().setpar(
            path="/".join([path, 'DATA', 'Par_file']),
            par='SIMULATION_TYPE',
            new_par='1'
        )
        
        SP = sp.run(
            ["./bin/xmeshfem2D"],
            cwd=path,
            capture_output=True,
        )
        
        SP = sp.run(
            ["./bin/xspecfem2D"],
            cwd=path,
            capture_output=True,
        )

    
    def adjoint(self, path):
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
            capture_output=True,
        )
    
    
    def smooth(self, path):
        return None
    
    
    def sum_kernels(self, path_list):
        return None







