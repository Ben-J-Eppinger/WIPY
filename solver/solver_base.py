import subprocess as sp
from typing import Callable
from joblib import Parallel, delayed
from base import paths, params # import the paths and params calsses so that we can use them in our constrcutor function {__init__()}


class solver_base:

    def __init__(self, PATHS: paths, PARAMS: params) -> None:
        self.PATHS: paths = PATHS
        self.PARAMS: params = PARAMS

    
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
        sp.run(
            ["cp", "-r", self.PATHS.scratch_traces_path, self.PATHS.OUTPUT],
            cwd=self.PATHS.wipy_root_path,
            capture_output=True,
        )









