from wipy.solver.solver_base import solver_base # import to solver base
from wipy.wipy_utils import utils
import subprocess as sp
import numpy as np

class specfem2d(solver_base):

    def forward(self, path: str) -> None:
        """
        Calls specfem2d forward solver
        inputs:
            path [string]: Will be the root the directory of the specfem call. It should contain the DATA, OUTPUT_FILES, bin, and SEM directories. 
        """

        # set that the simulation type is forward
        super().setpar(
            path="/".join([path, 'DATA', 'Par_file']),
            par='SIMULATION_TYPE',
            new_par='1'
        )

        # set save forrward to true in case we want to do an adjoint simulation later
        super().setpar(
            path="/".join([path, 'DATA', 'Par_file']),
            par='SAVE_FORWARD',
            new_par='.true.'
        )

        seismo_type_dict: dict[str: str] = {"d": "1", "v": "2", "a": "3", "p": "4"}
        super().setpar(
            path="/".join([path, 'DATA', 'Par_file']),
            par='seismotype',
            new_par=seismo_type_dict[self.PARAMS.seismotype]
        )
        
        print("Calling specfem2d forward solver")

        # run the mesher
        sp.run(
            ["./bin/xmeshfem2D"],
            cwd=path,
            capture_output=True,
        )
        
        # run the solver
        sp.run(
            ["./bin/xspecfem2D"],
            cwd=path,
            capture_output=True,
        )
        

    
    def adjoint(self, path: str) -> None:
        """
        Calls specfem2d adjoint solver
        inputs:
            path [string]: Will be the root the directory of the specfem call. It should contain the DATA, OUTPUT_FILES, bin, and SEM directories. 
        """
        
        super().setpar(
            path="/".join([path, 'DATA', 'Par_file']),
            par='SIMULATION_TYPE',
            new_par='3'
        )

        super().setpar(
            path="/".join([path, 'DATA', 'Par_file']),
            par='SAVE_FORWARD',
            new_par='.true.'
        )        

        if "approx_hessian" in self.PARAMS.precond:
            hess_par = ".true."
        else:
            hess_par = ".false."
        super().setpar(
            path="/".join([path, 'DATA', 'Par_file']),
            par="APPROXIMATE_HESS_KL",
            new_par=hess_par
        )
        
        print("Calling specfem2d adjoint solver")

        sp.run(
            ["./bin/xspecfem2D"],
            cwd=path,
            capture_output=True,
        )
    

    def smooth_kernels(self) -> None:
        """
        smooth the kerenels in the scratch/eval_grad/sum folder and 
        outputs them in the sctratch/eval_grad/smooth folder
        """

        input_path = "/".join([self.PATHS.scratch_eval_grad_path, "sum"])

        out_path_base = "/".join([self.PATHS.scratch_eval_grad_path, "sum_smooth"])

        pars = self.PARAMS.kernels_used

        m = utils.load_model(input_path, ["x", "z"] + pars)

        for par in pars:

            print("\nsmoothing " + par)

            g = utils.smooth_par(
                m, 
                par, 
                self.PARAMS.smooth_h/np.sqrt(8), 
                self.PARAMS.smooth_v/np.sqrt(8)
            )

            out_path = out_path_base + "/proc000000_" + par + "_smooth.bin"

            utils.write_fortran_binary(out_path, g)
