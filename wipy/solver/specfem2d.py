from wipy.solver.solver_base import solver_base # import to solver base
import subprocess as sp

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

        # if using elastic modeling output displacement seismograms and if using acoustic modeling output preasure seismograms
        if self.PARAMS.material == "elastic":
            seismotype = "1"
        elif self.PARAMS.material == "acoustic":
            seismotype = "4"
            
        super().setpar(
            path="/".join([path, 'DATA', 'Par_file']),
            par='seismotype',
            new_par=seismotype
        )
        
        print("Calling specfem2d forwad solver")

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

        if self.PARAMS.precond == "approx_hessian":
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
    

 
