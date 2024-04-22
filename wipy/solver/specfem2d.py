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
        
        print("Calling specfem2d adjoint solver")

        if self.PARAMS.material == "acoustic":
            # if the material is acoustic, then replace the Ux adjoint sources with the Uz adjount sources
            # this is becasue for a preasure source, specfem2d will only use the Ux file for the adjoint source
            sp.run(
                ["cp", "Uz_file_single.su.adj", "Ux_file_single.su.adj"],
                cwd="/".join([path, "SEM"]),
            )

        sp.run(
            ["./bin/xspecfem2D"],
            cwd=path,
            capture_output=True,
        )
    

 
