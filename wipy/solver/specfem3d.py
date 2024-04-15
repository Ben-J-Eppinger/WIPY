from wipy.solver.solver_base import solver_base # import to solver base
import subprocess as sp

class specfem2d(solver_base):

    def forward(self, path: str) -> None:
        """
        Calls specfem2d forward solver
        inputs:
            path [string]: Will be the root the directory of the specfem call. It should contain the DATA, OUTPUT_FILES, bin, and SEM directories. 
        """

        # specify the simulation type

        # run the solver (need to run the mesher beforehand and copy the files)

        # rename the output su file to have the .su suffix at the end

        return None
        

    
    def adjoint(self, path: str) -> None:
        """
        Calls specfem2d adjoint solver
        inputs:
            path [string]: Will be the root the directory of the specfem call. It should contain the DATA, OUTPUT_FILES, bin, and SEM directories. 
        """
        
        #
        # Add line here to make sure that the simulation_type = 3
        #
        # Do more stuff
        #
        
        return None
    
    
    def smooth(self, path: str) -> None:
        return None
    
    
    def sum_kernels(self, path) -> None:
        return None
 
