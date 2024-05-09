from wipy.base import paths, params, base
from wipy.preprocess.preprocess_base import preprocess_base
from wipy.adjoint.adjoint_base import adjoint_base
from wipy.solver.solver_base import solver_base 
import subprocess  as sp

class optimize_base:

    def __init__(self, base: base, PATHS: paths, PARAMS: params, preprocess: preprocess_base, adjoint: adjoint_base, solver: solver_base):
        """
        Create an optimize_base object which holds multiple wipy classes. The optmize object will you use functionality from these classes to ivnert data. 
        """
        
        self.base = base
        self.PATHS = PATHS
        self.PARMAS = PARAMS
        self.preprocess = preprocess
        self.adjoint = adjoint
        self.solver = solver
        self.iter: int = 0
        
        with open("/".join([self.PATHS.wipy_root_path, "scratch", "opt.log"]), "w") as fid:
            fid.write("iteration     step length     misfit\n")



    def eval_misfit(self):

        # run a forward simulation
        self.solver.call_solver(self.solver.forward)
        self.solver.export_traces()

        # preprocess the observed and synthetic data
        self.preprocess.call_preprocessor(data_type='obs')
        self.preprocess.call_preprocessor(data_type='syn')

        # compute the residuals
        self.adjoint.comp_all_misfits_and_adjoint_sources()
        misfit = self.adjoint.sum_residuals() 
        
        path = "/".join([self.PATHS.wipy_root_path, "scratch", "opt.log"])

        txt = "{:04d}".format(self.iter) + " "*10 + "-"*11 + " "*5 + "{:0.5e}".format(misfit) + "\n"

        with open(path, "a") as fid:
            fid.write(txt)



