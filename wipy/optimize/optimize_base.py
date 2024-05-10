from wipy.base import paths, params, base
from wipy.preprocess.preprocess_base import preprocess_base
from wipy.adjoint.adjoint_base import adjoint_base
from wipy.solver.solver_base import solver_base 
from wipy.wipy_utils import utils
import numpy as np
import subprocess as sp
from copy import deepcopy


class optimize_base:

    def __init__(self, base: base, PATHS: paths, PARAMS: params, preprocess: preprocess_base, adjoint: adjoint_base, solver: solver_base):
        """
        Create an optimize_base object which holds multiple wipy classes. The optmize object will you use functionality from these classes to ivnert data. 
        """
        
        self.base = base
        self.PATHS = PATHS
        self.PARAMS = PARAMS
        self.preprocess = preprocess
        self.adjoint = adjoint
        self.solver = solver
        self.iter: int = 0
        
        with open("/".join([self.PATHS.wipy_root_path, "scratch", "opt.log"]), "w") as fid:
            fid.write("iteration     step length     misfit\n")


    def eval_misfit(self):
        """"
        Calls the forward solver, preprocesses the data, computes the misfits, and writes
        misfit to the scratch/opt.log file 
        """

        # run a forward simulation
        self.solver.call_solver(self.solver.forward)
        self.solver.export_traces()

        # preprocess the observed and synthetic data
        self.preprocess.call_preprocessor(data_type='obs')
        self.preprocess.call_preprocessor(data_type='syn')

        # compute the residuals
        self.adjoint.comp_all_misfits_and_adjoint_sources()
        misfit = self.adjoint.sum_residuals() 
        
        # write misfits to the opt.log file
        path = "/".join([self.PATHS.wipy_root_path, "scratch", "opt.log"])
        txt = "{:04d}".format(self.iter) + " "*10 + "-"*11 + " "*5 + "{:0.3e}".format(misfit) + "\n"
        with open(path, "a") as fid:
            fid.write(txt)


    def comp_gradient(self) -> None:
        """
        Calls self.eval_misfit() which will in inturn call the forward solver, preprocesses the data, compute the misfits/adjiont sources
        Goes on to call the adjiont solver, process the kernels (sum kernels across events and smooths summed kernels) and precondititions the gradient
        * notes the gradient is written in the scratch/eval_grad/gradient folder after the preconditioner is applies
        """
        # generate observed synthetic data, preprocessing data, and compute the misfit and ajdiont sources
        self.eval_misfit()
        
        # prepare to run adjoint solver 
        self.base.import_adjoint_sources()

        # call adjoint solver and do basic kernel processing
        self.solver.call_solver(self.solver.adjoint)
        self.solver.export_kernels()
        self.solver.combine_kernels()
        self.solver.smooth_kernels()

        # apply preconditioner
        self.solver.apply_precond()

        # save gradient
        self.save_gadient()


    def get_descent_dir(self) -> dict[str: np.ndarray]:
        """
        compute the descent direction from the preconditiond gradient 
        (and past gradients/models if using LBGFS)
        inputs:
            None:
        outputs:
            h: a dictionary representation the descent direction. 
            the keys for the dictions are "grad_"+<par_name> (e.g., "grad_vp")
        """

        # using gradient descnet
        # note that the descent direction is just the negative of the gradient
        if self.PARAMS.optimize == "GD":

            g: dict[str: np.ndarray] = self.load_gradient()
            h: dict[str: np.ndarray] = {}
            for key in g.keys():
                h[key] = -g[key]

        # using LBGFS
        # not implemented yet
        elif self.PARAMS.optmize == "LBGFS":
            pass

        return h
    

    def load_gradient(self) -> dict[str: np.ndarray]:
        grad_path = "/".join([self.PATHS.scratch_eval_grad_path, "gradient"])
        pars = deepcopy(self.PARAMS.invert_params)
        for i in range(len(pars)):
            pars[i] = "grad_" + pars[i]

        g = utils.load_model(model_path=grad_path, pars=pars)

        return g


    def save_gadient(self) -> None: 
        """
        Copies preconditioned gradient from scratch/eval_grad/gradient to OUTPUT/gradient_<iter#>
        """

        # make directory for gradient to store in
        des_path: str = "/".join([self.PATHS.OUTPUT,"grad_"+"{:04d}".format(self.iter)])
        sp.run(["mkdir", des_path])

        # copy files from sctatch/eval_grad/gradient to OUTPUT/gradient_<iter#>
        src_path: str = "/".join([self.PATHS.scratch_eval_grad_path, "gradient"])+"/*"

        command: str = " ".join(["cp", src_path, des_path])
        sp.run(
            [command],
            shell=True
        ) 





