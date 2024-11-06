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
        Create an optimize_base object which holds multiple wipy classes. The optmize object will use functionality from these classes to ivnert data. 
        """
        
        self.base = base
        self.PATHS = PATHS
        self.PARAMS = PARAMS
        self.preprocess = preprocess
        self.adjoint = adjoint
        self.solver = solver
        self.iter: int = 0

        # special setup for LBFGS
        if self.PARAMS.optimize == "LBFGS":
            self.LBFGS_mem = 1
            self.LBFGS_mem_max = 6        

        # initialize the scratch/opt.log file
        with open("/".join([self.PATHS.wipy_root_path, "scratch", "opt.log"]), "w") as fid:
            fid.write("iter     step length     misfit\n")


    def eval_misfit(self, alpha: float = None):
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
        # note that the if statement below distinguishes between writing an opt.log before and during a line search
        path = "/".join([self.PATHS.wipy_root_path, "scratch", "opt.log"])
        if alpha == None:
            var = "-"*11
        else:
            var = "{:0.5e}".format(alpha)
        txt = "{:04d}".format(self.iter) + " "*5 + var + " "*5 + "{:0.3e}".format(misfit) + "\n"
        with open(path, "a") as fid:
            fid.write(txt)


    def comp_gradient(self) -> None:
        """
        Calls self.eval_misfit() which will in inturn call the forward solver, preprocesses the data, compute the misfits/adjiont sources
        Goes on to call the adjiont solver, process the kernels (sum kernels across events and smooths summed kernels) and create the gradient
        * notes the gradient is written in the scratch/eval_grad/gradient folder before the preconditioner is applied
        """

        # generate observed synthetic data, preprocessing data, and compute the misfit and ajdiont sources
        self.eval_misfit()
        
        # prepare to run adjoint solver 
        self.base.import_adjoint_sources()

        # call adjoint solver, do basic kernel processing, and save the gradient
        self.solver.call_solver(self.solver.adjoint)
        self.solver.export_kernels()
        self.solver.combine_kernels()
        self.solver.smooth_kernels()
        self.solver.export_smoothed_kernels()
        self.save_gadient()


    def get_descent_dir(self) -> dict[str: np.ndarray]:
        """
        compute the descent direction from the preconditiond gradient 
        (and past gradients/models if using LBFGS)
        inputs:
            None:
        outputs:
            h: a dictionary representation the descent direction. 
            the keys for the dictions are "grad_"+<par_name> (e.g., "grad_vp")
        """

        # using gradient descent
        if self.PARAMS.optimize == "GD":
            h = self.get_GD_descent_dir()

        # using LBFGS
        elif self.PARAMS.optimize == "LBFGS":
            if self.iter == 0:
                h = self.get_GD_descent_dir()
            elif self.iter > 0:
                h = self.get_LBFGS_descent_dir()
                # update LBFGS memory
                self.LBFGS_mem = min(self.LBFGS_mem+1, self.LBFGS_mem_max)

        # using conjugate gradient
        elif self.PARAMS.optimize == "CG":
            if self.iter == 0:
                h = self.get_GD_descent_dir()
            elif self.iter > 0:
                h = self.get_CG_descent_dir()

        # save descent direction
        des_path: str = "/".join([self.PATHS.OUTPUT,"desc_"+"{:04d}".format(self.iter)])
        sp.run(["mkdir", des_path])
        utils.write_model(des_path, h)

        return h
    

    def load_gradient(self) -> dict[str: np.ndarray]:
        """
        loads gradient from scratch/eval_grad/gradient and returns its dictionary representation
        inputs:
            None:
        outputs: 
            h: a dictionary representation of the gradient. 
            the keys for the dictionary are "grad_"+<par_name> (e.g., "grad_vp")
        """

        grad_path: str = "/".join([self.PATHS.scratch_eval_grad_path, "gradient"])

        # change parameter names to match those used for the gradient dictionary
        pars: list[str] = deepcopy(self.PARAMS.invert_params)
        for i in range(len(pars)):
            pars[i] = "grad_" + pars[i]

        g: dict[str: np.ndarray] = utils.load_model(model_path=grad_path, pars=pars)

        return g


    def save_gadient(self) -> None: 
        """
        Copies the binary files for the gradient from scratch/eval_grad/gradient to OUTPUT/gradient_<iter#>
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


    def get_GD_descent_dir(self) -> dict[str: np.ndarray]:
        """"
        Calculate descent direction for gradient descent method. Note that the descent dictionary is the negative of the gradient.
        inputs: 
            None
        outputs: 
            h: a dictionary representation the gradient. 
            the keys for the dictionary are "grad_"+<par_name> (e.g., "grad_vp")
        """
        
        g: dict[str: np.ndarray] = self.load_gradient()
        h: dict[str: np.ndarray] = {}
        for key in g.keys():
            v = -g[key]
            h[key] = v

        # apply preconditioner
        if self.PARAMS.precond is not None:
            h = self.solver.apply_precond(h)

        return h
    

    def get_LBFGS_descent_dir(self) -> dict[str: np.ndarray]:
        """
        Gets LBFGS descent direction using the method described by Modrak et al., 2016
        "Seismic waveforom inversion best practicess: regional and exploration test cases"
        inputs:
            None
        outputs:
            h: a dictionary representation the descent direction.
        """

        pars: list[str] = []
        for par in self.PARAMS.invert_params:
            pars.append("grad_" + par)

        it: int = self.iter
        l: int = self.LBFGS_mem           # maxium number of past gradients to use when constructing the descent direction

        # get j
        j: int = min(it, l)

        # fill out s and y
        s: dict[int: np.ndarray] = {}
        y: dict[int: np.ndarray] = {}

        for i in range(it-1, it-j-1, -1):
            s[i] = self.get_s_sub(i, self.PARAMS.invert_params)
            y[i] = self.get_y_sub(i, pars)

        # fill out lambda and create q
        lamb: dict[str: float] = {}
        grad_path: str = "/".join([self.PATHS.OUTPUT, "grad_" +  "{:04d}".format(it)])
        q: dict[str: np.ndarray] = utils.load_model(grad_path, pars)
        q: np.ndarray = self.dict2vec(q)

        for i in range(it-1, it-j-1, -1): 
            lamb[i] = np.inner(s[i], q)/np.inner(y[i], s[i])
            q -= lamb[i]*y[i]

        # create gamma and initialize r
        gamma: float = np.inner(s[it-1], y[it-1])/np.inner(y[it-1], y[it-1])
        r: np.ndarray = gamma*q

        # apply preconditioner
        if self.PARAMS.precond is not None:
            r = self.vec2dict(r, pars)          # convert r to a dictionary
            r = self.solver.apply_precond(r)    # apply preconditioner to  r
            r = self.dict2vec(r)                # convert r back to a vector

        # calc mu and r
        for i in range(it-j, it):
            mu: float = np.inner(y[i], r)/np.inner(y[i], s[i])
            r += s[i]*(lamb[i] - mu)

        # output the descent direction as a dictionary
        h: np.ndarray = -r
        h: dict[str: np.ndarray] = self.vec2dict(h, pars)

        return h


    def get_s_sub(self, i: int, pars) -> np.ndarray:
        """
        Helper function get_LBFGS_descent_dir: calculates the difference between two models from the ith+1 and ith iterations
        inputs: 
            i: the iteration number
            pars: the parameter keys
        outputs: 
            s_sub: the difference between the ith+1 and ith models as a vector representation 
        """

        model_path_1: str = "/".join([self.PATHS.OUTPUT, "model_" +  "{:04d}".format(i+1)])
        model_path_0: str = "/".join([self.PATHS.OUTPUT, "model_" +  "{:04d}".format(i)])

        m1: dict[str: np.ndarray] = utils.load_model(model_path_1, pars)
        m1: np.ndarray = self.dict2vec(m1)
        
        m0: dict[str: np.ndarray] = utils.load_model(model_path_0, pars)
        m0: np.ndarray = self.dict2vec(m0)

        s_sub = m1 - m0

        return s_sub


    def get_y_sub(self, i: int, pars) -> np.ndarray:
        """
        Helper function get_LBFGS_descent_dir: calculates the difference between two gradients from the ith+1 and ith iterations
        inputs: 
            i: the iteration number
            pars: the parameter keys
        outputs: 
            y_sub: the difference between the ith+1 and ith gradients as a vector representation 
        """

        grad_path_1: str = "/".join([self.PATHS.OUTPUT, "grad_" +  "{:04d}".format(i+1)])
        grad_path_0: str = "/".join([self.PATHS.OUTPUT, "grad_" +  "{:04d}".format(i)])

        g1: dict[str: np.ndarray] = utils.load_model(grad_path_1, pars)
        g1: np.ndarray = self.dict2vec(g1)
        
        g0: dict[str: np.ndarray] = utils.load_model(grad_path_0, pars)
        g0: np.ndarray = self.dict2vec(g0)

        y_sub = g1 - g0

        return y_sub


    def get_CG_descent_dir(self) -> dict[str: np.ndarray]:
        """
        Get the descent direction for the conjugate gradient method
        inputs:
            None
        outputs:
            h: a dictionary representation of the descent direction
        """

        # assign the iteration number
        it: int = self.iter

        # assign a beta min parameter to automatically reset by switching to gradient descent
        #  if conjugacy is lost
        beta_min: float = 0.2

        # get the parameters keys for gradient
        pars: list[str] = []
        for par in self.PARAMS.invert_params:
            pars.append("grad_" + par)

        # get the gradient
        grad_path: str = "/".join([self.PATHS.OUTPUT, "grad_" +  "{:04d}".format(it)])
        g: dict[str: np.ndarray] = utils.load_model(grad_path, pars)
        g: np.ndarray = self.dict2vec(g)

        # get the previous gradient
        prev_grad_path = "/".join([self.PATHS.OUTPUT, "grad_" +  "{:04d}".format(it-1)])
        g_prev: dict[str: np.ndarray] = utils.load_model(prev_grad_path, pars)
        g_prev: np.ndarray = self.dict2vec(g_prev)

        # get the previous descent direction   
        h_prev_path = "/".join([self.PATHS.OUTPUT, "desc_" +  "{:04d}".format(it-1)])
        h_prev: dict[str: np.ndarray] = utils.load_model(h_prev_path, pars)
        h_prev: np.ndarray = self.dict2vec(h_prev)

        # compute beta
        g_hat = g/np.linalg.norm(g) 
        g_prev_hat = g_prev/np.linalg.norm(g_prev) 
        h_prev_hat = h_prev/np.linalg.norm(h_prev) 
        beta: float = np.inner(g_hat, g_hat) / np.inner(g_hat-g_prev_hat, h_prev_hat)
        if beta <= beta_min:
            beta = 0.0

        # apply the preconditioner
        if self.PARAMS.precond is not None:
            g = self.vec2dict(g, pars)          # convert g to a dictionary
            g = self.solver.apply_precond(g)    # apply preconditioner to g
            g = self.dict2vec(g)                # convert g back to a vector

        # compute the descent direction
        h: np.ndarray = -g + beta*h_prev

        # output the descent direction as a dictionary
        h: dict[str: np.ndarray] = self.vec2dict(h, pars)

        return h


    def vec2dict(self, vec: np.ndarray, pars: list[str]) -> dict[str: np.ndarray]:
        """
        helper function for get_LBFGS_descent_dir: converts a dictionary represenation of a model/gradient to its vector representation
        """

        d: dict[str: np.ndarray] = {}
        N: int = int(len(vec)/len(pars))

        for par in pars:
            d[par] = vec[0:N]
            vec = vec[N:]
            
        return d
    

    def dict2vec(self, dict: dict[str: np.array]) -> np.array:
        """
        tranforms dictionary representation of model/gradients to vecocts 
        inputs:
            dict: a dictionary represenation of a model/gradient
        outputs: 
            vec: a vector representation of a dictionary 
            (with all the data for each parameter concatenated into one vector)
        """

        vec: np.ndarray = np.zeros(0)
        
        for key in dict.keys():
            vec = np.append(vec, dict[key])
        
        return vec
    

    def calc_theta(self, g: dict[str: np.ndarray], h: dict[str: np.ndarray]) -> float:
        """
        Measure how parralell the gradient and descent direction are (e.g., values closer to +/- one imply more parralel)
        inputs: 
            g: a dictionary represenation of the gradient
            h: a dictionary represenation of the descent direction
        outputs: 
            theta: measure of how parallel the gradient and descent direction are
        """
       
        h_vec: np.ndarray = self.dict2vec(h)
        g_vec: np.ndarray = self.dict2vec(g)

        theta: float = np.dot(h_vec, g_vec) / (np.linalg.norm(h_vec) * np.linalg.norm(g_vec))

        return theta
    

    def check_model(self, m: dict[str: np.ndarray]) -> dict[str: np.ndarray]:
        """
        Bounds a model to the upper and lower bounds supplied by the user in the parameters files
        inputs:
            m: dictionary represenation of a model
        outputs:
            m: the bounded model
        """

        # loop through keys
        for key in m.keys():

            # get the bounds for the parameter
            bounds: list[float] = getattr(self.PARAMS, key+"_bounds")

            # check the lower bound
            bol = m[key] < bounds[0]
            if sum(bol) > 0:
                print(f"Setting {key} to always be above {bounds[0]}")
                m[key][bol] = bounds[0]
            
            # check the upper bounds
            bol = m[key] > bounds[1]
            if sum(bol) > 0:
                print(f"Setting {key} to always be below {bounds[1]}")
                m[key][bol] = bounds[1]

        # check vp_vs_ratio 
        if self.PARAMS.material == "elastic":
    
            vp_vs_ratio = m["vp"] / m["vs"]

            # check lower bound of vp-vs ratio
            bol = vp_vs_ratio < self.PARAMS.vp_vs_ratio_bounds[0]
            if sum(bol) > 0: 
                if self.PARAMS.scale_vs_from_vp:
                    print(f"Scaling vs from vp so that the vp-vs ratio is grater than {self.PARAMS.vp_vs_ratio_bounds[0]}")
                    m["vs"][bol] = m["vp"][bol] / self.PARAMS.vp_vs_ratio_bounds[0]
                else:
                    print(f"Scaling vp from vs so that the vp-vs ratio is grater than {self.PARAMS.vp_vs_ratio_bounds[0]}")
                    m["vp"][bol] = m["vs"][bol] * self.PARAMS.vp_vs_ratio_bounds[0]

            # check upper bound of vp-vs ratio
            bol = vp_vs_ratio > self.PARAMS.vp_vs_ratio_bounds[1]
            if sum(bol) > 0: 
                if self.PARAMS.scale_vs_from_vp:
                    print(f"Scaling vs from vp so that the vp-vs ratio is less than {self.PARAMS.vp_vs_ratio_bounds[1]}")
                    m["vs"][bol] = m["vp"][bol] / self.PARAMS.vp_vs_ratio_bounds[1]
                else:
                    print(f"Scaling vp from vs so that the vp-vs ratio is less than {self.PARAMS.vp_vs_ratio_bounds[1]}")
                    m["vp"][bol] = m["vs"][bol] * self.PARAMS.vp_vs_ratio_bounds[1]

        return m
    

    def get_alpha_bounds(self, m: dict[str: np.ndarray], h: dict[str: np.ndarray]) -> tuple[float, float]:
        """
        For a given model (m), descent direction (h), and user-supplied max_update and min_update ratios, calculate the maximum 
        step lenght (alpha) for the update m := m + alpha*h
        inputs: 
            m: a dictionary represenatation of the current model
            h: a dictionary representation of the descent direction
        outputs: 
            alpha_max: the maximum steplength that will bound model updates such that ||m + alpha*h||_inf <= ||m||_inf + max_update*||m||_inf
            alpha_min: the minimum steplength that will bound model updates such that ||m + alpha*h||_inf >= ||m||_inf + min_update*||m||_inf
        """

        alpha_max: float = np.inf

        for key in m.keys():

            key1 = key
            key2 = "grad_" + key1

            if np.sum(np.abs(h[key2])) > 0.0:
                a: float = self.PARAMS.max_update * np.linalg.norm(m[key1], ord=np.inf) / np.linalg.norm(h[key2],ord=np.inf) 
                alpha_max = min(a, alpha_max)

        alpha_min: float = alpha_max * self.PARAMS.min_update / self.PARAMS.max_update

        return alpha_min, alpha_max
    

    def perturb_model(self, m: dict[str: np.ndarray], h: dict[str: np.ndarray], alpha: float) -> dict[str: np.ndarray]:
        """
        Adds a mdoel pertubation (update) based on the descent direction (h) and step length (alpha)
        inputs: 
            m: dictionary representation of the model
            h: dictionary representation of the descent direction
            alpha: the step length
        outputs:
            m_test: the perturbed model 
        """

        m_test: dict[str: np.ndarray] = deepcopy(m)

        for key in m.keys():
            grad_key = "grad_" + key
            m_test[key] += alpha*h[grad_key]

        return m_test


    def export_model(self, m: dict[str: np.ndarray]) -> None:
        """
        Saves a dictionary representation of a model to the the OUTPUT directory as binary files
        inputs: 
            m: a dictionary represntation of a model
        """

        save_path: str = "/".join([self.PATHS.OUTPUT, "model_{:04d}".format(self.iter)])

        sp.run(
            ["mkdir", "model_{:04d}".format(self.iter)],
            cwd=self.PATHS.OUTPUT,
            capture_output=True
        )

        utils.write_model(save_path, m)

    
    def save_traces(self):
        """
        copies traces from scratch to OUTPUT folder
        """

        save_path: str = "/".join([self.PATHS.OUTPUT, "traces_{:04d}".format(self.iter)])

        sp.run(
            ["mkdir", "traces_{:04d}".format(self.iter)],
            cwd=self.PATHS.OUTPUT,
            capture_output=True
        )

        sp.run(
            ["cp", "-r", self.PATHS.scratch_traces_path, save_path],
            cwd=self.PATHS.wipy_root_path,
        )

    
    def save_residuals(self):
        """
        copies residuals from scratch to OUTPUT folder
        """

        save_path: str = "/".join([self.PATHS.OUTPUT, "residuals_{:04d}".format(self.iter)])
        src_path: str = "/".join([self.PATHS.scratch_eval_misfit_path, "residuals"])

        sp.run(
            ["mkdir", "residuals_{:04d}".format(self.iter)], 
            cwd=self.PATHS.OUTPUT
        )

        sp.run(
            ["cp", "-r", src_path, save_path],
            cwd=self.PATHS.wipy_root_path
        )

    
    def get_iter_misfit(self, iter: int) -> float: 
        """
        reads the residuals in OUTPUT/residuals_<iter>/residuals/ folder and and sums the residuals together
        inputs:
            iter: the iteration number of the inversion from which we desire the residuals 
        """
        resid_paths: list[str] = ["/".join([self.PATHS.OUTPUT, "residuals_{:04d}".format(iter), "residuals", "{:06d}".format(num)]) for num in range(self.PARAMS.n_events)]

        misfit: float = 0.0
        for path in resid_paths:
            m: float = np.sum(np.loadtxt(path))
            misfit += m

        misfit /= self.PARAMS.n_events

        return misfit


    def backtrack_linesearch(self) -> str:
        """
        Updates the model via a backtracking line search. A good alpha is found by chacking the Armijo condition.
        When using LBGFS, there is a restart condition if the line search fails.
        outputs: 
            status which will equal either "Fail" or "Pass". 
            Fail means that no scaling of alpha could sufficently decrease the misfit
            Pass means that the model was updated successfully
        """

        # set tau (to update alpha)
        tau: float = 0.5   

        # load the model, gradient, and descent direction
        model_init_path: str = "/".join([self.PATHS.OUTPUT, "model_" + "{:04d}".format(self.iter)])
        model_path: str = "/".join([self.PATHS.scratch_eval_misfit_path, "model"])
        m: dict[str: np.ndarray] = utils.load_model(model_init_path, self.PARAMS.invert_params)
        g: dict[str: np.ndarray] = self.load_gradient()
        h: dict[str: np.ndarray] = self.get_descent_dir()

        # calculate theta
        theta: float = self.calc_theta(g, h)

        # get bounds for alpha
        alpha_min, alpha_max = self.get_alpha_bounds(m, h)

        # set parameter c for Armijo condition
        c: float = (10**-10)/alpha_max

        # initialize alpha and residuals
        alpha: float = alpha_max
        residuals: list[float] = [self.get_iter_misfit(self.iter)]

        # begin line search
        while alpha > alpha_min:

            # update model
            m_test: dict[str: np.ndarray] = self.perturb_model(m, h, alpha)

            # check model
            m_test: dict[str: np.ndarray] = self.check_model(m_test)

            # write model
            utils.write_model(model_path, m_test)

            # import model
            self.base.import_model(model_path)
            
            # eval misfit
            self.eval_misfit(alpha)
            
            # calc residuals
            residuals.append(self.adjoint.sum_residuals())
            
            # check Armijo condition
            if residuals[-1] < residuals[0] + c*alpha*theta:

                # if true, save the model and the residuals to the output directory
                self.iter += 1
                self.export_model(m_test)
                self.save_residuals()

                # add blank line to opt.log file
                with open("/".join([self.PATHS.wipy_root_path, "scratch", "opt.log"]), "a") as fid:
                    fid.write("\n")
                
                return "Pass"
            
            else:
                # update alpha
                alpha *= tau

        # cases for if we get out of the while loop without finding a good alpha 
                
        if self.PARAMS.optimize =="GD": 
            return "Fail"

        if self.PARAMS.optimize == "CG":
            return "Fail"
        
        elif self.PARAMS.optimize == "LBFGS":   #  restart using gradient descent
            # note in opt.log file
            with open("/".join([self.PATHS.wipy_root_path, "scratch", "opt.log"]), "a") as fid:
                fid.write("restart using GD \n")
                
            self.PARAMS.optimize = "GD"
            print("line search failed: switching to gradient descent \n")
            status: str = self.backtrack_linesearch() 
            
            if status == "Pass":
                self.PARAMS.optimize = "LBFGS"  # switch back to LBFGS
                self.LBFGS_mem = 1
                print("restart succeeded: switching back to LBFGS \n")
            
            return status

            


