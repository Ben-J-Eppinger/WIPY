from wipy.base import base, paths, params

# initialize the wipy paths and params classes (these will read your input files)
PATHS = paths()
PARAMS = params()

# import the solver specifieid in the parameters.py file 
_temp = __import__(".".join(["wipy", "solver", PARAMS.solver]), fromlist=PARAMS.solver)
solver_class = getattr(_temp, PARAMS.solver)

# import preprocessor
from wipy.preprocess.preprocess_base import preprocess_base

# import the adjoint module
from wipy.adjoint.adjoint_base import adjoint_base

# import the optmizse module
from wipy.optimize.optimize_base import optimize_base

# initialize the base variable
b = base(PATHS, PARAMS)
b.clean()
b.setup()

# intialize the sovler class
s = solver_class(PATHS, PARAMS)

# initialize the preprocess calss
p = preprocess_base(PATHS, PARAMS)

# inititialize the adjiont class
a = adjoint_base(PATHS, PARAMS)

# initialize the optimize class
o = optimize_base(base=b, 
                  PATHS=PATHS, 
                  PARAMS=PARAMS,
                  preprocess=p,
                  adjoint=a,
                  solver=s)

# do the invesion
while o.iter < PARAMS.max_iter:

    # add blank line to opt.log file
    with open("/".join([PATHS.wipy_root_path, "scratch", "opt.log"]), "a") as fid:
        fid.write("\n")
   
    # update the iteration number
    o.iter += 1

    # compute gradient
    o.comp_gradient()

    # update model with backtracking line search
    status = o.backtrack_linesearch()

    # save traces if requested
    if PARAMS.save_traces:
        o.save_traces()

    # quit if line search fails
    if status == "Fail":
        break
