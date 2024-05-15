from wipy.base import base, paths, params

# initialize the wipy paths and params classes (these will read your input files)
PATHS = paths()
PARAMS = params()

# create a wipy base object using the base class and run the clean and setup methods
b = base(PATHS, PARAMS)
b.clean()
b.setup()

# import the solver specifieid in the parameters.py file 
_temp = __import__(".".join(["wipy", "solver", PARAMS.solver]), fromlist=PARAMS.solver)
solver_class = getattr(_temp, PARAMS.solver)

# initialize a solver object and use it to do forward modeling
s = solver_class(PATHS, PARAMS)
s.call_solver(s.forward)
s.export_traces()

# import preprocessor
from wipy.preprocess.preprocess_base import preprocess_base

# create a preprocessor object
p = preprocess_base(PATHS, PARAMS)

# preprocess the observed and synthetic data
p.call_preprocessor(data_type='obs')
p.call_preprocessor(data_type='syn')


# import the adjoint class
from wipy.adjoint.adjoint_base import adjoint_base

# create a adjoint_base object
a = adjoint_base(PATHS, PARAMS)

# calculate misfits and adjoint sources and then import them
a.comp_all_misfits_and_adjoint_sources()
b.import_adjoint_sources()

# call adjoint solver and do basic kernel processing
s.call_solver(s.adjoint)
s.export_kernels()
s.combine_kernels()
s.smooth_kernels()