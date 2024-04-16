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

# import preprocess_base class, initialize a preprocessor, and use it to mute, filter, and/or normalize data
from wipy.preprocess.preprocess_base import preprocess_base
p = preprocess_base(PATHS, PARAMS)
p.call_preprocessor(data_type='obs')
p.call_preprocessor(data_type='syn')

# save traces if desired
if PARAMS.save_traces:
    s.save_traces()