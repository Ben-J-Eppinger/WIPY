import subprocess as sp


class paths:
    def __init__(self):
        name = "paths"
        p = __import__(name)

        for var in dir(p):
            if var[0:2] != "__":
                val = getattr(p, var)
                setattr(self, var, val)

        self.scratch_solver_path = "/".join([self.wipy_root_path, "scratch", "solver"])
        self.scratch_traces_path = "/".join([self.wipy_root_path, "scratch", "traces"])
        self.scratch_eval_misfit_path = "/".join([self.wipy_root_path, "scratch", "eval_misfit"])
        self.OUTPUT = "/".join([self.wipy_root_path, "OUTPUT"])


class params:
    def __init__(self):
        name = "parameters"
        p = __import__(name)

        for var in dir(p):
            if var[0:2] != "__":
                val = getattr(p, var)
                setattr(self, var, val)

        if self.solver == "specfem2d":
            self.gather_names = "U*_file_single_d.su"


class base:

    def __init__(self, PATHS: paths, PARAMS: params) -> None:
        self.PATHS: paths = PATHS
        self.PARAMS: params = PARAMS


    def clean(self) -> None:
        """
        Clean out the working directory
        """
        
        sp.run(
            ["rm", "-r", "scratch"],
            cwd=self.PATHS.wipy_root_path,
            capture_output=True,
        )
       
        sp.run(
            ["rm", "-r", "OUTPUT"],
            cwd=self.PATHS.wipy_root_path,
            capture_output=True,
        )


    def import_model(self, src_path: str) -> None:
        """
        copies the a model from its source path (src_path) to the scratch directories
        inputs: 
            src_path: the absolute path of the folder containing the desired model
        """

        # copy the desired model from its source path to scrach/eval_misfit/mode 

        sp.run(
            ["cp * " + "/".join([self.PATHS.scratch_eval_misfit_path, "model"])],
            cwd=src_path,
            shell=True,
            capture_output=True,
        )
        
        for i in range(self.PARAMS.n_events):    

            dir_num: str = "{:06d}".format(i)

            # copy the desired model from scrach/eval_misfit/model to each scratch/solver/event/DATA/ directory

            sp.run(
                ["cp " + "/".join([self.PATHS.scratch_eval_misfit_path, "model"]) + "/* ."],
                cwd="/".join([self.PATHS.scratch_solver_path, dir_num, 'DATA']),
                capture_output=True,
                shell=True
            )
        

    def setup(self) -> None:
        """
        Function prepares system to run the solver and other capbilites by initializing the needed directories
        """

        # export PATHS and PARAMERS to loca object to simplify syntax
        PATHS: paths = self.PATHS
        PARAMS: params = self.PARAMS

        # make the OUTPUT directory
        sp.run(
            ['mkdir', 'OUTPUT'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        # make the scratch directory
        sp.run(
            ['mkdir', 'scratch'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        # make the scratch/solver directory
        sp.run(
            ['mkdir', 'scratch/solver'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        # make the scratch/traces directory
        sp.run(
            ["mkdir", "scratch/traces"],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        # make the scratch/traces/obs
        sp.run(
            ["mkdir", "obs"],
            cwd=PATHS.scratch_traces_path,
            capture_output=True,
        )

        # make the scratch/traces/syn
        sp.run(
            ["mkdir", "syn"],
            cwd=PATHS.scratch_traces_path,
            capture_output=True,
        )

        # make scratch/traces/adj
        sp.run(
            ["mkdir", "adj"],
            cwd=PATHS.scratch_traces_path,
            capture_output=True,
        )

        # make the scratch/eval_misfit
        sp.run(
            ['mkdir', 'scratch/eval_misfit'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        # make the scratch/eval_misfit/residuals
        sp.run(
            ["mkdir", "residuals"],
            cwd=PATHS.scratch_eval_misfit_path,
            capture_output=True,
        )

        # make the scratch/eval_misfit/model
        sp.run(
            ["mkdir", "model"],
            cwd=PATHS.scratch_eval_misfit_path,
            capture_output=True,
        )
 
        for i in range(PARAMS.n_events):

            dir_num: str = "{:06d}".format(i)

            # make a  directry in scracth/traces/{obs, syn, or adj} for each event

            if hasattr(PATHS, "obs_data_path"):
                sp.run(
                    ["cp", "-r", dir_num, "/".join([PATHS.scratch_traces_path, "obs"])],
                    cwd=PATHS.obs_data_path,
                    capture_output=True,
                )

                sp.run(
                    ["mkdir", dir_num],
                    cwd="/".join([PATHS.scratch_traces_path, "adj"]),
                    capture_output=True,
                )   

            sp.run(
                ["mkdir", dir_num],
                cwd="/".join([PATHS.scratch_traces_path, "syn"]),
                capture_output=True,
            )


            # make a directry in scracth/solver for each event with solver executables and data
            sp.run(
                ["mkdir", dir_num],
                cwd=PATHS.scratch_solver_path,
                capture_output=True,
            )

            sp.run(
                ["cp", "-r", PATHS.solver_exec_path, "/".join([PATHS.scratch_solver_path, dir_num])], 
                cwd=PATHS.scratch_solver_path,
                capture_output=True,
            )

            sp.run(
                ["cp", "-r", PATHS.solver_data_path, "/".join([PATHS.scratch_solver_path, dir_num])], 
                cwd=PATHS.scratch_solver_path,
                capture_output=True,
            )

            # rename the source of the directory (event) number as SOURCE so specfem uses it as the source input
            sp.run(
                ["cp", "SOURCE_"+dir_num, "SOURCE"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num, 'DATA']),
                capture_output=True,
            )

            # Make a local OUTPUT_FILES in each scratch/solver/event/ dir (this is required by specfem) 
            sp.run(
                ["mkdir", "OUTPUT_FILES"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num]),
                capture_output=True,
            )

            # make an SEM directory in each scratch/solver/event/ dir (we will have to to put adjoint traces in here before running the adjoint solver)
            sp.run(
                ["mkdir", "SEM"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num]),
                capture_output=True,
            )

        self.import_model(src_path=PATHS.model_init_path)     