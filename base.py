import subprocess as sp


class paths:
    def __init__(self):
        # with open('paths.txt', 'r') as fid: 
        #     lines: list[str] = fid.readlines()

        # for line in lines:
        #     s = line.split("=")
        #     setattr(self, s[0].strip(), s[1].strip())
        name = "paths"
        p = __import__(name)

        for var in dir(p):
            if var[0:2] != "__":
                val = getattr(p, var)
                setattr(self, var, val)

        self.scratch_solver_path = "/".join([self.wipy_root_path, "scratch", "solver"])
        self.scratch_traces_path = "/".join([self.wipy_root_path, "scratch", "traces"])
        self.OUTPUT = "/".join([self.wipy_root_path, "OUTPUT"])


class params:
    def __init__(self):
        name = "parameters"
        p = __import__(name)

        for var in dir(p):
            if var[0:2] != "__":
                val = getattr(p, var)
                setattr(self, var, val)


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


    def setup(self) -> None:
        """
        Function prepares system to run the solver and other capbilites by initializing the needed directories
        """

        PATHS: paths = self.PATHS
        PARAMS: params = self.PARAMS

        # make basic directory structure
        sp.run(
            ['mkdir', 'OUTPUT'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        sp.run(
            ['mkdir', 'scratch'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        sp.run(
            ['mkdir', 'scratch/solver'],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        sp.run(
            ["mkdir", "scratch/traces"],
            cwd=PATHS.wipy_root_path,
            capture_output=True,
        )

        sp.run(
            ["mkdir", "obs"],
            cwd=PATHS.scratch_traces_path,
            capture_output=True,
        )

        sp.run(
            ["mkdir", "syn"],
            cwd=PATHS.scratch_traces_path,
            capture_output=True,
        )

        sp.run(
            ["mkdir", "adj"],
            cwd=PATHS.scratch_traces_path,
            capture_output=True,
        )
 
        for i in range(PARAMS.n_events):

            dir_num: str = "{:06d}".format(i)

            # make a  directry in scracth/traces/{obs, syn, or adj} for each event
            sp.run(
                ["mkdir", dir_num],
                cwd="/".join([PATHS.scratch_traces_path, "obs"]),
                capture_output=True,
            )

            sp.run(
                ["mkdir", dir_num],
                cwd="/".join([PATHS.scratch_traces_path, "syn"]),
                capture_output=True,
            )

            sp.run(
                ["mkdir", dir_num],
                cwd="/".join([PATHS.scratch_traces_path, "adj"]),
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

            sp.run(
                ["cp", "SOURCE_"+dir_num, "SOURCE"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num, 'DATA']),
                capture_output=True,
            )

            sp.run(
                ["mkdir", "OUTPUT_FILES"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num]),
                capture_output=True,
            )

            sp.run(
                ["mkdir", "SEM"],
                cwd="/".join([PATHS.scratch_solver_path, dir_num]),
                capture_output=True,
            )
