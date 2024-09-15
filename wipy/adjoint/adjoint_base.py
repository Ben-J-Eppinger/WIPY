from wipy.adjoint import misfits
from wipy.base import paths, params
import obspy
from obspy.core.stream import Stream
import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy

class adjoint_base:


    def __init__(self, PATHS: paths, PARAMS: params) -> None:
        self.PATHS: paths = PATHS
        self.PARAMS: params = PARAMS

        # import the desired misfit function from misfits
        self.misfit_func = getattr(misfits, self.PARAMS.misfit)

    
    def write_adjoint_sources(self, path: str, adj: Stream) -> None:
        """
        write adjoint sources for a shot gather to the disk
        inputs: 
            path: the directory and file name where the sources will be written
            data: an obspy stream object for a shot gahte
        """

        for trace in adj.traces:
            trace.data = trace.data.astype(np.float32)

        adj.write(filename=path, format="SU")


    def write_residuals(self, path: str, residuals: list[float]) -> None:
        """
        write the residuals for a particular gather to the disk
        inputs:
            path: the dicectory and file name of the residuals
            data: a list of residual values corresponding to each trace in a shot gather
        """
        
        np.savetxt(path, residuals)


    def make_mifits_and_adjoint_traces(self, event_num: int) -> None:
        """
        uses the event number to get the scratch/traces path of the observed, synthetic, and adjoint traces
        as well as the scratch/eval_misfit path of the residuals. Then calculates the adjoint sources and
        residuals and finaly writes the adjoint sources and residuals. 
        inputs: 
            event_num: the event number of the data to be processed
        """

        if self.PARAMS.solver == "specfem2d":
            if self.PARAMS.material == "elastic":
                components = ["x", "z"]
            elif self.PARAMS.material == "acoustic":
                components = ["p"]

        for comp in components:

            # get gather name based on the current compoenent
            gather_name = self.PARAMS.gather_names
            gather_name = list(gather_name)
            gather_name[1] = comp
            gather_name = "".join(gather_name)
            adj_name = "U" + comp + "_file_single.su.adj"

            # get paths of observed, synthetic, and the soon to be adjoint sources / misfits
            obs_path = "/".join([self.PATHS.scratch_traces_path, "obs", "{:06d}".format(event_num), gather_name])+"_proc"
            syn_path = "/".join([self.PATHS.scratch_traces_path, "syn", "{:06d}".format(event_num), gather_name])+"_proc"
            adj_path = "/".join([self.PATHS.scratch_traces_path, "adj", "{:06d}".format(event_num), adj_name])
            residuals_path = "/".join([self.PATHS.scratch_eval_misfit_path, "residuals", "{:06d}".format(event_num)])

            # load synthetic and observed data and the time increment
            obs = obspy.read(obs_path, format="SU")
            syn = obspy.read(syn_path, format="SU")
            
            if comp in self.PARAMS.components:
                # if the component being processed is one that is being inverted
                args = [syn, obs] + self.PARAMS.additional_misfit_parameters
                adj, residuals = self.misfit_func(*args)
                # write residuals
                self.write_residuals(residuals_path, residuals)
            else:
                # otherwise, set all adjoint sources for this component to be 0
                adj = deepcopy(syn) 
                for trace in adj.traces: 
                    trace.data *= 0 

            # write adjoint sources
            self.write_adjoint_sources(adj_path, adj)


    def comp_all_misfits_and_adjoint_sources(self) -> None:
        """
        Calls make_mifits_and_adjoint_traces in parralel so that as many events as possible can be processed at the same time
        """

        print("\ncomputing misfits and adjoint sources\n")

        Parallel(n_jobs=self.PARAMS.n_proc)(delayed(self.make_mifits_and_adjoint_traces)(event) for event in range(self.PARAMS.n_events))


    def sum_residuals(self) -> float: 
        """
        reads the residuals in scratch/eval_misfit/residuals/ and and summs the residuals together
        """
        resid_paths = ["/".join([self.PATHS.scratch_eval_misfit_path, "residuals", "{:06d}".format(num)]) for num in range(self.PARAMS.n_events)]

        misfit = 0
        for path in resid_paths:
            m = np.sum(np.loadtxt(path))
            misfit += m

        misfit /= self.PARAMS.n_events

        return misfit