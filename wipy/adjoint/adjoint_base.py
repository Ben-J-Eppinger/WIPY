from wipy.adjoint import misfits
from wipy.base import paths, params
import obspy
from obspy.core.stream import Stream
import numpy as np
from joblib import Parallel, delayed

class adjoint_base:


    def __init__(self, PATHS: paths, PARAMS: params) -> None:
        self.PATHS: paths = PATHS
        self.PARAMS: params = PARAMS

    
    def write_adjoint_traces(self, path: str, adj: Stream) -> None:
        """
        write adjoint traces to the disk
        inputs: 
            path: the directory and file name where the traces will be written
            data: an obspy stream object for a shot gahte
        """

        for trace in adj.traces:
            trace.data = trace.data.astype(np.float32)

        adj.write(filename=path, format="SU")


    def write_misfits(self, path: str, misfits: list[float]) -> None:
        """
        write the misfits for a particular gather to the disk
        inputs:
            path: the dicectory and file name of the misfits
            data: a list of misfit values corresponding to each trace in a shot gather
        """
        pass


    def make_adjoints_and_mifits(self, event_num: int) -> None:

        # for comp in self.PARAMS.components:

            # obs_path = 
            # syn_path = 
            # adj_path = 
            # misfit_path = 

            # calc adjoint sources and misfits

            # write adjoint sources
            # write misfits

        pass


    def call_adjoint(self):

        Parallel(n_jobs=self.PARAMS.n_proc)(delayed(self.make_adjoints_and_mifits(event)) for event in range(self.PARAMS.n_events))