from wipy.preprocess import filter, mute, normalize
from wipy.base import paths, params
import obspy
from obspy.core.stream import Stream
import numpy as np
from joblib import Parallel, delayed


class preprocess_base:


    def __init__(self, PATHS: paths, PARAMS: params) -> None:
        self.PATHS: paths = PATHS
        self.PARAMS: params = PARAMS


    def apply_mute(self, data: Stream) -> Stream:
        """
        applies the muting functions as perscribed in the parameters.py file
        inputs:
            data: an obspy stream object for a shot gather
        outputs: 
            data: the edited shot gather
        """
    
        if hasattr(self.PARAMS, "mute"):                                                                     

            if self.PARAMS.mute.count("mute_far_offsets") > 0:
                data = mute.mute_far_offsets(
                    stream=data, 
                    max_offset=self.PARAMS.max_offset
                )

            if self.PARAMS.mute.count("mute_short_offsets") > 0:
                data = mute.mute_short_offsets(
                    stream=data, 
                    min_offset=self.PARAMS.min_offset
                )

            if self.PARAMS.mute.count("mute_above_func") > 0:
                data = mute.mute_above_func(
                    stream=data, 
                    func=self.PARAMS.mute_above_func, 
                    t_taper=self.PARAMS.t_taper
                )

            if self.PARAMS.mute.count("mute_below_func") > 0:
                data = mute.mute_below_func(
                    stream=data,
                    func=self.PARAMS.mute_below_func, 
                    t_taper=self.PARAMS.t_taper
                )                                                           
        
        return data
    

    def apply_filter(self, data: Stream) -> Stream:
        """
        applies the filter function as perscribed in the parameters.py file
        inputs:
            data: an obspy stream object for a shot gather
        outputs: 
            data: the edited shot gather
        """

        if hasattr(self.PARAMS, "filter"):
            
            if self.PARAMS.filter == "bandpass":
                data = filter.bandpass_filter(
                    stream=data,
                    freq_min=self.PARAMS.freq_min,
                    freq_max=self.PARAMS.freq_max,
                    order=self.PARAMS.filter_order
                )

            if self.PARAMS.filter == "lowpass":
                data = filter.lowpass_filter(
                    stream=data,
                    freq_max=self.PARAMS.freq_max,
                    order=self.PARAMS.filter_order
                ) 

            if self.PARAMS.filter == "highpass":
                data = filter.lowpass_filter(
                    stream=data,
                    freq_min=self.PARAMS.freq_min,
                    order=self.PARAMS.filter_order
                )     
        
        return data


    def apply_normalize(self, data: Stream) -> Stream:
        """
        applies the normalizations functions as perscribed in the parameters.py file
        inputs:
            data: an obspy stream object for a shot gather
        outputs: 
            data: the edited shot gather
        """

        if hasattr(self.PARAMS, "normalize"): 
            
            if self.PARAMS.normalize.count("trace_normalize") > 0: 
                data = normalize.trace_normalize(
                    stream=data,
                )

            if self.PARAMS.normalize.count("event_normalize") > 0: 
                data = normalize.event_normalize(
                    stream=data,
                )

        return data


    def write_traces(self, path: str, data: Stream) -> None:
        """
        write preprocessed traces to the disk using the suffix "_proc"
        inputs: 
            path: the directory and file name where the traces will be written
            data: an obspy stream object for a shot gahter
        """

        for trace in data.traces:
            trace.data = trace.data.astype(np.float32)

        data.write(filename=path+"_proc", format="SU")


    def preprocess_traces(self, path: str) -> None:
        """
        executes the preprocessing flow as perscribed in the parameters.py file
        inputs: 
            path: the directory and name of the seismic unix (SU) file that will be preprocessed
        """

        # get the component from the file name
        comp: list[str] = path.split("/")
        comp:str = comp[-1]
        comp: str = comp[1]
        
        # load traces from path
        data: Stream = obspy.read(path, format="SU")

        if comp in self.PARAMS.components:
            # if the component for this data is one of the component we want to work with
            # preprocess traces
            data = self.apply_mute(data)
            data = self.apply_filter(data)
            data = self.apply_normalize(data)
        else:
            # otherwise, zero out all the traces
            data = mute.mute_far_offsets(data, -1)

        # save traces with _proc appended
        self.write_traces(path, data)

    
    def call_preprocessor(self, data_type: str) -> None:
        """
        executes preprocessing flows on shot gathers in parrallel. 
        inputs:
            data_type: option the preprocess synthetic or observed data. The only valid option are "syn" or "obs"
        """

        path_names: list[str] = []

        if self.PARAMS.solver == "specfem2d":
            if self.PARAMS.material == "elastic":
                components = ["x", "z"]
                mode = "d"
            elif self.PARAMS.material == "acoustic":
                components = ["p"]
                mode = "p"

        for comp in components:
            name: str = self.PARAMS.gather_names
            name: list[str] = list(name)
            name[1] = comp
            name[-4] = mode
            name = "".join(name)
            path_names  += ["/".join([self.PATHS.scratch_traces_path, data_type, "{:06d}".format(i), name]) for i in range(self.PARAMS.n_events)]

        print("\npreprocessing traces\n")
        
        Parallel(n_jobs=self.PARAMS.n_proc)(delayed(self.preprocess_traces)(path) for path in path_names)