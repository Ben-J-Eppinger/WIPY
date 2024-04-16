from obspy.core.stream import Stream
import warnings


def event_normalize(stream: Stream) -> Stream:
    """
    uses built in obspy stream methods to scale all the shot gather such that the maximum value in the shot gather is 1
    intputs:
        stream: obspy stream object for a shot gather
    outputs:
        stream: the edited stream object
    """
    stream.normalize(global_max=True)
    return stream


def trace_normalize(stream: Stream) -> Stream:
    """
    use built in obspy stream methods to scale each trace in the shot gather such that the maximum for each trace is 1
    intput:
        stream: obspy stream object for a shot gather
    outputs:
        stream: the edited stream object
    """
    with warnings.catch_warnings(action="ignore"):
        stream.normalize(global_max=False)
    return stream 
