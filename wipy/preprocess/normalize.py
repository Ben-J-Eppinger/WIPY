from obspy.core.stream import Stream


def event_normalize(stream: Stream) -> Stream:
    """
    uses built in obspy stream methods to scale all the shot gather such that the maximum value in the shot gather is 1
    """
    stream.normalize(global_max=True)
    return stream


def trace_normalize(stream: Stream) -> Stream:
    """
    use built in obspy stream methods to scale each trace in the shot gather such that the maximum for each trace is 1
    """
    stream.normalize(global_max=False)
    return stream 
