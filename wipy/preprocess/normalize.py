from obspy.core.stream import Stream


def event_normalize(stream: Stream) -> Stream:
    stream.normalize(global_max=True)
    return stream


def trace_normalize(stream: Stream) -> Stream:
    stream.normalize(global_max=False)
    return stream 
