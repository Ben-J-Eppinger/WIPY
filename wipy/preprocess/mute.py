import numpy as np
from obspy.core.stream import Stream
from typing import Callable


def calc_offset(trace: Stream) -> float:
    """
    calculates the distance between the source and reciever locations by accessing the obspy trace fields
    inputs: 
        trace: trace from an obspy stream object (is also an obspy stream object itself)
    outputs: 
        offset: the distance between the source and receiver 
    Notes:
        edits needed from 3d Implementation
    """

    delta_x = trace.stats.su.trace_header.group_coordinate_x - trace.stats.su.trace_header.source_coordinate_x
    delta_y = trace.stats.su.trace_header.group_coordinate_y - trace.stats.su.trace_header.source_coordinate_y
    offset = np.sqrt(delta_x**2 + delta_y**2)

    return offset


def mute_far_offsets(stream: Stream, max_offset: float) -> Stream:
    """
    sets traces at far offsets to 0
    inputs: 
        stream: obspy stream object for a shot gather
        max_offset: the offset beyond which traces will be set to 0
    outputs: 
        stream: the edited shot gather stream object
    """
    
    for trace in stream.traces:
        offset = calc_offset(trace)
        if offset > max_offset:
            trace.data *= 0

    return stream


def mute_short_offsets(stream: Stream, min_offset: float) -> Stream:
    """
    sets traces at short offsets to 0
    inputs: 
        stream: obspy stream object for a shot gather
        min_offset: the offset before which traces will be set to 0
    outputs: 
        stream: the edited shot gather stream object
    """
    
    for trace in stream.traces:
        offset = calc_offset(trace)
        if offset < min_offset:
            trace.data *= 0

    return stream


def mute_above_func(stream: Stream, func: Callable, t_taper: float):
    """
    mutes the traces at times before func where func is a lambda function taking offset as an input
    a cosine taper is applied prior to the cutoff point defined by func
    inputs: 
        stream: stream: obspy stream object for a shot gather
        func: a (lamda) function of offset determining the time before which each trace will be muted
        t_taper: the length of the cosine taper in time
    outputs: 
        stream: the edited shot gather stream object
    """

    dt: float = stream.traces[0].stats.delta
    T: float = stream.traces[0].stats.npts*dt
    t: np.ndarray = np.arange(start=0, stop=T, step=dt)

    for trace in stream.traces: 

        offset: float = calc_offset(trace)

        t1: float = func(offset)
        t0: float = t1 - t_taper 

        mask: np.ndarray = np.piecewise(
            t, 
            [t < t0, t > t0, t > t1],
            [0, lambda t: np.flip(np.cos(np.pi*(t-t0)/(2*t_taper))), 1],
        )

        trace.data *= mask

    return stream


def mute_below_func(stream: Stream, func: Callable, t_taper: float):
    """
    mutes the traces at times after func where func is a lambda function taking offset as an input
    a cosine taper is applied after to the cutoff point defined by func
    inputs: 
        stream: stream: obspy stream object for a shot gather
        func: a (lamda) function of offset determining the time after which  each trace will be muted
        t_taper: the length of the cosine taper in time
    outputs: 
        stream: the edited shot gather stream object
    """

    dt: float = stream.traces[0].stats.delta
    T:float = stream.traces[0].stats.npts*dt
    t: float = np.arange(start=0, stop=T, step=dt)

    for trace in stream.traces: 

        offset: float = calc_offset(trace)

        t0: float = func(offset)
        t1: float = t0 + t_taper 

        mask: np.ndarray = np.piecewise(
            t, 
            [t < t0, t > t0, t > t1],
            [1, lambda t: np.cos(np.pi*(t-t0)/(2*t_taper)), 0],
        )

        trace.data *= mask

    return stream
