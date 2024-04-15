import numpy as np
from obspy.core.stream import Stream
from typing import Callable

def calc_offset(trace: Stream):

    delta_x = trace.stats.su.trace_header.group_coordinate_x - trace.stats.su.trace_header.source_coordinate_x
    delta_y = trace.stats.su.trace_header.group_coordinate_y - trace.stats.su.trace_header.source_coordinate_y
    offset = np.sqrt(delta_x**2 + delta_y**2)

    return offset


def mute_far_offsets(stream: Stream, max_offset: float) -> Stream:
    
    for trace in stream.traces:
        offset = calc_offset(trace)
        if offset > max_offset:
            trace.data *= 0

    return stream


def mute_short_offsets(stream: Stream, min_offset: float) -> Stream:
    
    for trace in stream.traces:
        offset = calc_offset(trace)
        if offset < min_offset:
            trace.data *= 0

    return stream


def mute_above_func(stream: Stream, func: Callable, t_taper: float):

    dt = stream.traces[0].stats.delta
    T = stream.traces[0].stats.npts*dt
    t = np.arange(start=0, stop=T, step=dt)

    for trace in stream.traces: 

        offset = calc_offset(trace)

        t1 = func(offset)
        t0 = t1 - t_taper 

        mask = np.piecewise(
            t, 
            [t < t0, t > t0, t > t1],
            [0, lambda t: np.flip(np.cos(np.pi*(t-t0)/(2*t_taper))), 1],
        )

        trace.data *= mask

    return stream


def mute_below_func(stream: Stream, func: Callable, t_taper: float):

    dt = stream.traces[0].stats.delta
    T = stream.traces[0].stats.npts*dt
    t = np.arange(start=0, stop=T, step=dt)

    for trace in stream.traces: 

        offset = calc_offset(trace)

        t0 = func(offset)
        t1 = t0 + t_taper 

        mask = np.piecewise(
            t, 
            [t < t0, t > t0, t > t1],
            [1, lambda t: np.cos(np.pi*(t-t0)/(2*t_taper)), 0],
        )

        trace.data *= mask

    return stream
