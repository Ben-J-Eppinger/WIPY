from scipy import signal
from obspy.core.stream import Stream
import numpy as np


def bandpass_filter(stream: Stream, freq_min: float, freq_max: float, order: int = 2) -> Stream:
    """
    applies a zero phase bandpass filter to each trace in the stream object
    inputs:
        stream: the obspy stream object for a shot gather 
        freq_min: the lower cutoff frequency of the filter
        freq_max: the upper cutoff frequency of the filter
        order: the order of the filter
    outputs: 
        stream: the modified shot gather
    """

    fs: float = 1/stream.traces[0].stats.delta
    sos: np.ndarray = signal.butter(order, (freq_min, freq_max), btype='bandpass', output='sos', fs=fs)

    for trace in stream.traces:
        trace.data = signal.sosfiltfilt(sos, trace.data)

    return stream


def lowpass_filter(stream: Stream, freq_max: float, order: int = 2) -> Stream:
    """
    applies a zero phase lowpass filter to each trace in the stream object
    inputs:
        stream: the obspy stream object for a shot gather 
        freq_max: the upper cutoff frequency of the filter
        order: the order of the filter
    outputs: 
        stream: the modified shot gather
    """

    fs: float = 1/stream.traces[0].stats.delta
    sos: np.ndarray = signal.butter(order, freq_max, btype='lowpass', output='sos', fs=fs)

    for trace in stream.traces:
        trace.data = signal.sosfiltfilt(sos, trace.data)

    return stream


def highpass_filter(stream: Stream, freq_min: float, order: int = 2) -> Stream:
    """
    applies a zero phase highpass filter to each trace in the stream object
    inputs:
        stream: the obspy stream object for a shot gather 
        freq_min: the lower cutoff frequency of the filter
        order: the order of the filter
    outputs: 
        stream: the modified shot gather
    """

    fs: float = 1/stream.traces[0].stats.delta
    sos: np.ndarray = signal.butter(order, freq_min, btype='highpass', output='sos', fs=fs)

    for trace in stream.traces:
        trace.data = signal.sosfiltfilt(sos, trace.data)

    return stream
