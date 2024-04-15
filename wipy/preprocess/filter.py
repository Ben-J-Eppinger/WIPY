from scipy import signal
from obspy.core.stream import Stream


def bandpass_filter(freq_min: float, freq_max: float, stream: Stream, order: int = 2) -> Stream:

    fs = 1/stream.traces[0].stats.delta
    sos = signal.butter(order, (freq_min, freq_max), btype='bandpass', output='sos', fs=fs)

    for trace in stream.traces:
        trace.data = signal.sosfiltfilt(sos, trace.data)

    return stream


def lowpass_filter(freq_max: float, stream: Stream, order: int = 2) -> Stream:

    fs = 1/stream.traces[0].stats.delta
    sos = signal.butter(order, freq_max, btype='lowpass', output='sos', fs=fs)

    for trace in stream.traces:
        trace.data = signal.sosfiltfilt(sos, trace.data)

    return stream


def highpass_filter(freq_min: float, stream: Stream, order: int = 2) -> Stream:

    fs = 1/stream.traces[0].stats.delta
    sos = signal.butter(order, freq_min, btype='highpass', output='sos', fs=fs)

    for trace in stream.traces:
        trace.data = signal.sosfiltfilt(sos, trace.data)

    return stream
