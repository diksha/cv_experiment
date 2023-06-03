from typing import Callable

import numpy as np
import scipy.interpolate


def sample_interpolated_signal(
    signal: np.array,
    timestamps: np.array,
    sample_time: float = None,
    mode: str = "linear",
) -> tuple:
    """sample_interpolated_signal.

    Args:
        signal (np.array):  input signal
        timestamps (np.array): original signal timestamps
        sample_time (float): time to sample in the same units as timestamps
        mode (str): the interpolation mode (of linear, cubic, and quadratic)

    Returns:
        tuple: the interpolated sampled signal
    """
    if sample_time is None:
        sample_time = (timestamps[-1] - timestamps[0]) / len(timestamps)

    linearly_spaced_timestamps = np.linspace(
        timestamps[0],
        timestamps[-1],
        (timestamps[-1] - timestamps[0]) // sample_time + 1,
    )
    signal_interpolated = interpolate(timestamps, signal, mode)(
        linearly_spaced_timestamps
    )
    return signal_interpolated, linearly_spaced_timestamps


def interpolate(
    timestamps: np.array, signal: np.array, mode: str = "linear"
) -> Callable:
    """interpolate.

    Args:
        signal (np.array):  the signal to interpolate
        timestamps (np.array): the timestamps of the signal
        mode (str): the order of the interpolation. Must be of the list: "linear", "quadratic", "cubic"

    Returns:
        Callable: the callable interpolation function
    """
    return scipy.interpolate.interp1d(
        timestamps,
        signal,
        kind=mode,
        assume_sorted=True,
    )
