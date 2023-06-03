import numpy as np
import scipy.ndimage

from core.perception.smoothing.interpolation import (
    interpolate,
    sample_interpolated_signal,
)


def gaussian_smooth_signal(
    signal: np.array, timestamps: np.array, sample_time: float, sigma: float
) -> np.array:
    """gaussian_smooth_signal.

    Smooth the signal using a guassian kernel of the given variance(sigma) given in units of time.
    All input times should be in the same units. i.e. timestamps, sample time and sigma would all be
    the same units (ms/seconds)

    Args:
        signal (nd.array): signal
        timestamps (nd.array): timestamps
        sample_time (float): sample_time
        sigma (float): sigma

    Returns:
        np.array: the smoothed signal resampled at the input timestamps
    """

    # first interpolate
    signal_interpolated, timestamps_interpolated = sample_interpolated_signal(
        signal, timestamps, sample_time
    )
    signal_gaussian_smoothed = scipy.ndimage.gaussian_filter1d(
        signal_interpolated,
        axis=0,
        sigma=sigma / sample_time,
        mode="nearest",
    )
    resampled_signal = interpolate(
        timestamps_interpolated, signal_gaussian_smoothed, mode="linear"
    )(timestamps)
    # interpolate back using the original timestamps
    return resampled_signal
