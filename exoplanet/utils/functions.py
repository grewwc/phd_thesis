import warnings
from functools import wraps
import numpy as np
from lightkurve import LightCurve
from scipy.interpolate import interp1d

from config import *
from .bin import median_bin
from .fold import fold
import math


def almost_same(x, y):
    delta = 1e-2
    return math.fabs(x - y) <= delta


def main_tag(func):
    """
    api for clients, not internal usage
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def deperacated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(f"{func.__qualname__} is deperacated" +
                      "shouldn't be used")
        return func(*args, **kwargs)

    return wrapper


def internal_useage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def write_info(msg):
    sys.stdout.write(f'\r{msg}')
    sys.stdout.flush()


def choose_from_center(time, val_center, val_width):
    """
    val_center, val_width should have the same units (days in general)
    """
    half_width = val_width / 2.0
    hi = np.searchsorted(time, val_center + half_width)
    lo = np.searchsorted(time, val_center - half_width)
    lo, hi = int(lo), int(hi)
    return lo, hi


# sigma_clip is not good (by experiment)
def sigma_clip(time, flux, sigma=3.0):
    assert len(time) == len(flux)
    if len(time) < 1:
        return time, flux
        # don't do anything

    mask = np.abs(flux - np.median(flux)) <= np.std(flux) * sigma

    tce_time, tce = time[mask], flux[mask]
    if len(tce_time) < 2:
        return time, flux

    func = interp1d(tce_time, tce, fill_value='extrapolate', kind='linear')
    return time, func(time)


def remove_points_other_tce(all_time,
                            all_flux,
                            cur_period,
                            period_list,
                            t0_list,
                            duration_list):
    """
    all_time, all_flux: real time, not after folding \n
    return: time, flux after removing the other periods points
    """
    for period, t0, duration in zip(period_list, t0_list, duration_list):
        # don't remove the current period
        if almost_same(cur_period, period):
            continue
        fold_time = all_time % period
        t0 %= period
        half_duration = duration / 2.0
        mask = np.logical_or(fold_time > t0 + half_duration,
                             fold_time < t0 - half_duration)
        all_time = all_time[mask]
        all_flux = all_flux[mask]

    return all_time, all_flux


def __flatten_interp_transits(
        all_time, all_flux, period, t0, duration):
    """
    all_time, all_flux: real time of the lightcurve, before folding.\n
    duration: in Days
    return the flattened time, flux, not folding

    """

    fold_time = all_time % period
    t0 %= period
    half_duration = duration / 2.0
    mask = np.logical_and(fold_time <= t0 + half_duration,
                          fold_time >= t0 - half_duration)

    mask[0] = False
    mask[-1] = False

    other_time = all_time[~mask]
    other_flux = all_flux[~mask]

    if len(other_flux) > 2:
        func = interp1d(other_time, other_flux, fill_value='extrapolate')
        interp_flux = func(all_time[mask])
    else:
        interp_flux = all_flux[mask]

    flat_flux = all_flux.copy()
    flat_flux[mask] = interp_flux

    lc = LightCurve(all_time, flat_flux).flatten(
        window_length=51, polyorder=2, break_tolerance=40, sigma=5,
        niters=5
    )

    all_time, flat_flux = lc.time, lc.flux

    flat_flux[mask] = all_flux[mask]

    # keep the transit intact

    tce_time, tce = all_time[mask], all_flux[mask]
    tce_time, tce = sigma_clip(tce_time, tce, sigma=3.0)
    all_flux[mask] = tce
    # sigma clip the outliers in the transit

    return all_time, flat_flux


def flatten_interp_transits(
        all_time, all_flux, period, t0, duration):
    """
    all_time, all_flux: real time of the lightcurve, before folding.\n
    duration: in Days
    return the flattened time, flux, not folding

    """

    fold_time = all_time % period
    t0 %= period
    half_duration = duration / 2.0
    mask = np.logical_and(fold_time <= t0 + half_duration,
                          fold_time >= t0 - half_duration)

    tce_time, tce = all_time[mask], all_flux[mask]

    lc = LightCurve(all_time, all_flux).flatten(
        window_length=801, polyorder=2, break_tolerance=40, sigma=3
    )

    all_time, flat_flux = lc.time, lc.flux

    flat_flux[mask] = tce
    # keep the transit original

    # tce_time, tce = sigma_clip(tce_time, tce, sigma=3.0)
    # sigma clip the outliers in the transit
    return all_time, flat_flux


def process_global(time, flux, period, t0, duration):
    """
    time, flux: after removing the other period values\n
    duration: in DAYs \n
    return: binned flux
    """
    t0 %= period

    time, flux = flatten_interp_transits(time, flux, period, t0, duration)

    time, flux = fold(time, flux, period, t0)

    bin_width = period / num_bins
    # if period < 1:
    #     bin_width *= 40
    # elif period < 10:
    #     bin_width *= 20
    # elif period < 100:
    #     bin_width *= 10
    binned_flux = median_bin(
        time, flux,
        num_bins=num_bins,
        bin_width=bin_width
    )
    return binned_flux


def process_local(time, flux, period, t0, duration):
    """
    time, flux: after removing the other period values\n
    duration: in DAYs \n
    return: binned flux
    """
    t0 %= period

    time, flux = flatten_interp_transits(
        time, flux, period, t0, duration
    )

    time, flux = fold(time, flux, period, t0)

    lo, hi = choose_from_center(
        time, period / 2.0, 4 * duration
    )

    time, flux = time[lo:hi], flux[lo:hi]

    # experimental
    bin_width = bin_width_factor * duration

    # if duration < 5:
    #     bin_width *= 4
    # elif duration < 10:
    #     bin_width *= 2

    binned_flux = median_bin(
        time, flux,
        num_bins=num_local_bins,
        bin_width=bin_width
    )

    return binned_flux
