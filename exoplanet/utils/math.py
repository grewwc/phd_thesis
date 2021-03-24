import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from interfaces import *
from python_tools.NotEfficient.functions import Override
from astropy.stats import sigma_clip

from scipy import signal
from scipy.interpolate import interp1d
from copy import deepcopy
from astropy.stats import sigma_clip


def flatten(time, flux, window_length=101, polyorder=2, return_trend=False,
            break_tolerance=5, niters=3, sigma=3, mask=None, **kwargs):
    """Removes the low frequency trend using scipy's Savitzky-Golay filter.

    This method wraps `scipy.signal.savgol_filter`.

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e. the number of coefficients).
        ``window_length`` must be a positive odd integer.
    polyorder : int
        The order of the polynomial used to fit the samples. ``polyorder``
        must be less than window_length.
    return_trend : bool
        If `True`, the method will return a tuple of two elements
        (flattened_lc, trend_lc) where trend_lc is the removed trend.
    break_tolerance : int
        If there are large gaps in time, flatten will split the flux into
        several sub-lightcurves and apply `savgol_filter` to each
        individually. A gap is defined as a period in time larger than
        `break_tolerance` times the median gap.  To disable this feature,
        set `break_tolerance` to None.
    niters : int
        Number of iterations to iteratively sigma clip and flatten. If more than one, will
        perform the flatten several times, removing outliers each time.
    sigma : int
        Number of sigma above which to remove outliers from the flatten
    mask : boolean array with length of self.time
        Boolean array to mask data with before flattening. Flux values where
        mask is True will not be used to flatten the data. An interpolated
        result will be provided for these points. Use this mask to remove
        data you want to preserve, e.g. transits.
    **kwargs : dict
        Dictionary of arguments to be passed to `scipy.signal.savgol_filter`.

    Returns
    -------
    flatten_lc : LightCurve object
        Flattened lightcurve.
    If ``return_trend`` is `True`, the method will also return:
    trend_lc : LightCurve object
        Trend in the lightcurve data
    """

    if mask is None:
        mask = np.ones(len(time), dtype=bool)
    else:
        # Deep copy ensures we don't change the original.
        mask = deepcopy(~mask)
    # No NaNs
    mask &= np.isfinite(flux)
    # No outliers
    mask &= np.nan_to_num(
        np.abs(flux - np.nanmedian(flux))) <= (np.nanstd(flux) * sigma)
    for iter in np.arange(0, niters):
        if break_tolerance is None:
            break_tolerance = np.nan
        if polyorder >= window_length:
            polyorder = window_length - 1
            log.warning("polyorder must be smaller than window_length, "
                        "using polyorder={}.".format(polyorder))
        # Split the lightcurve into segments by finding large gaps in time
        # dt = time[mask][1:] - time[mask][0:-1]
        # with warnings.catch_warnings():  # Ignore warnings due to NaNs
        #     warnings.simplefilter("ignore", RuntimeWarning)
        #     cut = np.where(dt > break_tolerance * np.nanmedian(dt))[0] + 1
        cut = []
        low = np.append([0], cut)
        high = np.append(cut, len(time[mask]))
        # Then, apply the savgol_filter to each segment separately
        trend_signal = np.zeros(len(time[mask]))
        for l, h in zip(low, high):
            # Reduce `window_length` and `polyorder` for short segments;
            # this prevents `savgol_filter` from raising an exception
            # If the segment is too short, just take the median
            if np.any([window_length > (h - l), (h - l) < break_tolerance]):
                trend_signal[l:h] = np.nanmedian(flux[mask][l:h])
            else:
                # Scipy outputs a warning here that is not useful, will be fixed in version 1.2
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FutureWarning)
                    trend_signal[l:h] = signal.savgol_filter(x=flux[mask][l:h],
                                                             window_length=window_length,
                                                             polyorder=polyorder,
                                                             **kwargs)
        # No outliers
        mask1 = np.nan_to_num(np.abs(flux[mask] - trend_signal)) <\
            (np.nanstd(flux[mask] - trend_signal) * sigma)
        f = interp1d(time[mask][mask1],
                     trend_signal[mask1], fill_value='extrapolate')
        trend_signal = f(time)
        mask[mask] &= mask1

    flatten_lc = flux.copy()
    with warnings.catch_warnings():
        # ignore invalid division warnings
        warnings.simplefilter("ignore", RuntimeWarning)
        flatten_lc = flux / trend_signal
    if return_trend:
        trend_lc = trend_signal
        return flatten_lc, trend_lc
    return flatten_lc


