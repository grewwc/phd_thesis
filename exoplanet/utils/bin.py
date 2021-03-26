import numpy as np
# from config import *


def median_bin(x, y, num_bins, bin_width=None, normalize=True, bin_width_factor=1.0):
    """
    assume x is sorted 
    bin_width is the scale factor of (x_max-x_min)/num_bins, not the absolute value
    """
    x = x.ravel()
    y = y.ravel()
    x_min, x_max = x[0], x[-1]
    default_bin_width = (x_max - x_min) / num_bins
    bin_width = default_bin_width if bin_width is None else bin_width
    bin_width *= bin_width_factor

    bin_spacing = (x_max - x_min) / num_bins

    m = np.median(y)
    res = np.repeat(m, num_bins)
    bin_lo, bin_hi = x_min, x_min + bin_width

    for i in range(num_bins):
        indices = np.argwhere(
            (x < bin_hi) & (x >= bin_lo)).ravel()

        if len(indices) > 0:
            res[i] = np.median(y[indices])

        bin_lo += bin_spacing
        bin_hi += bin_spacing

    if normalize:
        res -= np.median(res)
        res /= np.max(np.abs(res))

    return res


def another_bin(x, y, num_bins, duration=None, normalize=True):
    """x, y represent folded light curves"""
    assert len(x) == len(y)
    
    tce_width = 0.5*duration
    x_min, x_max = x[0], x[-1]
    x_mid = (x_min + x_max) / 2
    tce_min, tce_max = x_mid - tce_width, x_mid + tce_width

    i1 = np.argwhere(x<tce_min).ravel()
    i2 = np.argwhere((x>=tce_min) & (x<=tce_max)).ravel()
    i3 = np.argwhere(x>tce_max).ravel()

    n2 = int(min(len(i2)/5, 0.4*num_bins))
    if n2 == 0:
        return median_bin(x, y, num_nins, normalize=normalize)

    n1 = n3 = (num_bins - n2) // 2 + 1

    a = median_bin(x[i1], y[i1], n1, bin_width_factor=1.0, normalize=False)
    b = median_bin(x[i2], y[i2], n2, bin_width_factor=4, normalize=False)
    c = median_bin(x[i3], y[i3], n3, bin_width_factor=1.0, normalize=False)

    # print(np.median(a), np.median(b), np.median(c))
    # print(np.median(x[i1]), np.median(x[i2]), np.median(x[i3]))
    # print(np.median(y[i1]), np.median(y[i2]), np.median(y[i3]))
    res = np.concatenate([a,b,c])
    if normalize:
        res -= np.median(res)
        res /= np.max(np.abs(res))
    return res[:num_bins]
