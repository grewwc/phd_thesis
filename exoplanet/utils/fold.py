import numpy as np
import matplotlib.pyplot as plt
from lightkurve.lightcurve import LightCurve


def fold(time, flux, period, t0=None):
    half_period = period / 2.0

    if t0 is None:
        t0 = half_period

    t0 %= period

    time = (time + half_period - t0) % period

    indices = np.argsort(time)
    time, flux = time[indices], flux[indices]

    return time, flux
