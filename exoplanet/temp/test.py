from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from lightkurve import LightCurve
import logging
import asyncio
from config import *

LEVEL = logging.INFO

logging.basicConfig(level=LEVEL)


def gen_data():
    N = 55000
    x = np.linspace(-5, 5, N)
    y = 1 + np.random.randn(N) * 0.1 + np.arange(N) / N / 2
    mask = np.logical_and(x >= -0.5, x <= 0.5)
    n = len(x[mask])
    x_transit = x[mask]
    x_transit[x_transit > 0] *= -1
    transit = -1 + x_transit ** 2
    y[mask] = transit
    return x, y


a, b = gen_data()

idx = np.arange(1000)


async def sigma_clip(time, flux, sigma=3):
    mask = np.abs(flux - np.median(flux)) <= np.std(flux) * sigma
    func = interp1d(time[mask], flux[mask], fill_value='extrapolate')
    return time, func(time)


def m1(a, b):
    GlobalVars.register('loop', asyncio.get_event_loop())

    mask = np.logical_and(a >= -0.5, a <= 0.5)
    mask[0] = False
    mask[-1] = False
    fu = sigma_clip(a[mask], b[mask])
    other_a = a[~mask]
    other_b = b[~mask]

    func = interp1d(other_a, other_b, kind='linear', fill_value='extrapolate')
    interp_b = func(a[mask])

    flat_b = b.copy()
    flat_b[mask] = interp_b

    lc, trend_lc = LightCurve(a, flat_b).flatten(
        return_trend=True, window_length=7, polyorder=3, niters=10
    )
    prev = b[mask].copy()
    a, b[mask] = GlobalVars.get_var('loop').run_until_complete(fu)

    a, flat_b = lc.time, lc.flux
    flat_b[mask] = b[mask]
    print(np.all(prev == b[mask]))
    return a, flat_b, trend_lc


def m2(a, b):
    b_original = b.copy()
    mask = np.logical_and(a >= -0.5, a <= 0.5)
    tce_time = a[mask]
    tce = b[mask]
    func = interp1d(tce_time, tce)
    interp_flux = func(tce_time)
    b[mask] = interp_flux

    logging.info(np.all(b == b_original))
    lc, trend_lc = LightCurve(a, b).flatten(
        break_tolerance=40, window_length=401, return_trend=True
    )

    flat_b = lc.flux
    flat_b[mask] = tce
    return a, flat_b, trend_lc


a, flat_b, trend_lc = m1(a, b)


# plt.plot(trend_lc.time, trend_lc.flux)
# plt.show()
