import numpy as np


def random_partial_reverse(flux):
    """
    reverse parts of the flux randomly
    """
    res = np.copy(flux)
    start, end = np.random.randint(0, len(res), size=2)
    if start > end:
        start, end = end, start
    res[start:end] = res[start:end][::-1]
    return res


