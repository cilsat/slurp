import math
import numpy as np

def calc_z(cx, cy, rh, rv, x, y):
    """
    Keyword arguments:
    cx, cy -- center x, center y
    rh, rv -- radius horizontal, radius vertical
    x, y -- pos where to calculate z
    """
    x_2, y_2 = math.pow(x-cx, 2), math.pow(y-cy, 2)
    rh_2, rv_2 = math.pow(rh, 2), math.pow(rv, 2)
    z_2 = float(rh_2-x_2-y_2)*rv_2/rh_2
    if z_2 < 0:
        return float('nan')
    else:
        return math.sqrt(z_2)

def calc_z_vectorized(cx, cy, rh, rv, gridxy):
    rh_2, rv_2 = rh*rh, rv*rv

    _gridxy = gridxy.astype(np.float, copy=True)
    x, y = _gridxy[:, 0], _gridxy[:, 1]

    x -= cx
    y -= cy

    _gridxy[:] = np.square(_gridxy)
    z_2 = (rh_2 - x - y) * rv_2 / rh_2
    z_2[z_2<0] = np.nan
    z_2[:] = np.sqrt(z_2)

    return z_2
