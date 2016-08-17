import math
import numpy as np

def calc_z(cx, cy, rh, rv, gridxy):
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
