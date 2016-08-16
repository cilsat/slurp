import math
import numpy as np

def calc_z(cx, cy, rh, rv, gridxy):
    rh_2, rv_2 = rh*rh, rv*rv
    _gridxy = gridxy.astype(np.float, copy=True)
    xy = np.diff(_gridxy, axis=-1) - cx - cy
    _gridxy[:] = np.square(_gridxy)
    z_2 = (rh_2 - xy) * rv_2 / rh_2
    z_2[z_2<0] = np.nan
    z_2[:] = np.sqrt(z_2)

    return z_2
