import math

def calc_z(cx, cy, rh, rv, x, y, blocksize):
    """
    Keyword arguments:
    cx, cy -- center x, center y
    rh, rv -- radius horizontal, radius vertical
    x, y -- pos where to calculate z
    blocksize -- mesh grid block size
    """
    trux, truy = x-cx, y-cy
    x_2, y_2 = math.pow(trux, 2), math.pow(truy, 2)
    if rh-math.sqrt(math.pow(blocksize, 2)*2) < math.sqrt(x_2+y_2) <= rh: # idenya sih supaya yg pinggir banget pasti 0 tapi gak tau itungannya bener apa nggak
        return 0
    else:
        rh_2, rv_2 = math.pow(rh, 2), math.pow(rv, 2)
        z_2 = float(rh_2-x_2-y_2)*rv_2/rh_2
        if z_2 < 0:
            return float('nan')
        else:
            return math.sqrt(z_2)
