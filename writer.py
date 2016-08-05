import os
import time
import random
import numpy as np

from config import config

class Writer:
    def __init__(self, folder):
        self.folder = folder

    def write(self, ncols, nrows, xllcorner, yllcorner, surface):
        name = '{}-{}'.format(time.time(), random.random())

        header =  'ncols         {}\n'.format(ncols)
        header += 'nrows         {}\n'.format(nrows)
        header += 'xllcorner     {:.12g}\n'.format(xllcorner)
        header += 'yllcorner     {:.12g}\n'.format(yllcorner)
        header += 'cellsize      {:.12g}\n'.format(config['cellsize'])
        header += 'NODATA_value  {:.12g}'.format(config['nodata_value'])

        for jenis in ['top', 'bottom']:
            grid = surface[jenis]
            grid[grid!=grid] = config['nodata_value']
            path = os.path.join(self.folder, '{}-{}.asc'.format(name, jenis))
            with open(path, 'w') as f:
                np.savetxt(f, grid, fmt='%.6g', delimiter=' ', header=header, comments='')

