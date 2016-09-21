import os
import time
import random
import numpy as np

from config import config

class Writer:
    def __init__(self, folder):
        self.folder = folder
        self.reset()

    def write(self, ncols, nrows, xllcorner, yllcorner, surface):
        self.counter += 1
        name = '{}-{:0>4}'.format(self.prefix, self.counter)

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

    def reset(self):
        self.prefix = int(time.time())
        self.counter = 0
