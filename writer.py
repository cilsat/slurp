import os
import time
import random
import numpy as np

from config import config

class Writer:
    def __init__(self, folder):
        self.folder = folder

    def create_header(self, ncols, nrows, xllcorner, yllcorner):
        self.header =  'ncols         {}\r\n'.format(ncols)
        self.header += 'nrows         {}\r\n'.format(nrows)
        self.header += 'xllcorner     {}\r\n'.format(xllcorner)
        self.header += 'yllcorner     {}\r\n'.format(yllcorner)
        self.header += 'cellsize      {}\r\n'.format(config['cellsize'])
        self.header += 'NODATA_value  {}'.format(config['nodata_value'])

    def write(self, surface):
        name = '{}-{}'.format(time.time(), random.random())
        for jenis in ['top', 'bottom']:
            grid = surface[jenis]
            grid[grid!=grid] = config['nodata_value']
            path = os.path.join(self.folder, '{}-{}.asc'.format(name, jenis))
            with open(path, 'w') as f:
                np.savetxt(f, grid, fmt='%.6g', delimiter=' ', newline='\r\n', header=self.header, comments='')

