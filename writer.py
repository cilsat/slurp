import os
import time
import random
import numpy as np

from config import config

class Writer:
    def __init__(self, folder):
        self.folder = folder

    def write(self, surface):
        name = '{}-{}'.format(time.time(), random.random())

        header =  'ncols         {}\r\n'.format(self.ncols)
        header += 'nrows         {}\r\n'.format(self.nrows)
        header += 'xllcorner     {}\r\n'.format(self.xllcorner)
        header += 'yllcorner     {}\r\n'.format(self.yllcorner)
        header += 'cellsize      {}\r\n'.format(config['cellsize'])
        header += 'NODATA_value  {}'.format(config['nodata_value'])

        for jenis in ['top', 'bottom']:
            grid = surface[jenis]
            grid[grid!=grid] = config['nodata_value']
            path = os.path.join(self.folder, '{}-{}.asc'.format(name, jenis))
            with open(path, 'w') as f:
                np.savetxt(f, surface['top'], fmt='%.6g', delimiter=' ', newline='\r\n', header=header, comments='')

