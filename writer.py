import time
import random

class Writer:
    def __init__(self, folder):
        self.folder = folder

    def write(self, surface):
        name = '{}-{}'.format(time.time(), random.random())
        for jenis in ['top', 'bottom']:
            path = '{}/{}-{}.asc'.format(self.folder, name, jenis)
            time.sleep(1)
            print '\tTODO write ascii'

