import time
import random
import threading

class Writer:
    def __init__(self, folder):
        self.folder = folder

    def write(self, surface):
        name = '{}-{}'.format(time.time(), random.random())

        def write_one(path, z):
            time.sleep(1)
            print '\tTODO'

        threads = []
        for jenis in ['top', 'bottom']:
            path = '{}/{}-{}.asc'.format(self.folder, name, jenis)
            threads.append(threading.Thread(target=write_one, args=(path, surface[jenis])))

        [thread.start() for thread in threads]
        [thread.join() for thread in threads]

