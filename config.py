import ConfigParser
from os import path

config = {
    'buffersize': 10, # diameter/height
    'gradient': 1,
    'cellsize': 25, # ascii cell size
    'nodata_value': -9999, # for NODATA_value in ascii file
}

def parse():
    parser = ConfigParser.RawConfigParser(allow_no_value=False)
    file = path.join(path.dirname(path.abspath(__file__)), '..', 'config.txt')
    with open(file, 'r') as f:
        parser.readfp(f)
    for jenis in ['bore_buff', 'screen_buff', 'gradient', 'cellsize', 'nodata_value']:
        config[jenis] = parser.getfloat('config', jenis)
    config['soil'] = {}
    for soil in parser.get('config', 'fer').split(', '):
        config['soil'][soil] = 'fer'
    for soil in parser.get('config', 'tar').split(', '):
        config['soil'][soil] = 'tar'
