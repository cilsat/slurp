import ConfigParser

config = {
    'buffersize': 10, # diameter/height
    'cellsize': 25, # ascii cell size
    'nodata_value': -9999, # for NODATA_value in ascii file
}

def parse():
    parser = ConfigParser.RawConfigParser(allow_no_value=False)
    with open('config.cfg', 'r') as f:
        parser.readfp(f)
    for jenis in ['buffersize', 'cellsize', 'nodata_value']:
        config[jenis] = parser.getfloat('config', jenis)
    config['soil'] = {}
    for soil in parser.get('config', 'sand').split(', '):
        config['soil'][soil] = 'sand'
    for soil in parser.get('config', 'clay').split(', '):
        config['soil'][soil] = 'clay'
