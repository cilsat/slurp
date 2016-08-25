import os
import re
import numpy as np

def read_asc(file):
    data = {'meta':{}, 'top':None, 'bottom':None}
    meta = data['meta']
    with open(file+'-top.asc', 'r') as f:
        asc = f.read()
    # read metadata
    lines = asc.replace('\r', '').split('\n')
    for line in lines:
        match = re.search('^([A-z_]+?)\s+?(.+?)$', line)
        if match:
            meta[match.group(1)] = float(match.group(2))
        else:
            break
    meta['xurcorner'] = meta['xllcorner']+(meta['cellsize']*meta['ncols'])
    meta['yurcorner'] = meta['yllcorner']+(meta['cellsize']*meta['nrows'])
    # read grid
    for jenis in ['top', 'bottom']:
        data[jenis] = np.loadtxt(file+'-'+jenis+'.asc', comments=[
            'ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'NODATA_value'
        ])
    return data

def get_value(meta, grid, x, y):
    if x < meta['xllcorner'] or x > meta['xurcorner']:
        return None
    if y < meta['yllcorner'] or y > meta['yurcorner']:
        return None
    tx = int(round((x-meta['xllcorner'])/meta['cellsize']))
    ty = int(round((y-meta['yllcorner'])/meta['cellsize']))
    value = grid[tx, ty]
    if value == meta['NODATA_value']:
        return None
    else:
        return value

def calc(points, directory):
    acqs = [[] for i in range(0, len(points))]
    files = set([])
    for index, filename in enumerate(sorted(os.listdir(directory))):
        files.add(re.search('^(.+?\-.+?)\-[A-z]+?\.', filename).group(1))
    for file in files:
        data = read_asc(os.path.join(directory, file))
        for i, p in enumerate(points):
            z1 = get_value(data['meta'], data['top'], p[0], p[1])
            z2 = get_value(data['meta'], data['bottom'], p[0], p[1])
            if z1 != None or z2 != None:
                acqs[i].append((z1, z2))
    print acqs

if __name__ == '__main__':
    calc([(2675,16237856),(699363,9312135),(45683475,345876)], 'data/2000')
