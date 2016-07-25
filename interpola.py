import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import ellipsoid

config = {
    'buffersize': 2, # diameter/height
    'blocksize': 0.25, # ascii block size

    # dimension
    'xmin': 0,
    'xmax': 50,
    'ymin': 0,
    'ymax': 50,
}

def interpola():
    data = [
        {'x': 20, 'y': 10, 'z1': 21, 'z2': 13},
        {'x': 30, 'y': 10, 'z1': 19, 'z2': 11},
        {'x': 40, 'y': 20, 'z1': 19, 'z2': 11},
        {'x': 40, 'y': 30, 'z1': 19, 'z2': 11},
        {'x': 30, 'y': 40, 'z1': 19, 'z2': 11},
        {'x': 20, 'y': 40, 'z1': 19, 'z2': 9},
        {'x': 10, 'y': 30, 'z1': 19, 'z2': 11},
        {'x': 10, 'y': 20, 'z1': 19, 'z2': 11},
    ]

    widthx = float(config['xmax']-config['xmin']) / config['blocksize']
    widthy = float(config['ymax']-config['ymin']) / config['blocksize']

    linx = np.linspace(config['xmin'], config['xmax'], widthx)
    liny = np.linspace(config['ymin'], config['ymax'], widthy)
    gridx, gridy = np.meshgrid(linx, liny)
    gridxy = zip(np.ravel(gridx), np.ravel(gridy))

    # hitung ellipsoid masing-masing
    for datum in data:
        height = datum['z1']-datum['z2']
        rv = float(height)/2
        mid = datum['z1']-rv
        rh = config['buffersize']*rv
        zs = np.array([ellipsoid.calc_z(datum['x'], datum['y'], rh, rv, x, y, config['blocksize']) for x, y in gridxy]) # lama di sini, harusnya bisa vectorized
        linz_top = mid+zs
        linz_bottom = mid-zs
        datum['surface_top'] = linz_top.reshape(gridx.shape)
        datum['surface_bottom'] = linz_bottom.reshape(gridx.shape)

    # interpolasi alias gabungin weh
    surface_top = np.full(gridx.shape, np.nan)
    surface_bottom = np.full(gridx.shape, np.nan)
    for datum in data:
        compara_top = np.asarray([surface_top, datum['surface_top']])
        compara_bottom = np.asarray([surface_bottom, datum['surface_bottom']])
        surface_top[:] = np.nanmax(compara_top, axis=0)
        surface_bottom[:] = np.nanmin(compara_bottom, axis=0)

    # habis ini ide gw sih surface_top sama surface_bottom dismoothing gitu semacam blurring pake konvolusi biar agak smooth beloknya

    ax = plt.gca(projection='3d')
    ax.plot_surface(gridx, gridy, surface_top)
    ax.plot_surface(gridx, gridy, surface_bottom)

    ax.set_zlim(0, 50) # supaya skalanya samar
    plt.show()

def nyoba():
    kiri, kanan = -15, 15
    rh, rv = 10, 5
    width = 200
    blocksize = float(kanan-kiri)/width

    ax = plt.gca(projection='3d')
    linx = np.linspace(kiri, kanan, width)
    liny = np.linspace(kiri, kanan, width)
    gridx, gridy = np.meshgrid(linx, liny)

    linz = np.array([ellipsoid.calc_z(1, 1, rh, rv, x, y, blocksize) for x, y in zip(np.ravel(gridx), np.ravel(gridy))])
    gridz = linz.reshape(gridx.shape)
    ax.plot_surface(gridx, gridy, gridz)

    linz_minus = np.array([-ellipsoid.calc_z(1, 1, rh, rv, x, y, blocksize) for x, y in zip(np.ravel(gridx), np.ravel(gridy))])
    gridz_minus = linz_minus.reshape(gridx.shape)
    ax.plot_surface(gridx, gridy, gridz_minus)

    ax.set_zlim(kiri, kanan) # supaya keliatan elips
    plt.show()

if __name__ == '__main__':
    interpola()
