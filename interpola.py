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
    gridxy = np.array(zip(np.ravel(gridx), np.ravel(gridy)))

    # hitung ellipsoid masing-masing
    for datum in data:
        height = datum['z1']-datum['z2']
        rv = float(height)/2
        mid = datum['z1']-rv
        rh = config['buffersize']*rv
        z = ellipsoid.calc_z_vectorized(datum['x'], datum['y'], rh, rv, gridxy)
        gridz = z.reshape(gridx.shape)
        zerosides(gridz)
        datum['surface_top'] = mid+gridz
        datum['surface_bottom'] = mid-gridz

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

def zerosides(surface):
    left = np.roll(surface, 1, axis=1)
    left[:, 0] = 0
    right = np.roll(surface, -1, axis=1)
    right[:, -1] = 0
    top = np.roll(surface, 1, axis=0)
    top[0, :] = 0
    bottom = np.roll(surface, -1, axis=0)
    bottom[-1, :] = 0

    # set 0 yang pinggiran
    nonnan = surface==surface
    surface[nonnan*(left!=left)] = 0
    surface[nonnan*(right!=right)] = 0
    surface[nonnan*(top!=top)] = 0
    surface[nonnan*(bottom!=bottom)] = 0

if __name__ == '__main__':
    interpola()
