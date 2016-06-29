#!/usr/bin/python

import sys, os
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, SmoothBivariateSpline

def showScatterplot(x, y, z):
    ax = plt.figure().gca(projection='3d')
    ax.scatter(x, y, z)
    plt.show()

def getData(filename):
    data = open(filename).read().replace('\r','').split('\n')
    data = [d for d in data if len(d.split(',')) == 5]

    i = -1
    ids = []
    wells = []
    rows = []
    for d in data:
        x,y,q,z1,z2 = d.split(',')
        rows.append([float(x), float(y), float(z1), float(z2)])
        if [x,y] not in wells:
            wells.append([x,y])
            i += 1
        ids.append(i)
    data = np.array(rows)
    #data[:,:2] -= [data[:,0].min(), data[:,1].min()]

    df = pd.DataFrame(data, index=ids, columns=['x', 'y', 'z1', 'z2'])
    df.index.name = 'wid'
    df.drop_duplicates(inplace=True)
    return df

def getLayers(df):
    top = []
    for n in range(df.index[-1]+1):
        first = df.loc[n].get_values()
        if first.ndim > 1:
            top.append(first[np.argmax(first[:,2]),:3])
        else:
            top.append(first[:3])
    top = np.array(top)
    return top

def interpolasurface(data, ip='linear'):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    kiri, kanan = x.min(), x.max()
    bawah, atas = y.min(), y.max()

    if ip == 'linear':
        interpolator = LinearNDInterpolator(np.asarray([x, y]).T, z)
    elif ip == 'bispline':
        interpolator = SmoothBivariateSpline(
            x, y, z,
            kx=1, ky=1, bbox=[kiri, kanan, bawah, atas]
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    linx = np.linspace(kiri, kanan, 200)
    liny = np.linspace(bawah, atas, 200)
    gridx, gridy = np.meshgrid(linx, liny)
    linz = np.array([interpolator(ptx, pty) for ptx, pty in zip(np.ravel(gridx), np.ravel(gridy))])
    gridz = linz.reshape(gridx.shape)

    ax.plot_surface(gridx, gridy, gridz)
    ax.scatter(x, y, z, c='r')

    plt.show()

if __name__ == "__main__":
    data = getData(sys.argv[1])
    top = getLayers(data)
    # showScatterplot(data[:,0], data[:,1], data[:,2])
    interpolasurface(top, sys.argv[2])
