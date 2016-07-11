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

def showBores(wells):
    x, y, z1, z2 = np.array(wells).T
    ax = plt.gca(projection='3d')
    [ax.plot([i,i],[j,j],[k,h], c='black') for i,j,k,h in zip(x,y,z1,z2)]
    ax.scatter(x, y, z1, c='b')
    ax.scatter(x, y, z2, c='r')
    plt.show()

def showLayers(wells, lays_uniq):
    df = pd.concat((wells, lays_uniq), join='inner', axis=1)
    fer = df[df['type'] == 'fer'].loc[:, ['x','y','z1','z2']].get_values()
    tard = df[df['type'] == 'tard'].loc[:, ['x','y','z1','z2']].get_values()
    dots = wells.get_values()

    ax = plt.gca(projection='3d')
    [ax.plot([i,i],[j,j],[k,h], c='black') for i,j,k,h in tard]
    [ax.plot([i,i],[j,j],[k,h], c='orange') for i,j,k,h in fer]
    x, y, z1, z2 = dots.T
    ax.scatter(x, y, z1, c='b')
    ax.scatter(x, y, z2, c='r')
    plt.show()

def getBores(path='data/Imod Jakarta'):
    # read data
    raw = open(os.path.join(path, 'Boreholes_Jakarta.ipf')).read().split('\n')[10:-1]
    wel = []
    lay = []
    lay_uniq = []
    wid = []
    lid = []
    luid = []
    for r in raw:
        x, y, name, z1, z2 = r.split(',')[:5]
        wel.append([float(x), float(y), float(z1), float(z2)])
        nim = name.split('\\')[-1]
        wid.append(nim)
        d = open(os.path.join(path, name.replace('\\', '/')+'.txt')).read().split('\n')[4:-1]
        layer = []
        types = []
        for l in d:
            z, t = l.split(',')
            lay.append([float(z), t])
            layer.append(float(z))
            types.append(t)
            lid.append(nim)
        for n,t in enumerate(types):
            types[n] = "fer" if t != "clay" else "tard"
        s = 0
        for n in range(len(layer)):
            if n+1 < len(layer):
                if types[n+1] != types[n]:
                    lay_uniq.append([layer[n-s], layer[n+1], types[n]])
                    luid.append(nim)
                    s = 0
                else:
                    s += 1
            else:
                lay_uniq.append([layer[n-s], layer[n], types[n]])
                luid.append(nim)
        
    wells = pd.DataFrame(wel, index=wid, columns=['x', 'y', 'surface', 'bottom'])
    lays = pd.DataFrame(lay, index=lid, columns=['z', 'type'])
    lays_uniq = pd.DataFrame(lay_uniq, index=luid, columns=['z1', 'z2', 'type'])

    return wells, lays, lays_uniq

def getWells(filename):
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

def readConf(conf):
    try:
        f = open(conf)
    except:
        print("Could not find slurp.conf file: using default config")
    
    if f:
        f = f.read().split('\n')

def interpolasurface(data, ip='linear'):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    kiri, kanan = x.min(), x.max()
    bawah, atas = y.min(), y.max()

    if ip == 'linear':
        interpolator = LinearNDInterpolator(np.asarray([x, y]).T, z)
    elif ip == 'bispline':
        interpolator = SmoothBivariateSpline(
            x, y, z,
            kx=2, ky=2,
            bbox=[kiri, kanan, bawah, atas]
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    linx = np.linspace(kiri, kanan, 200)
    liny = np.linspace(bawah, atas, 200)
    gridx, gridy = np.meshgrid(linx, liny)
    linz = np.array([interpolator(ptx, pty) for ptx, pty in zip(np.ravel(gridx), np.ravel(gridy))])
    gridz = linz.reshape(gridx.shape)

    ax.scatter(x, y, z, c='r')
    ax.plot_surface(gridx, gridy, gridz)

    plt.show()

if __name__ == "__main__":
    data = getData(sys.argv[1])
    top = getLayers(data)
    # showScatterplot(data[:,0], data[:,1], data[:,2])
    interpolasurface(top, sys.argv[2])
