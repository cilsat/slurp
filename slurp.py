#!/usr/bin/python

import sys, os
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, SmoothBivariateSpline
from scipy.spatial.distance import cdist
from numpy.linalg import norm

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

def getBores(file='data/Imod Jakarta/Boreholes_Jakarta.ipf'):
    # read data
    path = os.path.dirname(file)
    raw = open(file).read().split('\n')[10:-1]

    wells = []
    layers = []
    for r in raw:
        x, y, name, top, btm = r.split(',')[:5]
        wname = name.split('\\')[-1]
        wells.append([wname, float(x), float(y)])

        well = open(os.path.join(path, name.replace('\\','/')+'.txt')).read().split('\n')[4:-1]
        for layer in well:
            depth, soil = layer.split(',')
            layers.append([wname, float(depth), soil])

    wells = pd.DataFrame(wells, columns=['name', 'x', 'y']).set_index('name')
    layers = pd.DataFrame(layers, columns=['name', 'top', 'soil']).set_index('name')
    df = pd.concat((wells, layers), axis=1, join='inner')
    # map soil types to aquafer/tard
    df['fer'] = df.soil.str.startswith('sand').astype(np.int8)
    # get depth for all aquafer layers
    dfg = df.groupby(df.index)
    df['dep'] = dfg.top.transform(lambda x: x.diff())
    # concatenate adjacent layers of the same type
    df['lay'] = dfg.fer.transform(lambda x: (x.diff(1).abs() == 1).cumsum())
    # get centers and radius of each layer
    points = df.loc[df.fer == 1, ['lay','x','y','top','dep']]
    pg = points.groupby([points.index, points.lay])
    points['z'] = pg.top.transform(lambda x: x.mean())
    points['r'] = pg.dep.transform(lambda x: 0.5*np.abs(x.sum()))
    print('points:')
    print(points.head())
    points = points.groupby([points.index, points.lay])['x','y','z','r'].first()

    return df, points

def getGroupies(dfp, grad=1.0, f=10):
    p = dfp.copy()
    xy = p[['x','y']]
    z = p.z.values
    r = p.r.values

    # xy distance
    dxy = cdist(xy.values, xy.values)
    # z difference
    dz = np.subtract.outer(z, z.T)
    # sum of radii
    sr = np.add.outer(r, r.T)
    # gradient between points
    gxyz = dz/dxy
    # point pairs for which gradient less than max specified and distance less
    # than sum of radii times some multiple
    d = (np.abs(gxyz) < grad)*(dxy < f*sr)
    np.fill_diagonal(d, False)
    wi = np.argwhere(np.triu(d)).tolist()

    # dfs to find connected wells
    i = []
    while wi:
        stack = [wi[0]]
        count = [wi[0]]
        while stack:
            w = stack.pop()
            wi.remove(w)
            if len(wi) == 0: break
            d = ((w - np.array(wi)) == 0).T
            ns = np.argwhere(np.logical_xor(d[0], d[1])).flatten().tolist()
            for n in ns:
                if wi[n] not in stack:
                    stack.append(wi[n])
                    count.append(wi[n])
        i.append([count, list(set(np.array(count).flatten()))])

    return i

"""
def _getGroups(dfp, grad=2.0, f=10):
    # get xy coordinates of wells and calculate distances
    p = dfp.copy()
    p.x -= p.x.min()
    p.y -= p.y.min()
    pg = p.groupby(level=0)
    xy = pg[['x','y']].first()
    dxy = cdist(xy.values, xy.values)

    # get max radii of wells and calculate needed distance to overcome
    # constant 'f' is the multiplication factor of the radius
    r = f*pg.r.max().values
    rr = np.add.outer(r, r.T)

    # find wells that potentially intersect, remove redundant and self
    d = dxy < rr
    np.fill_diagonal(d, False)
    wi = np.argwhere(np.triu(d)).tolist()

    # dfs to find connected wells
    i = []
    while wi:
        stack = [wi[0]]
        count = [wi[0]]
        while stack:
            w = stack.pop()
            wi.remove(w)
            if len(wi) == 0: break
            d = ((w - np.array(wi)) == 0).T
            ns = np.argwhere(np.logical_xor(d[0], d[1])).flatten().tolist()
            for n in ns:
                if wi[n] not in stack:
                    stack.append(wi[n])
                    count.append(wi[n])
        i.append([count, list(set(np.array(count).flatten()))])

    # for each group, assume nodes connect at the largest aquafir
    dfgraph = []
    adj = {}
    for group in i:
        order, nodes = group
        dfnodes = p.loc[p.loc[xy.iloc[nodes].index.tolist()].groupby(level=0).r.idxmax()]
        dfnodes['lbl'] = nodes
        dfgraph.append(dfnodes)
        for i in nodes:
            adj[i] = []
        for n in order:
            a, b = n
            adj[a].append(b)
            adj[b].append(a)

    dfgraph = pd.concat(dfgraph)
    xy = dfgraph[['x','y']].values
    dxy = cdist(xy, xy)
    z = dfgraph.z.values
    dz = np.subtract.outer(z, z.T)
    gxyz = dz/dxy
    return dfgraph, adj
"""

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
