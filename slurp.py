#!/usr/bin/python

import sys, os
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, SmoothBivariateSpline
from scipy.spatial.distance import cdist
from numpy.linalg import norm

def show_scatterplot(x, y, z):
    ax = plt.figure().gca(projection='3d')
    ax.scatter(x, y, z)
    plt.show()

def show_bores(wells):
    x, y, z1, z2 = np.array(wells).T
    ax = plt.gca(projection='3d')
    [ax.plot([i,i],[j,j],[k,h], c='black') for i,j,k,h in zip(x,y,z1,z2)]
    ax.scatter(x, y, z1, c='b')
    ax.scatter(x, y, z2, c='r')
    plt.show()

def show_layers(wells, lays_uniq):
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

def get_screens(file='data/wells_M_z_known.ipf'):
    df = pd.read_csv(file, delimiter=',', skiprows=7, header=0, names=['x','y','q','z1','z2'], usecols=[0,1,3,4])
    df['name'] = (df.x.astype(str)+df.y.astype(str)).astype(int)
    df.set_index('name', inplace=True)

    # drop duplicate entries
    df.drop_duplicates(inplace=True)
    # we define 'layers' by their centre points 'z' and thickness 'r'
    # radius r
    df['r'] = 0.5*(df.z1 - df.z2).abs()
    # centre z
    df['z'] = 0.5*(df.z1 + df.z2)
    # drop negative z entries: probably some kind of data entry error
    #df.drop(df.z < 0, inplace=True)
    df.drop(['z1','z2'], axis=1, inplace=True)

    # method to merge overlapping layers in a given well
    # overlapping layers are averaged w.r.t. depth and thickness
    def merge_layers(dfc):
        r = dfc.r.values
        z = dfc.z.values
        rr = np.add.outer(r, r.T)
        zz = np.abs(np.subtract.outer(z, z.T))
        # 2 layers are said to overlap iff the sum of their radii is larger
        # than the distance between their centres
        fg = (zz < rr)
        np.fill_diagonal(fg, False)
        # merge overlapping layers into groups: a single well may have layers
        # that overlap at different depths, not just one.
        g = np.argwhere(np.triu(fg)).tolist()
        i = []
        while g:
            stack = [g[0]]
            count = [g[0]]
            while stack:
                w = stack.pop()
                g.remove(w)
                if len(g) == 0: break
                d = ((w - np.array(g)) == 0).T
                ns = np.argwhere(np.logical_xor(d[0], d[1])).flatten().tolist()
                for n in ns:
                    if g[n] not in stack:
                        stack.append(g[n])
                        count.append(g[n])
            i.append(list(set(np.array(count).flatten())))
        xy = [dfc.iloc[0,0], dfc.iloc[0,1]]
        joint = list(set(np.array(i).flatten()))
        rz = np.dstack((r,z))[0]
        rzl = []
        [rzl.append(xy + np.mean(rz[idx], axis=0).tolist()) for idx in i]
        [rzl.append(xy + rn) for n, rn in enumerate(rz.tolist()) if n not in joint]
        dfr = pd.DataFrame(rzl, columns=['x', 'y', 'r','z'], index=[dfc.index[0]]*len(rzl))
        return dfr

    # loop through the unique wells: if there is more than one screen there may
    # be overlapping layers
    idx = df.index.unique()
    dfn = []
    for d in df.index.unique():
        dfc = df.loc[d]
        if dfc.ndim > 1:
            dfn.append(merge_layers(dfc))
            df.drop(d, inplace=True)
            
    dfn = pd.concat((df, pd.concat(dfn)))
    dfn['lay'] = dfn.groupby(dfn.index).z.transform(lambda x: (x.diff() != 0).cumsum() - 1)
    return dfn

def get_bores(file='data/Imod Jakarta/Boreholes_Jakarta.ipf', soilmap=None):
    # read data
    df = pd.read_csv(file, delimiter=',', skiprows=10, header=0, names=['x','y','name'], usecols=[0,1,2]).set_index('name')
    boredir = df.index[0].split('\\')[0]
    df.index = df.index.str[len(boredir)+1:]
    path = os.path.join(os.path.dirname(file), boredir)

    layers = []
    for name in df.index:
        well = open(os.path.join(path, name+'.txt')).read().split('\n')[4:-1]
        for layer in well:
            depth, soil = layer.split(',')
            layers.append([name.split('/')[-1], float(depth), soil])

    layers = pd.DataFrame(layers, columns=['name', 'top', 'soil']).set_index('name')
    df = pd.concat((df, layers), axis=1, join='inner')
    # map soil types to aquafer/tard
    df['fer'] = (df.soil.map(soilmap) == 'fer').astype(np.int8)
    # get depth for all aquafer layers
    dfg = df.groupby(df.index)
    df['dep'] = dfg.top.transform(lambda x: x.diff())
    # concatenate adjacent layers of the same type
    df['lay'] = dfg.fer.transform(lambda x: (x.diff().abs() == 1).cumsum())
    print(df.head())
    # get centers and radius of each layer
    points = df.loc[df.fer == 1, ['lay','x','y','top','dep']]
    pg = points.groupby([points.index, points.lay])
    points['z'] = pg.top.transform(lambda x: x.mean())
    points['r'] = pg.dep.transform(lambda x: 0.5*np.abs(x.sum()))
    print('points:')
    print(points.head())
    points.dropna(inplace=True)
    points = points.groupby([points.index, points.lay])['x','y','z','r'].first()

    return df, points

def get_groupies(dfp, grad=1.0, f=10):
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

if __name__ == "__main__":
    data = getData(sys.argv[1])
    top = getLayers(data)
    # showScatterplot(data[:,0], data[:,1], data[:,2])
    interpolasurface(top, sys.argv[2])
