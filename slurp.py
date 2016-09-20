#!/usr/bin/python

import sys, os
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, SmoothBivariateSpline
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from numpy.linalg import norm

def dfs_array(inp):
    src = inp.copy()
    ln = src.shape[0]
    out = []
    nr = 0
    while nr < ln:
        if not np.any(src[nr]):
            nr += 1
            continue
        coord, visit, stack = [], set(), [nr]
        while stack:
            x = stack.pop()
            if x not in visit:
                visit.add(x)
                y = list(set(np.nonzero(src[x])[0]) - set(stack))
                stack.extend(y)
                [coord.append((x, yt)) for yt in y]
                src[x,:] = False
                src[:,x] = False
        out.append([coord, list(visit)])
    return out

def first_nonzero(a):
    for n,l in enumerate(a):
        if np.any(l):
            return [n, np.argmax(l)]

def dfs(src):
    srcpy = np.triu(src)
    #g = np.argwhere(srcpy).tolist()
    out = []
    # global dfs loop
    while np.any(srcpy):
        first = first_nonzero(srcpy)
        stack = [first]
        count = [first]
        # main dfs loop
        while stack:
            # get first pair from stack and remove from global
            w = stack.pop()
            srcpy.itemset(tuple(w), False)
            if not np.any(srcpy): break
            # find all pairs connected to w
            ns = [[w[n[0]], n[1]] for n in np.argwhere(srcpy[w])]
            # push new pairs if not already in stack
            new = [n for n in ns if n not in stack]
            stack.extend(new)
            count.extend(new)
        out.append([count, list(set(np.array(count).flatten()))])
    return out

def get_screens(file='data/wells_M_z_known.ipf'):
    df = pd.read_csv(file, delimiter=',', skiprows=7, header=0, names=['x','y','q','z1','z2'], usecols=[0,1,3,4])
    df['name'] = (df.x.astype(str)+df.y.astype(str))
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
        i = dfs_array(fg)

        xy = [dfc.iloc[0,0], dfc.iloc[0,1]]
        joint = list(set([m for ix in i for m in ix[1]]))
        rz = np.dstack((r,z))[0]
        rzl = []
        [rzl.append(xy + np.mean(rz[idx], axis=0).tolist()) for _,idx in i]
        [rzl.append(xy + rn) for n, rn in enumerate(rz.tolist()) if n not in joint]
        dfr = pd.DataFrame(rzl, columns=['x', 'y', 'r','z'], index=[dfc.index[0]]*len(rzl))
        return dfr

    # loop through the unique wells: if there is more than one screen in a
    # particular well there may be overlapping layers
    idx = df.index.unique()
    dfn = []
    for d in df.index.unique():
        dfc = df.loc[d]
        if dfc.ndim > 1:
            dfn.append(merge_layers(dfc))
            df.drop(d, inplace=True)

    dfn = pd.concat((df, pd.concat(dfn)))
    dfn['lay'] = dfn.groupby(dfn.index).z.transform(lambda x: (x.diff() != 0).cumsum())
    return dfn.groupby([dfn.index, dfn.lay])[dfn.columns[:-1]].first()

def get_bores(file='data/Imod Jakarta/Boreholes_Jakarta.ipf', soilmap=None):
    # read data
    df = pd.read_csv(
        file,
        delimiter=',',
        skiprows=10,
        header=0,
        names=['x','y','path','name'],
        usecols=[0,1,2,5]
    ).set_index('name')

    data_path = os.path.dirname(file)

    layers = []
    for name, path in df.path.iteritems():
        well = open(os.path.join(data_path, path.replace('\\','/')+'.txt')).read().split('\n')[4:-1]
        for layer in well:
            depth, soil = layer.split(',')
            layers.append([name, float(depth), soil])

    layers = pd.DataFrame(layers, columns=['name', 'top', 'soil']).set_index('name')
    df = pd.concat((df, layers), axis=1, join='inner')
    # map soil types to aquafer/tard
    df['fer'] = (df.soil.map(soilmap) == 'fer').astype(np.int8)
    # get depth for all aquafer layers
    dfg = df.groupby(df.index)
    df['dep'] = dfg.top.transform(lambda x: x.diff())
    # concatenate adjacent layers of the same type
    df['lay'] = dfg.fer.transform(lambda x: (x.diff().abs() == 1).cumsum())
    # get centers and radius of each layer
    points = df.loc[df.fer == 1, ['lay','x','y','top','dep']]
    points['z'] = points.top + 0.5*points.dep
    pg = points.groupby([points.index, points.lay])
    points['z'] = pg.z.transform(lambda x: x.mean())
    points['r'] = pg.dep.transform(lambda x: 0.5*np.abs(x.sum()))
    points.dropna(inplace=True)
    points = points.groupby([points.index, points.lay])['x','y','z','r'].first()

    return points

def get_groupies(dfp, grad=1.0):
    xy = dfp[['x','y']].values
    z = dfp.z.values
    r = dfp.rh.values

    # xy distance
    dxy = cdist(xy, xy)
    #print(np.mean(dxy, axis=-1))
    # z difference
    dz = np.subtract.outer(z, z.T)
    # sum of radii
    sr = np.add.outer(r, r.T)
    # gradient between points
    gxyz = dz/dxy
    # point pairs for which gradient less than max specified and distance less
    # than sum of radii times some multiple
    d = (np.abs(gxyz) < grad)*(dxy < sr)
    np.fill_diagonal(d, False)

    # dfs to find connected wells
    adj = dfs_array(d)
    return adj

def prep(fbore='data/Imod Jakarta/Boreholes_Jakarta.ipf', fscreen='data/wells_M_z_known.ipf', config=None, log=lambda msg: sys.stdout.write(msg)):
    # read files
    log('Reading files... ')
    p = get_bores(file=fbore, soilmap=config.config['soil'])
    sp = get_screens(file=fscreen)
    # discard outliers
    #p.drop(p[p.r > (p.r.mean() + 2*p.r.std())].index, inplace=True)
    #sp.drop(sp[sp.r > (sp.r.mean() + 2*sp.r.std())].index, inplace=True)
    log('Done\n')

    # get buffered size of layers and join
    p['rh'] = p.r*config.config['bore_buff']
    sp['rh'] = sp.r*config.config['screen_buff']
    pp = pd.concat((p, sp))
    pp.sort_values(['x','y'], inplace=True)
    # log(pp.head().to_string())
    # log('\n')
    # log(pp.tail().to_string())
    log('\nPre-processing... ')
    #pp.drop(pp[pp.rh > 2000].index, inplace=True)
    rh_min = 1.6*config.config['cellsize']
    pp.loc[pp.rh < rh_min, 'rh'] = rh_min
    adj = get_groupies(pp, grad=config.config['gradient'])
    log('Done\n')
    return pp, adj

if __name__ == "__main__":
    from interpolator import Interpolator
    from writer import Writer
    import config

    config.parse()
    if len(sys.argv) > 1:
        df, adj = prep(fbore=sys.argv[1], fscreen=sys.argv[2], config=config)
        w = Writer(sys.argv[3])
        i = Interpolator(df, adj, w)
        i.interpolate()
    else:
        df, adj = prep(config=config)
        i = Interpolator(df, adj)
        i.interpolate()

