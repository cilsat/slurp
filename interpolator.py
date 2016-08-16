import math
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from config import config
import ellipsoid
import sys
from writer import Writer

class Interpolator:
    def __init__(self, p, adj, writer=Writer('test'), log=lambda msg:sys.stdout.write(msg)):
        self.p = p
        self.adj = adj
        self.writer = writer
        self.log = log

    def interpolate(self):
        idx_groupies = [idx for group in self.adj for idx in group[1]]
        idx_nongroup = list(set(range(0, len(self.p)))-set(idx_groupies))
        nongroup = self.p.iloc[idx_nongroup]
        counter, total = 0, len(self.adj)+len(nongroup)

        for group in self.adj:
            counter += 1
            self.log('Processing {} of {}...'.format(counter, total))

            dfg = self.p.iloc[group[1]]
            xmin, xmax = dfg['x'].min(), dfg['x'].max()
            ymin, ymax = dfg['y'].min(), dfg['y'].max()
            gutter = np.ceil(dfg['rh'].max()+(2*config['cellsize']))
            ncols, nrows, xllcorner, yllcorner, gridx, gridy, gridxy = self.make_params(xmin, xmax, ymin, ymax, gutter)
            surface_top = np.full(gridx.shape, np.nan)
            surface_bottom = np.full(gridx.shape, np.nan)

            for i in range(0, len(dfg)):
                item = dfg.iloc[i]
                gridz = self.make_gridz(item, gridxy, gridx.shape)
                surface_top[:] = np.nanmax([surface_top, item['z']+gridz], axis=0)
                surface_bottom[:] = np.nanmin([surface_bottom, item['z']-gridz], axis=0)

            # interpolation
            surface_top[surface_top!=surface_top] = -999999
            surface_bottom[surface_bottom!=surface_bottom] = 999999
            #for m in range(0, len(group[1])):       # point to point...
            #    for n in range(m+1, len(group[1])): # ...exhaustive search
            #        p1, p2 = self.p.iloc[group[1][m]], self.p.iloc[group[1][n]]
            self.log('\n')
            for i1, i2 in group[0]:
                self.log('.')
                p1, p2 = self.p.iloc[i1], self.p.iloc[i2]
                angle = -math.atan2(p2['y']-p1['y'], p2['x']-p1['x'])
                points, values_top, values_bottom = [], [], []
                for p in [p1, p2]:
                    points.append([p['x'], p['y']])
                    values_top.append(p['z']+p['r'])
                    values_bottom.append(p['z']-p['r'])
                    dx, dy = p['rh']*math.sin(angle), p['rh']*math.cos(angle)
                    points.append([p['x']+dx, p['y']+dy])
                    values_top.append(p['z'])
                    values_bottom.append(p['z'])
                    points.append([p['x']-dx, p['y']-dy])
                    values_top.append(p['z'])
                    values_bottom.append(p['z'])
                try:
                    gridz_top = griddata(points, values_top, (gridx, gridy), method='cubic', fill_value=-999999)
                    gridz_bottom = griddata(points, values_bottom, (gridx, gridy), method='cubic', fill_value=999999)
                    surface_top[:] = np.nanmax([surface_top, gridz_top], axis=0)
                    surface_bottom[:] = np.nanmin([surface_bottom, gridz_bottom], axis=0)
                except:
                    qhul = 'error'
            surface_top[surface_top==-999999] = np.nan
            surface_bottom[surface_bottom==999999] = np.nan

            self.smoothing(surface_top, surface_bottom)
            self.writer.write(ncols, nrows, xllcorner, yllcorner,
                              {'top':surface_top, 'bottom':surface_bottom})

            self.log(' Done\n')

        for i in range(0, len(nongroup)):
            counter += 1
            self.log('Processing {} of {}...'.format(counter, total))

            item = nongroup.iloc[i]
            xmin = xmax = item['x']
            ymin = ymax = item['y']
            gutter = np.ceil(item['rh']+(2*config['cellsize']))
            ncols, nrows, xllcorner, yllcorner, gridx, gridy, gridxy = self.make_params(xmin, xmax, ymin, ymax, gutter)
            gridz = self.make_gridz(item, gridxy, gridx.shape)

            surface_top = item['z']+gridz
            surface_bottom = item['z']-gridz

            self.smoothing(surface_top, surface_bottom)
            self.writer.write(ncols, nrows, xllcorner, yllcorner,
                              {'top':surface_top, 'bottom':surface_bottom})

            self.log(' Done\n')

    def make_params(self, xmin, xmax, ymin, ymax, gutter):
        xllcorner, yllcorner = np.floor(xmin)-gutter, np.floor(ymin)-gutter
        xhrcorner, yhrcorner = np.ceil(xmax)+gutter, np.ceil(ymax)+gutter

        widthx = float(xhrcorner-xllcorner)/config['cellsize']
        widthy = float(yhrcorner-yllcorner)/config['cellsize']
        linx = np.linspace(xllcorner, xhrcorner, widthx)
        liny = np.linspace(yllcorner, yhrcorner, widthy)

        gridx, gridy = np.meshgrid(linx, liny)
        gridxy = np.array(zip(np.ravel(gridx), np.ravel(gridy)))

        return linx.shape[0], liny.shape[0], xllcorner, yllcorner, gridx, gridy, gridxy

    def make_gridz(self, item, gridxy, shape):
        z = ellipsoid.calc_z(item['x'], item['y'], item['rh'], item['r'], gridxy)
        gridz = z.reshape(shape)
        return gridz

    def smoothing(self, surface_top, surface_bottom):
        mean = np.mean([surface_top, surface_bottom], axis=0)

        left = np.roll(mean, 1, axis=1)
        right = np.roll(mean, -1, axis=1)
        top = np.roll(mean, 1, axis=0)
        bottom = np.roll(mean, -1, axis=0)
        topleft = np.roll(top, 1, axis=1)
        topright = np.roll(top, -1, axis=1)
        bottomleft = np.roll(bottom, 1, axis=1)
        bottomright = np.roll(bottom, -1, axis=1)

        nonnans = mean==mean
        for neighbor in [left, right, top, bottom]:
            select = nonnans*(neighbor!=neighbor)
            surface_top[select] = mean[select]
            surface_bottom[select] = mean[select]
        for neighbor in [topleft, topright, bottomleft, bottomright]:
            select = nonnans*(neighbor!=neighbor)
            surface_top[select] = mean[select]
            surface_bottom[select] = mean[select]

def test():
    import sys
    import slurp
    from writer import Writer
    from config import parse

    parse()

    w, p = slurp.getBores(soilmap=config['soil'])
    p.dropna(inplace=True)
    p['rh'] = p['r']*config['buffersize'] # r horizontal

    # set minimum r horizontal
    rh_min = 1.6*config['cellsize']
    p.set_value(p['rh'] < rh_min, 'rh', rh_min)
    adj = slurp.getGroupies(p, config['gradient'], config['buffersize'])

    writer = Writer('data/sampah')
    log = lambda message: sys.stdout.write(message)
    interpolator = Interpolator(p, adj, writer, log)

    return interpolator.interpolate()

if __name__ == '__main__':
    test()
