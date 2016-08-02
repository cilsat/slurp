import threading
import numpy as np

from config import config
import ellipsoid

class Interpolator:
    def __init__(self, p, df, adj, writer, log):
        self.p = p
        self.df = df
        self.adj = adj
        self.writer = writer
        self.log = log

    def interpolate(self):
        idx_nongroup = set(self.p.index.unique())-set(self.df.index.unique())
        nongroup = self.p.ix[idx_nongroup]
        groups = self.get_groups()
        counter, total = 0, len(groups)+len(nongroup)

        for group in groups:
            counter += 1
            self.log('Processing {} of {}...'.format(counter, total))

            dfg = self.df[self.df['lbl'].isin(group)]
            xmin, xmax = dfg['x'].min(), dfg['x'].max()
            ymin, ymax = dfg['y'].min(), dfg['y'].max()
            gutter = np.ceil(dfg['rh'].max()+(2*config['cellsize']))
            ncols, nrows, xllcorner, yllcorner, gridxy, shape = self.make_params(xmin, xmax, ymin, ymax, gutter)
            surface_top = np.full(shape, np.nan)
            surface_bottom = np.full(shape, np.nan)

            for i in range(0, len(dfg)):
                item = dfg.iloc[i]
                gridz = self.make_gridz(item, gridxy, shape)
                compara_top = np.asarray([surface_top, item['z']+gridz])
                compara_bottom = np.asarray([surface_bottom, item['z']-gridz])
                surface_top[:] = np.nanmax(compara_top, axis=0)
                surface_bottom[:] = np.nanmin(compara_bottom, axis=0)

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
            ncols, nrows, xllcorner, yllcorner, gridxy, shape = self.make_params(xmin, xmax, ymin, ymax, gutter)
            gridz = self.make_gridz(item, gridxy, shape)

            self.writer.write(ncols, nrows, xllcorner, yllcorner,
                              {'top':item['z']+gridz, 'bottom':item['z']-gridz})

            self.log(' Done\n')

    def get_groups(self):
        is_checked = [False]*(self.df['lbl'].max()+1)
        lbl_groups = []

        def get_groups_recursive(i, members):
            for j in self.adj[i]:
                if not is_checked[j]:
                    is_checked[j] = True
                    members.append(j)
                    get_groups_recursive(j, members)

        for i in self.adj:
            if not is_checked[i]:
                is_checked[i] = True
                members = [i]
                get_groups_recursive(i, members)
                lbl_groups.append(members)

        return lbl_groups

    def make_params(self, xmin, xmax, ymin, ymax, gutter):
        xllcorner, yllcorner = np.floor(xmin)-gutter, np.floor(ymin)-gutter
        xhrcorner, yhrcorner = np.ceil(xmax)+gutter, np.ceil(ymax)+gutter

        widthx = float(xhrcorner-xllcorner)/config['cellsize']
        widthy = float(yhrcorner-yllcorner)/config['cellsize']
        linx = np.linspace(xllcorner, xhrcorner, widthx)
        liny = np.linspace(yllcorner, yhrcorner, widthy)

        gridx, gridy = np.meshgrid(linx, liny)
        gridxy = np.array(zip(np.ravel(gridx), np.ravel(gridy)))

        return linx.shape[0], liny.shape[0], xllcorner, yllcorner, gridxy, gridx.shape

    def make_gridz(self, item, gridxy, shape):
        z = ellipsoid.calc_z(item['x'], item['y'], item['rh'], item['r'], gridxy)
        gridz = z.reshape(shape)

        left = np.roll(gridz, 1, axis=1)
        left[:, 0] = np.nan
        right = np.roll(gridz, -1, axis=1)
        right[:, -1] = np.nan
        top = np.roll(gridz, 1, axis=0)
        top[0, :] = np.nan
        bottom = np.roll(gridz, -1, axis=0)
        bottom[-1, :] = np.nan

        nans = gridz!=gridz

        gridz[nans*(left==left)] = 0
        gridz[nans*(right==right)] = 0
        gridz[nans*(top==top)] = 0
        gridz[nans*(bottom==bottom)] = 0

        return gridz

def test():
    import sys
    import slurp
    from writer import Writer

    w, p = slurp.getBores()
    p = p[p['r']==p['r']] # remove points with r is NaN
    p['rh'] = p['r']*config['buffersize'] # r horizontal

    # set minimum r horizontal
    rh_min = 1.6*config['cellsize']
    p.set_value(p['rh'] < rh_min, 'rh', rh_min)

    df, adj = slurp.getGroups(p, config['buffersize'])

    writer = Writer('data/sampah')
    log = lambda message: sys.stdout.write(message)
    interpolator = Interpolator(p, df, adj, writer, log)

    return interpolator.interpolate()

if __name__ == '__main__':
    test()
