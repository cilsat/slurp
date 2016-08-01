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
        self.log('Preparing data...')

        gutter = np.ceil(self.p['r'].max()*config['buffersize'])
        xmin, xmax = self.p['x'].min(), self.p['x'].max()
        ymin, ymax = self.p['y'].min(), self.p['y'].max()

        self.df['x'] += xmin
        self.df['y'] += xmax

        xllcorner = np.floor(xmin)-gutter
        yllcorner = np.floor(ymin)-gutter
        xhrcorner = np.ceil(xmax)+gutter
        yhrcorner = np.ceil(ymax)+gutter

        widthx = float(xhrcorner-xllcorner)/config['cellsize']
        widthy = float(yhrcorner-yllcorner)/config['cellsize']
        linx = np.linspace(xllcorner, xhrcorner, widthx)
        liny = np.linspace(yllcorner, yhrcorner, widthy)
        gridx, gridy = np.meshgrid(linx, liny)
        self.gridxy = np.array(zip(np.ravel(gridx), np.ravel(gridy)))
        self.gridx_shape = gridx.shape

        self.writer.create_header(linx.shape[0], liny.shape[0], xllcorner, yllcorner)

        self.log(' Done\n')

        writers = []
        idx_nondf = set(self.p.index.unique())-set(self.df.index.unique())
        nondf = self.p.ix[idx_nondf]
        counter, total = 0, len(self.adj)+len(nondf)

        def write(surface):
            self.writer.write(surface)

        # groups
        self.is_processed = [False]*(self.df['lbl'].max()+1)
        for i in self.adj:
            counter += 1
            self.log('Processing {} of {}...'.format(counter, total))
            if not self.is_processed[i]:
                surface_top = np.full(self.gridx_shape, np.nan)
                surface_bottom = np.full(self.gridx_shape, np.nan)
                self.grouping(i, surface_top, surface_bottom)

                thread = threading.Thread(target=write, args=({'top':surface_top, 'bottom':surface_bottom}, ))
                thread.start()
                writers.append(thread)
            self.log(' Done\n')

        # individuals
        for i in range(0, len(nondf)):
            counter += 1
            self.log('Processing {} of {}...'.format(counter, total))

            item = nondf.iloc[i]
            gridz = self.make_gridz(item)
            surface_top = item['z']+gridz
            surface_bottom = item['z']-gridz

            thread = threading.Thread(target=write, args=({'top':surface_top, 'bottom':surface_bottom}, ))
            thread.start()
            writers.append(thread)

            self.log(' Done\n')

        # wait for ascii writer
        self.log('Writing files...')
        [thread.join() for thread in writers]

    def grouping(self, i, surface_top, surface_bottom):
        self.is_processed[i] = True
        item = self.df[self.df['lbl']==i].iloc[0]
        gridz = self.make_gridz(item)

        compara_top = np.asarray([surface_top, item['z']+gridz])
        compara_bottom = np.asarray([surface_bottom, item['z']-gridz])
        surface_top[:] = np.nanmax(compara_top, axis=0)
        surface_bottom[:] = np.nanmin(compara_bottom, axis=0)

        for j in self.adj[i]:
            if not self.is_processed[j]:
                self.grouping(j, surface_top, surface_bottom)

    def make_gridz(self, item):
        z = ellipsoid.calc_z(item['x'], item['y'], item['rh'], item['r'], self.gridxy)
        gridz = z.reshape(self.gridx_shape)

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
