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
        self.prepare()
        self.log(' Done\n')

        writers = []
        idx_nondf = set(self.p.index.unique())-set(self.df.index.unique())
        nondf = self.p.ix[idx_nondf]
        counter, total = 0, len(self.adj)+len(nondf)

        def write(surface):
            self.writer.write(surface)

        # grouping
        self.is_processed = [False]*(self.df['lbl'].max()+1)
        for i in self.adj:
            counter += 1
            self.log('Processing {} of {}...'.format(counter, total))
            if not self.is_processed[i]:
                surface_top = np.full(self.gridx_shape, np.nan)
                surface_bottom = np.full(self.gridx_shape, np.nan)
                self.grouping(i, surface_top, surface_bottom)

                thread = threading.Thread(target=write, args=({'top':surface_top, 'bottom':surface_bottom}, ))
                writers.append(thread)
                thread.start()
            self.log(' Done\n')

        # individual
        for i in range(0, len(nondf)):
            counter += 1
            self.log('Processing {} of {}...'.format(counter, total))

            item = nondf.iloc[i]
            gridz = self.make_gridz(item)
            surface_top = item['z']+gridz
            surface_bottom = item['z']-gridz

            thread = threading.Thread(target=write, args=({'top':surface_top, 'bottom':surface_bottom}, ))
            writers.append(thread)
            thread.start()

            self.log(' Done\n')

        # wait for ascii writer
        [thread.join() for thread in writers]

    def prepare(self):
        gutter = np.ceil(self.p['r'].max()*config['buffersize'])
        xmin = np.floor(self.p['x'].min())-gutter
        xmax = np.ceil(self.p['x'].max())+gutter
        ymin = np.floor(self.p['y'].min())-gutter
        ymax = np.ceil(self.p['y'].max())+gutter

        widthx = float(xmax-xmin)/config['blocksize']
        widthy = float(ymax-ymin)/config['blocksize']
        linx = np.linspace(xmin, xmax, widthx)
        liny = np.linspace(ymin, ymax, widthy)
        gridx, gridy = np.meshgrid(linx, liny)
        self.gridxy = np.array(zip(np.ravel(gridx), np.ravel(gridy)))
        self.gridx_shape = gridx.shape

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
        rv = item['r']
        rh = config['buffersize']*rv
        z = ellipsoid.calc_z(item['x'], item['y'], rh, rv, self.gridxy)
        gridz = z.reshape(self.gridx_shape)
        self.zerosides(gridz)
        return gridz

    def zerosides(self, surface):
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

def test():
    import sys
    import slurp
    from writer import Writer
    w, p = slurp.getBores()
    p = p[p['r']==p['r']] # remove points with r is NaN
    df, adj = slurp.getGroups(p, config['buffersize'])
    writer = Writer('folder/output')
    log = lambda message: sys.stdout.write(message)
    interpolator = Interpolator(p, df, adj, writer, log)
    return interpolator.interpolate()

if __name__ == '__main__':
    test()
