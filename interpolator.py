import numpy as np

from config import config
import ellipsoid
import slurp

class Interpolator:
    def __init__(self, path, writer):
        self.path = path
        self.writer = writer

    def interpolate(self):
        print 'preparing...'
        self.prepare()

        # grouping
        counter = 0
        self.is_processed = [False]*(self.df['lbl'].max()+1)
        for i in self.adj:
            if not self.is_processed[i]:
                counter += 1
                print 'group #{}'.format(counter)
                print '\tprocessing...'

                surface_top = np.full(self.gridx_shape, np.nan)
                surface_bottom = np.full(self.gridx_shape, np.nan)
                self.grouping(i, surface_top, surface_bottom)

                print '\twriting ascii files...'
                self.writer.write({'top':surface_top, 'bottom':surface_bottom})

        # individual
        # TODO

    def prepare(self):
        w, p = slurp.getBores(self.path)
        p = p[p['r']==p['r']] # remove points with r is NaN
        self.df, self.adj = slurp.getGroups(p, config['buffersize'])

        gutter = np.ceil(p['r'].max()*config['buffersize'])
        xmin = np.floor(p['x'].min())-gutter
        xmax = np.ceil(p['x'].max())+gutter
        ymin = np.floor(p['y'].min())-gutter
        ymax = np.ceil(p['y'].max())+gutter

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

        rv = item['r']
        rh = config['buffersize']*rv
        z = ellipsoid.calc_z(item['x'], item['y'], rh, rv, self.gridxy)
        gridz = z.reshape(self.gridx_shape)
        self.zerosides(gridz)

        compara_top = np.asarray([surface_top, item['z']+gridz])
        compara_bottom = np.asarray([surface_bottom, item['z']-gridz])
        surface_top[:] = np.nanmax(compara_top, axis=0)
        surface_bottom[:] = np.nanmin(compara_bottom, axis=0)

        for j in self.adj[i]:
            if not self.is_processed[j]:
                self.grouping(j, surface_top, surface_bottom)

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
    from writer import Writer
    writer = Writer('folder/output')
    interpolator = Interpolator(None, writer)
    interpolator.interpolate()

if __name__ == '__main__':
    test()
