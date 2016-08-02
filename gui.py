import sys
import time
import traceback
from PyQt4 import QtGui
from PyQt4.QtCore import QThread, SIGNAL

import design
import config
import slurp
from interpolator import Interpolator
from writer import Writer

class ProgramRunner(QThread):
    def run(self):
        log = lambda message: self.emit(SIGNAL('logging(QString)'), message)
        writer = Writer(str(self.text_output))

        try:
            log('Reading files...')

            config.parse()

            w, p = slurp.getBores(str(self.text_input))
            p.dropna(inplace=True)
            p['rh'] = p['r']*config.config['buffersize'] # r horizontal

            # set minimum r horizontal
            rh_min = 1.6*config.config['cellsize']
            p.set_value(p['rh'] < rh_min, 'rh', rh_min)

            df, adj = slurp.getGroups(p, config.config['buffersize'])
            df['x'] += p['x'].min()
            df['y'] += p['y'].min()

            log(' Done\n')

            interpolator = Interpolator(p, df, adj, writer, log)
            interpolator.interpolate()

            log('\n[DONE]')
        except Exception as e:
            log('\n\n[ERROR] {}'.format(e))
            traceback.print_exc()

        self.emit(SIGNAL('finish_program()'))

class Window(QtGui.QMainWindow, design.Ui_MainWindow):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.button_input.clicked.connect(self.browse_input)
        self.button_output.clicked.connect(self.browse_output)
        self.button_run.clicked.connect(self.run_program)

    def browse_input(self):
        file = QtGui.QFileDialog.getOpenFileName(self, 'Open File', '', 'IPF (*.ipf)')
        if file:
            self.text_input.setText(file)
        self.enable_button_run()

    def browse_output(self):
        directory = QtGui.QFileDialog.getExistingDirectory(self, 'Folder Output', '')
        if directory:
            self.text_output.setText(directory)
        self.enable_button_run()

    def enable_button_run(self):
        if self.text_input.text() != '' and self.text_output.text() != '':
            self.button_run.setEnabled(True)
        else:
            self.button_run.setEnabled(False)

    def run_program(self):
        self.log.setText('')
        for button in [self.button_input, self.button_output, self.button_run]:
            button.setEnabled(False)

        self.program_runner = ProgramRunner()
        self.program_runner.text_input = self.text_input.text()
        self.program_runner.text_output = self.text_output.text()
        self.connect(self.program_runner, SIGNAL('logging(QString)'), self.logging)
        self.connect(self.program_runner, SIGNAL('finish_program()'), self.finish_program)
        self.program_runner.start()

    def logging(self, text):
        self.log.moveCursor(QtGui.QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QtGui.QTextCursor.End)

    def finish_program(self):
        for button in [self.button_input, self.button_output]:
            button.setEnabled(True)
        self.enable_button_run()

def main():
    app = QtGui.QApplication(sys.argv)
    form = Window()
    form.show()
    app.exec_()

if __name__ == '__main__':
    main()
