import sys
import time
from PyQt4 import QtGui
from PyQt4.QtCore import QThread, SIGNAL

import design

class ProgramRunner(QThread):

    def run(self):
        self.emit(SIGNAL('logging(QString)'), 'Reading files...')
        time.sleep(1)
        self.emit(SIGNAL('logging(QString)'), ' [DONE]\n')
        self.emit(SIGNAL('logging(QString)'), 'Processing...')
        time.sleep(3)
        self.emit(SIGNAL('logging(QString)'), ' [FAILED]\n')
        self.emit(SIGNAL('logging(QString)'), '\nERROR: dum dumdum dum\n')
        time.sleep(1)
        self.emit(SIGNAL('finish_program()'))

class Window(QtGui.QMainWindow, design.Ui_MainWindow):

    def __init__(self, parent=None):
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.button_input.clicked.connect(self.browse_input)
        self.button_output.clicked.connect(self.browse_output)
        self.button_run.clicked.connect(self.run_program)
        # self.button_run.setEnabled(True)

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
        if self.text_input.text()+self.text_output.text() != '':
            self.button_run.setEnabled(True)
        else:
            self.button_run.setEnabled(False)

    def run_program(self):
        self.log.setText('')
        for button in [self.button_input, self.button_output, self.button_run]:
            button.setEnabled(False)

        self.program_runner = ProgramRunner()
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
