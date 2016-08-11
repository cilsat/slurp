# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gooey.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(640, 480)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.log = QtGui.QTextBrowser(self.centralwidget)
        self.log.setObjectName(_fromUtf8("log"))
        self.gridLayout.addWidget(self.log, 0, 0, 1, 2)
        self.button_input = QtGui.QPushButton(self.centralwidget)
        self.button_input.setObjectName(_fromUtf8("button_input"))
        self.gridLayout.addWidget(self.button_input, 1, 0, 1, 1)
        self.button_screen = QtGui.QPushButton(self.centralwidget)
        self.button_screen.setObjectName(_fromUtf8("button_screen"))
        self.gridLayout.addWidget(self.button_screen, 2, 0, 1, 1)
        self.button_output = QtGui.QPushButton(self.centralwidget)
        self.button_output.setObjectName(_fromUtf8("button_output"))
        self.gridLayout.addWidget(self.button_output, 3, 0, 1, 1)
        self.button_run = QtGui.QPushButton(self.centralwidget)
        self.button_run.setObjectName(_fromUtf8("button_run"))
        self.button_run.setEnabled(False)
        self.gridLayout.addWidget(self.button_run, 4, 1, 1, 1, QtCore.Qt.AlignRight)
        self.text_input = QtGui.QLineEdit(self.centralwidget)
        self.text_input.setEnabled(False)
        self.text_input.setObjectName(_fromUtf8("text_input"))
        self.gridLayout.addWidget(self.text_input, 1, 1, 1, 1)
        self.text_screen = QtGui.QLineEdit(self.centralwidget)
        self.text_screen.setEnabled(False)
        self.text_screen.setObjectName(_fromUtf8("text_screen"))
        self.gridLayout.addWidget(self.text_screen, 2, 1, 1, 1)
        self.text_output = QtGui.QLineEdit(self.centralwidget)
        self.text_output.setEnabled(False)
        self.text_output.setObjectName(_fromUtf8("text_output"))
        self.gridLayout.addWidget(self.text_output, 3, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Slurp", None))
        self.button_input.setText(_translate("MainWindow", "Input Borehole", None))
        self.button_screen.setText(_translate("MainWindow", "Input Screen", None))
        self.button_output.setText(_translate("MainWindow", "Output Folder", None))
        self.button_run.setText(_translate("MainWindow", "Run", None))

