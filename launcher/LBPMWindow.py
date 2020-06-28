from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIntValidator

class IntLineEdit(QLineEdit):
    def __init__(self, maxLength=3):
        QLineEdit.__init__(self)
        self.setValidator(QIntValidator())
        self.setMaxLength(maxLength)
        self.setFixedWidth(40)

class LBPMWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(100, 100))
        self.setWindowTitle("LBPM-Slim") 
        
        menu = self.menuBar().addMenu('Menu')
        action = menu.addAction('Quit')
        action.triggered.connect(QtWidgets.QApplication.quit)
