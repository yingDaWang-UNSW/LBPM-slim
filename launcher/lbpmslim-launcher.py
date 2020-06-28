import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QPushButton, QLabel

from LBPMWindow import LBPMWindow
from LBPMSinglePhaseWindow import SinglePhaseWindow
from LBPMTwoPhaseWindow import TwoPhaseWindow

     
class MainWindow(LBPMWindow):
    def __init__(self):
        LBPMWindow.__init__(self)
 
       
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   
 
        mainLayout = QVBoxLayout(self)     
        centralWidget.setLayout(mainLayout)  
 
        title = QLabel("Choose flow:", self) 
        title.setAlignment(QtCore.Qt.AlignCenter)
        mainLayout.addWidget(title)
        
        singlephase = QPushButton('Single Phase', self)
        singlePhaseWindow = SinglePhaseWindow(self)
        def openSinglePhaseFlow():
            singlePhaseWindow.openWindow()
            singlePhaseWindow.show()
            self.hide()

        singlephase.clicked.connect(openSinglePhaseFlow)
        mainLayout.addWidget(singlephase)
        
        twophase = QPushButton('Two Phase', self)
        twoPhaseWindow = TwoPhaseWindow(self)
        
        def openTwoPhaseFlow():
            twoPhaseWindow.openWindow()
            twoPhaseWindow.show()
            self.hide()

        twophase.clicked.connect(openTwoPhaseFlow)
        mainLayout.addWidget(twophase)
     
    
def run_app():
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    app.exec_()
    
    
if(__name__ == '__main__'):
    run_app()