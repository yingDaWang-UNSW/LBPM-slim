import sys
import os

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFormLayout
from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtWidgets import QPushButton, QLabel, QLineEdit

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
        
        instfs = QHBoxLayout()
        self.instfn = QLineEdit()
        self.instfn.setReadOnly(True)
        instbtn = QPushButton("Select Folder")
        def oninstbtnpush():
            path = os.path.dirname(self.instfn.text())
            if not path:
                path = "~"
            dialog = QtWidgets.QFileDialog(self)
            dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
            dialog.setDirectory(path)
            dialog.setFileMode(QtWidgets.QFileDialog.Directory)
            dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
            if dialog.exec_() == QtWidgets.QFileDialog.Accepted:
                self.instfn.setText(dialog.selectedFiles()[0])
        
        instbtn.clicked.connect(oninstbtnpush)
        instfs.addWidget(self.instfn)
        instfs.addWidget(instbtn)
        form = QFormLayout()
        form.addRow("Install Location", instfs)
        mainLayout.addLayout(form)
        
 
        title = QLabel("Choose flow:", self) 
        title.setAlignment(QtCore.Qt.AlignCenter)
        mainLayout.addWidget(title)
        
        singlephase = QPushButton('Single Phase', self)
        singlePhaseWindow = SinglePhaseWindow(self)
        def openSinglePhaseFlow():
            inst = self.instfn.text()
            if(not inst):
                QMessageBox.about(self, "Missing data", "Please select an install location.")
                return
            singlePhaseWindow.openWindow(inst)
            singlePhaseWindow.show()
            self.hide()

        singlephase.clicked.connect(openSinglePhaseFlow)
        mainLayout.addWidget(singlephase)
        
        twophase = QPushButton('Two Phase', self)
        twoPhaseWindow = TwoPhaseWindow(self)
        
        def openTwoPhaseFlow():
            inst = self.instfn.text()
            if(not inst):
                QMessageBox.about(self, "Missing data", "Please select an install location.")
                return
            twoPhaseWindow.openWindow(inst)
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