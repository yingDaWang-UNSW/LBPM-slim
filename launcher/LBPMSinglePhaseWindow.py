import scipy.io
import numpy as np
import os
import subprocess

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QObject
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFormLayout
from PyQt5.QtWidgets import QWidget, QLabel
from PyQt5.QtWidgets import QLineEdit, QPushButton
from PyQt5.QtGui import QDoubleValidator

from LBPMWindow import LBPMWindow, IntLineEdit
import removeDisconnections as rd
import runLBPMSinglePhase as rls

class SinglePhaseWindow(LBPMWindow):
    def __init__(self, backWindow):
        LBPMWindow.__init__(self)
        self.backWindow = backWindow
        

         
    def openWindow(self, installLocation):
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)
        self.installLocation = installLocation
        fullLayout = QVBoxLayout(self)
        mainLayout = QHBoxLayout(self)
        centralWidget.setLayout(fullLayout)
        form = QFormLayout()
        fullLayout.addLayout(mainLayout)
        mainLayout.addLayout(form)
        
        domfs = QHBoxLayout()
        self.fsfn = QLineEdit()
        self.fsfn.setReadOnly(True)
        fsbtn = QPushButton("Select File")
        def onfsbtnpush():
            path = os.path.dirname(self.fsfn.text())
            if not path:
                path = "~"

            dialog = QtWidgets.QFileDialog(self)
            dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
            dialog.setDirectory(path)
            dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            if dialog.exec_() == QtWidgets.QFileDialog.Accepted:
                self.fsfn.setText(dialog.selectedFiles()[0])
        
        fsbtn.clicked.connect(onfsbtnpush)
        domfs.addWidget(self.fsfn)
        domfs.addWidget(fsbtn)
        form.addRow("Domain File", domfs)
        
        self.viif = IntLineEdit(10)
        self.viif.setText('1000000000')
        self.viif.setFixedWidth(100)
        form.addRow("Visualisation Interval", self.viif)
                
        self.aniif = IntLineEdit(10)
        self.aniif.setText('1000')
        self.aniif.setFixedWidth(100)
        form.addRow("Analysis Interval", self.aniif)
        
        self.ptif = QLineEdit()
        self.ptif.setValidator(QDoubleValidator())
        self.ptif.setFixedWidth(100)
        self.ptif.setText(str(1e-5))
        form.addRow("Perm Tolerance", self.ptif)
        
        self.tsif = IntLineEdit(10)
        self.tsif.setFixedWidth(100)
        form.addRow("Timesteps", self.tsif)
        
        self.npif = IntLineEdit()
        form.addRow("Number of Processors", self.npif)
        
        outfs = QHBoxLayout()
        self.ofsfn = QLineEdit()
        self.ofsfn.setReadOnly(True)
        ofsbtn = QPushButton("Select Folder")
        def onofsbtnpush():
            path = os.path.dirname(self.ofsfn.text())
            if not path:
                path = "~"
            dialog = QtWidgets.QFileDialog(self)
            dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
            dialog.setDirectory(path)
            dialog.setFileMode(QtWidgets.QFileDialog.Directory)
            dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
            if dialog.exec_() == QtWidgets.QFileDialog.Accepted:
                self.ofsfn.setText(dialog.selectedFiles()[0])
        
        ofsbtn.clicked.connect(onofsbtnpush)
        outfs.addWidget(self.ofsfn)
        outfs.addWidget(ofsbtn)
        form.addRow("Output Location", outfs)
        
        self.vxsif = QLineEdit()
        self.vxsif.setValidator(QDoubleValidator())
        self.vxsif.setFixedWidth(100)
        form.addRow("Voxel Size (microns)", self.vxsif)
        
        self.muif = QLineEdit()
        self.muif.setValidator(QDoubleValidator())
        self.muif.setFixedWidth(100)
        self.muif.setText("0.07")
        form.addRow("Viscosity", self.muif)
        
        form.addRow("Periodic Pressure Gradient", None)
        self.fxif = QLineEdit()
        self.fxif.setValidator(QDoubleValidator())
        self.fxif.setText("0")
        self.fxif.setFixedWidth(100)
        self.fyif = QLineEdit()
        self.fyif.setValidator(QDoubleValidator())
        self.fyif.setText("0")
        self.fyif.setFixedWidth(100)
        self.fzif = QLineEdit()
        self.fzif.setValidator(QDoubleValidator())
        self.fzif.setText("0")
        self.fzif.setFixedWidth(100)
        form.addRow("x:", self.fxif)
        form.addRow("y:", self.fyif)
        form.addRow("z:", self.fzif)
       
        self.fluxif = QLineEdit()
        self.fluxif.setValidator(QDoubleValidator())
        self.fluxif.setText("1")
        self.fluxif.setFixedWidth(100)
        form.addRow("Injection Rate (voxels/timestep)", self.fluxif)
        
        self.pinif = QLineEdit()
        self.pinif.setValidator(QDoubleValidator())
        self.pinif.setText("0")
        self.pinif.setFixedWidth(100)
        self.poutif = QLineEdit()
        self.poutif.setValidator(QDoubleValidator())
        self.poutif.setText("0.3333333333")
        self.poutif.setFixedWidth(100)
        form.addRow("Inlet Pressure", self.pinif)
        form.addRow("Outlet Pressure", self.poutif)
          
        btnlayout = QHBoxLayout()
        runbtn = QPushButton("Run Solver")
        runThread = QThread()
        observerThread = QThread()
        def on_run_clicked():
            #self.hide()
            #self.backWindow.hide()
            self.saveValues()
            runObj = SinglePhaseRunObj(self)
            runObj.moveToThread(runThread)
            runThread.started.connect(runObj.startRun)
            observerObj = SinglePhaseObserverObj(runObj)
            observerObj.moveToThread(observerThread)
            observerObj.finished.connect(observerThread.quit)
            observerObj.finished.connect(runThread.quit)
            runObj.subprocessStarted.connect(observerObj.setSubprocess)
            runObj.updated.connect(observerObj.updateWindowText)
            runThread.start()
            observerThread.start()
        
                    
        runbtn.clicked.connect(on_run_clicked)
        btnlayout.addWidget(runbtn)
        backbtn = QPushButton("Back")
        def on_back_clicked():
            self.backWindow.show()
            self.hide()
                
        backbtn.clicked.connect(on_back_clicked)
        btnlayout.addWidget(backbtn)
        fullLayout.addLayout(btnlayout)
        

    def saveValues(self):
        self.domainPath = self.fsfn.text()
        self.visInterval = int(self.viif.text())
        self.analysisInterval = int(self.aniif.text())
        self.permTolerance = float(self.ptif.text())
        self.timesteps = int(self.tsif.text())
        numproc = int(self.npif.text())
        self.npx = 1
        self.npy = 1
        self.npz = numproc
        self.outpath = self.ofsfn.text()
        self.voxelSize = float(self.vxsif.text())
        self.mu = float(self.muif.text())
        self.fx = float(self.fxif.text())
        self.fy = float(self.fyif.text())
        self.fz = float(self.fzif.text())
        self.flux = float(self.fluxif.text())
        self.pin = float(self.pinif.text())
        self.pout = float(self.poutif.text())
        

class SinglePhaseStatusWindow(LBPMWindow):
    terminated = pyqtSignal()
    
    def __init__(self, observerObject):
        LBPMWindow.__init__(self)
        self.observerObject = observerObject
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   
 
        mainLayout = QVBoxLayout(self)     
        centralWidget.setLayout(mainLayout)
        self.status = QLabel("Running...", self) 
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        mainLayout.addWidget(self.status)
        
        abortbtn = QPushButton('Abort', self)
        def abort():
            self.terminated.emit()
        abortbtn.clicked.connect(abort)
        mainLayout.addWidget(abortbtn)
        return
    
    def updateText(self, text):
        self.status.setText(text)
        

class SinglePhaseRunObj(QObject):
    subprocessStarted = pyqtSignal(subprocess.Popen)
    updated = pyqtSignal(str)
    
    def __init__(self, window):
        QObject.__init__(self)
        self.window = window
        
    
    def startRun(self):
        print("Run Object run called")
        self.updated.emit("Reading domain file...")
        mat = scipy.io.loadmat(self.window.domainPath)
        keyname = ''
        for key in list(mat.keys()):
            if(not key.startswith('__') and not key.endswith('__')):
                keyname = key
                break
            
        image = mat[keyname]#[:200,:200,:200]
        image = np.array(image).astype(bool)
        self.updated.emit("Removing disconnections...")
        domain = rd.removeDisconnections(image)
        domain[:,:,0] = True	
        domain[:,:,-1] = True	
        domain[:,0,:] = True	
        domain[:,-1,:] = True
        
        #simulation parameters
        visInterval=self.window.visInterval 
        restartFq=False;
        analysisInterval = self.window.analysisInterval;
        permTolerance = self.window.permTolerance;
        terminal = True;
        timesteps = self.window.timesteps;
        gpuIDs=[];
        npx=self.window.npx;
        npy=self.window.npy;
        npz=self.window.npz;
        
        targetdir = self.window.outpath
        if(not os.path.exists(targetdir)):
            os.mkdir(targetdir)
        os.chdir(targetdir)
        
        #physical parameters
        voxelSize=self.window.voxelSize;
        
        mu=self.window.mu; 
        Fx = self.window.fx;
        Fy = self.window.fy;
        Fz = self.window.fz;
        flux = self.window.flux;
        Pin = self.window.pin;
        Pout = self.window.pout;
        
        install = self.window.installLocation
        
        rls.runLBPMSinglePhase(self, domain, targetdir, npx, npy, npz, voxelSize, 
                               timesteps, gpuIDs, Fx, Fy, Fz, flux, Pin, Pout, 
                               mu, restartFq, visInterval, analysisInterval, permTolerance, 
                               terminal, install)




class SinglePhaseObserverObj(QObject):
    finished = pyqtSignal()
    
    @pyqtSlot(subprocess.Popen)
    def setSubprocess(self, subprocess):
        self.subprocess = subprocess
    
    @pyqtSlot()
    def terminateRunThread(self):
        if(self.subprocess):
            os.killpg(os.getpgid(self.subprocess.pid), 15)
        print("Aborted.")
        self.finished.emit()
        self.statusWindow.close()
    
    @pyqtSlot(str)
    def updateWindowText(self, text):
        self.statusWindow.updateText(text)
        self.statusWindow.show()
    
    
    def __init__(self, runObject):
        QtCore.QThread.__init__(self)
        self.runObject = runObject
        self.subprocess = None
        self.statusWindow = SinglePhaseStatusWindow(self)
        self.statusWindow.terminated.connect(self.terminateRunThread)
        self.statusWindow.show()
        
        
        
        

                
        
    
