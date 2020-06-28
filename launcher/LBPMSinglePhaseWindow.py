import scipy.io
import numpy as np
import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFormLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLineEdit, QPushButton, QCheckBox
from PyQt5.QtGui import QDoubleValidator

from LBPMWindow import LBPMWindow, IntLineEdit
import removeDisconnections as rd
import runLBPMSinglePhase as rls

class SinglePhaseWindow(LBPMWindow):
    def __init__(self, backWindow):
        LBPMWindow.__init__(self)
        self.backWindow = backWindow
        
         
    def openWindow(self):
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)
        fullLayout = QVBoxLayout(self)
        mainLayout = QHBoxLayout(self)
        centralWidget.setLayout(fullLayout)
        form = QFormLayout()
        fullLayout.addLayout(mainLayout)
        mainLayout.addLayout(form)
        
        domfs = QHBoxLayout()
        fsfn = QLineEdit()
        fsfn.setReadOnly(True)
        fsbtn = QPushButton("Select File")
        domfs.addWidget(fsfn)
        domfs.addWidget(fsbtn)
        form.addRow("Domain File", domfs)
        
        viif = IntLineEdit(10)
        form.addRow("visInterval", viif)
        
        rfqcbx = QCheckBox()
        form.addRow("RestartFq", rfqcbx)
        
        aniif = IntLineEdit(10)
        form.addRow("Analysis Interval", aniif)
        
        ptif = QLineEdit()
        ptif.setValidator(QDoubleValidator())
        ptif.setFixedWidth(120)
        form.addRow("Perm Tolerance", ptif)
        
        tsif = IntLineEdit(10)
        tsif.setFixedWidth(120)
        form.addRow("Timesteps", tsif)
        
        form.addRow("Num Processors", None)
        npxif = IntLineEdit()
        npyif = IntLineEdit()
        npzif = IntLineEdit()
        form.addRow("x:", npxif)
        form.addRow("y:", npyif)
        form.addRow("z:", npzif)
        form.addRow(None, None)
        
        outfs = QHBoxLayout()
        ofsfn = QLineEdit()
        ofsfn.setReadOnly(True)
        ofsbtn = QPushButton("Select Folder")
        outfs.addWidget(ofsfn)
        outfs.addWidget(ofsbtn)
        form.addRow("Output Location", outfs)
        
        vxsif = IntLineEdit()
        form.addRow("Voxel Size", vxsif)
        
        muif = QLineEdit()
        muif.setValidator(QDoubleValidator())
        muif.setFixedWidth(120)
        form.addRow("Mu", muif)
        
        fxif = IntLineEdit()
        fyif = IntLineEdit()
        fzif = IntLineEdit()
        form.addRow("Fx", fxif)
        form.addRow("Fy", fyif)
        form.addRow("Fz", fzif)
       
        fluxif = IntLineEdit(10)
        fluxif.setFixedWidth(120)
        form.addRow("Flux", fluxif)
        
        pinif = QLineEdit()
        pinif.setValidator(QDoubleValidator())
        pinif.setFixedWidth(120)
        poutif = QLineEdit()
        poutif.setValidator(QDoubleValidator())
        poutif.setFixedWidth(120)
        form.addRow("Pin", pinif)
        form.addRow("Pout", poutif)
        
        rform = QFormLayout()
        mainLayout.addLayout(rform)
        
        bgkcbx = QCheckBox()
        rform.addRow("BGK Flag", bgkcbx)
        thermalcbx = QCheckBox()
        rform.addRow("Thermal", thermalcbx)
        vstcbx = QCheckBox()
        rform.addRow("Vis Tolerance", vstcbx)
        dfcif = QLineEdit()
        dfcif.setValidator(QDoubleValidator())
        dfcif.setFixedWidth(120)
        rform.addRow("Diff Coefficient", dfcif)
        
        Lif = QLineEdit()
        rform.addRow("L", Lif)
        
        rtif = QLineEdit()
        rform.addRow("ReadType", rtif)
        
        rvif = QLineEdit()
        rform.addRow("ReadValues", rvif)
        
        wvif = QLineEdit()
        rform.addRow("WriteValues", wvif)
        
        rstcbx = QCheckBox()
        rform.addRow("Restart", rstcbx)
        
        btnlayout = QHBoxLayout()
        runbtn = QPushButton("Run Solver")
        def on_run_clicked():
            self.hide()
            self.backWindow.hide()
            SinglePhase()
            QtWidgets.QApplication.quit()
            
        runbtn.clicked.connect(on_run_clicked)
        btnlayout.addWidget(runbtn)
        backbtn = QPushButton("Back")
        def on_back_clicked():
            self.backWindow.show()
            self.hide()
                
        backbtn.clicked.connect(on_back_clicked)
        btnlayout.addWidget(backbtn)
        fullLayout.addLayout(btnlayout)
        
def SinglePhase():
    mat = scipy.io.loadmat("bentheimer.mat")
    
    image = mat['bentheimer'][:200,:200,:200]
    image = np.array(image).astype(bool)
    
    domain = rd.removeDisconnections(image)
    
    #simulation parameters
    visInterval=1e4; 
    restartFq=False;
    analysisInterval = 1e3;
    permTolerance = 1e-5;
    terminal = False;
    timesteps = 1e6;
    gpuIDs=[];
    npx=3;
    npy=2;
    npz=4;
    
    targetdir = '/mnt/c/Users/THOMAS/Documents/Projects/Uni/lbpm-test'
    if(not os.path.exists(targetdir)):
        os.mkdir(targetdir)
    os.chdir(targetdir)
    
    #physical parameters
    voxelSize=1e-6;
    domain[0,:,:] = True
    domain[-1,:,:] = True
    domain[:,0,:] = True
    domain[:,-1,:] = True
    
    
    mu=1/15; 
    Fx = 0;
    Fy = 0;
    Fz = 0;
    flux = 100;
    Pin = 0;
    Pout = 1/3;
    
    rls.runLBPMSinglePhase(domain, targetdir, npx, npy, npz, voxelSize, timesteps, gpuIDs, Fx, Fy, Fz, flux, Pin, Pout, mu, restartFq, visInterval, analysisInterval, permTolerance, terminal)
 


