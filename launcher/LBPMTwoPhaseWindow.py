import scipy.io
import numpy as np
import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFormLayout
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLineEdit, QPushButton, QCheckBox, QComboBox
from PyQt5.QtGui import QDoubleValidator

from LBPMWindow import LBPMWindow, IntLineEdit
import removeDisconnections as rd
import runLBPMTwoPhase as rlt

class TwoPhaseWindow(LBPMWindow):
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
        rform = QFormLayout()
        fullLayout.addLayout(mainLayout)
        mainLayout.addLayout(form)
        mainLayout.addLayout(rform)
        
        domfs = QHBoxLayout()
        fsfn = QLineEdit()
        fsfn.setReadOnly(True)
        fsbtn = QPushButton("Select File")
        domfs.addWidget(fsfn)
        domfs.addWidget(fsbtn)
        form.addRow("Domain File", domfs)
        
        hpccbx = QCheckBox()
        form.addRow("HPC Flag", hpccbx)
        
        rfqcbx = QCheckBox()
        form.addRow("Restart", rfqcbx)
        
        tsif = IntLineEdit(10)
        form.addRow("Timesteps", tsif)
        
        viif = IntLineEdit(10)
        form.addRow("visInterval", viif)
        
        aniif = IntLineEdit(10)
        form.addRow("Analysis Interval", aniif)
        
        ptif = QLineEdit()
        ptif.setValidator(QDoubleValidator())
        form.addRow("Perm Tolerance", ptif)
        
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
        rform.addRow("Output Location", outfs)
        
        vxsif = IntLineEdit()
        form.addRow("Voxel Size", vxsif)
        
        muaif = QLineEdit()
        muaif.setValidator(QDoubleValidator())
        rform.addRow("Mu A", muaif)
        mubif = QLineEdit()
        mubif.setValidator(QDoubleValidator())
        rform.addRow("Mu B", mubif)
        rhoaif = QLineEdit()
        rhoaif.setValidator(QDoubleValidator())
        rform.addRow("Rho A", rhoaif)
        rhobif = QLineEdit()
        rhobif.setValidator(QDoubleValidator())
        rform.addRow("Rho B", rhobif)
        alphif = QLineEdit()
        alphif.setValidator(QDoubleValidator())
        rform.addRow("Alpha", alphif)
        betaif = QLineEdit()
        betaif.setValidator(QDoubleValidator())
        rform.addRow("Beta", betaif)
        
        fxif = IntLineEdit()
        fyif = IntLineEdit()
        fzif = IntLineEdit()
        form.addRow("Fx", fxif)
        form.addRow("Fy", fyif)
        form.addRow("Fz", fzif)
       
        fluxif = IntLineEdit(10)
        form.addRow("Flux", fluxif)
        
        pinif = QLineEdit()
        pinif.setValidator(QDoubleValidator())
        poutif = QLineEdit()
        poutif.setValidator(QDoubleValidator())
        form.addRow("Pin", pinif)
        form.addRow("Pout", poutif)
        
        simcbb = QComboBox()
        simcbb.addItem("colour")
        simcbb.addItem("dfh")
        form.addRow("Simulation Type", simcbb)
        
        limswif = IntLineEdit()
        rform.addRow("LimSW", limswif)
        injif = IntLineEdit()
        rform.addRow("InjType", injif)
        amphcbx = QCheckBox()
        rform.addRow("Automorph", amphcbx)
        smphcbx = QCheckBox()
        rform.addRow("Spinomorph", smphcbx)
        fmphcbx = QCheckBox()
        rform.addRow("Flux Morph", fmphcbx)
        cinjcbx = QCheckBox()
        rform.addRow("Coinjection", cinjcbx)
        
        
        
        
        btnlayout = QHBoxLayout()
        runbtn = QPushButton("Run Solver")
        def on_run_clicked():
            self.hide()
            self.backWindow.hide()
            TwoPhase()
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

   
def TwoPhase():
    mat = scipy.io.loadmat("bentheimer.mat")
    
    image = mat['bentheimer'][:200,:200,:200]
    image = np.array(image).astype(bool)
    
    domain = rd.removeDisconnections(image)
    
    HPCFlag = False
    restart = False
    timesteps = 1e6
    visInterval = 1e4
    analysisInterval = 1e3
    permTolerance = 1e-5
    terminal = False
    gpuIDs = [0, 1, 2]
    npx = 2
    npy = 2
    npz = 3
    
    voxelSize = 1e-6
    
    muA = 1/15
    muB = 1/15
    rhoA = 1
    rhoB = 1
    alpha = 1e-3
    beta = 0.95
    
    Fx = 0
    Fy = 0
    Fz = 0
    flux = 50
    Pin = 0
    Pout = 1/3
    simType = 'colour'
    
    inputIDs=[6, 0, 1, 2, 3, 4, 5]
    readIDs=[1, 2, 0, -1, -2, -3, -4]
    solidIDs=[0, -1, -2, -3, -4]
    contactAngles=[0.34, -0.5, -0.34, 0, 0]
    
    targetdir = '/mnt/c/Users/THOMAS/Documents/Projects/Uni/lbpm-test'
    if(not os.path.exists(targetdir)):
        os.mkdir(targetdir)
    os.chdir(targetdir)
    
    rlt.runLBPMTwoPhase(domain, targetdir, npx, npy, npz,
    voxelSize, timesteps, gpuIDs, simType,
    Fx, Fy, Fz, flux, Pin, Pout, muA, muB, rhoA, rhoB, alpha, beta,
    inputIDs, readIDs, solidIDs, contactAngles,
    restart, visInterval, analysisInterval, permTolerance, terminal, HPCFlag)