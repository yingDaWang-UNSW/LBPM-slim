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
        
        self.tsif = IntLineEdit(10)
        self.tsif.setFixedWidth(100)
        form.addRow("Timesteps", self.tsif)
        
        self.viif = IntLineEdit(10)
        self.viif.setText('1000000000000')
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
        
        self.muaif = QLineEdit()
        self.muaif.setValidator(QDoubleValidator())
        self.muaif.setText("0.07")
        rform.addRow("Viscosity A", self.muaif)
        self.mubif = QLineEdit()
        self.mubif.setValidator(QDoubleValidator())
        self.mubif.setText("0.07")
        rform.addRow("Viscosity B", self.mubif)
        self.rhoaif = QLineEdit()
        self.rhoaif.setValidator(QDoubleValidator())
        rform.addRow("Density A", self.rhoaif)
        self.rhobif = QLineEdit()
        self.rhobif.setValidator(QDoubleValidator())
        rform.addRow("Density B", self.rhobif)
        self.alphif = QLineEdit()
        self.alphif.setValidator(QDoubleValidator())
        rform.addRow("Interfacial Tension", self.alphif)
        self.betaif = QLineEdit()
        self.betaif.setValidator(QDoubleValidator())
        rform.addRow("Beta", self.betaif)
        
        self.scncbx = QCheckBox()
        rform.addRow("Set Capillary Number", self.scncbx)
        
        self.inidif = QLineEdit()
        self.inidif.setFixedWidth(100)
        rform.addRow("Input IDs", self.inidif)
        self.rdidif = QLineEdit()
        self.rdidif.setFixedWidth(100)
        rform.addRow("Read IDs", self.rdidif)
        self.slidif = QLineEdit()
        self.slidif.setFixedWidth(100)
        rform.addRow("Solid IDs", self.slidif)
        self.ctaif = QLineEdit()
        self.ctaif.setFixedWidth(100)
        rform.addRow("Contact Angles", self.ctaif)
        
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
        
        self.simcbb = QComboBox()
        self.simcbb.addItem("colour")
        self.simcbb.addItem("dfh")
        form.addRow("Simulation Type", self.simcbb)
        
    
               
        btnlayout = QHBoxLayout()
        runbtn = QPushButton("Run Solver")
        def on_run_clicked():
            self.hide()
            self.backWindow.hide()
            self.saveValues()
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
        self.muA = float(self.muaif.text())
        self.muB = float(self.mubif.text())
        self.rhoA = float(self.rhoaif.text())
        self.rhoB = float(self.rhob.text())
        self.alpha = float(self.alphaif.text())
        self.beta = float(self.betaif.text())
        self.setCapillaryNumber = self.scncbx.isChecked()
        self.fx = int(self.fxif.text())
        self.fy = int(self.fyif.text())
        self.fz = int(self.fzif.text())
        self.simtype = str(self.simcbb.currentText())
        self.flux = int(self.fluxif.text())
        self.pin = float(self.pinif.text())
        self.pout = float(self.poutif.text())
        self.inputIDs = self.inidif.text()
        self.readIDs = self.rdidif.text()
        self.solidIDs = self.slidif.text()
        self.contactAngles = self.ctaif.text()

   
def TwoPhase(window):
    mat = scipy.io.loadmat(window.domainPath)
    keyname = ''
    for key in list(mat.keys()):
        if(not key.startswith('__') and not key.endswith('__')):
            keyname = key
            break
        
    image = mat[keyname]#[:200,:200,:200]
    image = np.array(image).astype(bool)
    
    domain = rd.removeDisconnections(image)
    
    HPCFlag = False
    restart = False
    timesteps = window.timesteps
    visInterval = window.visInterval
    analysisInterval = window.analysisInterval
    permTolerance = window.permTolerance
    terminal = True
    gpuIDs = []
    npx = window.npx
    npy = window.npy
    npz = window.npz
    
    voxelSize = window.voxelSize
    
    muA = window.muA
    muB = window.muB
    rhoA = window.rhoA
    rhoB = window.rhoB
    alpha = window.alpha
    beta = window.beta
    
    Fx = window.fx;
    Fy = window.fy;
    Fz = window.fz;
    flux = window.flux;
    Pin = window.pin;
    Pout = window.pout;
    simType = window.simType
    
    inputIDs = window.inputIDs
    readIDs = window.readIDs
    solidIDs = window.solidIDs
    contactAngles = window.contactAngles
    
    setCapillaryNumber = window.setCapillaryNumber
    
    targetdir = window.outpath
    if(not os.path.exists(targetdir)):
        os.mkdir(targetdir)
    os.chdir(targetdir)
    
    rlt.runLBPMTwoPhase(domain, targetdir, npx, npy, npz,
    voxelSize, timesteps, gpuIDs, simType,
    Fx, Fy, Fz, flux, Pin, Pout, muA, muB, rhoA, rhoB, alpha, beta,
    inputIDs, readIDs, solidIDs, contactAngles,
    restart, visInterval, analysisInterval, permTolerance, terminal, HPCFlag, setCapillaryNumber)