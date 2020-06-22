import os
from math import ceil
import subprocess
import sys

def runLBPMTwoPhase(domain, targetdir, npx, npy, npz,
                    voxelSize, timesteps, gpuIDs, simType,
                    Fx, Fy, Fz, flux, Pin, Pout, muA, muB, rhoA, rhoB, alpha, beta,
                    inputIDs, readIDs, solidIDs, contactAngles,
                    restart, visInterval, analysisInterval, permTolerance, terminal, HPCFlag):
    
    LBPM_CPU_Install = "/mnt/c/Users/THOMAS/Documents/Projects/Uni/LBPM-CPU"
    LBPM_GPU_Install = "/mnt/c/Users/THOMAS/Documents/Projects/Uni/LBPM-GPU"
    
    if(not os.path.exists(targetdir)):
        os.mkdir(targetdir)
    os.chdir(targetdir)
    
    fileName= 'geopack';
    setCapillaryNumber = False;
    if not setCapillaryNumber:
        tag='//';
    else:
        tag=''; 
    targetCapillaryNumber = 5e-4;

    tauA=3*muA+0.5;
    tauB=3*muB+0.5;

    Nx, Ny, Nz = domain.shape 

    nx=ceil(Nx/npx);
    ny=ceil(Ny/npy);
    nz=ceil(Nz/npz);

    if flux>0:
        BC=4;
    elif Pin>0:
        BC=3;
    else:
        BC=0;

    if restart:
        restartFq = 'true';
    else:
        restartFq='false';

    limSw=2;
    injType=1;
    autoMorph =  'false';
    spinoMorph = 'false';
    fluxMorph =  'false';
    coinjection = 'false';

    if (simType == 'colour'):
        model='lbpm_color_simulator';
    elif (simType == 'dfh'):
        model='lbpm_dfh_simulator';
        
    inputfile = ('Domain {', '\n', 
               '    Filename = "', fileName, '.raw"', '\n',
               '    nproc = ', str(npx), ', ', str(npy), ', ', str(npz), '\n',
               '    n = ', str(nx), ', ', str(ny), ', ', str(nz), '\n',
               '    N = ', str(Nx), ', ', str(Ny), ', ', str(Nz), '\n',
               '    L = 1, 1, 1', '\n',
               '    BC = ', str(BC), '\n',   #Boundary condition type, 0=PBC, 1 or 2 are? ,3 is pressure, 4 is flux
               '    voxel_length = ', str(voxelSize*1e6), '\n',
               '    ReadType = "8bit"', '\n',
               '    ReadValues = ', str(inputIDs), '\n',
               '    WriteValues = ', str(readIDs), '\n',
               '}', '\n',
               '', '\n',

               'Color {', '\n',
               '    tauA = ', str(tauA), '\n',
               '    tauB = ', str(tauB), '\n',
               '    rhoA = ', str(rhoA), '\n',
               '    rhoB = ', str(rhoB), '\n',
               '    alpha = ', str(alpha), '\n',
               '    beta = ', str(beta), '\n',
               '    F = ', str(Fx), ', ', str(Fy), ', ', str(Fz), '\n',
               '    Restart = ', str(restartFq), '\n',           
               '    din = ', str(Pin*3), '\n',
               '    dout = ', str(Pout*3), '\n',
               '    timestepMax = ', str(timesteps), '\n',
               '    flux = ', str(flux), '\n',
               '    inletA = 1.0', '\n',
               '    inletB = 0.0', '\n',
               '    outletA = 0.0', '\n',
               '    outletB = 1.0', '\n',
               '    fluxReversalFlag = false', '\n',
               '    fluxReversalType = 1', '\n', # 1 for flip IO phases, 2 for flip flow direction
               '    fluxReversalSat = 0.15', '\n', # if neg, then use settling
               '    settlingTolerance = 1e-6', '\n', # 
               '    ComponentLabels = ',str(solidIDs), '\n',
               '    ComponentAffinity = ',str(contactAngles), '\n', # affinity to A, -1 (B) is water, 1 (A) is oil
               '    affinityRampupFlag = false', '\n',
               '    affinityRampupSteps = 10000', '\n',
               '    ',tag,'capillary_number = ', str(targetCapillaryNumber), '\n', # target capillary number
               '    //target_saturation = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0', '\n', # manual morph points
               '}', '\n',
               '', '\n',
               'Analysis {', '\n', 
               '    tolerance = 1e20', '\n', # morpho steady state tolerance level (set high for consistent morph)
               '    ramp_timesteps = 1000', '\n', # timesteps before morph is activated
               '    //morph_interval = 10000', '\n', # manual morph
               '    //morph_delta = -0.1', '\n',
               '    autoMorphFlag = ',autoMorph, '\n', # automorph using shells
               '    spinoMorphFlag = ',spinoMorph, '\n', # automorph using shells               
               '    fluxMorphFlag = ',fluxMorph, '\n', # automorph using injection this sucks
               '    coinjectionFlag = ',coinjection, '\n', # automatic coinjection so does this
               '    satInit = ', str(limSw), '\n', # the saturation set to start automorph. reach this saturation by flux injection
               '    satInc = 0.1', '\n', # push by (set low for single step)
               '    injectionType = ', str(injType), '\n', # 1 for drain (morph+) 2 for imb(morph-)
               '    stabilisationRate = 50000', '\n', # time between steady checks
               '    accelerationRate = 3000', '\n', # time between morph events
               '    blobid_interval = 10000000', '\n',
               '    analysis_interval = ', str(analysisInterval), '\n',          
               '    restart_interval = 100000', '\n',
               '    visualization_interval = 10000000000', '\n',
               '    raw_visualisation_interval = ', str(visInterval), '\n',     
               '    restart_file = "Restart"', '\n',
               '    N_threads = ',str(npx*npy*npz), '\n',
               '    load_balance = "none"', '\n', # "none", "default", "independent"
               '}', '\n'
               )
    fid = open('inputFile.db', 'wt')
    fid.write(''.join(inputfile))
    fid.close()
    
    if HPCFlag:
        runfile=('#!/bin/bash', '\n',
                 '#PBS -P m65', '\n',
                 '#PBS -q normal', '\n',
                 '#PBS -l walltime=10:00:00', '\n',
                 '#PBS -l mem=2TB', '\n',
                 '#PBS -l jobfs=1GB', '\n',
                 '#PBS -l ncpus=',str(npx*npy*npz), '\n',
                 '#PBS -l storage=scratch/m65', '\n',
                 '#PBS -l software=my_program', '\n',
                 '#PBS -l wd', '\n',
                 'cd $PBS_O_WORKDIR', '\n',
                 'export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmBuild', '\n',
                 'module load gcc/system openmpi/4.0.1', '\n',
                 'export NUMPROCS=',str(npx*npy*npz), '\n',
                 'mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db', '\n',
                 'mpirun -np $NUMPROCS $LBPM_DIR/bin/',model,' inputFile.db', '\n'
                 )
    else:
        if gpuIDs:
            runfile=('#!/bin/bash', '\n',
                     'export LBPM_DIR="',LBPM_CPU_Install,'"', '\n',
                     'export NUMPROCS=',str(npx*npy*npz), '\n',
                     'mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db', '\n',
                     'mpirun -np $NUMPROCS $LBPM_DIR/bin/',model,' inputFile.db', '\n'
                     )
        else:
            runfile=('#!/bin/bash', '\n',
                     'export LBPM_DIR="',LBPM_GPU_Install,'"', '\n',
                     'export NUMPROCS=',str(npx*npy*npz), '\n',
                     'mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db', '\n',
                     'CUDA_VISIBLE_DEVICES=', gpuIDs,' mpirun -np $NUMPROCS $LBPM_DIR/bin/',model,' inputFile.db', '\n'
                     )
    fid = open('runfile.db', 'wt')
    fid.write(''.join(runfile))
    fid.close()
    
    fid = open(fileName +'.raw','wb');
    domain.tofile(fid)
    fid.close()
    
    pathname = os.path.abspath('runfile.db')
    cmd = ["bash", pathname]

    print("Running solver...")
    
    subprocess.Popen(cmd, stdout=sys.stdout)