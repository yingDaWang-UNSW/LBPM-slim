function [] = runLBPMColorGrad(domain, dir, npx, npy, npz,...
    voxelSize, timesteps, gpuIDs, simType, pcProtocol, pcRelaxationTime, pcStepUpRatio, morphType, max_stabilisation, injectionType,...
    Fx, Fy, Fz, flux, Pin, Pout, Nca, colourBCs, labelBCs, labelBCA, nucleationFactor, muA, muB, rhoA, rhoB, alpha, beta,...
    inputIDs, readIDs, solidIDs, contactAngles,...
    restart, restartInterval, visInterval, satVisInterval, analysisInterval, terminal, HPCFlag, HPCDir, projectID, ngpus)
    cd /
    folderName=strsplit(dir,'/');
    folderName=folderName{end};
    if ngpus==0
        walltime = 48;
        if npx*npy*npz>=720
            walltime = 24;
        end
        if npx*npy*npz>=1488
            walltime = 10;
        end
        if npx*npy*npz>=3024
            walltime = 5;
        end
%         mem=min((npx*npy*npz/48)*190,mem);
%         mem=(npx*npy*npz/48)*190;
        mem=(npx*npy*npz/48)*190;
        ncpus=ceil(npx*npy*npz/48)*48;
    else
        walltime = 48;
        if ngpus>4
            walltime = 24;
        end
        if ngpus>=20
            walltime = 5;
        end
        mem=ngpus*382/4;%min((npx*npy*npz)*46,mem);
    end
    logFile = 'false';
    mem=round(mem);
    %% auto drain
    forcedDrainFlag = 'false';
    drainTolerance = '1e-4';
    %% auto pc
    if pcProtocol
        pcProtocolFlag = 'true';
    else
        pcProtocolFlag = 'false';
    end
    %% automorph
    limSw=-20;
    injType=injectionType;
    autoMorph =  'false';
    spinoMorph = 'false';
    fluxMorph =  'false'; % these are ineffective.
    coinjection = 'false';
    if strcmp(morphType,'shell')
        autoMorph =  'true';
    elseif strcmp(morphType,'spino')
        autoMorph =  'true';
        spinoMorph = 'true';
    elseif strcmp(morphType,'coinjection')
        coinjection = 'true';
    end


    %%
    if Nca>0
        setCapillaryNumber=1;
        targetCapillaryNumber = Nca;
    else
        setCapillaryNumber=0;
        targetCapillaryNumber = 0;
    end   

    fileName='geopack';
    if ~setCapillaryNumber
        tag='//';
    else
        tag=''; 
    end

    tauA=(3*muA/rhoA+0.5);
    tauB=(3*muB/rhoB+0.5);
    Nx=size(domain, 1);
    Ny=size(domain, 2);
    Nz=size(domain, 3); 
    nx=ceil(Nx/npx);
    ny=ceil(Ny/npy);
    nz=ceil(Nz/npz);
    if flux>0
        BC=4;
    elseif Pin>0 && Pout>0 && Pin~=Pout
        BC=3;
    else
        BC=0;
        Pin=1/3;
        Pout=1/3;
        flux=0;
    end
    if restart
        restartFq = 'true';
    else
        try
            rmdir(dir,'s')
        catch
            disp('Folder doesnt exist')
        end
        mkdir(dir)
        restartFq='false';
    end
    cd(dir) 
    inletA = colourBCs(3,1);
    inletB = colourBCs(3,2);
    outletA = colourBCs(3,3);
    outletB = colourBCs(3,4);
    
    inletAx = colourBCs(1,1);
    inletBx = colourBCs(1,2);
    outletAx = colourBCs(1,3);
    outletBx = colourBCs(1,4);
    
    inletAy = colourBCs(2,1);
    inletBy = colourBCs(2,2);
    outletAy = colourBCs(2,3);
    outletBy = colourBCs(2,4);
%% sim types
if strcmp(simType,'colour')
    model='lbpm_color_simulator';
elseif strcmp(simType,'dfh')
    model='lbpm_dfh_simulator';
end
    %% construct inputfile, testfile, and geometry file
    inputFile={['Domain {']; 
               ['    Filename = "', fileName, '.raw"'];
               ['    nproc = ', num2str(npx), ', ', num2str(npy), ', ', num2str(npz)];
               ['    n = ', num2str(nx), ', ', num2str(ny), ', ', num2str(nz)];
               ['    N = ', num2str(Nx), ', ', num2str(Ny), ', ', num2str(Nz)];
               ['    L = 1, 1, 1'];
               ['    BC = ', num2str(BC)];  % Boundary condition type, 0=PBC, 1 or 2 are? ,3 is pressure, 4 is flux
               ['    voxel_length = ', num2str(voxelSize*1e6)];
               ['    ReadType = "8bit"'];
               ['    ReadValues = ', inputIDs];
               ['    WriteValues = ', readIDs];
               ['}'];
               [''];

               ['Color {'];
               ['    tauA = ', num2str(tauA)];
               ['    tauB = ', num2str(tauB)];
               ['    rhoA = ', num2str(rhoA)];
               ['    rhoB = ', num2str(rhoB)];
               ['    alpha = ', num2str(alpha)];
               ['    beta = ', num2str(beta)];
               ['    F = ', num2str(Fx), ', ', num2str(Fy), ', ', num2str(Fz)];
               ['    Restart = ', restartFq];           
               ['    din = ', num2str(Pin*3)];
               ['    dout = ', num2str(Pout*3)];
               ['    timestepMax = ', num2str(timesteps)];
               ['    flux = ', num2str(flux)];
               %['    fluxRampup = ', num2str(100000)];
               ['    inletA = ', num2str(inletA)];
               ['    inletB = ', num2str(inletB)];
               ['    outletA = ', num2str(outletA)];
               ['    outletB = ', num2str(outletB)];
               ['    inletAx = ', num2str(inletAx)];
               ['    inletBx = ', num2str(inletBx)];
               ['    outletAx = ', num2str(outletAx)];
               ['    outletBx = ', num2str(outletBx)];
               ['    inletAy = ', num2str(inletAy)];
               ['    inletBy = ', num2str(inletBy)];
               ['    outletAy = ', num2str(outletAy)];
               ['    outletBy = ', num2str(outletBy)];
               ['    labelBCs = ', labelBCs];
               ['    labelBCValsA = ', labelBCA];
               ['    nucleationFactor = ', num2str(nucleationFactor)];
               ['    ComponentLabels = ',solidIDs];
               ['    ComponentAffinity = ',contactAngles]; % affinity to A, -1 (B) is water, 1 (A) is oil
               ['    fluxReversalFlag = false'];
               ['    fluxReversalType = 1']; % 1 for flip IO phases, 2 for flip flow direction
               ['    fluxReversalSat = 0.5']; % if neg, then use settling
               ['    settlingTolerance = 1e-60']; % 
               ['    affinityRampupFlag = false'];
               ['    affinityRampupSteps = 10000'];
               ['    ',tag,'capillary_number = ', num2str(targetCapillaryNumber)]; % target capillary number               ['}'];
               ['}'];

               ['Analysis {']; 
               ['    tolerance = 1e-16']; % morpho steady state tolerance level
               ['    ramp_timesteps = 10000']; % timesteps before morph is activated
               ['    pcProtocolFlag = ',pcProtocolFlag]; % autodrain protocol will 2x flux when saturation plateaus
               ['    pcRelaxationTime = ',num2str(pcRelaxationTime)]; % autodrain protocol will 2x flux when saturation plateaus
               ['    pcStepUpRatio = ',num2str(pcStepUpRatio)]; % autodrain protocol will 2x flux when saturation plateaus
               ['    autoForcedDrain = ',forcedDrainFlag]; % autodrain protocol will 2x flux when saturation plateaus
               ['    drainTolerance = ',drainTolerance]; % autodrain saturation change per analysis step threshold
               ['    autoMorphFlag = ',autoMorph]; % automorph using shells
               ['    spinoMorphFlag = ',spinoMorph]; % automorph using spino               
               ['    fluxMorphFlag = ',fluxMorph]; % automorph using injection
               ['    coinjectionFlag = ',coinjection]; % automatic coinjection
               ['    satInit = ', num2str(limSw)] % the saturation set to start automorph. reach this saturation by flux injection
               ['    satInc = 0.1']; % push by (set low for single step)
               ['    injectionType = ', num2str(injType)]; % 1 for drain (morph+) 2 for imb(morph-)
               ['    stabilisationRate = ', num2str(analysisInterval)];  % time between steady checks
               ['    max_stabilisation = ', num2str(max_stabilisation)]; % max time before forced morph
               ['    accelerationRate = ', num2str(analysisInterval)];  % time between morph events
               ['    analysis_interval = ', num2str(analysisInterval)];          
               ['    restart_interval = ', num2str(restartInterval)];
               ['    raw_visualisation_interval = ', num2str(visInterval)];     
               ['    satVisInterval = ', num2str(satVisInterval)];     
               ['    restart_file = "Restart"'];
               ['    logFile = ', logFile];           
               ['}']};


    fid = fopen('inputFile.db', 'wt');
    fprintf(fid, '%s\n', string(inputFile));
    fclose(fid);
    %% construct HPC PBS files
    if HPCFlag
        if ngpus==0
            runFile={['#!/bin/bash'];
                     ['#PBS -P ', projectID];
                     ['#PBS -q normal'];
                     ['#PBS -l walltime=',num2str(walltime),':00:00'];
                     ['#PBS -l mem=', num2str(mem),'GB'];
                     ['#PBS -l jobfs=100GB'];
                     ['#PBS -l ncpus=',num2str(ncpus)];
                     ['#PBS -l storage=scratch/m65'];
                     ['#PBS -l software=my_program'];
                     ['#PBS -l wd'];
                     ['cd $PBS_O_WORKDIR'];
                     ['export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmSlimBuild'];
                     ['module load openmpi/4.1.2 ucx/1.12'];
                     ['export NUMPROCS=',num2str(npx*npy*npz)];
                     ['mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db'];
                     ['mpirun -np $NUMPROCS $LBPM_DIR/bin/',model,' inputFile.db']};
        else
            workerspernode=npx*npy*npz/(ngpus/4);
            runFile={['#!/bin/bash'];
                     ['#PBS -P ', projectID];
                     ['#PBS -q gpuvolta'];
                     ['#PBS -l walltime=',num2str(walltime),':00:00'];
                     ['#PBS -l mem=',num2str(mem),'GB'];
                     ['#PBS -l jobfs=100GB'];
                     ['#PBS -l ncpus=',num2str(min(ngpus*12,12*npx*npy*npz))];
                     ['#PBS -l ngpus=',num2str(min(ngpus,npx*npy*npz))];
                     ['#PBS -l storage=scratch/m65'];
                     ['#PBS -l software=my_program'];
                     ['#PBS -l wd'];
                     ['cd $PBS_O_WORKDIR'];
                     ['echo "Job is running on node(s): "'];
                     ['cat $PBS_NODEFILE | sort | uniq'];
                     ['cat $PBS_NODEFILE | sort | uniq > nodes.txt'];
                     ['rm nodeList.txt'];
                     ['limit=',num2str(workerspernode)];
                     ['for ((i=0; i<limit; i++)); do cat nodes.txt >> nodeList.txt; done'];
%                      ['export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmSlimGPUBuild'];
%                      ['module load openmpi/2.1.6 cuda/10.1'];
%                      ['export NUMPROCS=',num2str(npx*npy*npz)];
%                      ['mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db'];
%                      ['mpirun -np $NUMPROCS $LBPM_DIR/bin/',model,' inputFile.db']};
                     ['export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmSlimGPUucxBuild'];
                     ['module load openmpi/4.1.2 ucx/1.12 cuda/11.4.1'];
                     ['export NUMPROCS=',num2str(npx*npy*npz)];
                     ['mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db'];
                     ['mpirun -np $NUMPROCS --mca pml ob1 --machinefile nodeList.txt $LBPM_DIR/bin/',model,' inputFile.db']};
        end
    else
        if isempty(gpuIDs)
            runFile={['#!/bin/bash;'];
                     ['export LBPM_DIR="/home/user/LBPMYDW/lbpmSlimBuild"'];
%                      ['export LBPM_DIR="/mnt/c/Users/mutris4/LBPMYDW/lbpmSlimBuild";'];
                     ['export NUMPROCS=',num2str(npx*npy*npz),';'];
                     ['mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db;'];
                     ['bin/mpirun -np $NUMPROCS $LBPM_DIR/bin/',model,' inputFile.db;']};
        else
            runFile={['#!/bin/bash;'];
                     ['export LBPM_DIR="/home/user/sourceCodesGit/LBPMYDW/lbpmSlimGPUBuild"'];
%                      ['export LBPM_DIR="/mnt/c/Users/mutris4/LBPMYDW/lbpmSlimGPUBuild";'];
                     ['export NUMPROCS=',num2str(npx*npy*npz),';'];
                     ['mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db;'];
                     ['CUDA_VISIBLE_DEVICES=', gpuIDs,' mpirun -np $NUMPROCS $LBPM_DIR/bin/',model,' inputFile.db;']};
        end
    end
    fid = fopen('runFile.db', 'wt');
    fprintf(fid, '%s\n', string(runFile));
    fclose(fid);
    %% export geom
    fileID = fopen([fileName,'.raw'],'w');
    fwrite(fileID,domain(:),'int8');
    fclose(fileID);

    %% run solver or upload to gadi
%     if ~HPCFlag
%         %         [stat, cmdOut] = system(['bash ', pathToScript], '-echo');
%         %     %% rel perm
%         %     relperm=load('relperm.csv');
%         %     figure(1)
%         %     hold on
%         %     plot(relperm(:,16),relperm(:,17))
%         %     plot(relperm(:,16),relperm(:,18))
% 
% %         cmdStr       = ['bash runFile.db'];
% %         if terminal
% %             system(['wsl -e ',cmdStr,' &'])    
% %         else
% %             [stat, cmdOut] = system(['bash ', pathToScript], '-echo');
% %         end
% %     else
% %         pathToScript = ['../',dir];  % assumes script is in curent directory
% %         cmdStr       = ['sshpass -p 29051994Taneehga rsync -rvPu ',pathToScript ,' yw5484@gadi.nci.org.au:/scratch/m65/yw5484/',HPCDir];
% % %         if terminal
% %             system(['wsl -e ',cmdStr,' &']) 
% %     end
% 
% 
%         pathToScript = fullfile(pwd,'runFile.db');  % assumes script is in curent directory
%         cmdStr       = ['"bash ''' pathToScript,'''"'];
%         if terminal
%             system(['gnome-terminal -e ',cmdStr])    
%         else
%             [stat, cmdOut] = system(['bash ', pathToScript], '-echo');
%         end
%     else
% %         pathToScript = fullfile(pwd,'runFile.db');  % assumes script is in curent directory
%         cmdStr       = ['"sshpass -p 29051994Taneehga rsync -rvPu ',pwd ,' yw5484@gadi.nci.org.au:/scratch/m65/yw5484/',HPCDir,'"'];
% %         if terminal
%         [status, cmdout] = system(cmdStr, '-echo');
% 
%             % system(['gnome-terminal -e ',cmdStr], '-echo')    
% %             [stat, cmdOut] = system(cmdStr, '-echo');
% %             system(['gnome-terminal -e "sshpass -p 0406476058Taneehga ssh yw5484@gadi.nci.org.au bash ~/submitGadiJobs.sh /scratch/',projectID,'/yw5484/',HPCDir,'/',folderName,'"'])    
% 
% %         if submitFlag
% %             cmdStr       = ['"sshpass -p 29051994Taneehga ssh yw5484@gadi.nci.org.au; cd /scratch/m65/yw5484/',HPCDir,';qsub runFile.db"'];
% %     %         if terminal
% %             system(['gnome-terminal -e ',cmdStr])  
% %         end
% %         else
% %             [stat, cmdOut] = system(['bash ', cmdStr], '-echo');
% %         end
%     end
end
