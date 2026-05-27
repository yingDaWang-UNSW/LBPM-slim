function [K, cmdOut] = runLBMSinglePhase(domain, dir, npx, npy, npz, modelType,...
    voxelSize, timesteps, gpuIDs,...
    Fx, Fy, Fz, flux, Pin, Pout, mu,...
    restart, restartInterval, visInterval, analysisInterval, permTolerance, terminal, HPCFlag, HPCDir, projectID, ngpus)
    %%
    % example: 
    % load bentheimer.mat
    % 
    % [K, cmdOut] = runLBMSinglePhase(bentheimer, 'E:\lbm\bentPerm', 2, 2, 2, 'mrt',...
    %     1e-6, 1e6, '0,1,2,3',...
    %     0, 0, 0, 100, 1/3, 1/3, 1/15,...
    %     0, 1e5, 1e5, 1e3, 1e-6, 0, 1, 'bentPerm', 'm65', 8);
    % 

% cd C:\Users\Whydee\Downloads\bentperm
% [vx, vy, vz, density] = loadLBPMVelP(107000,1);
% 
% vmag=sqrt(vx.*vx+vy.*vy+vz.*vz);
% 
% vmagnorm=vmag./mean(vmag(vmag~=0));
% 
% vmagnorm=uint8(vmagnorm);
% 
% load bentheimer.mat
% 
% writeTiffStackYDW('solid.tiff',bentheimer);
% writeTiffStackYDW('velmagnorm.tiff',vmagnorm*10);



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
    %% load 
    if strcmp(modelType,'srt')
        bgkFlag = 'true';
    elseif strcmp(modelType,'mrt')
        bgkFlag = 'false';
    end
    thermalFlag = 'false';
    velDeltaTrackingFlag = 'false';
    visTolerance='true';
    fqFlag = 'false';
    logFile ='false'; % generally, you can just obtain this data from terminal output
    DiffCoeff=1e-1;

    tau=3*mu+0.5;
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

    fileName='domain';

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
    
%     mpiloc='/opt/openmpi-3.0.0/bin/';
    mpiloc='';


    %% construct inputfile, testfile, and geometry file
    inputFile={['Domain {']; 
               ['    Filename = "', fileName, '.raw"'];
               ['    nproc = ', num2str(npx), ', ', num2str(npy), ', ', num2str(npz)];
               ['    n = ', num2str(nx), ', ', num2str(ny), ', ', num2str(nz)];
               ['    N = ', num2str(Nx), ', ', num2str(Ny), ', ', num2str(Nz)];
               ['    L = 1, 1, 1'];
               ['    BC = ', num2str(BC)];
               ['    voxel_length = ', num2str(voxelSize*1e6)];
               ['    ReadType = "8bit"'];
               ['    ReadValues = 0, 1'];
               ['    WriteValues = 2, 0'];
               ['}'];
               [''];
               ['MRT {'];
               ['    bgkFlag = ', bgkFlag];
               ['    thermalFlag = ', thermalFlag];
               ['    timestepMax = ', num2str(timesteps)];
               ['    tau = ', num2str(tau)]; 
               ['    F = ', num2str(Fx), ', ', num2str(Fy), ', ', num2str(Fz)];
               ['    Restart = false'];
               ['    din = ', num2str(Pin*3,'%e')];
               ['    dout = ', num2str(Pout*3,'%e')];
               ['    flux = ', num2str(flux)];
               ['    visInterval = ', num2str(visInterval)];  
               ['    fqFlag = ', fqFlag];
               ['    restartFq = ', restartFq];           
               ['    analysis_interval = ', num2str(analysisInterval)];           
               ['    restart_interval = ', num2str(restartInterval)];           
               ['    permTolerance = ', num2str(permTolerance)];           
               ['    velDeltaTrackingFlag = ', velDeltaTrackingFlag];           
               ['    visTolerance = ', visTolerance];           
               ['    logFile = ', logFile];           
               ['}'];
               [''];
               ['Thermal {'];
               ['    DiffCoeff = ', num2str(DiffCoeff)];      
               ['}']};
    fid = fopen('inputFile.db', 'wt');
    fprintf(fid, '%s\n', string(inputFile));
    fclose(fid);
    
    if HPCFlag
        if ngpus==0
            runFile={['#!/bin/bash'];
                     ['#PBS -P ', projectID];
                     ['#PBS -q normal'];
                     ['#PBS -l walltime=',num2str(walltime),':00:00'];
                     ['#PBS -l mem=',num2str(mem),'GB'];
                     ['#PBS -l jobfs=1GB'];
                     ['#PBS -l ncpus=',num2str(npx*npy*npz)];
                     ['#PBS -l storage=scratch/m65'];
                     ['#PBS -l software=my_program'];
                     ['#PBS -l wd'];
                     ['cd $PBS_O_WORKDIR'];
                     ['export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmSlimBuild'];
                     ['module load openmpi/4.1.2 ucx/1.12'];
                     ['export NUMPROCS=',num2str(npx*npy*npz)];
                     ['mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db'];
                     ['mpirun -np $NUMPROCS $LBPM_DIR/bin/lbpm_permeability_simulator inputFile.db']};
        else
            workerspernode=npx*npy*npz/(ngpus/4);
            runFile={['#!/bin/bash'];
                     ['#PBS -P ', projectID];
                     ['#PBS -q gpuvolta'];
                     ['#PBS -l walltime=',num2str(walltime),':00:00'];
                     ['#PBS -l mem=',num2str(mem),'GB'];
                     ['#PBS -l jobfs=1GB'];
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
                     ['export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmSlimGPUucxBuild'];
                     ['module load openmpi/4.1.2 ucx/1.12 cuda/11.4.1'];
                     ['export NUMPROCS=',num2str(npx*npy*npz)];
                     ['mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db'];
                     ['mpirun -np $NUMPROCS --mca pml ob1 --machinefile nodeList.txt $LBPM_DIR/bin/lbpm_permeability_simulator inputFile.db']};
%                      ['export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmSlimGPUucxBuild'];
%                      ['module load openmpi/4.1.0 cuda/11.2.2'];
%                      ['export NUMPROCS=',num2str(npx*npy*npz)];
%                      ['mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db'];
%                      ['mpirun -np $NUMPROCS --mca pml ob1 $LBPM_DIR/bin/lbpm_permeability_simulator inputFile.db']};
  
        end
    else
        if isempty(gpuIDs)
            runFile={['#!/bin/bash'];
                     ['export LBPM_DIR="/home/user/LBPMYDW/lbpmSlimBuild"'];
                     ['export NUMPROCS=',num2str(npx*npy*npz)];
                     [mpiloc,'mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db'];
                     [mpiloc,'mpirun -np $NUMPROCS $LBPM_DIR/tests/lbpm_permeability_simulator inputFile.db']};
        else
            runFile={['#!/bin/bash'];
                     ['export LBPM_DIR="/home/user/LBPMYDW/lbpmSlimGPUBuild"'];
                     ['export NUMPROCS=',num2str(npx*npy*npz)];
                     [mpiloc,'mpirun -np 1 $LBPM_DIR/bin/lbpm_serial_decomp inputFile.db'];
                     ['CUDA_VISIBLE_DEVICES=', gpuIDs,' ',mpiloc,'mpirun -np $NUMPROCS $LBPM_DIR/tests/lbpm_permeability_simulator inputFile.db']};
        end
    end
    fid = fopen('runFile.db', 'wt');
    fprintf(fid, '%s\n', string(runFile));
    fclose(fid);

    fileID = fopen([fileName,'.raw'],'w');
    fwrite(fileID,domain(:),'int8');
    fclose(fileID);

    %% run solver
    if ~HPCFlag
        pathToScript = fullfile(pwd,'runFile.db');  % assumes script is in curent directory
        cmdStr       = ['"bash ' pathToScript,'"'];
        if terminal
            system(['gnome-terminal -e ',cmdStr])    
            K=nan;
            cmdOut=[];
        else
            [stat, cmdOut] = system(['bash ', pathToScript], '-echo');
            % get perm
            T=readtable('Permeability.csv');
            K=table2array(T(:,end));
            if iscell(K)
                K=cellfun(@str2num,K,'UniformOutput',false);
                K=cell2mat(K);
            end
            K=K.*9.87e11;
            lastnan=find(isnan(K));
            if lastnan>0
                K=K(lastnan(end)+1:end);
            end
        end
    else
        % cmdStr       = ['"sshpass -p 29051994Taneehga rsync -rvPu ',pwd ,' yw5484@gadi.nci.org.au:/scratch/m65/yw5484/',HPCDir,'"'];
        % system(['gnome-terminal -e ',cmdStr])   
        % K=nan;
        % cmdOut=[];
        % Define variables
        username = 'yw5484';
        remoteDir = '/scratch/m65/yw5484/';
        localDir = pwd;
        % HPCDir = 'your_HPC_directory';  % Replace with your actual HPC directory
        
        % Build the pscp command to copy files to the remote directory
        cmdStr = ['scp -r "', localDir, '" ', username, '@gadi.nci.org.au:', remoteDir, HPCDir];
        
        % Execute the command in PowerShell
        system(['powershell -Command "', cmdStr, '"']);
        
        K = nan;
        cmdOut = [];


    end

end
%% move IO around to generate the dataset