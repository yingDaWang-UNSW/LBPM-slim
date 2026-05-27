#!/bin/bash
# Generate a per-case PBS script matching the example/*.m gpuvolta convention.
# Internal helper used by submit_all.sh (and re-runnable on demand).
#
# Usage:
#   bash _mkpbs.sh <case_dir> <npx> <npy> <npz>
#
# Sizing rules (from runLBMSinglePhase.m):
#   ngpus       = npx*npy*npz
#   ncpus       = ngpus * 12        (gpuvolta is 12 cpus / gpu)
#   mem         = ngpus * 382/4 GB  (382 GB per 4-GPU node)
#   workerspernode = ngpus / (ngpus/4) = 4
#   walltime    = 48 if ngpus<=4, 24 if 4<ngpus<20, 5 if ngpus>=20
set -eu
CASE=$1; NPX=$2; NPY=$3; NPZ=$4
NGPUS=$((NPX*NPY*NPZ))
NCPUS=$((NGPUS*12))
MEM=$((NGPUS*382/4))
WORKERSPN=4
if [ $NGPUS -le 4 ]; then WALL=48
elif [ $NGPUS -lt 20 ]; then WALL=24
else                       WALL=5
fi

cat > "${CASE}/run.pbs" <<EOF
#!/bin/bash
#PBS -P m65
#PBS -q gpuvolta
#PBS -l walltime=${WALL}:00:00
#PBS -l mem=${MEM}GB
#PBS -l jobfs=1GB
#PBS -l ncpus=${NCPUS}
#PBS -l ngpus=${NGPUS}
#PBS -l storage=scratch/m65
#PBS -l software=my_program
#PBS -l wd
#PBS -N tort_${CASE}
cd \$PBS_O_WORKDIR
echo "Job is running on node(s):"
cat \$PBS_NODEFILE | sort | uniq
cat \$PBS_NODEFILE | sort | uniq > nodes.txt
rm -f nodeList.txt
for ((i=0; i<${WORKERSPN}; i++)); do cat nodes.txt >> nodeList.txt; done
export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmSlimGPUucxBuild
module load openmpi/4.1.2 ucx/1.12 cuda/11.4.1
export NUMPROCS=${NGPUS}
# step 1: mirror_pp (single-rank, just adds 16-voxel smoothing slabs)
mpirun -np 1 \$LBPM_DIR/bin/lbpm_mirror_pp mirror.db
# step 2: serial_decomp -> per-rank ID.xxxxx files
mpirun -np 1 \$LBPM_DIR/bin/lbpm_serial_decomp inputFile.db
# step 3: tortuosity simulator
mpirun -np \$NUMPROCS --mca pml ob1 --machinefile nodeList.txt \\
    \$LBPM_DIR/bin/lbpm_tortuosity_simulator inputFile.db
EOF
chmod +x "${CASE}/run.pbs"
echo "wrote ${CASE}/run.pbs  (ngpus=${NGPUS}, mem=${MEM}GB, walltime=${WALL}h)"
