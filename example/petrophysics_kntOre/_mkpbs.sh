#!/bin/bash
# Generate a per-case PBS script.  Each case's run.pbs is self-contained:
# converts TIF -> binary raw with the right phase mask, mirrors, decomposes,
# then runs the multi-GPU tortuosity simulator.
#
# Usage:  bash _mkpbs.sh <case_dir>  <tif_path>  <phase>  <npx> <npy> <npz>
#
# Sizing rules (matching example/runLBMSinglePhase.m gpuvolta tier):
#   ngpus       = npx*npy*npz
#   ncpus       = ngpus * 12   (gpuvolta has 12 cpus/gpu)
#   mem         = ngpus * 382/4 GB
#   workerspernode = 4         (4 V100 / gpuvolta node)
#   walltime    = 48 if ngpus<=4, 24 if 4<ngpus<20, 5 if ngpus>=20
set -eu
CASE_DIR=$1; TIF_PATH=$2; PHASE=$3; NPX=$4; NPY=$5; NPZ=$6
NGPUS=$((NPX*NPY*NPZ))
NCPUS=$((NGPUS*12))
MEM=$((NGPUS*382/4))
WORKERSPN=4
if [ $NGPUS -le 4 ]; then WALL=48
elif [ $NGPUS -lt 20 ]; then WALL=24
else                       WALL=5
fi
CASE_NAME=$(basename "$CASE_DIR")
SRC_DIR=$(cd "$(dirname "$0")" && pwd)   # where convert_tif_phase.py lives

cat > "${CASE_DIR}/run.pbs" <<EOF
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
#PBS -N tort_${CASE_NAME}
cd \$PBS_O_WORKDIR
echo "Job is running on node(s):"
cat \$PBS_NODEFILE | sort | uniq
cat \$PBS_NODEFILE | sort | uniq > nodes.txt
rm -f nodeList.txt
for ((i=0; i<${WORKERSPN}; i++)); do cat nodes.txt >> nodeList.txt; done
export LBPM_DIR=/home/561/yw5484/LBPMYDW/lbpmSlimGPUucxBuild
module load python3/3.10.0 openmpi/4.1.2 ucx/1.12.0 cuda/11.4.1
export NUMPROCS=${NGPUS}
# step 0: convert TIF -> single-phase raw  (~15 GB peak host RAM; fine on a
# gpuvolta node which has 382 GB).  Skips if in.raw already exists.
if [ ! -f in.raw ]; then
    python3 ${SRC_DIR}/convert_tif_phase.py ${TIF_PATH} ${PHASE} in.raw
fi
# step 1: mirror_pp -- adds 16-voxel smoothing slab per axis for periodic seam
if [ ! -f in_mirrored.raw ]; then
    mpirun -np 1 \$LBPM_DIR/bin/lbpm_mirror_pp mirror.db
fi
# step 2: serial_decomp -> per-rank ID.xxxxx files
if [ ! -f "ID.\$(printf '%05d' \$((NUMPROCS-1)))" ]; then
    mpirun -np 1 \$LBPM_DIR/bin/lbpm_serial_decomp inputFile.db
fi
# step 3: tortuosity simulator across all ranks
mpirun -np \$NUMPROCS --mca pml ob1 --machinefile nodeList.txt \\
    \$LBPM_DIR/bin/lbpm_tortuosity_simulator inputFile.db
EOF
chmod +x "${CASE_DIR}/run.pbs"
echo "wrote ${CASE_DIR}/run.pbs  (TIF=$(basename $TIF_PATH) phase=${PHASE}  ngpus=${NGPUS}  mem=${MEM}GB  wall=${WALL}h)"
