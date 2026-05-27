#!/bin/bash
# Set up the four knt-ore tortuosity cases.  Creates per-case subdirs under
# $RUN_ROOT (default /scratch/m65/yw5484/kntOre_runs) and drops in the static
# inputs:
#
#   <RUN_ROOT>/<case>/{ mirror.db, inputFile.db }
#
# The actual TIF -> raw conversion (~15 GB peak RAM) happens INSIDE the PBS
# job because Gadi login nodes will OOM-kill it.  See _mkpbs.sh.
#
# All physics parameters in inputFile.db are STATIC across every sample;
# only the geometry-dependent dims and the per-case nproc are filled in here.

set -eu
RUN_ROOT=${RUN_ROOT:-/scratch/m65/yw5484/kntOre_runs}

# Source TIF dims (1708 x 1704 x 1704 for both lot3 and lot4 -- if a new
# sample has different dims, edit these three.  axes follow tifffile order:
# Nz is the slowest = axis 0 = number of slices.)
NZ_SRC=1708; NY_SRC=1704; NX_SRC=1704

# Pad applied by lbpm_mirror_pp (kept on the static side -- 16 voxels is
# the protocol).
PAD=16

# Single decomp for every case.  Must divide each mirrored axis.
# Mirrored dims: NX = 1720, NY = 1720, NZ = 1724.
# 4 x 4 x 2 -> per-rank 430 x 430 x 862 voxels.
# 32 ranks total -> 8 V100 gpuvolta nodes.
NPX=4; NPY=4; NPZ=2

mkdir -p "$RUN_ROOT"

write_inputs() {
    local case_dir=$1
    mkdir -p "$case_dir"
    cat > "$case_dir/mirror.db" <<EOF
Domain {
    Filename     = "in.raw"
    N            = ${NX_SRC}, ${NY_SRC}, ${NZ_SRC}
    ReadType     = "8bit"
    MirrorPad    = ${PAD}, ${PAD}, ${PAD}
    nproc        = 1, 1, 1
}
EOF
    local Mx=$((NX_SRC + PAD))
    local My=$((NY_SRC + PAD))
    local Mz=$((NZ_SRC + PAD))
    local nx=$((Mx / NPX))
    local ny=$((My / NPY))
    local nz=$((Mz / NPZ))
    if [ $((nx*NPX)) -ne $Mx ] || [ $((ny*NPY)) -ne $My ] || [ $((nz*NPZ)) -ne $Mz ]; then
        echo "ERROR: nproc=($NPX,$NPY,$NPZ) does not divide mirrored dims ($Mx,$My,$Mz)"
        return 1
    fi
    cat > "$case_dir/inputFile.db" <<EOF
Domain {
    Filename     = "in_mirrored.raw"
    nproc        = ${NPX}, ${NPY}, ${NPZ}
    n            = ${nx}, ${ny}, ${nz}
    N            = ${Mx}, ${My}, ${Mz}
    L            = 1.0, 1.0, 1.0
    voxel_length = 1.0
    BC           = 1
    ReadType     = "8bit"
    ReadValues   = 0, 1
    WriteValues  = 0, 1
}

Poisson {
    tau               = 0.75
    timestepMax       = 2000000
    analysis_interval = 500
    tolerance         = 1.0e-9
    Vin               = 1.0
    Vout              = 0.0
    BC_Inlet          = 1
    BC_Outlet         = 1
}
EOF
}

for lot in lot3 lot4; do
    for phase in solid solidbinder; do
        case_dir=${RUN_ROOT}/${lot}_${phase}
        echo "writing $case_dir/{mirror.db, inputFile.db}"
        write_inputs "$case_dir"
    done
done

echo
echo "Cases laid out under $RUN_ROOT"
echo "Submit with:  bash submit_all.sh"
