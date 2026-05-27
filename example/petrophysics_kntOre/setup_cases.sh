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

# Per-case decomp.  Must divide each mirrored axis (1720, 1720, 1724).
# Why split the two phases: ScaLBL's MemoryOptimizedLayoutAA stores
# `idx + 18*Np` in int32.  At Np > 119M the value wraps -> garbage
# neighbor index -> wild GPU read -> SIGSEGV in ScaLBL_Poisson::Create.
# Sized so per-rank Np stays well under 100M for the worst rank:
#   solid       (phi~0.55):  4x4x2  -> per-rank 430x430x862 = 159M voxels
#                                       Np_max ~87M  -> safe.
#   solidbinder (phi~0.66):  4x4x4  -> per-rank 430x430x431 =  80M voxels
#                                       Np_max ~53M  -> safe (32 ranks crashed).
SOLID_NPROC=(4 4 2)         #  32 ranks ->  8 V100 nodes
SOLIDBIN_NPROC=(4 4 4)      #  64 ranks -> 16 V100 nodes

mkdir -p "$RUN_ROOT"

write_inputs() {
    local case_dir=$1
    local phase=$2
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
    local NPX NPY NPZ
    if [ "$phase" = "solid" ]; then
        NPX=${SOLID_NPROC[0]}; NPY=${SOLID_NPROC[1]}; NPZ=${SOLID_NPROC[2]}
    else
        NPX=${SOLIDBIN_NPROC[0]}; NPY=${SOLIDBIN_NPROC[1]}; NPZ=${SOLIDBIN_NPROC[2]}
    fi
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
        write_inputs "$case_dir" "$phase"
    done
done

echo
echo "Cases laid out under $RUN_ROOT"
echo "Submit with:  bash submit_all.sh"
