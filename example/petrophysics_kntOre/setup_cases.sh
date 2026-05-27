#!/bin/bash
# Set up four case subdirectories under this folder:
#   lot3_solid  lot3_solidbinder  lot4_solid  lot4_solidbinder
# Each case dir gets:
#   - in.raw           (1704^3 binary mask: 1 = transport phase, 0 = blocking)
#   - mirror.db        (lbpm_mirror_pp input)
#   - inputFile.db     (Domain + Poisson; used by BOTH serial_decomp and
#                       lbpm_tortuosity_simulator -- one-file convention matching
#                       example/runLBMSinglePhase.m)
#
# The per-case decomposition (nproc) is BAKED INTO inputFile.db here, sized to
# fit the V100 32 GB GPUs on gpuvolta:
#   solid       (Np ~ 20% * 1720^3 = 1.0e9) -> 8 ranks  (1x1x8)   ~26 GB/rank
#   solidbinder (Np ~ 80% * 1720^3 = 4.0e9) -> 32 ranks (2x2x8)   ~21 GB/rank
# Change SOLID_NPROC / SOLIDBINDER_NPROC below to retune.

set -eu
TIF_DIR=/scratch/m65/yw5484/kntOre
PY=python3

# (npx, npy, npz) for each case type
SOLID_NPX=1;       SOLID_NPY=1;       SOLID_NPZ=8
SOLIDBIN_NPX=2;    SOLIDBIN_NPY=2;    SOLIDBIN_NPZ=8

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

declare -A TIFFOR=( [lot3]=FinalSEG_Lot3.tif [lot4]=Finalseg_Lot4.tif )

write_inputs() {
    local case_dir=$1 N=$2 npx=$3 npy=$4 npz=$5
    cat > "$case_dir/mirror.db" <<EOF
Domain {
    Filename     = "in.raw"
    N            = ${N}, ${N}, ${N}
    ReadType     = "8bit"
    MirrorPad    = 16, 16, 16
    nproc        = 1, 1, 1
}
EOF
    local Nm=$((N + 16))
    local nx=$((Nm / npx))
    local ny=$((Nm / npy))
    local nz=$((Nm / npz))
    [ $((nx * npx)) -eq $Nm ] && [ $((ny * npy)) -eq $Nm ] && [ $((nz * npz)) -eq $Nm ] \
        || { echo "ERROR: nproc=($npx,$npy,$npz) does not divide Nm=$Nm evenly"; return 1; }
    cat > "$case_dir/inputFile.db" <<EOF
Domain {
    Filename     = "in_mirrored.raw"
    nproc        = ${npx}, ${npy}, ${npz}
    n            = ${nx}, ${ny}, ${nz}
    N            = ${Nm}, ${Nm}, ${Nm}
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
    tif_name=${TIFFOR[$lot]}
    tif_path=${TIF_DIR}/${tif_name}
    [ -f "$tif_path" ] || { echo "missing $tif_path"; exit 1; }
    for phase in solid solidbinder; do
        case_dir=${lot}_${phase}
        echo "================================================================"
        echo "  $case_dir   <- $tif_name  (phase=$phase)"
        echo "================================================================"
        mkdir -p "$case_dir"
        cd "$case_dir"
        if [ -f in.raw ]; then
            echo "  in.raw exists -- skipping conversion"
        else
            ${PY} ../convert_tif_phase.py "$tif_path" "$phase" in.raw
        fi
        Nx=$(awk '/^Nx/{print $2}' in.dims)
        Ny=$(awk '/^Ny/{print $2}' in.dims)
        Nz=$(awk '/^Nz/{print $2}' in.dims)
        [ "$Nx" = "$Ny" ] && [ "$Nx" = "$Nz" ] \
            || { echo "ERROR: non-cubic input"; exit 1; }
        cd ..
        if [ "$phase" = "solid" ]; then
            write_inputs "$case_dir" "$Nx" "$SOLID_NPX" "$SOLID_NPY" "$SOLID_NPZ"
        else
            write_inputs "$case_dir" "$Nx" "$SOLIDBIN_NPX" "$SOLIDBIN_NPY" "$SOLIDBIN_NPZ"
        fi
    done
done

echo
echo "All four cases set up.  Submit with: bash submit_all.sh"
