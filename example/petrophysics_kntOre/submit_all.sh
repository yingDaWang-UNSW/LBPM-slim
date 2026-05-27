#!/bin/bash
# Generate per-case PBS scripts (calling _mkpbs.sh with the right TIF +
# phase + nproc) and qsub each.  Run AFTER setup_cases.sh has materialised
# the per-case dirs under $RUN_ROOT.
set -eu
RUN_ROOT=${RUN_ROOT:-/scratch/m65/yw5484/kntOre_runs}
TIF_DIR=${TIF_DIR:-/scratch/m65/yw5484/kntOre}

# Must match setup_cases.sh (per-phase decomp -- solidbinder needs more
# ranks to keep per-rank Np under the int32 idx-overflow threshold).
SOLID_NPROC=(4 4 2)
SOLIDBIN_NPROC=(4 4 4)

declare -A TIFFOR=( [lot3]=FinalSEG_Lot3.tif [lot4]=Finalseg_Lot4.tif )

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

for lot in lot3 lot4; do
    for phase in solid solidbinder; do
        c=${lot}_${phase}
        case_dir="$RUN_ROOT/$c"
        [ -d "$case_dir" ] || { echo "missing $case_dir -- run setup_cases.sh first"; exit 1; }
        if [ "$phase" = "solid" ]; then
            NPX=${SOLID_NPROC[0]}; NPY=${SOLID_NPROC[1]}; NPZ=${SOLID_NPROC[2]}
        else
            NPX=${SOLIDBIN_NPROC[0]}; NPY=${SOLIDBIN_NPROC[1]}; NPZ=${SOLIDBIN_NPROC[2]}
        fi
        bash _mkpbs.sh "$case_dir" "${TIF_DIR}/${TIFFOR[$lot]}" "$phase" "$NPX" "$NPY" "$NPZ"
    done
done

echo
for lot in lot3 lot4; do
    for phase in solid solidbinder; do
        c=${lot}_${phase}
        pushd "$RUN_ROOT/$c" > /dev/null
        qsub run.pbs
        popd > /dev/null
    done
done
echo
echo "qstat -u \$USER  to monitor."
