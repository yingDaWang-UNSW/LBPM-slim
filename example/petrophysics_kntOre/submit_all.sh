#!/bin/bash
# Generate per-case PBS scripts using the same sizing as setup_cases.sh,
# then qsub each one.  Run this AFTER setup_cases.sh (which materialises
# the four case subdirs with mirror.db + inputFile.db + in.raw).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# Must match setup_cases.sh
SOLID_NPX=1; SOLID_NPY=1; SOLID_NPZ=8
SOLIDBIN_NPX=2; SOLIDBIN_NPY=2; SOLIDBIN_NPZ=8

for c in lot3_solid lot4_solid; do
    [ -d "$c" ] || { echo "missing $c -- run setup_cases.sh first"; exit 1; }
    bash _mkpbs.sh "$c" $SOLID_NPX $SOLID_NPY $SOLID_NPZ
done
for c in lot3_solidbinder lot4_solidbinder; do
    [ -d "$c" ] || { echo "missing $c -- run setup_cases.sh first"; exit 1; }
    bash _mkpbs.sh "$c" $SOLIDBIN_NPX $SOLIDBIN_NPY $SOLIDBIN_NPZ
done

echo
for c in lot3_solid lot4_solid lot3_solidbinder lot4_solidbinder; do
    pushd "$c" > /dev/null
    qsub run.pbs
    popd > /dev/null
done
echo
echo "qstat -u \$USER  to monitor."
