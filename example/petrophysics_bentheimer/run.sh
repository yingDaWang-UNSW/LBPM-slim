#!/bin/bash
# Master petrophysics pipeline for one segmented image.
#
# Steps (each is part of the IMMUTABLE protocol):
#   1. mirror_pp        -- add 16-voxel transition slabs so periodic seam is smooth
#   2. serial_decomp    -- pad with 1-voxel halo, flip pore/solid into LBPM convention
#   3a. tortuosity_simulator    -- D3Q7 LBM Laplace -> tau_zz
#   3b. permeability_simulator  -- D3Q19 MRT body force -> k_zz [Darcy]
#
# To use on a new sample:
#   - point convert_mat_to_raw.py at the new .mat (or supply a .raw directly)
#   - update Filename + N in mirror.db
#   - update Filename + N + n in decomp.db (n = N + 16 per padded axis)
#   - update n in tort.db and perm.db (= N + 16 per padded axis)
#   - run this script

set -u
BIN=/home/user/sourceCodesGit/LBPMYDW/lbpmSlimGPUBuild/tests
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

echo "================================================================"
echo "  step 1/4  mirror_pp   (smooth periodic seam)"
echo "================================================================"
${BIN}/lbpm_mirror_pp mirror.db | tail -10

echo
echo "================================================================"
echo "  step 2/4  serial_decomp   (halo + relabel into LBPM convention)"
echo "================================================================"
${BIN}/lbpm_serial_decomp decomp.db | tail -10

echo
echo "================================================================"
echo "  step 3/4  tortuosity_simulator   (D3Q7 Laplace, tau_zz)"
echo "================================================================"
/usr/bin/time -v -o tort.time ${BIN}/lbpm_tortuosity_simulator tort.db | tee tort.log | tail -15
echo "  wall: $(grep 'Elapsed (wall clock)' tort.time | sed -E 's/^.*: //')"
echo "  rss : $(grep 'Maximum resident set size' tort.time | awk '{print $NF}') kB"

echo
echo "================================================================"
echo "  step 4/4  permeability_simulator   (D3Q19 MRT, k_zz)"
echo "================================================================"
/usr/bin/time -v -o perm.time ${BIN}/lbpm_permeability_simulator perm.db | tee perm.log | tail -15
echo "  wall: $(grep 'Elapsed (wall clock)' perm.time | sed -E 's/^.*: //')"
echo "  rss : $(grep 'Maximum resident set size' perm.time | awk '{print $NF}') kB"
