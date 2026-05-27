"""
Convert a multi-label 8-bit segmented TIF into a single-phase binary raw
(LBPM input convention: 0 = blocking, 1 = transport phase).

For these knt-ore SR images:
    0   = pore
    128 = binder
    255 = solid

    case=solid          -> transport=255, blocking={0,128}
    case=solidbinder    -> transport={128,255}, blocking=0
    case=pore           -> transport=0, blocking={128,255}   (for completeness)

Usage:
    python3 convert_tif_phase.py <input.tif> <case> <output.raw>
        case in {solid, solidbinder, pore}

Output: 1704^3 (or whatever the input is) uint8 raw, k-slowest, byte order
matching LBPM's i*Nx*Ny + j*Nx + i indexing. Also prints the dims.
"""
import os, sys
import numpy as np

if len(sys.argv) != 4:
    print(__doc__); sys.exit(1)

src, case, dst = sys.argv[1], sys.argv[2].lower(), sys.argv[3]

# Streaming read of the TIF -- tifffile keeps memory low because we only
# touch the volume once when we threshold.  For 1704^3 the bare array is
# ~5 GB; threshold to bool then cast to uint8 doubles transient memory
# briefly, peak ~15 GB.
try:
    import tifffile as tiff
except ImportError:
    print("ERROR: pip install tifffile  (or load a python env that has it)")
    sys.exit(2)

print(f"reading {src} ...")
vol = tiff.imread(src)          # numpy ndarray, ordering depends on TIF
print(f"  shape={vol.shape} dtype={vol.dtype} uniques={np.unique(vol)[:6]}")
if vol.ndim != 3:
    raise RuntimeError(f"expected 3D, got shape {vol.shape}")
if vol.dtype != np.uint8:
    # Force 8-bit -- tifffile sometimes returns uint16 for 8-bit data on some
    # platforms; truncate, do NOT scale.
    print(f"  WARN: casting {vol.dtype} -> uint8 (no scaling)")
    vol = vol.astype(np.uint8)

# Pick the transport mask
if case == "solid":
    transport = (vol == 255)
elif case == "solidbinder":
    transport = (vol >= 128)
elif case == "pore":
    transport = (vol == 0)
else:
    print(f"ERROR: unknown case {case!r}"); sys.exit(3)
del vol

# tifffile usually returns (z, y, x) order -- which is already the
# k-slowest byte order LBPM wants.  Don't transpose.
out = transport.astype(np.uint8)   # 1 = transport, 0 = blocking
del transport
print(f"  transport voxel fraction = {out.sum() / out.size:.6f}")

print(f"writing {dst} ({out.shape}, {out.nbytes/1e9:.2f} GB) ...")
out.tofile(dst)
# Drop a tiny dims file so the shell scripts can pick up N
with open(os.path.splitext(dst)[0] + ".dims", "w") as f:
    f.write(f"Nx {out.shape[2]}\nNy {out.shape[1]}\nNz {out.shape[0]}\n")
print(f"done.")
