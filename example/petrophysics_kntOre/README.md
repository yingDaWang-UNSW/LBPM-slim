# knt-ore SR images -- 4-case tortuosity pipeline (gpuvolta)

Runs `lbpm_tortuosity_simulator` on the two SR images sitting at
`/scratch/m65/yw5484/kntOre/` on Gadi:

| TIF (1704³, 3-label: 0/128/255) | lot |
|---------------------------------|-----|
| `FinalSEG_Lot3.tif`             | 3   |
| `Finalseg_Lot4.tif`             | 4   |

For each lot, two cases:

| case          | transport phase            | blocking phase           |
|---------------|----------------------------|--------------------------|
| `solid`       | solid only (TIF=255)       | pore (0) + binder (128)  |
| `solidbinder` | solid + binder (TIF ≥ 128) | pore (0) only            |

Four runs total: `lot3_solid`, `lot3_solidbinder`, `lot4_solid`,
`lot4_solidbinder`.

## Files

| File | Role |
|------|------|
| `convert_tif_phase.py` | TIF (0/128/255) → single-phase binary raw (1 = transport, 0 = blocking) |
| `setup_cases.sh`       | Creates 4 case subdirs, converts, writes `mirror.db` + `inputFile.db` per case |
| `_mkpbs.sh`            | Internal: generates a per-case `run.pbs` matching `example/runLBMSinglePhase.m` sizing rules |
| `submit_all.sh`        | Calls `_mkpbs.sh` per case, then `qsub`s all four jobs |

## How to drive it on Gadi

```bash
cd ~/LBPMYDW/LBPM-slim/example/petrophysics_kntOre
bash setup_cases.sh        # converts TIFs + writes per-case input dbs (a few min total)
bash submit_all.sh         # generates run.pbs per case and submits all four
qstat -u $USER             # monitor
```

## Sizing (matching `runLBMSinglePhase.m`)

gpuvolta nodes are **4 × V100 (32 GB) per node, 48 cpus, 382 GB RAM**.
Sizing baked into `setup_cases.sh` + `_mkpbs.sh`:

| case type      | nproc   | ngpus | nodes | mem     | walltime |
|----------------|---------|-------|-------|---------|----------|
| `solid`        | 4, 4, 2 |  32   |   8   | 3056 GB | 5 h      |
| `solidbinder`  | 4, 4, 4 |  64   |  16   | 6112 GB | 5 h      |

Why solidbinder needs more ranks: LBPM's `MemoryOptimizedLayoutAA`
stores `idx + 18*Np` in int32 — when per-rank `Np` exceeds ~119 M voxels
the value wraps and the GPU dereferences a garbage neighbor index
(SIGSEGV in `ScaLBL_Poisson::Create`). For phi ~ 0.66 (solid+binder),
the 32-rank cube (Np_max ~130 M) crashes; the 4×4×4 split keeps
Np_max ~ 53 M, well clear. Solid (phi ~ 0.55, Np_max ~ 87 M) is safe
at 32 ranks.

## Per-case file layout (after `setup_cases.sh`)

```
lot3_solid/
├── in.raw                 (1704³ uint8, transport=1, blocking=0)
├── in.dims                (Nx Ny Nz dims)
├── mirror.db              (lbpm_mirror_pp input)
└── inputFile.db           (Domain + Poisson; serial_decomp AND tortuosity use this)
```

After `submit_all.sh` also adds `run.pbs`, and the job produces:
`in_mirrored.raw`, `ID.00000..ID.NPROCS-1`, `Tortuosity.csv`,
plus stdout/stderr in `tort_<case>.o<jobid>`.

## Static-protocol parameters (do not retune per sample)

Hardcoded in `setup_cases.sh::write_inputs`:

- `MirrorPad = 16, 16, 16` — periodic-seam smoothing slab per axis.
- Poisson: `tau = 0.75`, `tolerance = 1e-9`, `Vin = 1`, `Vout = 0`,
  `BC_Inlet = BC_Outlet = 1`.
- `ReadValues = (0, 1)`, `WriteValues = (0, 1)` — identity relabel
  (the conversion script already writes 1 in the transport phase).
- `voxel_length = 1.0` µm (tortuosity is dimensionless; only matters for
  perm).

Per-sample knobs: `Filename` and `N` (auto-detected from the TIF dims by
`convert_tif_phase.py`); the decomposition (`SOLID_NPROC`, `SOLIDBIN_NPROC`)
at the top of `setup_cases.sh` if the sample size changes.

## Syncing this folder to Gadi

These files (and the LBPM-slim source tree including the int32 overflow
fix + new `lbpm_tortuosity_simulator`) are not yet on Gadi. From your
workstation:

```bash
cd ~/sourceCodesGit/LBPMYDW/LBPM-slim
git add example/petrophysics_kntOre
git commit -m "kntOre SR pellet tortuosity (4 cases, gpuvolta)"
git push

# on Gadi
ssh gadi
cd ~/LBPMYDW/LBPM-slim && git pull
bash installer/configureGadiGPU         # rebuild with the int32 fix
```

(Or rsync if you'd rather skip GitHub — see the matching section in
`example/petrophysics_bentheimer/README.md`.)
