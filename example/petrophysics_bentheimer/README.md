# Petrophysics master example -- Bentheimer 500³

Self-contained example of the LBPM-slim petrophysics workflow. The four
input `.db` files here are the **static protocol** — they should not be
re-tuned per sample. Only the geometry fields (Filename, N, n, nproc)
change between samples; the physics parameters in the Poisson and MRT
blocks are fixed.

## Files

| File | Role |
|------|------|
| `bentheimer.mat` | Source 500³ uint8 (0 = pore, 1 = solid) segmented CT, ~22.3 % porosity |
| `convert_mat_to_raw.py` | One-off conversion: `bentheimer.mat` → `bentheimer.raw` |
| `mirror.db` | Step 1 — `lbpm_mirror_pp` adds a 16-voxel periodic-seam-smoothing slab per axis |
| `decomp.db` | Step 2 — `lbpm_serial_decomp` writes the per-rank `ID.xxxxx` files + relabels into LBPM convention |
| `tort.db` | Step 3a — `lbpm_tortuosity_simulator` (D3Q7 LBM Laplace) → τ_zz |
| `perm.db` | Step 3b — `lbpm_permeability_simulator` (D3Q19 MRT body force) → k_zz [Darcy] |
| `run.sh` | Driver that runs steps 1 → 4 in order |

## Reference numbers (this 500³ Bentheimer at 4 µm/voxel, 1-GPU on a 4090)

| protocol | result | wall | host RSS |
|----------|--------|------|----------|
| `tort`   | τ_zz = 3.127,  φ = 0.224, D_eff/D_0 = 0.072, F = 13.97 | 18 min | 8.2 GB |
| `perm`   | k_zz = 3.016 D (Z), 3.018 D (RMS)                       | 18 min | 7.0 GB |

Both squarely within published Bentheimer ranges (φ 22–25 %, τ 2–4,
k 1–3 D at µm scale). k scales as `voxel_length²` — running with the
wrong resolution gives a `(actual/used)²` error in the Darcy value, so
setting `voxel_length` correctly in `perm.db` is the one sample-physical
knob that actually matters (tortuosity is dimensionless and doesn't care).

## How to use on a new sample

1. Drop the new `<sample>.mat` (or `<sample>.raw`) in this folder.
2. If `.mat`: `python3 convert_mat_to_raw.py <sample>.mat`.
3. Edit `mirror.db`:
   - `Filename = "<sample>.raw"`
   - `N = Nx, Ny, Nz` (the source dimensions)
4. Edit `decomp.db`:
   - `Filename = "<sample>_mirrored.raw"`
   - `N = Nx+16, Ny+16, Nz+16` (mirror adds 16 per axis)
   - `n = N` for single rank; `n = N / nproc` per axis for multi-rank
5. Edit `tort.db` and `perm.db`:
   - `n = same as decomp.db`
   - `nproc = same as decomp.db`
   - `voxel_length = µm/voxel of the sample` (perm.db uses it for Darcy
     units; tort doesn't depend on it).  For Bentheimer here: 4.0.
6. `./run.sh`

## Static-protocol parameters (do not tune)

- **Mirror pad = 16 voxels** per axis: thick enough to smooth the periodic
  seam, thin enough to leave the bulk sample unchanged.
- **Tortuosity Poisson block**: `tau = 0.75`, MSE `tolerance = 1e-9`,
  `Vin = 1`, `Vout = 0`, Dirichlet z-BCs.
- **Permeability MRT block**: `tau = 1.0`, `F = (0, 0, 1e-5)`,
  `permTolerance = 1e-5` (|dK/K| between analysis windows), periodic BCs.
- Relabel in `decomp.db`: `ReadValues = (0,1)`, `WriteValues = (1,0)` —
  flips segmented-image convention (0=pore) into LBPM convention (id>0=fluid).
  This is fixed; only the input `Filename` changes per sample.
