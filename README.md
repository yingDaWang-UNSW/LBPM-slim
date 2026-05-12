LBPM-slim — Petrophysics
========================

This `petrophysics` branch is a focused slice of LBPM-slim that contains
only the protocols needed for routine petrophysical characterisation of
porous-media samples:

| Protocol | What it measures              | Driver binary                  | Underlying model |
|----------|-------------------------------|--------------------------------|------------------|
| `k`      | Absolute permeability         | `lbpm_permeability_simulator`  | MRT / BGK single-phase |
| `pc`     | Capillary pressure (Sw → pc)  | `lbpm_color_simulator` (+ `_SI`) | Colour-gradient LBM with `pcProtocolFlag` |
| `kr`     | Relative permeability (kr–Sw) | `lbpm_relperm_simulator`       | Colour LBM + automorph + per-event single-phase BGK on connected components |

Everything else (DFH, fuel-cell, greyscale, thermal, the assorted micro-tests
and example bubbles) has been removed from this branch to keep the repo
focused on petrophysical workflows. Reach for the upstream `dev` branch when
those are needed.

Pre-processors
--------------

- `lbpm_serial_decomp`     — decompose a raw 8-bit segmented geometry into per-rank `ID.xxxxx` files.
- `lbpm_serial_uCT_decomp` — decompose a uCT (16-bit) volume.
- `lbpm_uCT_pp`            — post-segmentation cleanup for uCT volumes.
- `lbpm_morphopen_pp`      — morphological drainage to a target Sw. Used both
                             as an initialiser for `kr` and as a standalone
                             Sw vs critical-radius generator that, via
                             Young–Laplace, yields a morphological `pc(Sw)` curve.

Typical workflows
-----------------

**Absolute permeability (k)**:

    mpirun -np 1 lbpm_serial_decomp        inputFile.db
    mpirun -np N lbpm_permeability_simulator inputFile.db

**Capillary pressure (pc)** — colour-model drainage with the pc protocol:

    mpirun -np 1 lbpm_serial_decomp        inputFile.db
    mpirun -np N lbpm_color_simulator      inputFile.db
    # (use `lbpm_color_simulator_SI inputFile.db [--dimless]` for SI-unit input)

For a quick `pc(Sw)` proxy without running LBM, sweep `Domain.Sw` over
`lbpm_morphopen_pp` and convert the final critical radius to pc via
Young–Laplace.

**Relative permeability (kr)**:

    mpirun -np 1 lbpm_serial_decomp        inputFile.db
    mpirun -np N lbpm_morphopen_pp         inputFile.db   # initialise to Sw_init (e.g. 0.9)
    mpirun -np N lbpm_relperm_simulator    inputFile.db   # color + automorph + single-phase BGK per event

`lbpm_relperm_simulator` rides the colour-model automorph loop. Every time a
steady state is detected, `OnSteadyStatePoint()` runs `ComputeGlobalBlobIDs`
on both phases, intersects the inlet- and outlet-touching blob lists to find
the percolating components, and then runs a single-phase BGK with body force
under periodic (BC=0) on each connected component to record `k_eff` (a row
per phase per event is appended to `RelPermSummary.csv`). The colour LBM
then resumes, the next morph step is applied, and the cycle repeats until
the automorph reaches its terminal saturation.

Reference layout: `/media/user/Data0/lbpmslimRuns/relpermBereaTest/`
(see `inputFile.db` and `runFile.db`).

Build
-----

    mkdir build && cd build
    cmake ..                       # CPU
    cmake .. -DUSE_CUDA=1          # GPU (needs MPI + CUDA)
    make -j

Requirements: CMake ≥ 3.9, C++14, MPI; for GPU, a CUDA-aware MPI and CUDA.
