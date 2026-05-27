/*
  Copyright 2013--2018 James E. McClure, Virginia Polytechnic & State University

  This file is part of the Open Porous Media project (OPM).
  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/
/*
  ScaLBL_Poisson: D3Q7 LBM Laplace solver. See PoissonSolver.h for scope.
*/
#include "models/PoissonSolver.h"

#include <algorithm>   // std::max for the defensive Npad sizing in Create()

using namespace std;

ScaLBL_Poisson::ScaLBL_Poisson(int RANK, int NP, MPI_Comm COMM)
    : timestep(0), timestepMax(0), analysis_interval(0),
      BoundaryConditionInlet(0), BoundaryConditionOutlet(0),
      tau(0), tolerance(0), Vin(0), Vout(0),
      Nx(0), Ny(0), Nz(0), N(0), Np(0),
      rank(RANK), nprocs(NP), nprocx(0), nprocy(0), nprocz(0),
      Lx(0), Ly(0), Lz(0), h(1.0),
      NeighborList(nullptr), dvcMap(nullptr), fq(nullptr), Psi(nullptr),
      ElectricField(nullptr), ChargeDensity(nullptr),
      comm(COMM)
{
}

ScaLBL_Poisson::~ScaLBL_Poisson()
{
    if (NeighborList)  ScaLBL_FreeDeviceMemory(NeighborList);
    if (dvcMap)        ScaLBL_FreeDeviceMemory(dvcMap);
    if (Psi)           ScaLBL_FreeDeviceMemory(Psi);
    if (ElectricField) ScaLBL_FreeDeviceMemory(ElectricField);
    if (ChargeDensity) ScaLBL_FreeDeviceMemory(ChargeDensity);
    if (fq)            ScaLBL_FreeDeviceMemory(fq);
}

// =========================================================================
// ReadParams: pull "Poisson" + "Domain" sections from the input db.
// =========================================================================
void ScaLBL_Poisson::ReadParams(std::string filename) {
    db          = std::make_shared<Database>(filename);
    domain_db   = db->getDatabase("Domain");
    electric_db = db->getDatabase("Poisson");

    // D3Q7 inverse speed of sound squared = 4; tau-default puts diffusivity at 1
    const double k2_inv = 4.0;
    tau                  = 0.5 + 1.0 / k2_inv;
    timestepMax          = 100000;
    tolerance            = 1.0e-6;
    analysis_interval    = 1000;
    Vin                  = 1.0;
    Vout                 = 0.0;
    BoundaryConditionInlet  = 1;   // Dirichlet at inlet (-z)
    BoundaryConditionOutlet = 1;   // Dirichlet at outlet (+z)
    h = 1.0;

    if (electric_db->keyExists("timestepMax"))
        timestepMax = electric_db->getScalar<int>("timestepMax");
    if (electric_db->keyExists("tau"))
        tau = electric_db->getScalar<double>("tau");
    if (electric_db->keyExists("analysis_interval"))
        analysis_interval = electric_db->getScalar<int>("analysis_interval");
    if (electric_db->keyExists("tolerance"))
        tolerance = electric_db->getScalar<double>("tolerance");
    if (electric_db->keyExists("Vin"))
        Vin = electric_db->getScalar<double>("Vin");
    if (electric_db->keyExists("Vout"))
        Vout = electric_db->getScalar<double>("Vout");
    if (electric_db->keyExists("BC_Inlet"))
        BoundaryConditionInlet = electric_db->getScalar<int>("BC_Inlet");
    if (electric_db->keyExists("BC_Outlet"))
        BoundaryConditionOutlet = electric_db->getScalar<int>("BC_Outlet");

    if (domain_db->keyExists("voxel_length"))
        h = domain_db->getScalar<double>("voxel_length");

    if (rank == 0) {
        printf("================================================================\n");
        printf("  ScaLBL_Poisson: D3Q7 LBM Laplace solver\n");
        printf("================================================================\n");
        printf("  tau              = %.5f\n", tau);
        printf("  tolerance (MSE)  = %.3e\n", tolerance);
        printf("  timestepMax      = %d\n", timestepMax);
        printf("  analysis interv. = %d\n", analysis_interval);
        printf("  Vin / Vout       = %.4f / %.4f\n", Vin, Vout);
        printf("  BC inlet/outlet  = %d / %d\n", BoundaryConditionInlet, BoundaryConditionOutlet);
        printf("================================================================\n");
    }
}

// =========================================================================
// SetDomain: build analysis (Dm) and LBM (Mask) domains.
// =========================================================================
void ScaLBL_Poisson::SetDomain() {
    Dm   = std::make_shared<Domain>(domain_db, comm);
    Mask = std::make_shared<Domain>(domain_db, comm);

    Nx = Dm->Nx;  Ny = Dm->Ny;  Nz = Dm->Nz;
    Lx = Dm->Lx;  Ly = Dm->Ly;  Lz = Dm->Lz;
    N  = Nx * Ny * Nz;

    Psi_host.resize(Nx, Ny, Nz);
    Psi_previous.resize(Nx, Ny, Nz);

    for (int i = 0; i < N; i++) Dm->id[i] = 1;
    MPI_Barrier(comm);

    if (BoundaryConditionInlet == 0 && BoundaryConditionOutlet == 0) {
        Dm->BoundaryCondition   = 0;
        Mask->BoundaryCondition = 0;
    }
    else if (BoundaryConditionInlet > 0 && BoundaryConditionOutlet > 0) {
        Dm->BoundaryCondition   = 1;
        Mask->BoundaryCondition = 1;
    }
    else {
        ERROR("ScaLBL_Poisson::SetDomain -- inlet/outlet BCs must both be periodic or both Dirichlet\n");
    }
    Dm->CommInit();
    MPI_Barrier(comm);

    rank   = Dm->rank();
    nprocx = Dm->nprocx();
    nprocy = Dm->nprocy();
    nprocz = Dm->nprocz();
}

// =========================================================================
// ReadInput: load per-rank ID files (slim's only supported input format).
// =========================================================================
void ScaLBL_Poisson::ReadInput() {
    sprintf(LocalRankString,   "%05d", Dm->rank());
    sprintf(LocalRankFilename, "%s%s", "ID.", LocalRankString);

    Mask->ReadIDs();

    if (rank == 0) cout << "ScaLBL_Poisson: ID file loaded." << endl;
}

// =========================================================================
// Create: ScaLBL communicator, memory-optimised layout, GPU buffers.
// =========================================================================
void ScaLBL_Poisson::Create() {
    // Mirror Mask->id into Dm->id (Dm is used for analysis; Mask is the LBM mask).
    for (int i = 0; i < N; i++) Dm->id[i] = Mask->id[i];
    Mask->CommInit();
    Np = Mask->PoreCount();

    if (rank == 0) printf("ScaLBL_Poisson: Create ScaLBL_Communicator\n");
    ScaLBL_Comm = std::make_shared<ScaLBL_Communicator>(Mask);

    // Defensive Npad: MemoryOptimizedLayoutAA's *internal* Np grows up to the
    // per-rank interior voxel count (not just Mask->PoreCount()), and it writes
    // to neighborList[q*internal_Np + idx] for q up to 17.  Sizing only for the
    // input PoreCount lets those writes go OOB on uneven-phase ranks.  Match
    // the MRTModel.cpp:163 sizing and use size_t throughout to avoid int32
    // overflow at large per-rank cubes (18 * 159M = 2.87e9 > INT_MAX).
    const size_t subdomain_voxels = (size_t)(Nx - 2) * (size_t)(Ny - 2) * (size_t)(Nz - 2);
    const size_t Npad_sz = std::max((size_t)Np, subdomain_voxels) + 16;
    if (rank == 0) printf("ScaLBL_Poisson: build memory-efficient layout (Np = %d)\n", Np);
    Map.resize(Nx, Ny, Nz);
    Map.fill(-2);
    auto neighborList = new int[18UL * Npad_sz];
    // MemoryOptimizedLayoutAA still takes the limit as int; bounded by per-rank
    // interior voxel count which is well under INT_MAX here.
    Np = ScaLBL_Comm->MemoryOptimizedLayoutAA(Map, neighborList, Mask->id, (int)Npad_sz);
    MPI_Barrier(comm);

    if (rank == 0) printf("ScaLBL_Poisson: allocate distributions (Np = %d)\n", Np);
    // size_t throughout: at N>=600 the byte counts (Np*sizeof(double)*7,
    // 18*Np*sizeof(int), etc.) exceed 2^31 and silently wrap to ~30x smaller
    // values when stored in int -> cudaMalloc allocates an undersized buffer
    // -> kernel writes go OOB -> "illegal memory access". Same overflow bug
    // hit MRTModel.cpp:163; cast everything to size_t before the multiplication.
    const size_t Np_sz         = (size_t)Np;
    const size_t dist_mem_size = Np_sz * sizeof(double);
    const size_t neighborSize  = 18UL * Np_sz * sizeof(int);
    const size_t psi_size      = (size_t)Nx * (size_t)Ny * (size_t)Nz * sizeof(double);

    ScaLBL_AllocateDeviceMemory((void **)&NeighborList,  neighborSize);
    ScaLBL_AllocateDeviceMemory((void **)&dvcMap,        Np_sz * sizeof(int));
    ScaLBL_AllocateDeviceMemory((void **)&Psi,           psi_size);
    ScaLBL_AllocateDeviceMemory((void **)&ElectricField, 3UL * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&ChargeDensity, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&fq,            7UL * dist_mem_size);

    // Build the inverse-of-Map (compressed-index -> cartesian-index) on host, copy to device.
    // CRITICAL: default-init to a safe sentinel. MemoryOptimizedLayoutAA returns a
    // padded Np that's > the number of real pore voxels; the slots in
    // [actual_pore_count, Np) are never assigned by the loop below and would
    // otherwise contain `new int[]` uninitialized garbage. At 600^3 single-rank
    // those slots are inside the kernel's exterior range and the kernel reads
    // Psi[garbage_ijk] -> "illegal memory access". Pin the safe sentinel at the
    // last valid cartesian index so any kernel access lands in-bounds.
    int *TmpMap = new int[Np];
    const int safe_ijk = Nx * Ny * Nz - 1;
    for (size_t i = 0; i < Np_sz; i++) TmpMap[i] = safe_ijk;
    for (int k = 1; k < Nz - 1; k++) {
        for (int j = 1; j < Ny - 1; j++) {
            for (int i = 1; i < Nx - 1; i++) {
                int idx = Map(i, j, k);
                if (!(idx < 0) && idx < Np) TmpMap[idx] = k * Nx * Ny + j * Nx + i;
            }
        }
    }
    // Belt-and-braces: any value >= Nx*Ny*Nz (or negative) collapses to safe_ijk.
    for (int idx = 0; idx < ScaLBL_Comm->LastExterior(); idx++) {
        if (TmpMap[idx] < 0 || TmpMap[idx] >= Nx * Ny * Nz) TmpMap[idx] = safe_ijk;
    }
    for (int idx = ScaLBL_Comm->FirstInterior(); idx < ScaLBL_Comm->LastInterior(); idx++) {
        if (TmpMap[idx] < 0 || TmpMap[idx] >= Nx * Ny * Nz) TmpMap[idx] = safe_ijk;
    }
    ScaLBL_CopyToDevice(dvcMap,       TmpMap,       Np_sz * sizeof(int));
    ScaLBL_CopyToDevice(NeighborList, neighborList, neighborSize);
    ScaLBL_DeviceBarrier();
    MPI_Barrier(comm);

    delete[] TmpMap;
    delete[] neighborList;

    // Zero-init ChargeDensity (pure-Laplace mode is the default).
    double *zeros = new double[Np];
    for (size_t i = 0; i < Np_sz; i++) zeros[i] = 0.0;
    ScaLBL_CopyToDevice(ChargeDensity, zeros, dist_mem_size);
    delete[] zeros;
}

// =========================================================================
// Potential_Init: linear interpolation Vin (-z) -> Vout (+z) on pore voxels.
// Uses GLOBAL z so multi-rank decompositions start from a consistent gradient
// (otherwise every rank ramps Vin->Vout across its own slab, producing
// inter-rank discontinuities that take O(Nz_g^2) iterations to relax).
// =========================================================================
void ScaLBL_Poisson::Potential_Init(double *psi_init) {
    const int interior = Nz - 2;                          // interior slices per rank
    const int kproc    = Dm->kproc();
    const long Nz_g    = (long)interior * nprocz;          // global interior length
    const double slope = (Nz_g > 1) ? (Vout - Vin) / double(Nz_g - 1) : 0.0;
    for (int k = 0; k < Nz; k++) {
        const int global_k = kproc * interior + k;        // 0=halo of rank 0, 1=first interior of rank 0, ..., Nz_g=last interior of last rank
        double psi_lin;
        // Snap to exactly Vin/Vout on the Dirichlet slices (only on the outermost ranks)
        if (kproc == 0 && k <= 1)                       psi_lin = Vin;
        else if (kproc == nprocz - 1 && k >= Nz - 2)    psi_lin = Vout;
        else                                            psi_lin = Vin + slope * (global_k - 1);
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int n = k * Nx * Ny + j * Nx + i;
                psi_init[n] = (Mask->id[n] > 0) ? psi_lin : 0.0;
            }
        }
    }
}

// =========================================================================
// Initialize: seed Psi linearly and project onto the D3Q7 equilibrium fq.
// =========================================================================
void ScaLBL_Poisson::Initialize() {
    if (rank == 0) printf("ScaLBL_Poisson: initializing D3Q7 distributions\n");

    double *psi_host = new double[Nx * Ny * Nz];
    Potential_Init(psi_host);
    ScaLBL_CopyToDevice(Psi, psi_host, (size_t)Nx * (size_t)Ny * (size_t)Nz * sizeof(double));
    ScaLBL_SyncAndCheck("after Psi copy");

    ScaLBL_D3Q7_Poisson_Init(dvcMap, fq, Psi, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
    ScaLBL_DeviceBarrier();
    ScaLBL_D3Q7_Poisson_Init(dvcMap, fq, Psi, 0, ScaLBL_Comm->LastExterior(), Np);
    ScaLBL_DeviceBarrier();

    delete[] psi_host;
}

// =========================================================================
// Half-step kernels: stream + apply BCs (SolveElectricPotential) and
// collide (SolvePoisson). Mirrors gaslbm's split (one updates Psi from
// fq, the other relaxes fq toward its equilibrium given Psi).
// =========================================================================
void ScaLBL_Poisson::SolveElectricPotentialAAodd() {
    ScaLBL_Comm->SendD3Q7AA(fq, 0);
    ScaLBL_D3Q7_AAodd_Poisson_ElectricPotential(NeighborList, dvcMap, fq, Psi,
                                                ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
    ScaLBL_Comm->RecvD3Q7AA(fq, 0);
    ScaLBL_DeviceBarrier();
    if (BoundaryConditionInlet > 0)
        ScaLBL_Comm->D3Q7_Poisson_Potential_BC_z(NeighborList, fq, Vin, timestep);
    if (BoundaryConditionOutlet > 0)
        ScaLBL_Comm->D3Q7_Poisson_Potential_BC_Z(NeighborList, fq, Vout, timestep);
    ScaLBL_D3Q7_AAodd_Poisson_ElectricPotential(NeighborList, dvcMap, fq, Psi,
                                                0, ScaLBL_Comm->LastExterior(), Np);
}

void ScaLBL_Poisson::SolveElectricPotentialAAeven() {
    ScaLBL_Comm->SendD3Q7AA(fq, 0);
    ScaLBL_D3Q7_AAeven_Poisson_ElectricPotential(dvcMap, fq, Psi,
                                                 ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
    ScaLBL_Comm->RecvD3Q7AA(fq, 0);
    ScaLBL_DeviceBarrier();
    if (BoundaryConditionInlet > 0)
        ScaLBL_Comm->D3Q7_Poisson_Potential_BC_z(NeighborList, fq, Vin, timestep);
    if (BoundaryConditionOutlet > 0)
        ScaLBL_Comm->D3Q7_Poisson_Potential_BC_Z(NeighborList, fq, Vout, timestep);
    ScaLBL_D3Q7_AAeven_Poisson_ElectricPotential(dvcMap, fq, Psi,
                                                 0, ScaLBL_Comm->LastExterior(), Np);
}

void ScaLBL_Poisson::SolvePoissonAAodd() {
    // EnforceElectroneutrality = false; ChargeDensity is zero so rho_e=0 (pure Laplace).
    ScaLBL_D3Q7_AAodd_Poisson(NeighborList, dvcMap, fq, ChargeDensity, Psi, ElectricField,
                              tau, 1.0, false,
                              ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
    ScaLBL_D3Q7_AAodd_Poisson(NeighborList, dvcMap, fq, ChargeDensity, Psi, ElectricField,
                              tau, 1.0, false,
                              0, ScaLBL_Comm->LastExterior(), Np);
}

void ScaLBL_Poisson::SolvePoissonAAeven() {
    ScaLBL_D3Q7_AAeven_Poisson(dvcMap, fq, ChargeDensity, Psi, ElectricField,
                               tau, 1.0, false,
                               ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
    ScaLBL_D3Q7_AAeven_Poisson(dvcMap, fq, ChargeDensity, Psi, ElectricField,
                               tau, 1.0, false,
                               0, ScaLBL_Comm->LastExterior(), Np);
}

// =========================================================================
// Run: iterate to steady state; convergence on mean-square change in Psi
// between successive analysis windows, MPI-reduced over fluid voxels.
// =========================================================================
void ScaLBL_Poisson::Run() {
    timestep = 0;
    double error = 1.0;

    if (rank == 0) {
        printf("ScaLBL_Poisson: solving (timestepMax = %d, tolerance = %.3e)\n",
               timestepMax, tolerance);
    }

    while (timestep < timestepMax && error > tolerance) {
        // odd half-step: stream + BCs, then collide
        timestep++;
        SolveElectricPotentialAAodd();
        SolvePoissonAAodd();
        ScaLBL_DeviceBarrier(); MPI_Barrier(comm);

        // even half-step: stream + BCs, then collide
        timestep++;
        SolveElectricPotentialAAeven();
        SolvePoissonAAeven();
        ScaLBL_DeviceBarrier(); MPI_Barrier(comm);

        if (timestep == 2) {
            ScaLBL_CopyToHost(Psi_previous.data(), Psi, sizeof(double) * Nx * Ny * Nz);
        }
        if (timestep % analysis_interval == 0) {
            ScaLBL_CopyToHost(Psi_host.data(), Psi, sizeof(double) * Nx * Ny * Nz);
            double mse_loc = 0.0, count_loc = 0.0;
            for (int k = 1; k < Nz - 1; k++) {
                for (int j = 1; j < Ny - 1; j++) {
                    for (int i = 1; i < Nx - 1; i++) {
                        int n = k * Nx * Ny + j * Nx + i;
                        if (Mask->id[n] > 0) {
                            double d = Psi_host(i, j, k) - Psi_previous(i, j, k);
                            mse_loc   += d * d;
                            count_loc += 1.0;
                        }
                    }
                }
            }
            double mse_global = 0.0, count_global = 0.0;
            MPI_Allreduce(&mse_loc,   &mse_global,   1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(&count_loc, &count_global, 1, MPI_DOUBLE, MPI_SUM, comm);
            error = (count_global > 0.0) ? (mse_global / count_global) : 0.0;
            if (rank == 0) printf("ScaLBL_Poisson:   timestep = %6d   MSE = %.4e\n", timestep, error);
            ScaLBL_CopyToHost(Psi_previous.data(), Psi, sizeof(double) * Nx * Ny * Nz);
        }
    }

    if (rank == 0) {
        if (error <= tolerance)
            printf("ScaLBL_Poisson: converged at t = %d (MSE = %.4e <= tol = %.3e)\n",
                   timestep, error, tolerance);
        else
            printf("ScaLBL_Poisson: WARNING -- exited at t = %d without converging "
                   "(MSE = %.4e > tol = %.3e)\n", timestep, error, tolerance);
    }
}

// =========================================================================
// Output: extract Psi (Nx, Ny, Nz) and the LB-units electric field
// (Ex, Ey, Ez), de-compressed via RegularLayout. Solid voxels read zero.
// =========================================================================
void ScaLBL_Poisson::getElectricPotential(DoubleArray &ReturnValues) {
    ScaLBL_CopyToHost(ReturnValues.data(), Psi, sizeof(double) * Nx * Ny * Nz);
}

void ScaLBL_Poisson::getElectricField(DoubleArray &Ex, DoubleArray &Ey, DoubleArray &Ez) {
    ScaLBL_Comm->RegularLayout(Map, &ElectricField[0 * Np], Ex);
    ScaLBL_Comm->RegularLayout(Map, &ElectricField[1 * Np], Ey);
    ScaLBL_Comm->RegularLayout(Map, &ElectricField[2 * Np], Ez);
}
