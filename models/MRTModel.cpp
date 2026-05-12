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
  ScaLBL_MRTModel: single-phase D3Q19 MRT permeability solver.
*/
#include "models/MRTModel.h"

#include <cmath>

using namespace std;

ScaLBL_MRTModel::ScaLBL_MRTModel(int RANK, int NP, MPI_Comm COMM)
    : Restart(false), timestep(0), timestepMax(0), BoundaryCondition(0),
      tau(0), Fx(0), Fy(0), Fz(0), flux(0), din(0), dout(0),
      Nx(0), Ny(0), Nz(0), N(0), Np(0),
      rank(RANK), nprocs(NP), nprocx(0), nprocy(0), nprocz(0),
      Lx(0), Ly(0), Lz(0), voxelSize(0), porosity(0),
      analysis_interval(1000), visInterval(0), restart_interval(0),
      permTolerance(1e-5), visTolerance(true), fqFlag(false),
      restartFq(false), logFile(true),
      NeighborList(nullptr), fq(nullptr), Velocity(nullptr), Pressure(nullptr),
      comm(COMM)
{
}

ScaLBL_MRTModel::~ScaLBL_MRTModel() {
    if (NeighborList) ScaLBL_FreeDeviceMemory(NeighborList);
    if (fq)           ScaLBL_FreeDeviceMemory(fq);
    if (Velocity)     ScaLBL_FreeDeviceMemory(Velocity);
    if (Pressure)     ScaLBL_FreeDeviceMemory(Pressure);
}

// =========================================================================
// ReadParams: pull the "MRT" + "Domain" sections from the input db.
// =========================================================================
void ScaLBL_MRTModel::ReadParams(std::string filename) {
    db        = std::make_shared<Database>(filename);
    domain_db = db->getDatabase("Domain");
    mrt_db    = db->getDatabase("MRT");

    // ---- Required MRT parameters ----
    timestepMax = mrt_db->getScalar<int>("timestepMax");
    tau         = mrt_db->getScalar<double>("tau");
    auto F      = mrt_db->getVector<double>("F");
    Fx = F[0]; Fy = F[1]; Fz = F[2];
    Restart     = mrt_db->getScalar<bool>("Restart");
    din         = mrt_db->getScalar<double>("din");
    dout        = mrt_db->getScalar<double>("dout");
    flux        = mrt_db->getScalar<double>("flux");

    // ---- Optional MRT parameters (analysis / I/O) ----
    if (mrt_db->keyExists("analysis_interval"))
        analysis_interval = mrt_db->getScalar<int>("analysis_interval");
    if (mrt_db->keyExists("visInterval"))
        visInterval = mrt_db->getScalar<int>("visInterval");
    if (mrt_db->keyExists("restart_interval"))
        restart_interval = mrt_db->getScalar<int>("restart_interval");
    if (mrt_db->keyExists("permTolerance"))
        permTolerance = mrt_db->getScalar<double>("permTolerance");
    if (mrt_db->keyExists("visTolerance"))
        visTolerance = mrt_db->getScalar<bool>("visTolerance");
    if (mrt_db->keyExists("fqFlag"))
        fqFlag = mrt_db->getScalar<bool>("fqFlag");
    if (mrt_db->keyExists("restartFq"))
        restartFq = mrt_db->getScalar<bool>("restartFq");
    if (mrt_db->keyExists("logFile"))
        logFile = mrt_db->getScalar<bool>("logFile");

    // ---- Domain parameters ----
    auto L     = domain_db->getVector<double>("L");
    auto size  = domain_db->getVector<int>("n");
    auto nproc = domain_db->getVector<int>("nproc");
    BoundaryCondition = domain_db->getScalar<int>("BC");
    Nx = size[0];  Ny = size[1];  Nz = size[2];
    Lx = L[0];     Ly = L[1];     Lz = L[2];
    nprocx = nproc[0]; nprocy = nproc[1]; nprocz = nproc[2];

    if (domain_db->keyExists("voxel_length")) {
        voxelSize = domain_db->getScalar<double>("voxel_length") * 1.0e-6; // um -> m
    } else {
        voxelSize = Lz / double(Nz * nprocz);
    }

    if (rank == 0) {
        printf("================================================================\n");
        printf("  ScaLBL_MRTModel: single-phase MRT permeability solver\n");
        printf("================================================================\n");
        printf("  tau           = %f  (mu_LB = %.6e)\n", tau, (tau - 0.5) / 3.0);
        printf("  F             = (%.4e, %.4e, %.4e)\n", Fx, Fy, Fz);
        printf("  BC            = %d\n", BoundaryCondition);
        if (BoundaryCondition == 3) printf("  din/dout      = %.4f / %.4f\n", din, dout);
        if (BoundaryCondition == 4) printf("  flux          = %.4e\n", flux);
        printf("  timestepMax   = %d\n", timestepMax);
        printf("  voxel size    = %.4f microns\n", voxelSize * 1.0e6);
        printf("================================================================\n");
    }
}

// =========================================================================
// SetDomain: build the Mask domain and resize cartesian scratch arrays.
// =========================================================================
void ScaLBL_MRTModel::SetDomain() {
    Mask = std::make_shared<Domain>(domain_db, comm);
    Nx += 2; Ny += 2; Nz += 2;
    N = Nx * Ny * Nz;

    Geom.resize(Nx, Ny, Nz);
    Velocity_x.resize(Nx, Ny, Nz);
    Velocity_y.resize(Nx, Ny, Nz);
    Velocity_z.resize(Nx, Ny, Nz);
    Pressure_Cart.resize(Nx, Ny, Nz);
    fqTemp.resize(Nx, Ny, Nz);

    if (rank == 0) cout << "Cartesian scratch arrays allocated." << endl;
}

// =========================================================================
// ReadInput: load the per-rank ID file and build the Geom indicator.
// =========================================================================
void ScaLBL_MRTModel::ReadInput() {
    Mask->ReadIDs();
    sprintf(LocalRankString,   "%05d", Mask->rank());
    sprintf(LocalRankFilename, "ID.%s",         LocalRankString);
    sprintf(LocalRestartFile,  "Restart.%s",    LocalRankString);

    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int n = k * Nx * Ny + j * Nx + i;
                Geom(i, j, k) = (Mask->id[n] > 0) ? 1.0 : 0.0;
            }
        }
    }
    if (rank == 0) cout << "Geometry loaded." << endl;
}

// =========================================================================
// Create: build ScaLBL communicator, memory-optimised layout, and GPU buffers.
// =========================================================================
void ScaLBL_MRTModel::Create() {
    Mask->CommInit();
    if (rank == 0) printf("Create ScaLBL_Communicator\n");
    ScaLBL_Comm = std::make_shared<ScaLBL_Communicator>(Mask);

    Np = Mask->PoreCount();
    porosity = Mask->Porosity();
    int Npad = (Np / 16 + 2) * 16;
    Map.resize(Nx, Ny, Nz);
    auto neighborList = new int[18 * Npad];
    Np = ScaLBL_Comm->MemoryOptimizedLayoutAA(Map, neighborList, Mask->id, Np);
    MPI_Barrier(comm);

    if (rank == 0) printf("Allocating distributions (Np = %d, porosity = %.4f)\n", Np, porosity);
    ScaLBL_AllocateDeviceMemory((void **)&NeighborList, 18 * Np * sizeof(int));
    ScaLBL_AllocateDeviceMemory((void **)&fq,           19 * Np * sizeof(double));
    ScaLBL_AllocateDeviceMemory((void **)&Pressure,          Np * sizeof(double));
    ScaLBL_AllocateDeviceMemory((void **)&Velocity,      3 * Np * sizeof(double));

    ScaLBL_CopyToDevice(NeighborList, neighborList, 18 * Np * sizeof(int));
    delete[] neighborList;
    MPI_Barrier(comm);
}

// =========================================================================
// Initialize: zero-init fq, optionally restart from disk.
// =========================================================================
void ScaLBL_MRTModel::Initialize() {
    if (rank == 0) printf("Initializing fq distribution\n");
    ScaLBL_D3Q19_Init(fq, Np);
    if (!restartFq) return;

    if (rank == 0) printf("Reading restart file\n");
    ifstream rst("Restart.txt");
    if (rst.is_open()) {
        rst >> timestep;
        rst.close();
    }
    if (rank == 0) printf("Restarting from timestep %d\n", timestep);

    MPI_Barrier(comm);

    char fname[120];
    sprintf(fname, "restartFq_Part_%d_%d_%d_%d_%d_%d_%d.txt",
            rank, Nx, Ny, Nz, nprocx, nprocy, nprocz);
    FILE *in = fopen(fname, "rb");
    if (in == nullptr) {
        if (rank == 0) printf("WARNING: restart file %s not found; starting from t=0 init\n", fname);
        timestep = 0;
        return;
    }

    double *mrtDist = new double[19 * Np];
    for (int d = 0; d < 19; d++) {
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    int idx = Map(i, j, k);
                    if (idx < 0) continue;
                    double val = 0.0;
                    size_t nr = fread(&val, sizeof(double), 1, in);
                    if (nr != 1) {
                        if (rank == 0) printf("WARNING: short read on restart file %s\n", fname);
                    }
                    mrtDist[d * Np + idx] = val;
                }
            }
        }
    }
    fclose(in);
    ScaLBL_CopyToDevice(fq, mrtDist, 19 * Np * sizeof(double));
    ScaLBL_DeviceBarrier();
    MPI_Barrier(comm);
    delete[] mrtDist;
}

// =========================================================================
// applyBoundaryConditions: BC=3 starts in flux mode and "ramps" din up
// to the user-specified target as the flow develops; BC=4 is a fixed
// flux at the inlet with fixed dout at the outlet.  Called between the
// streaming and collision halves of each half-step.
// =========================================================================
void ScaLBL_MRTModel::applyBoundaryConditions() {
    static double dinTarget = 0.0;          // set once, on first call
    static bool   initialized = false;
    if (BoundaryCondition == 3 && !initialized) {
        dinTarget = din;
        din = dout;
        if (flux == 0.0) flux = 1000.0 * porosity * porosity * porosity;
        initialized = true;
    }
    if (BoundaryCondition == 3) {
        if (din > dinTarget)
            ScaLBL_Comm->D3Q19_Pressure_BC_z(NeighborList, fq, din, timestep);
        else
            din = ScaLBL_Comm->D3Q19_Flux_BC_z(NeighborList, fq, flux, timestep);
        ScaLBL_Comm->D3Q19_Pressure_BC_Z(NeighborList, fq, dout, timestep);
    } else if (BoundaryCondition == 4) {
        din = ScaLBL_Comm->D3Q19_Flux_BC_z(NeighborList, fq, flux, timestep);
        ScaLBL_Comm->D3Q19_Pressure_BC_Z(NeighborList, fq, dout, timestep);
    }
}

// =========================================================================
// Run: D3Q19 MRT main loop with A-A streaming.  Records permeability
// every analysis_interval and exits early on |dK/K| < permTolerance.
// =========================================================================
void ScaLBL_MRTModel::Run() {
    const double rlxA   = 1.0 / tau;
    const double rlxB   = 8.0 * (2.0 - rlxA) / (8.0 - rlxA);
    const double mu_LB  = (tau - 0.5) / 3.0;
    const double DARCY_M2 = 9.869233e-13;
    double Kold = 0.0;

    if (rank == 0 && logFile) {
        FILE *log = fopen("Permeability.csv", "w");
        fprintf(log, "timestep Fx Fy Fz din dout mu vax vay vaz absperm_m2\n");
        fclose(log);
    }

    ScaLBL_DeviceBarrier();
    MPI_Barrier(comm);
    double startTime = MPI_Wtime();
    double prevTime  = startTime;

    if (rank == 0) {
        printf("No. of timesteps: %d, Boundary Condition: %d\n", timestepMax, BoundaryCondition);
        printf("********************************************************\n");
    }

    while (timestep < timestepMax) {
        // ---- ODD half-step (collide interior, then BCs, then collide exterior) ----
        timestep++;
        ScaLBL_Comm->SendD3Q19AA(fq);
        ScaLBL_D3Q19_AAodd_MRT(NeighborList, fq,
                               ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(),
                               Np, rlxA, rlxB, Fx, Fy, Fz);
        ScaLBL_Comm->RecvD3Q19AA(fq);
        ScaLBL_DeviceBarrier();
        applyBoundaryConditions();
        ScaLBL_D3Q19_AAodd_MRT(NeighborList, fq,
                               0, ScaLBL_Comm->LastExterior(),
                               Np, rlxA, rlxB, Fx, Fy, Fz);
        ScaLBL_DeviceBarrier(); MPI_Barrier(comm);

        // ---- EVEN half-step (single pass) ----
        timestep++;
        ScaLBL_Comm->SendD3Q19AA(fq);
        ScaLBL_Comm->RecvD3Q19AA(fq);
        ScaLBL_DeviceBarrier();
        applyBoundaryConditions();
        ScaLBL_D3Q19_AAeven_MRT(fq, 0, ScaLBL_Comm->LastInterior(),
                                Np, rlxA, rlxB, Fx, Fy, Fz);
        ScaLBL_DeviceBarrier(); MPI_Barrier(comm);

        // ---- Analysis ----
        if (timestep % analysis_interval == 0 || timestep == 2) {
            ScaLBL_D3Q19_Momentum(fq, Velocity, Np);
            ScaLBL_D3Q19_Pressure(fq, Pressure, Np);
            ScaLBL_DeviceBarrier(); MPI_Barrier(comm);

            ScaLBL_Comm->RegularLayout(Map, &Velocity[0],      Velocity_x);
            ScaLBL_Comm->RegularLayout(Map, &Velocity[Np],     Velocity_y);
            ScaLBL_Comm->RegularLayout(Map, &Velocity[2 * Np], Velocity_z);

            // Sum local velocities over fluid voxels.
            double vax_loc = 0, vay_loc = 0, vaz_loc = 0;
            for (int k = 1; k < Nz - 1; k++) {
                for (int j = 1; j < Ny - 1; j++) {
                    for (int i = 1; i < Nx - 1; i++) {
                        if (Geom(i, j, k) <= 0) continue;
                        vax_loc += Velocity_x(i, j, k);
                        vay_loc += Velocity_y(i, j, k);
                        vaz_loc += Velocity_z(i, j, k);
                    }
                }
            }
            double vax, vay, vaz;
            MPI_Allreduce(&vax_loc, &vax, 1, MPI_DOUBLE, MPI_SUM, Mask->Comm);
            MPI_Allreduce(&vay_loc, &vay, 1, MPI_DOUBLE, MPI_SUM, Mask->Comm);
            MPI_Allreduce(&vaz_loc, &vaz, 1, MPI_DOUBLE, MPI_SUM, Mask->Comm);
            double cellsGlob = double(Nx - 2) * double(Ny - 2) * double(Nz - 2) * double(nprocs);
            vax /= cellsGlob;  vay /= cellsGlob;  vaz /= cellsGlob;

            double gradP   = sqrt(Fx*Fx + Fy*Fy + Fz*Fz)
                           + (din - dout) / ((Nz - 2) * nprocz) / 3.0;
            double vMag    = sqrt(vax*vax + vay*vay + vaz*vaz);
            double absperm  = voxelSize * voxelSize * mu_LB * vMag / gradP;   // m^2
            double abspermZ = voxelSize * voxelSize * mu_LB * vaz  / gradP;   // m^2
            double convRate = (Kold > 0.0) ? fabs((absperm - Kold) / Kold) : 1.0;

            if (std::isnan(vMag) || vMag == 0.0 || std::isnan(gradP)) {
                if (rank == 0) printf("NaN/zero velocity or gradP -- terminating\n");
                break;
            }

            double stopTime  = MPI_Wtime();
            double cpuTime   = stopTime - startTime;
            double deltaTime = stopTime - prevTime;
            prevTime         = stopTime;
            double MLUPS_loc  = double(Np) * analysis_interval / deltaTime / 1.0e6;
            double MLUPS_glob;
            MPI_Allreduce(&MLUPS_loc, &MLUPS_glob, 1, MPI_DOUBLE, MPI_SUM, Mask->Comm);

            if (rank == 0) {
                printf("Timestep: %d, MLUPS: %.1f, K = %.6f D (RMS), %.6f D (Z), "
                       "wall = %.2fs, dK/K = %.4e, gradP = %.4e\n",
                       timestep, MLUPS_glob,
                       absperm / DARCY_M2, abspermZ / DARCY_M2,
                       cpuTime, convRate, gradP);
                if (logFile) {
                    FILE *log = fopen("Permeability.csv", "a");
                    fprintf(log, "%d %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n",
                            timestep, Fx, Fy, Fz, din, dout, mu_LB,
                            vax, vay, vaz, absperm);
                    fclose(log);
                }
            }
            Kold = absperm;
            if (convRate < permTolerance) {
                if (rank == 0) printf("Convergence reached (|dK/K| = %.4e < %.4e)\n",
                                      convRate, permTolerance);
                if (visTolerance) {
                    if (fqFlag) writeFqField();
                    else        writeVelocityPressure();
                }
                break;
            }
        }

        // ---- Periodic snapshots ----
        if (visInterval > 0 && timestep % visInterval == 0) {
            if (fqFlag) writeFqField();
            else        writeVelocityPressure();
        }
        if (restart_interval > 0 && timestep % restart_interval == 0) {
            writeRestart();
        }
    }

    // Final dump even if loop exits via timestepMax.
    if (fqFlag) writeFqField();
    else        writeVelocityPressure();
}

// =========================================================================
// I/O: rank-local binary dumps.  All output paths are timestep-tagged so
// repeated runs in the same directory do not clobber one another.
// =========================================================================
static void mkdirRank0(const char *path, int rank, MPI_Comm comm) {
    if (rank == 0) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    MPI_Barrier(comm);
}

void ScaLBL_MRTModel::writeVelocityPressure() {
    char folder[120];
    sprintf(folder, "./rawVisVelP%d", timestep);
    mkdirRank0(folder, rank, comm);

    char fname[200];
    sprintf(fname, "rawVisVelP%d/Part_%d_%d_%d_%d_%d_%d_%d.txt",
            timestep, rank, Nx, Ny, Nz, nprocx, nprocy, nprocz);
    FILE *out = fopen(fname, "wb");

    ScaLBL_Comm->RegularLayout(Map, &Pressure[0], Pressure_Cart);
    double temp;
    auto dumpField = [&](DoubleArray &fld) {
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    if (Map(i, j, k) < 0) continue;
                    temp = fld(i, j, k);
                    fwrite(&temp, sizeof(double), 1, out);
                }
            }
        }
    };
    dumpField(Velocity_x);
    dumpField(Velocity_y);
    dumpField(Velocity_z);
    dumpField(Pressure_Cart);
    fclose(out);
    MPI_Barrier(comm);
}

void ScaLBL_MRTModel::writeFqField() {
    char folder[120];
    sprintf(folder, "./rawVisFq%d", timestep);
    mkdirRank0(folder, rank, comm);

    char fname[200];
    sprintf(fname, "rawVisFq%d/Part_%d_%d_%d_%d_%d_%d_%d.txt",
            timestep, rank, Nx, Ny, Nz, nprocx, nprocy, nprocz);
    FILE *out = fopen(fname, "wb");

    double temp;
    for (int d = 0; d < 19; d++) {
        ScaLBL_Comm->RegularLayout(Map, &fq[d * Np], fqTemp);
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    if (Map(i, j, k) < 0) continue;
                    temp = fqTemp(i, j, k);
                    fwrite(&temp, sizeof(double), 1, out);
                }
            }
        }
    }
    fclose(out);
    MPI_Barrier(comm);
}

void ScaLBL_MRTModel::writeRestart() {
    if (rank == 0) {
        FILE *rst = fopen("Restart.txt", "w");
        fprintf(rst, "%d\n", timestep);
        fclose(rst);
    }
    MPI_Barrier(comm);

    char fname[200];
    sprintf(fname, "restartFq_Part_%d_%d_%d_%d_%d_%d_%d.txt",
            rank, Nx, Ny, Nz, nprocx, nprocy, nprocz);
    FILE *out = fopen(fname, "wb");

    double temp;
    for (int d = 0; d < 19; d++) {
        ScaLBL_Comm->RegularLayout(Map, &fq[d * Np], fqTemp);
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    if (Map(i, j, k) < 0) continue;
                    temp = fqTemp(i, j, k);
                    fwrite(&temp, sizeof(double), 1, out);
                }
            }
        }
    }
    fclose(out);
    MPI_Barrier(comm);
}
