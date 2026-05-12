/*
  LBPMRelPermSimulator: rides the color model + automorph loop and runs
  per-phase single-phase BGK at each steady-state point to measure the
  effective absolute permeability of each phase's inlet->outlet
  connected component.
*/
#include "models/RelPermSimulator.h"

#include <algorithm>
#include <bits/stdc++.h>
#include <cmath>

using namespace std;

ScaLBL_LBPMRelPermSimulator::ScaLBL_LBPMRelPermSimulator(int RANK, int NP, MPI_Comm COMM)
    : ScaLBL_ColorModel(RANK, NP, COMM),
      rp_tau(0.7), rp_Fx(0), rp_Fy(0), rp_Fz(0),
      rp_timestepMax(200000), rp_analysis_interval(1000),
      rp_permTolerance(1e-5), rp_bgkFlag(true), rp_writeVisOnConv(false),
      rp_event_count(0),
      rp_NeighborList(nullptr), rp_fq(nullptr), rp_Velocity(nullptr),
      rp_Pressure(nullptr), rp_Np(0)
{
}

ScaLBL_LBPMRelPermSimulator::~ScaLBL_LBPMRelPermSimulator() {
    freePhaseLayout();
}

void ScaLBL_LBPMRelPermSimulator::ReadParamsRelPerm(std::string filename) {
    // Color-model side (Domain + Color + Analysis sections).
    ScaLBL_ColorModel::ReadParams(filename);

    // Layer on the RelPerm section.
    auto rp_db = db->getDatabase("RelPerm");
    rp_tau         = rp_db->getScalar<double>("tau");
    auto F         = rp_db->getVector<double>("F");
    rp_Fx = F[0]; rp_Fy = F[1]; rp_Fz = F[2];
    rp_timestepMax = rp_db->getScalar<int>("timestepMax");
    if (rp_db->keyExists("analysis_interval"))
        rp_analysis_interval = rp_db->getScalar<int>("analysis_interval");
    if (rp_db->keyExists("permTolerance"))
        rp_permTolerance = rp_db->getScalar<double>("permTolerance");
    if (rp_db->keyExists("bgkFlag"))
        rp_bgkFlag = rp_db->getScalar<bool>("bgkFlag");
    if (rp_db->keyExists("writeVisOnConv"))
        rp_writeVisOnConv = rp_db->getScalar<bool>("writeVisOnConv");

    if (rank == 0) {
        printf("================================================================\n");
        printf(" LBPMRelPermSimulator: single-phase BGK config\n");
        printf("================================================================\n");
        printf("  rp_tau              = %f (mu = %.6e)\n", rp_tau, (rp_tau - 0.5) / 3.0);
        printf("  rp_F                = (%.4e, %.4e, %.4e)\n", rp_Fx, rp_Fy, rp_Fz);
        printf("  rp_timestepMax      = %d\n", rp_timestepMax);
        printf("  rp_analysis_interval= %d\n", rp_analysis_interval);
        printf("  rp_permTolerance    = %.2e\n", rp_permTolerance);
        printf("  rp_bgkFlag          = %s\n", rp_bgkFlag ? "true" : "false");
        printf("  rp_writeVisOnConv   = %s\n", rp_writeVisOnConv ? "true" : "false");
        printf("================================================================\n");

        FILE *log = fopen("RelPermSummary.csv", "w");
        fprintf(log, "event timestep phase Sw vol_total vol_conn k_eff_m2 k_eff_Darcies F\n");
        fclose(log);
    }
}

int ScaLBL_LBPMRelPermSimulator::collateBlobs(int *&blobsGlob,
                                              std::vector<int> blobsLoc) {
    blobsLoc.erase(std::remove(blobsLoc.begin(), blobsLoc.end(), -2), blobsLoc.end());
    blobsLoc.erase(std::remove(blobsLoc.begin(), blobsLoc.end(), -1), blobsLoc.end());
    int sizeLocal = blobsLoc.size();

    std::vector<int> recvcounts(nprocs);
    MPI_Allgather(&sizeLocal, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, Dm->Comm);

    std::vector<int> displs(nprocs);
    displs[0] = 0;
    int totlen = recvcounts[0];
    for (int i = 1; i < nprocs; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
        totlen += recvcounts[i];
    }

    blobsGlob = (int *)malloc(totlen * sizeof(int));
    std::vector<int> temp(blobsLoc.begin(), blobsLoc.end());
    MPI_Allgatherv(temp.data(), sizeLocal, MPI_INT,
                   blobsGlob, recvcounts.data(), displs.data(), MPI_INT, Dm->Comm);
    return totlen;
}

long ScaLBL_LBPMRelPermSimulator::buildPhaseMask(int phaseID,
                                                 const IntArray &blob_label,
                                                 const std::vector<int> &connectedBlobs) {
    // Fresh Domain; previous one (if any) is released via shared_ptr.
    rp_Mask = std::make_shared<Domain>(domain_db, Dm->Comm);

    long localCount = 0;
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int n = k * Nx * Ny + j * Nx + i;
                bool isPhase = (rp_id_full[n] == phaseID);
                bool isKept  = std::find(connectedBlobs.begin(), connectedBlobs.end(),
                                         blob_label(i, j, k)) != connectedBlobs.end();
                if (isPhase && isKept) {
                    rp_Mask->id[n] = 1;
                    if (i >= 1 && i < Nx - 1 && j >= 1 && j < Ny - 1 && k >= 1 && k < Nz - 1)
                        localCount++;
                } else {
                    rp_Mask->id[n] = 0;
                }
            }
        }
    }
    long globalCount = 0;
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_LONG, MPI_SUM, Dm->Comm);
    return globalCount;
}

void ScaLBL_LBPMRelPermSimulator::createPhaseLayout() {
    rp_Mask->CommInit();
    rp_Comm = std::make_shared<ScaLBL_Communicator>(rp_Mask);
    rp_Np = rp_Mask->PoreCount();
    int Npad = (rp_Np / 16 + 2) * 16;
    rp_Map.resize(Nx, Ny, Nz);
    auto neighborList = new int[18 * Npad];
    rp_Np = rp_Comm->MemoryOptimizedLayoutAA(rp_Map, neighborList, rp_Mask->id, rp_Np);
    MPI_Barrier(Dm->Comm);

    ScaLBL_AllocateDeviceMemory((void **)&rp_NeighborList, 18 * rp_Np * sizeof(int));
    ScaLBL_AllocateDeviceMemory((void **)&rp_fq,           19 * rp_Np * sizeof(double));
    ScaLBL_AllocateDeviceMemory((void **)&rp_Pressure,          rp_Np * sizeof(double));
    ScaLBL_AllocateDeviceMemory((void **)&rp_Velocity,      3 * rp_Np * sizeof(double));

    ScaLBL_CopyToDevice(rp_NeighborList, neighborList, 18 * rp_Np * sizeof(int));
    delete[] neighborList;

    rp_Vx.resize(Nx, Ny, Nz);
    rp_Vy.resize(Nx, Ny, Nz);
    rp_Vz.resize(Nx, Ny, Nz);
    rp_P.resize(Nx, Ny, Nz);
    MPI_Barrier(Dm->Comm);
}

void ScaLBL_LBPMRelPermSimulator::freePhaseLayout() {
    if (rp_NeighborList) { ScaLBL_FreeDeviceMemory(rp_NeighborList); rp_NeighborList = nullptr; }
    if (rp_fq)           { ScaLBL_FreeDeviceMemory(rp_fq);           rp_fq = nullptr; }
    if (rp_Pressure)     { ScaLBL_FreeDeviceMemory(rp_Pressure);     rp_Pressure = nullptr; }
    if (rp_Velocity)     { ScaLBL_FreeDeviceMemory(rp_Velocity);     rp_Velocity = nullptr; }
}

double ScaLBL_LBPMRelPermSimulator::runPhaseBGK(int phaseID) {
    double rlx_setA = 1.0 / rp_tau;
    double rlx_setB = 8.0 * (2.0 - rlx_setA) / (8.0 - rlx_setA);
    double Kold     = 0.0;
    double absperm  = 0.0;
    double mu       = (rp_tau - 0.5) / 3.0;

    ScaLBL_D3Q19_Init(rp_fq, rp_Np);
    MPI_Barrier(Dm->Comm);

    if (rank == 0) {
        printf("[RelPerm event %d phase %d]: BGK loop start (Np=%d)\n",
               rp_event_count, phaseID, rp_Np);
    }

    double starttime = MPI_Wtime();
    double temptime  = starttime;

    double voxel = (domain_db->keyExists("voxel_length"))
                      ? domain_db->getScalar<double>("voxel_length") * 1.0e-6
                      : Lz / double(Nz * nprocz);

    int local_timestep = 0;
    char phase_log_name[80];
    sprintf(phase_log_name, "RelPermPhase%d_event%d.csv", phaseID, rp_event_count);
    if (rank == 0) {
        FILE *log = fopen(phase_log_name, "w");
        fprintf(log, "timestep Fx Fy Fz mu vax vay vaz absperm_m2 dKdK\n");
        fclose(log);
    }

    while (local_timestep < rp_timestepMax) {
        // ---- ODD ----
        local_timestep++;
        rp_Comm->SendD3Q19AA(rp_fq);
        if (rp_bgkFlag) {
            ScaLBL_D3Q19_AAodd_BGK(rp_NeighborList, rp_fq, rp_Comm->FirstInterior(),
                                   rp_Comm->LastInterior(), rp_Np, rlx_setA,
                                   rp_Fx, rp_Fy, rp_Fz);
        } else {
            ScaLBL_D3Q19_AAodd_MRT(rp_NeighborList, rp_fq, rp_Comm->FirstInterior(),
                                   rp_Comm->LastInterior(), rp_Np, rlx_setA, rlx_setB,
                                   rp_Fx, rp_Fy, rp_Fz);
        }
        rp_Comm->RecvD3Q19AA(rp_fq);
        ScaLBL_DeviceBarrier();
        if (rp_bgkFlag) {
            ScaLBL_D3Q19_AAodd_BGK(rp_NeighborList, rp_fq, 0, rp_Comm->LastExterior(),
                                   rp_Np, rlx_setA, rp_Fx, rp_Fy, rp_Fz);
        } else {
            ScaLBL_D3Q19_AAodd_MRT(rp_NeighborList, rp_fq, 0, rp_Comm->LastExterior(),
                                   rp_Np, rlx_setA, rlx_setB, rp_Fx, rp_Fy, rp_Fz);
        }
        ScaLBL_DeviceBarrier(); MPI_Barrier(Dm->Comm);

        // ---- EVEN ----
        local_timestep++;
        rp_Comm->SendD3Q19AA(rp_fq);
        rp_Comm->RecvD3Q19AA(rp_fq);
        ScaLBL_DeviceBarrier();
        if (rp_bgkFlag) {
            ScaLBL_D3Q19_AAeven_BGK(rp_fq, 0, rp_Comm->LastInterior(), rp_Np,
                                    rlx_setA, rp_Fx, rp_Fy, rp_Fz);
        } else {
            ScaLBL_D3Q19_AAeven_MRT(rp_fq, 0, rp_Comm->LastInterior(), rp_Np,
                                    rlx_setA, rlx_setB, rp_Fx, rp_Fy, rp_Fz);
        }
        ScaLBL_DeviceBarrier(); MPI_Barrier(Dm->Comm);

        if (local_timestep % rp_analysis_interval == 0 || local_timestep == 2) {
            ScaLBL_D3Q19_Momentum(rp_fq, rp_Velocity, rp_Np);
            ScaLBL_D3Q19_Pressure(rp_fq, rp_Pressure, rp_Np);
            ScaLBL_DeviceBarrier(); MPI_Barrier(Dm->Comm);

            rp_Comm->RegularLayout(rp_Map, &rp_Velocity[0],          rp_Vx);
            rp_Comm->RegularLayout(rp_Map, &rp_Velocity[rp_Np],      rp_Vy);
            rp_Comm->RegularLayout(rp_Map, &rp_Velocity[2 * rp_Np],  rp_Vz);

            double vax_loc = 0, vay_loc = 0, vaz_loc = 0;
            for (int k = 1; k < Nz - 1; k++) {
                for (int j = 1; j < Ny - 1; j++) {
                    for (int i = 1; i < Nx - 1; i++) {
                        int n = k * Nx * Ny + j * Nx + i;
                        if (rp_Mask->id[n] > 0) {
                            vax_loc += rp_Vx(i, j, k);
                            vay_loc += rp_Vy(i, j, k);
                            vaz_loc += rp_Vz(i, j, k);
                        }
                    }
                }
            }
            double vax, vay, vaz;
            MPI_Allreduce(&vax_loc, &vax, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);
            MPI_Allreduce(&vay_loc, &vay, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);
            MPI_Allreduce(&vaz_loc, &vaz, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);
            double cellsGlob = double(Nx - 2) * double(Ny - 2) * double(Nz - 2) * double(nprocs);
            vax /= cellsGlob; vay /= cellsGlob; vaz /= cellsGlob;

            double gradP = sqrt(rp_Fx*rp_Fx + rp_Fy*rp_Fy + rp_Fz*rp_Fz);
            double vMag  = sqrt(vax*vax + vay*vay + vaz*vaz);
            absperm  = voxel * voxel * mu * vMag / gradP;          // m^2
            double abspermZ = voxel * voxel * mu * vaz / gradP;    // m^2

            double convRate = (Kold > 0) ? fabs((absperm - Kold) / Kold) : 1.0;
            double stoptime  = MPI_Wtime();
            double cputime   = stoptime - starttime;
            double deltatime = stoptime - temptime;
            temptime = stoptime;
            double MLUPS_loc  = double(rp_Np) * rp_analysis_interval / deltatime / 1.0e6;
            double MLUPS_glob;
            MPI_Allreduce(&MLUPS_loc, &MLUPS_glob, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);

            if (rank == 0) {
                printf("[RelPerm event %d phase %d] t=%d, MLUPS=%.1f, "
                       "K=%.4f D (RMS), %.4f D (Z), wall=%.2fs, dK/K=%.4e, "
                       "v_D=(%.3e %.3e %.3e), |F|=%.3e\n",
                       rp_event_count, phaseID, local_timestep, MLUPS_glob,
                       absperm / 9.869233e-13, abspermZ / 9.869233e-13,
                       cputime, convRate, vax, vay, vaz, gradP);
                FILE *log = fopen(phase_log_name, "a");
                fprintf(log, "%d %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g %.8g\n",
                        local_timestep, rp_Fx, rp_Fy, rp_Fz, mu,
                        vax, vay, vaz, absperm, convRate);
                fclose(log);
            }
            if (std::isnan(vMag) || vMag == 0.0) {
                if (rank == 0)
                    printf("[RelPerm event %d phase %d]: NaN/zero velocity, terminating.\n",
                           rp_event_count, phaseID);
                break;
            }
            if (convRate < rp_permTolerance && local_timestep > 4 * rp_analysis_interval) {
                if (rank == 0)
                    printf("[RelPerm event %d phase %d]: converged (dK/K=%.4e).\n",
                           rp_event_count, phaseID, convRate);
                if (rp_writeVisOnConv) writeVelPField(phaseID);
                break;
            }
            Kold = absperm;
        }
    }
    return absperm;
}

void ScaLBL_LBPMRelPermSimulator::writeVelPField(int phaseID) {
    char folder[120];
    if (rank == 0) {
        sprintf(folder, "./rawVisVelP_phase%d_event%d", phaseID, rp_event_count);
        mkdir(folder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    MPI_Barrier(Dm->Comm);

    char fname[200];
    sprintf(fname, "rawVisVelP_phase%d_event%d/Part_%d_%d_%d_%d_%d_%d_%d.txt",
            phaseID, rp_event_count, rank, Nx, Ny, Nz, nprocx, nprocy, nprocz);
    FILE *out = fopen(fname, "wb");

    rp_Comm->RegularLayout(rp_Map, &rp_Pressure[0], rp_P);
    double temp;
    auto dumpField = [&](DoubleArray &fld) {
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    if (rp_Map(i, j, k) >= 0) {
                        temp = fld(i, j, k);
                        fwrite(&temp, sizeof(double), 1, out);
                    }
                }
            }
        }
    };
    dumpField(rp_Vx);
    dumpField(rp_Vy);
    dumpField(rp_Vz);
    dumpField(rp_P);
    fclose(out);
    MPI_Barrier(Dm->Comm);
}

// =========================================================================
// Steady-state hook: called by ScaLBL_ColorModel::Run() each time the
// automorph detects a steady state.  We:
//   1. Build a snapshot id (0/1/2) from PhaseField (which is up-to-date
//      on host because the color-model analysis block just copied Phi).
//   2. Run connected-component analysis on both phases.
//   3. For each phase: build a single-phase Mask, allocate ScaLBL
//      state, run BGK with body force under BC=0 periodic to
//      convergence, record k_eff to RelPermSummary.csv.
// =========================================================================
void ScaLBL_LBPMRelPermSimulator::OnSteadyStatePoint() {
    rp_event_count++;
    if (rank == 0)
        printf("\n========== [RelPerm event %d] steady-state @ t=%d, Sw=%.4f ==========\n",
               rp_event_count, timestep,
               (volA + volB > 0) ? volB / (volA + volB) : 0.0);

    // 1. Build id snapshot from PhaseField + the original solid mask.
    //    PhaseField was refreshed from device Phi in the analysis block
    //    earlier this iteration.  Color-model Dm->id[n] still flags
    //    solid voxels (immobile labels set Dm->id[n]=0 in
    //    AssignComponentLabels), so we read that for solid detection.
    rp_id_full.assign(N, 0);
    for (int n = 0; n < N; n++) {
        if (Dm->id[n] == 0) {
            rp_id_full[n] = 0;        // solid
        } else if (PhaseField(n) > 0.0) {
            rp_id_full[n] = 1;        // phase A (NWP)
        } else if (PhaseField(n) < 0.0) {
            rp_id_full[n] = 2;        // phase B (WP)
        } else {
            // phi == 0 along the interface or wherever -- bin into the
            // phase whose mass is larger nearby; default to solid so
            // the voxel does not contribute spuriously.
            rp_id_full[n] = 0;
        }
    }

    // 2. Connected-component analysis on the steady-state phase field.
    const RankInfoStruct rank_info(rank, nprocx, nprocy, nprocz);
    IntArray NWP_label(Nx, Ny, Nz), WP_label(Nx, Ny, Nz);
    double vF = 0.f, vS = 0.f;
    ComputeGlobalBlobIDs(Nx - 2, Ny - 2, Nz - 2, rank_info, PhaseField, Distance,
                         vF, vS, NWP_label, Dm->Comm);
    MPI_Barrier(Dm->Comm);
    PhaseField.scale(-1);
    ComputeGlobalBlobIDs(Nx - 2, Ny - 2, Nz - 2, rank_info, PhaseField, Distance,
                         vF, vS, WP_label, Dm->Comm);
    PhaseField.scale(-1);
    MPI_Barrier(Dm->Comm);

    // 3. Inlet (kproc=0, k=1) and outlet (kproc=nprocz-1, k=Nz-2) blob
    //    IDs per rank.
    int kproc = Dm->kproc();
    std::vector<int> inA{-1, -2}, outA{-1, -2}, inB{-1, -2}, outB{-1, -2};
    if (kproc == 0) {
        int k = 1;
        for (int j = 1; j < Ny - 1; j++) {
            for (int i = 1; i < Nx - 1; i++) {
                int a = NWP_label(i, j, k);
                int b = WP_label(i, j, k);
                if (std::find(inA.begin(), inA.end(), a) == inA.end()) inA.push_back(a);
                if (std::find(inB.begin(), inB.end(), b) == inB.end()) inB.push_back(b);
            }
        }
    }
    if (kproc == nprocz - 1) {
        int k = Nz - 2;
        for (int j = 1; j < Ny - 1; j++) {
            for (int i = 1; i < Nx - 1; i++) {
                int a = NWP_label(i, j, k);
                int b = WP_label(i, j, k);
                if (std::find(outA.begin(), outA.end(), a) == outA.end()) outA.push_back(a);
                if (std::find(outB.begin(), outB.end(), b) == outB.end()) outB.push_back(b);
            }
        }
    }
    MPI_Barrier(Dm->Comm);

    int *inAG = nullptr, *outAG = nullptr, *inBG = nullptr, *outBG = nullptr;
    int nInA  = collateBlobs(inAG,  inA);
    int nOutA = collateBlobs(outAG, outA);
    int nInB  = collateBlobs(inBG,  inB);
    int nOutB = collateBlobs(outBG, outB);

    std::sort(inAG,  inAG  + nInA);
    std::sort(outAG, outAG + nOutA);
    std::sort(inBG,  inBG  + nInB);
    std::sort(outBG, outBG + nOutB);

    std::vector<int> connNWP, connWP;
    {
        std::vector<int> v(nInA + nOutA);
        auto it = std::set_intersection(inAG, inAG + nInA, outAG, outAG + nOutA, v.begin());
        for (auto st = v.begin(); st != it; ++st) connNWP.push_back(*st);
        std::sort(connNWP.begin(), connNWP.end());
        connNWP.erase(std::unique(connNWP.begin(), connNWP.end()), connNWP.end());
    }
    {
        std::vector<int> v(nInB + nOutB);
        auto it = std::set_intersection(inBG, inBG + nInB, outBG, outBG + nOutB, v.begin());
        for (auto st = v.begin(); st != it; ++st) connWP.push_back(*st);
        std::sort(connWP.begin(), connWP.end());
        connWP.erase(std::unique(connWP.begin(), connWP.end()), connWP.end());
    }
    free(inAG); free(outAG); free(inBG); free(outBG);

    if (rank == 0) {
        printf("[RelPerm event %d]: connected phase A blob IDs: ", rp_event_count);
        for (int b : connNWP) printf("%d ", b);
        printf("\n[RelPerm event %d]: connected phase B blob IDs: ", rp_event_count);
        for (int b : connWP)  printf("%d ", b);
        printf("\n");
    }

    double Sw = (volA + volB > 0) ? volB / (volA + volB) : 0.0;
    double Fmag = sqrt(rp_Fx*rp_Fx + rp_Fy*rp_Fy + rp_Fz*rp_Fz);

    // 4. Per-phase single-phase BGK run.
    for (int phaseID = 1; phaseID <= 2; phaseID++) {
        const std::vector<int> &conn = (phaseID == 1) ? connNWP : connWP;
        const IntArray &lab          = (phaseID == 1) ? NWP_label : WP_label;
        double phase_vol_total       = (phaseID == 1) ? volA : volB;

        long phaseCount = buildPhaseMask(phaseID, lab, conn);
        if (phaseCount == 0) {
            if (rank == 0) {
                printf("[RelPerm event %d phase %d]: no inlet->outlet connected voxels -- k_eff = 0\n",
                       rp_event_count, phaseID);
                FILE *log = fopen("RelPermSummary.csv", "a");
                fprintf(log, "%d %d %d %.6f %.0f %.0f 0.0 0.0 %.6e\n",
                        rp_event_count, timestep, phaseID, Sw,
                        phase_vol_total, double(phaseCount), Fmag);
                fclose(log);
            }
            continue;
        }

        createPhaseLayout();
        double k_eff = runPhaseBGK(phaseID);
        freePhaseLayout();
        rp_Mask.reset();
        rp_Comm.reset();

        if (rank == 0) {
            printf("[RelPerm event %d phase %d]: k_eff = %.6e m^2 (%.4f D), "
                   "Sw = %.4f, connected voxels = %ld\n",
                   rp_event_count, phaseID, k_eff, k_eff / 9.869233e-13,
                   Sw, phaseCount);
            FILE *log = fopen("RelPermSummary.csv", "a");
            fprintf(log, "%d %d %d %.6f %.0f %.0f %.6e %.6f %.6e\n",
                    rp_event_count, timestep, phaseID, Sw,
                    phase_vol_total, double(phaseCount),
                    k_eff, k_eff / 9.869233e-13, Fmag);
            fclose(log);
        }
        MPI_Barrier(Dm->Comm);
    }

    if (rank == 0)
        printf("========== [RelPerm event %d] complete, resuming color LBM ==========\n\n",
               rp_event_count);
}
