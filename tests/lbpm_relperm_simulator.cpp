/*
  lbpm_relperm_simulator: rel-perm driver.

  Expected workflow (see /media/user/Data0/lbpmslimRuns/relpermBereaTest/runFile.db):
    1. lbpm_serial_decomp  <input.db>   -- write per-rank ID.xxxxx
    2. lbpm_morphopen_pp   <input.db>   -- morphological drainage to
                                            an *initial* Sw (e.g. 0.9)
    3. lbpm_relperm_simulator <input.db>   -- this binary

  Step 3 runs the existing ScaLBL_ColorModel two-phase + automorph
  loop.  At every automorph steady-state detection event, the
  LBPMRelPermSimulator override of OnSteadyStatePoint() identifies the
  inlet->outlet connected component for each phase and runs a
  single-phase BGK with body force + periodic BC=0 on each phase to
  measure the effective absolute permeability.  Results land in
  RelPermSummary.csv (one row per phase per event).

  Usage: mpirun -np N lbpm_relperm_simulator inputFile.db
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "models/RelPermSimulator.h"

int main(int argc, char **argv)
{
    int rank, nprocs;
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (argc < 2) {
        if (rank == 0) printf("Usage: lbpm_relperm_simulator <input.db>\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("********************************************************\n");
        printf("Running LBPMRelPermSimulator\n");
        printf("  (color LBM + automorph + per-event single-phase BGK)\n");
        printf("********************************************************\n");
    }

    ScaLBL_SetDevice(rank);
    ScaLBL_DeviceBarrier();
    MPI_Barrier(comm);

    Utilities::setErrorHandlers();

    {
        ScaLBL_LBPMRelPermSimulator sim(rank, nprocs, comm);
        sim.ReadParamsRelPerm(argv[1]);
        sim.SetDomain();
        sim.ReadInput();
        sim.Create();
        sim.Initialize();
        sim.Run();
        MPI_Barrier(comm);
    }

    MPI_Finalize();
    return 0;
}
