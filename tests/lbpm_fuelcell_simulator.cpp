/*
  lbpm_fuelcell_simulator.cpp — Test driver for the Fuel Cell model

  Full MEA fuel cell operando simulator:
    - Multi-component multiphase (Color-Gradient + Shan-Chen + Greyscale)
    - Carnahan-Starling EoS
    - Butler-Volmer electrochemistry
    - Non-isothermal energy transport
    - Springer membrane model
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <fstream>

#include "models/FuelCellModel.h"

using namespace std;

int main(int argc, char **argv)
{
    int rank, nprocs;
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (rank == 0) printf("MPI Initialised\n");
    {
        if (rank == 0) {
            printf("********************************************************\n");
            printf("Running LBPM Fuel Cell Simulator\n");
            printf("********************************************************\n");
        }

        // Initialize compute device
        int device = ScaLBL_SetDevice(rank);
        ScaLBL_DeviceBarrier();
        MPI_Barrier(comm);

        Utilities::setErrorHandlers();

        auto filename = argv[1];
        ScaLBL_FuelCellModel FuelCellModel(rank, nprocs, comm);
        FuelCellModel.ReadParams(filename);
        FuelCellModel.SetDomain();
        FuelCellModel.ReadInput();
        FuelCellModel.Create();
        FuelCellModel.Initialize();
        FuelCellModel.Run();
        //FuelCellModel.WriteDebug();

        MPI_Barrier(comm);
    }
    MPI_Finalize();
}
