#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <fstream>
#include <cstring>

#include "models/ColorModelSI.h"

/*
 * Two-phase flow simulator with SI unit inputs
 * Uses ColorModelSI to convert physical parameters to lattice units
 *
 * Usage:
 *   mpirun -np N lbpm_color_simulator_SI input.db              (SI mode)
 *   mpirun -np N lbpm_color_simulator_SI input.db --dimless     (dimensionless mode)
 */

using namespace std;

int main(int argc, char **argv)
{
	int rank, nprocs;
	MPI_Init(&argc, &argv);

	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &nprocs);

	// Check for --dimless flag
	bool dimensionless_mode = false;
	for (int i = 2; i < argc; i++) {
		if (strcmp(argv[i], "--dimless") == 0) {
			dimensionless_mode = true;
		}
	}

	if (rank == 0) printf("MPI Initialised\n");
	{
		if (rank == 0) {
			printf("********************************************************\n");
			if (dimensionless_mode)
				printf("Running Color LBM (Dimensionless Number Matching)\n");
			else
				printf("Running Color LBM (SI Unit Interface)\n");
			printf("********************************************************\n");
		}
		int device = ScaLBL_SetDevice(rank);
		ScaLBL_DeviceBarrier();
		MPI_Barrier(comm);

		Utilities::setErrorHandlers();

		auto filename = argv[1];
		ScaLBL_ColorModelSI ColorModel(rank, nprocs, comm);
		if (dimensionless_mode)
			ColorModel.ReadParamsDimensionless(filename);
		else
			ColorModel.ReadParams(filename);
		ColorModel.SetDomain();
		ColorModel.ReadInput();
		ColorModel.Create();
		ColorModel.Initialize();
		ColorModel.Run();

		MPI_Barrier(comm);
	}
	MPI_Finalize();
}
