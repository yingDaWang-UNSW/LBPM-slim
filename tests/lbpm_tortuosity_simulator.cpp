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
  lbpm_tortuosity_simulator -- diffusive tortuosity of a porous sample.

  Drives ScaLBL_Poisson (D3Q7 LBM Laplace solver) on the pore phase with
  Dirichlet potentials Vin at z = 0 and Vout at z = L, no-flux on solids.
  At steady state, integrates the lattice-units electric field E_z over
  the full domain to recover D_eff/D_0, and reports:

    porosity         phi  = N_fluid / V_total
    diffusivity      D_eff/D_0 = <E_z>_vol * Lz / (Vin - Vout)
    tortuosity       tau  = phi / (D_eff/D_0)         (geometric)
    formation factor F    = D_0 / D_eff               (= tau / phi)

  Direction: the BC infrastructure ports from gaslbm only supports the
  z-axis. To compute x- or y-direction tortuosity, rotate the input
  geometry so the desired flow direction maps to z, then re-run.

  Input db:
    Domain { n, nproc, Filename, voxel_length, BC }
    Poisson {
      tau, tolerance, timestepMax, analysis_interval,
      Vin = 1.0, Vout = 0.0,
      BC_Inlet  = 1,   // Dirichlet
      BC_Outlet = 1,
    }
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#include "common/ScaLBL.h"
#include "common/Communication.h"
#include "common/MPI_Helpers.h"
#include "models/PoissonSolver.h"

using namespace std;

int main(int argc, char **argv) {
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    {
        if (argc < 2) {
            if (rank == 0)
                printf("usage: lbpm_tortuosity_simulator <input.db>\n");
            MPI_Finalize();
            return 1;
        }

        if (rank == 0) {
            printf("********************************************************\n");
            printf("  Running Tortuosity Calculation (D3Q7 LBM Laplace)\n");
            printf("********************************************************\n");
        }

        ScaLBL_SetDevice(rank);
        ScaLBL_DeviceBarrier();
        MPI_Barrier(comm);

        // -- solve the steady-state Laplace problem --
        ScaLBL_Poisson Poisson(rank, nprocs, comm);
        Poisson.ReadParams(argv[1]);
        Poisson.SetDomain();
        Poisson.ReadInput();
        Poisson.Create();
        Poisson.Initialize();
        Poisson.Run();

        // -- post-process: pull E and psi back, compute integrals --
        const int Nx = Poisson.Nx, Ny = Poisson.Ny, Nz = Poisson.Nz;
        DoubleArray Psi(Nx, Ny, Nz);
        DoubleArray Ex (Nx, Ny, Nz), Ey(Nx, Ny, Nz), Ez(Nx, Ny, Nz);
        Poisson.getElectricPotential(Psi);
        Poisson.getElectricField(Ex, Ey, Ez);

        const int    nprocz = Poisson.nprocz;
        const int    nprocx = Poisson.nprocx;
        const int    nprocy = Poisson.nprocy;
        const long   Nx_g = (long)(Nx - 2) * nprocx;
        const long   Ny_g = (long)(Ny - 2) * nprocy;
        const long   Nz_g = (long)(Nz - 2) * nprocz;
        const long   V_g  = Nx_g * Ny_g * Nz_g;
        const double dV   = (Poisson.Vin - Poisson.Vout);
        // Distance between Dirichlet slices: BC is applied at the first interior
        // slice (global k=1) and the last interior slice (global k=Nz_g), so the
        // macroscopic gradient is dV / (Nz_g - 1). This matches the MATLAB
        // Laplace_Solver convention (gradP = (Pin-Pout)/(Lz-1)).
        const double L_eff = double(Nz_g - 1);

        // local sums over interior voxels (skip the +/-1 halo)
        double sumEz_loc = 0.0;
        double nFluid_loc = 0.0;
        for (int k = 1; k < Nz - 1; k++) {
            for (int j = 1; j < Ny - 1; j++) {
                for (int i = 1; i < Nx - 1; i++) {
                    int n = k * Nx * Ny + j * Nx + i;
                    if (Poisson.Mask->id[n] > 0) {
                        sumEz_loc  += Ez(i, j, k);
                        nFluid_loc += 1.0;
                    }
                }
            }
        }

        double sumEz_g = 0.0, nFluid_g = 0.0;
        MPI_Allreduce(&sumEz_loc,  &sumEz_g,  1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&nFluid_loc, &nFluid_g, 1, MPI_DOUBLE, MPI_SUM, comm);

        if (dV == 0.0) {
            if (rank == 0)
                printf("ERROR: Vin == Vout; no potential drop to compute tortuosity\n");
            MPI_Finalize();
            return 2;
        }

        const double meanEz   = sumEz_g / double(V_g);    // <E_z> over the full volume
        const double porosity = nFluid_g / double(V_g);
        const double Drel     = meanEz * L_eff / dV;       // D_eff / D_0
        const double tau_geom = (Drel > 0.0) ? porosity / Drel : 0.0;
        const double formationFactor = (Drel > 0.0) ? 1.0 / Drel : 0.0;

        if (rank == 0) {
            printf("\n");
            printf("================================================================\n");
            printf("  Tortuosity result (z-direction)\n");
            printf("================================================================\n");
            printf("  global domain size      = %ld x %ld x %ld\n", Nx_g, Ny_g, Nz_g);
            printf("  fluid voxel count       = %.0f\n",            nFluid_g);
            printf("  porosity      phi       = %.6f\n",            porosity);
            printf("  <E_z>_vol               = %.6e   [V/lu]\n",   meanEz);
            printf("  imposed E_macro = dV/L  = %.6e   [V/lu]   (L_eff = %.0f lu)\n",
                   dV / L_eff, L_eff);
            printf("  D_eff / D_0             = %.6f\n",            Drel);
            printf("  tortuosity    tau       = %.6f\n",            tau_geom);
            printf("  formation factor F      = %.6f\n",            formationFactor);
            printf("================================================================\n");

            // CSV output
            const char *csv = "Tortuosity.csv";
            ifstream chk(csv);
            const bool writeHeader = !chk.good();
            chk.close();
            FILE *f = fopen(csv, "a");
            if (f) {
                if (writeHeader)
                    fprintf(f, "Nx Ny Nz porosity D_eff_over_D0 tortuosity formationFactor Vin Vout tau\n");
                fprintf(f, "%ld %ld %ld %.6f %.6f %.6f %.6f %.4f %.4f %.4f\n",
                        Nx_g, Ny_g, Nz_g, porosity, Drel, tau_geom, formationFactor,
                        Poisson.Vin, Poisson.Vout, Poisson.tau);
                fclose(f);
            }
        }
    }

    MPI_Barrier(comm);
    MPI_Finalize();
    return 0;
}
