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
  ScaLBL_Poisson: D3Q7 lattice-Boltzmann Laplace / Poisson solver on the
  pore phase of a segmented geometry. Stripped-down port of the gaslbm
  electrokinetic Poisson solver for use by lbpm_tortuosity_simulator.

  Drops (compared to gaslbm/models/PoissonSolver.h):
    - D3Q19 lattice scheme (D3Q7 only)
    - charge-density / electroneutrality coupling
    - slipping-velocity BC
    - time-periodic Dirichlet BCs
    - solid-label dependent Dirichlet/Neumann surface BC list
      (we use the implicit no-flux that ScaLBL bounce-back gives)
    - restart, dummy-charge-density debugging, MSE_max tolerance variant

  Solver loop:  ReadParams -> SetDomain -> ReadInput -> Create -> Initialize
                -> Run -> getElectricPotential / getElectricField.
*/
#ifndef SCALBL_POISSON_H
#define SCALBL_POISSON_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/stat.h>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <fstream>
#include <cmath>
#include <iterator>

#include "common/ScaLBL.h"
#include "common/Communication.h"
#include "common/MPI_Helpers.h"

class ScaLBL_Poisson {
public:
    ScaLBL_Poisson(int RANK, int NP, MPI_Comm COMM);
    ~ScaLBL_Poisson();

    // Run in order:
    //   ReadParams -> SetDomain -> ReadInput -> Create -> Initialize -> Run
    void ReadParams(std::string filename);
    void SetDomain();
    void ReadInput();
    void Create();
    void Initialize();
    void Run();

    // Field extractors -- ReturnValues must be sized (Nx, Ny, Nz) on host.
    void getElectricPotential(DoubleArray &ReturnValues);
    void getElectricField(DoubleArray &Ex, DoubleArray &Ey, DoubleArray &Ez);

    // ---- LBM parameters (read from "Poisson" section of the db) ----
    int    timestep, timestepMax;
    int    analysis_interval;
    int    BoundaryConditionInlet;   // 0 = periodic, 1 = Dirichlet Vin
    int    BoundaryConditionOutlet;  // 0 = periodic, 1 = Dirichlet Vout
    double tau;                       // D3Q7 relaxation time, defaults to 1/2 + 1/(D3Q7 cs^2) inverse
    double tolerance;                 // steady-state convergence threshold on MSE(psi)
    double Vin, Vout;                 // Dirichlet potentials at the -z / +z faces

    // ---- Geometry / decomposition ----
    int    Nx, Ny, Nz, N, Np;
    int    rank, nprocs;
    int    nprocx, nprocy, nprocz;
    double Lx, Ly, Lz;
    double h;                          // voxel size, microns/lu (cosmetic; unused for ratios)

    // ---- ScaLBL state ----
    std::shared_ptr<Domain>              Dm;     // analysis domain
    std::shared_ptr<Domain>              Mask;   // LBM domain (immobile = solid)
    std::shared_ptr<ScaLBL_Communicator> ScaLBL_Comm;

    std::shared_ptr<Database> db;
    std::shared_ptr<Database> domain_db;
    std::shared_ptr<Database> electric_db;

    IntArray    Map;
    DoubleArray Psi_host;
    DoubleArray Psi_previous;

    int    *NeighborList;
    int    *dvcMap;
    double *fq;
    double *Psi;
    double *ElectricField;
    double *ChargeDensity;   // owned by this class; zero-filled for pure Laplace

private:
    MPI_Comm comm;
    char     LocalRankString[8];
    char     LocalRankFilename[40];

    void Potential_Init(double *psi_init);
    void SolveElectricPotentialAAodd();
    void SolveElectricPotentialAAeven();
    void SolvePoissonAAodd();
    void SolvePoissonAAeven();
};

#endif
