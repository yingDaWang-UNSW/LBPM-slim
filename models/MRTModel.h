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
  ScaLBL_MRTModel: single-phase D3Q19 MRT lattice Boltzmann driver.

  This is the absolute-permeability ("k") protocol: drive an LBM run on
  a segmented geometry with either a body force (BC=0 periodic),
  pressure BCs (BC=3), or a flux BC (BC=4), measure the steady-state
  superficial velocity, and report the Darcy permeability.

  Convergence is on |dK/K| < permTolerance at each analysis interval.
*/
#ifndef MRTMODEL_H
#define MRTMODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <fstream>

#include "common/ScaLBL.h"
#include "common/Communication.h"
#include "common/MPI_Helpers.h"

class ScaLBL_MRTModel {
public:
    ScaLBL_MRTModel(int RANK, int NP, MPI_Comm COMM);
    ~ScaLBL_MRTModel();

    // Run in order: ReadParams -> SetDomain -> ReadInput -> Create
    //               -> Initialize -> Run
    void ReadParams(std::string filename);
    void SetDomain();
    void ReadInput();
    void Create();
    void Initialize();
    void Run();

    // Field dumps (called automatically on convergence / at vis intervals).
    void writeVelocityPressure();   // raw rank-local velocity + pressure
    void writeFqField();             // full fq distribution
    void writeRestart();             // restart binary + Restart.txt timestep

    // ---- LBM parameters (read from "MRT" section of the db) ----
    bool   Restart;
    int    timestep, timestepMax;
    int    BoundaryCondition;        // 0 = periodic + body force,
                                     // 3 = pressure BCs, 4 = flux BC
    double tau;                      // MRT relaxation time
    double Fx, Fy, Fz;               // body force (lattice units)
    double flux;                     // BC=4 only
    double din, dout;                // BC=3 only

    // ---- Geometry / decomposition ----
    int    Nx, Ny, Nz;               // local interior + 2 halos
    int    N;                        // Nx * Ny * Nz
    int    Np;                       // memory-optimised fluid voxel count
    int    rank, nprocs;
    int    nprocx, nprocy, nprocz;
    double Lx, Ly, Lz;
    double voxelSize;                // metres per voxel
    double porosity;

    // ---- Analysis / I/O knobs (read from "MRT" section) ----
    int    analysis_interval;        // recompute K every N steps
    int    visInterval;              // dump vel/pressure every N steps (0 = off)
    int    restart_interval;         // dump restart every N steps   (0 = off)
    double permTolerance;            // |dK/K| convergence threshold
    bool   visTolerance;             // dump vel/pressure on convergence
    bool   fqFlag;                   // dump fq (vs vel/pressure) on vis events
    bool   restartFq;                // load fq from Restart.* at startup
    bool   logFile;                  // append rows to Permeability.csv

    // ---- ScaLBL state ----
    std::shared_ptr<Domain>                Mask;
    std::shared_ptr<ScaLBL_Communicator>   ScaLBL_Comm;
    std::shared_ptr<Database>              db;
    std::shared_ptr<Database>              domain_db;
    std::shared_ptr<Database>              mrt_db;

    IntArray     Map;
    DoubleArray  Geom;                       // 1 = fluid, 0 = solid
    DoubleArray  Pressure_Cart;              // cartesian pressure scratch
    DoubleArray  Velocity_x, Velocity_y, Velocity_z;
    DoubleArray  fqTemp;                     // cartesian fq scratch (for dumps)

    int    *NeighborList;
    double *fq;
    double *Velocity;
    double *Pressure;

private:
    MPI_Comm comm;
    char LocalRankString[8];
    char LocalRankFilename[40];
    char LocalRestartFile[40];

    // Drive the body force / pressure BC ramp logic at each timestep.
    void applyBoundaryConditions();
};

#endif
