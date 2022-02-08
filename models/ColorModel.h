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
Implementation of color lattice boltzmann model
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <fstream>

//#include "analysis/TwoPhase.h"
#include "analysis/analysis.h" //only used for blob identification in morph
#include "analysis/distance.h" //for distance map calculation
#include "common/ScaLBL.h"
#include "common/Communication.h"
#include "common/MPI_Helpers.h"
using namespace std;
class ScaLBL_ColorModel{
public:
	ScaLBL_ColorModel(int RANK, int NP, MPI_Comm COMM);
	~ScaLBL_ColorModel();	
	
	// functions in they should be run
	void ReadParams(string filename);
	void ReadParams(shared_ptr<Database> db0);
	void SetDomain();
	void ReadInput();
	void Create();
	void Initialize();
	void Run();
	void WriteDebug();
	void WriteDebugYDW();
	void WriteRestartYDW();
	double approxRollingAverage(double avg, double new_sample, int timestep);
    int collateBoundaryBlobs(int *&inletNWPBlobsGlob, vector<int> inletNWPBlobsLoc);
	
	bool Restart,pBC;
	int timestep,timestepMax;
	int BoundaryCondition;
	double tauA,tauB,rhoA,rhoB,alpha,beta;
	double Fx,Fy,Fz,flux;
	double din,dout,inletA,inletB,outletA,outletB;
	
	int Nx,Ny,Nz,N,Np;
	double poro;
    double volB;// = Averages->Volume_w();
    double volA;// = Averages->Volume_n();
    double volB_H;// = Averages->Volume_w();
    double volA_H;// = Averages->Volume_n();
	int rank,nprocx,nprocy,nprocz,nprocs;
	double Lx,Ly,Lz;

	std::shared_ptr<Domain> Dm;   // this domain is for analysis
	std::shared_ptr<Domain> Mask; // this domain is for lbm //keep these legacy domains for ctesting
	std::shared_ptr<ScaLBL_Communicator> ScaLBL_Comm;
	std::shared_ptr<ScaLBL_Communicator> ScaLBL_Comm_Regular;
    //std::shared_ptr<TwoPhase> Averages; //for ctest only, remove this asap
    
    // input database
    std::shared_ptr<Database> db;
    std::shared_ptr<Database> domain_db;
    std::shared_ptr<Database> color_db;
    std::shared_ptr<Database> analysis_db;

    IntArray Map;
    // the poreindexed arrays
    char *id;    
	int *NeighborList;
	int *dvcMap;
	int *surfaceBCInds;
	double *surfaceBCValsA;
	double *surfaceBCValsB;
	double *fq, *Aq, *Bq;
	double *Den, *Phi;
	double *ColorGrad;
	double *Velocity;
	double *Pressure;
	
	// the cartesian arrays
    DoubleArray Distance;
    DoubleArray DistanceLabelBCs;
    DoubleArray Pressure_Cart;
    DoubleArray Density_A_Cart;
    DoubleArray Density_B_Cart;
    DoubleArray Velocity_x;
    DoubleArray Velocity_y;
    DoubleArray Velocity_z;
    DoubleArray PhaseField;
    // the porespace arrays
	DoubleArray cDen;
	DoubleArray cfq;
private:
	MPI_Comm comm;
    
	int dist_mem_size;
	int neighborSize;
	// filenames
    char LocalRankString[8];
    char LocalRankFilename[40];
    char LocalRestartFile[40];
   
    //int rank,nprocs;
    void LoadParams(std::shared_ptr<Database> db0);
    void AssignComponentLabels(double *phase);
    int IndexSurfaceBoundaries(int *surfaceInds);
    double MorphInit(const double beta, const double morph_delta);
    double SpinoInit(const double delta_sw);
};

