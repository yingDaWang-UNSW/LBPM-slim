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
color lattice boltzmann model
 */
 
 // add routine to freeze interfaces, and solve perm within the analysis loop.
 // setup automorph, and fluxmorph, and implement flux reversal option
#include "models/ColorModel.h"
#include <sys/stat.h>
#include <bits/stdc++.h> 
#include <algorithm>
using namespace std;
ScaLBL_ColorModel::ScaLBL_ColorModel(int RANK, int NP, MPI_Comm COMM):
rank(RANK), nprocs(NP),  Restart(0),timestep(0),timestepMax(0),tauA(0),tauB(0),rhoA(0),rhoB(0),alpha(0),beta(0),
Fx(0),Fy(0),Fz(0),flux(0),din(0),dout(0),inletA(0),inletB(0),outletA(0),outletB(0),
Nx(0),Ny(0),Nz(0),N(0),Np(0),poro(0),nprocx(0),nprocy(0),nprocz(0),BoundaryCondition(0),Lx(0),Ly(0),Lz(0),comm(COMM)
{

}
ScaLBL_ColorModel::~ScaLBL_ColorModel(){

}

void ScaLBL_ColorModel::ReadParams(string filename){
	// read the input database 
	db = make_shared<Database>( filename );
	domain_db = db->getDatabase( "Domain" );
	color_db = db->getDatabase( "Color" );
	analysis_db = db->getDatabase( "Analysis" );

	// Color Model parameters
	timestepMax = color_db->getScalar<int>( "timestepMax" );
	tauA = color_db->getScalar<double>( "tauA" );
	tauB = color_db->getScalar<double>( "tauB" );
	rhoA = color_db->getScalar<double>( "rhoA" );
	rhoB = color_db->getScalar<double>( "rhoB" );
	Fx = color_db->getVector<double>( "F" )[0];
	Fy = color_db->getVector<double>( "F" )[1];
	Fz = color_db->getVector<double>( "F" )[2];
	alpha = color_db->getScalar<double>( "alpha" );
	beta = color_db->getScalar<double>( "beta" );
	Restart = color_db->getScalar<bool>( "Restart" );
	din = color_db->getScalar<double>( "din" );
	dout = color_db->getScalar<double>( "dout" );
	flux = color_db->getScalar<double>( "flux" );
	
	if (color_db->keyExists( "inletA" )){
		inletA = color_db->getScalar<double>( "inletA" );
	}
	else{
		inletA=1.0;
	}
	
	if (color_db->keyExists( "inletB" )){
		inletB = color_db->getScalar<double>( "inletB" );
	}
	else{
		inletB=0.0;
	}
	
	if (color_db->keyExists( "outletA" )){
		outletA = color_db->getScalar<double>( "outletA" );
	}
	else{
		outletA=0.0;
	}
	
	if (color_db->keyExists( "outletB" )){
		outletB = color_db->getScalar<double>( "outletB" );
	}
	else{
		outletB=1.0;
	}
	// Read domain parameters
	auto L = domain_db->getVector<double>( "L" );
	auto size = domain_db->getVector<int>( "n" );
	auto nproc = domain_db->getVector<int>( "nproc" );
	BoundaryCondition = domain_db->getScalar<int>( "BC" );
	Nx = size[0];
	Ny = size[1];
	Nz = size[2];
	Lx = L[0];
	Ly = L[1];
	Lz = L[2];
	nprocx = nproc[0];
	nprocy = nproc[1];
	nprocz = nproc[2];
	
	if (BoundaryCondition==4) flux *= rhoA; // mass flux must adjust for density (see formulation for details)

}
void ScaLBL_ColorModel::SetDomain(){
	//Dm  = shared_ptr<Domain>(new Domain(domain_db,comm));      // full domain for analysis
	Dm  = shared_ptr<Domain>(new Domain(domain_db,comm));    // mask domain removes immobile phases
	Nx+=2; Ny+=2; Nz += 2;
	N = Nx*Ny*Nz;
	id = new char [N];
	for (int i=0; i<Nx*Ny*Nz; i++) Dm->id[i] = 1;               // initialize this way
	//Averages = shared_ptr<TwoPhase> ( new TwoPhase(Dm) ); // TwoPhase analysis object
	Distance.resize(Nx,Ny,Nz);
	MPI_Barrier(Dm->Comm);
	//Dm->CommInit();
	MPI_Barrier(Dm->Comm);
	rank = Dm->rank();
}

void ScaLBL_ColorModel::ReadInput(){
	size_t readID;
	Dm->ReadIDs();
	for (int i=0; i<Nx*Ny*Nz; i++) id[i] = Dm->id[i];  // save what was read
	char LocalRankFilename[100];
	sprintf(LocalRankString,"%05d",rank);
	sprintf(LocalRankFilename,"%s%s","ID.",LocalRankString);
	sprintf(LocalRestartFile,"%s%s","Restart.",LocalRankString);
	
	// Generate the signed distance map
	// Initialize the domain and communication
	Array<char> id_solid(Nx,Ny,Nz);
	int count = 0;
	// Solve for the position of the solid phase
	for (int k=0;k<Nz;k++){
		for (int j=0;j<Ny;j++){
			for (int i=0;i<Nx;i++){
				int n = k*Nx*Ny+j*Nx+i;
				// Initialize the solid phase
				if (Dm->id[n] > 0)	id_solid(i,j,k) = 1;
				else	     	      	id_solid(i,j,k) = 0;
			}
		}
	}
	// Initialize the signed distance function
	for (int k=0;k<Nz;k++){
		for (int j=0;j<Ny;j++){
			for (int i=0;i<Nx;i++){
				int n=k*Nx*Ny+j*Nx+i;
				// Initialize distance to +/- 1
				Distance(i,j,k) = 2.0*double(id_solid(i,j,k))-1.0;
			}
		}
	}
//	MeanFilter(Averages->SDs);
	if (rank==0) printf("Initialized solid phase -- Converting to Signed Distance function \n");
	CalcDist(Distance,id_solid,*Dm);
	
	if (rank == 0) cout << "Domain set." << endl;
	
	//Averages->SetParams(rhoA,rhoB,tauA,tauB,Fx,Fy,Fz,alpha);
}

void ScaLBL_ColorModel::AssignComponentLabels(double *phase)
{
	size_t NLABELS=0;
	char VALUE=0;
	double AFFINITY=0.f;
    bool affinityRampupFlag = false;
	auto LabelList = color_db->getVector<char>( "ComponentLabels" );
	auto AffinityList = color_db->getVector<double>( "ComponentAffinity" );
	if (color_db->keyExists( "affinityRampupFlag" )){
		affinityRampupFlag = color_db->getScalar<bool>( "affinityRampupFlag" );
		if (rank==0 && affinityRampupFlag) printf("Affinities are set to ramp up. Initialising the domain with neutral wetting \n");
	}
	else{
		affinityRampupFlag=false;  //redundancy for no sake is such a human thing to do
	}
	
	NLABELS=LabelList.size();
	if (NLABELS != AffinityList.size()){
		ERROR("Error: ComponentLabels and ComponentAffinity must be the same length! \n");
	}

	int label_count[NLABELS];
	int label_count_global[NLABELS];
	// Assign the labels

	for (int idx=0; idx<NLABELS; idx++) label_count[idx]=0;

	for (int k=0;k<Nz;k++){
		for (int j=0;j<Ny;j++){
			for (int i=0;i<Nx;i++){
				int n = k*Nx*Ny+j*Nx+i;
				VALUE=id[n];
				// Assign the affinity from the paired list
				for (unsigned int idx=0; idx < NLABELS; idx++){
				      //printf("idx=%i, value=%i, %i, \n",idx, VALUE,LabelList[idx]);
					if (VALUE == LabelList[idx]){
						AFFINITY=AffinityList[idx];
						label_count[idx]++;
						idx = NLABELS;
						Dm->id[n] = 0; // set mask to zero since this is an immobile component
					}
				}
				// fluid labels are reserved
				if (VALUE == 1) AFFINITY=1.0;
				else if (VALUE == 2) AFFINITY=-1.0;
				else if (affinityRampupFlag == true) AFFINITY=0;//overwrite the affinity if the flag is active
				phase[n] = AFFINITY;
			}
		}
	}
	// Set Dm to match Mask
	//for (int i=0; i<Nx*Ny*Nz; i++) Dm->id[i] = Mask->id[i]; 

	MPI_Allreduce(&label_count[0],&label_count_global[0],NLABELS,MPI_INT,MPI_SUM,Dm->Comm);
    //poro=0.0;
    // because I am stupid, just do this for all cores
	if (rank==0) printf("Components labels: %lu \n",NLABELS);
	for (unsigned int idx=0; idx<NLABELS; idx++){
		VALUE=LabelList[idx];
		AFFINITY=AffinityList[idx];
		double volume_fraction  = double(label_count_global[idx])/double((Nx)*(Ny)*(Nz)*nprocz*nprocy*nprocx);
		if (rank==0) printf("-- label=%i, affinity=%f, volume fraction==%f\n",int(VALUE),AFFINITY,volume_fraction); 
		poro=poro+volume_fraction;
	}
	if (rank==0) printf("Porosity: %f\n",1-poro); 
	
	//MPI_Scatter(&poro, 1, MPI_DOUBLE, &poro, 1, MPI_DOUBLE, 0, Dm->Comm);
    poro=1-poro;
	//printf("Core: %d, Porosity: %f\n",rank, 1-poro); 
}


void ScaLBL_ColorModel::Create(){
	/*
	 *  This function creates the variables needed to run a LBM 
	 */
	//.........................................................
	// Initialize communication structures in averaging domain
	//for (int i=0; i<Nx*Ny*Nz; i++) Dm->id[i] = Mask->id[i];
	Dm->CommInit();
	Np=Dm->PoreCount();
	//...........................................................................
	if (rank==0)    printf ("Create ScaLBL_Communicator \n");
	// Create a communicator for the device (will use optimized layout)
	// ScaLBL_Communicator ScaLBL_Comm(Mask); // original
	ScaLBL_Comm  = shared_ptr<ScaLBL_Communicator>(new ScaLBL_Communicator(Dm));
	ScaLBL_Comm_Regular  = shared_ptr<ScaLBL_Communicator>(new ScaLBL_Communicator(Dm));

	int Npad=(Np/16 + 2)*16;
	if (rank==0)    printf ("Set up memory efficient layout, %i | %i | %i \n", Np, Npad, N);
	Map.resize(Nx,Ny,Nz);       Map.fill(-2);
	auto neighborList= new int[18*Npad];
	Np = ScaLBL_Comm->MemoryOptimizedLayoutAA(Map,neighborList,Dm->id,Np);
	MPI_Barrier(Dm->Comm);

	//...........................................................................
	//                MAIN  VARIABLES ALLOCATED HERE
	//...........................................................................
	// LBM variables
	if (rank==0)    printf ("Allocating distributions \n");
	//......................device distributions.................................
	dist_mem_size = Np*sizeof(double);
	neighborSize=18*(Np*sizeof(int));
	//...........................................................................
	ScaLBL_AllocateDeviceMemory((void **) &NeighborList, neighborSize);
	ScaLBL_AllocateDeviceMemory((void **) &dvcMap, sizeof(int)*Np);
	ScaLBL_AllocateDeviceMemory((void **) &fq, 19*dist_mem_size);
	ScaLBL_AllocateDeviceMemory((void **) &Aq, 7*dist_mem_size);
	ScaLBL_AllocateDeviceMemory((void **) &Bq, 7*dist_mem_size);
	ScaLBL_AllocateDeviceMemory((void **) &Den, 2*dist_mem_size);
	ScaLBL_AllocateDeviceMemory((void **) &Phi, sizeof(double)*Nx*Ny*Nz);		
	//ScaLBL_AllocateDeviceMemory((void **) &Pressure, sizeof(double)*Np);
	ScaLBL_AllocateDeviceMemory((void **) &Velocity, 3*dist_mem_size);
	//ScaLBL_AllocateDeviceMemory((void **) &ColorGrad, 3*dist_mem_size);
	//...........................................................................
	// Update GPU data structures
	if (rank==0)	printf ("Setting up device map and neighbor list \n");
	fflush(stdout);
	int *TmpMap;
	TmpMap=new int[Np];
	for (int k=1; k<Nz-1; k++){
		for (int j=1; j<Ny-1; j++){
			for (int i=1; i<Nx-1; i++){
				int idx=Map(i,j,k);
				if (!(idx < 0))
					TmpMap[idx] = k*Nx*Ny+j*Nx+i;
			}
		}
	}
	// check that TmpMap is valid
	for (int idx=0; idx<ScaLBL_Comm->LastExterior(); idx++){
		int n = TmpMap[idx];
		if (n > Nx*Ny*Nz){
			printf("Bad value! idx=%i \n");
			TmpMap[idx] = Nx*Ny*Nz-1;
		}
	}
	for (int idx=ScaLBL_Comm->FirstInterior(); idx<ScaLBL_Comm->LastInterior(); idx++){
		int n = TmpMap[idx];
		if (n > Nx*Ny*Nz){
			printf("Bad value! idx=%i \n");
			TmpMap[idx] = Nx*Ny*Nz-1;
		}
	}
	ScaLBL_CopyToDevice(dvcMap, TmpMap, sizeof(int)*Np);
	ScaLBL_DeviceBarrier();
	delete [] TmpMap;
	
	// copy the neighbor list 
	ScaLBL_CopyToDevice(NeighborList, neighborList, neighborSize);
	// initialize phi based on PhaseLabel (include solid component labels)
	double *PhaseLabel;
	PhaseLabel = new double[N];
	AssignComponentLabels(PhaseLabel);
	ScaLBL_CopyToDevice(Phi, PhaseLabel, N*sizeof(double));
	
	Density_A_Cart.resize(Nx,Ny,Nz);
	Density_B_Cart.resize(Nx,Ny,Nz);
    Velocity_x.resize(Nx,Ny,Nz);
    Velocity_y.resize(Nx,Ny,Nz);
    Velocity_z.resize(Nx,Ny,Nz);
}        

/********************************************************
 * AssignComponentLabels                                 *
 ********************************************************/

void ScaLBL_ColorModel::Initialize(){
	
	if (rank==0)	printf ("Initializing distributions \n");
	ScaLBL_D3Q19_Init(fq, Np);
	/*
	 * This function initializes model
	 */
	if (Restart == true){
		if (rank==0){
			printf("Reading restart file! \n");
			ifstream restart("Restart.txt");
			if (restart.is_open()){
				restart  >> timestep;
				printf("Restarting from timestep =%i \n",timestep);
			}
			else{
				printf("WARNING:No Restart.txt file, setting timestep=0 \n");
				timestep=0;
			}
		}
		MPI_Bcast(&timestep,1,MPI_INT,0,Dm->Comm);
		// Read in the restart file to CPU buffers
		int *TmpMap;
		TmpMap = new int[Np];
		
		double *cPhi, *cDist, *cDen;
		cPhi = new double[N];
		cDen = new double[2*Np];
		cDist = new double[19*Np];
		ScaLBL_CopyToHost(TmpMap, dvcMap, Np*sizeof(int));
        ScaLBL_CopyToHost(cPhi, Phi, N*sizeof(double));
    	
		ifstream File(LocalRestartFile,ios::binary);
		int idx;
		double value,va,vb;
		for (int n=0; n<Np; n++){
			File.read((char*) &va, sizeof(va));
			File.read((char*) &vb, sizeof(vb));
			cDen[n]    = va;
			cDen[Np+n] = vb;
		}
		for (int n=0; n<Np; n++){
			// Read the distributions
			for (int q=0; q<19; q++){
				File.read((char*) &value, sizeof(value));
				cDist[q*Np+n] = value;
			}
		}
		File.close();
		
		for (int n=0; n<ScaLBL_Comm->LastExterior(); n++){
			va = cDen[n];
			vb = cDen[Np + n];
			value = (va-vb)/(va+vb);
			idx = TmpMap[n];
			if (!(idx < 0) && idx<N)
				cPhi[idx] = value;
		}
		for (int n=ScaLBL_Comm->FirstInterior(); n<ScaLBL_Comm->LastInterior(); n++){
		  va = cDen[n];
		  vb = cDen[Np + n];
		  	value = (va-vb)/(va+vb);
		  	idx = TmpMap[n];
		  	if (!(idx < 0) && idx<N)
		  		cPhi[idx] = value;
		}
		
		// Copy the restart data to the GPU
		ScaLBL_CopyToDevice(Den,cDen,2*Np*sizeof(double));
		ScaLBL_CopyToDevice(fq,cDist,19*Np*sizeof(double));
		ScaLBL_CopyToDevice(Phi,cPhi,N*sizeof(double));
		ScaLBL_DeviceBarrier();

		MPI_Barrier(Dm->Comm);
	}

	if (rank==0)	printf ("Initializing phase field \n");
	ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq, 0, ScaLBL_Comm->LastExterior(), Np);
	ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);

//	if (BoundaryCondition > 0 ){
//		if (Dm->kproc()==0){
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,0);
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,1);
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,2);
//		}
//		if (Dm->kproc() == nprocz-1){
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-1);
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-2);
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-3);
//		}
//	}

}
// the flux reversal should work both ways as a vector input
void ScaLBL_ColorModel::Run(){
	int nprocs=nprocx*nprocy*nprocz;
	const RankInfoStruct rank_info(rank,nprocx,nprocy,nprocz);
	// raw visualisations
    int visualisation_interval=0;
    // manual morph parameters
	bool SET_CAPILLARY_NUMBER = false;
	int ramp_timesteps = 50000;
	double capillary_number = 0;
	double tolerance = 1.f;
	double Ca_previous = 0.f;
    double Ca_EMA = 0.f;
    double Ca_EMA_previous = 0.f;
    double dCadt = 0.f;
    double dCadtEMA = 0.f;
    double absperm1_EMA = 0.0;
    double absperm2_EMA = 0.0;
    double absperm1_H_EMA = 0.0;
    double absperm2_H_EMA = 0.0;
	int steadycounter=0;
	// for mixed wetting problems
    bool affinityRampupFlag = false;
    int affinityRampSteps = 0;
    // for forced injection problems
    bool fluxReversalFlag = false;
    int fluxReversalType = 1;
    double fluxReversalSat = -1.0;
    // the settling parameter is a general tool
    double settlingTolerance = 3e-5;
    DoubleArray oldPhase(Nx,Ny,Nz); //stores the settling parameter info
    // for steady state problems (borrows params from above as well) 
    bool autoMorphFlag = false; //this will setup morpho automatically if this is off, default to manual mode or no morph
	bool autoMorphAdapt = false; //dynamic flag, dont touch.
	bool spinoMorphFlag = false;
    bool fluxMorphFlag = false; // this will replace shell aggregation with traditional flux injection
    bool coinjectionFlag = false;
    double satInit = -5.0;
    double satInc = 0.05; // a replacement for the saturation vector, the increment to use during automorph
    int injectionType = 1; // 1 for drainage and 2 for imbibition
    double targetSaturation = 0.f;
    int stabilityCounter = 0;
    int accelerationCounter = 0;
    int stabilisationRate = 10000; // interval for checking stabilisation
    int accelerationRate = 1000; // interval for doing automorph acceleration
    double shellRadius = 0.0;


    double voxelSize = 0.0;
	if (analysis_db->keyExists( "autoMorphFlag" )){
		autoMorphFlag = analysis_db->getScalar<bool>( "autoMorphFlag" );
	}
	if (analysis_db->keyExists( "spinoMorphFlag" )){
		spinoMorphFlag = analysis_db->getScalar<bool>( "spinoMorphFlag" );
	}
	if (analysis_db->keyExists( "fluxMorphFlag" )){
		fluxMorphFlag = analysis_db->getScalar<bool>( "fluxMorphFlag" );
	}
	if (analysis_db->keyExists( "coinjectionFlag" )){
		coinjectionFlag = analysis_db->getScalar<bool>( "coinjectionFlag" );
	}
	if (analysis_db->keyExists( "satInit" )){
		satInit = analysis_db->getScalar<double>( "satInit" );
	}
	if (analysis_db->keyExists( "satInc" )){
		satInc = analysis_db->getScalar<double>( "satInc" );
	}
	if (analysis_db->keyExists( "injectionType" )){
		injectionType = analysis_db->getScalar<int>( "injectionType" );
	}
	if (analysis_db->keyExists( "stabilisationRate" )){
		stabilisationRate = analysis_db->getScalar<int>( "stabilisationRate" );
	}
	if (analysis_db->keyExists( "accelerationRate" )){
		accelerationRate = analysis_db->getScalar<int>( "accelerationRate" );
	}
//setup steadyFlux routine, fluxdrainmorphimb, and others


	if (domain_db->keyExists( "voxel_length" )){
		voxelSize = domain_db->getScalar<double>( "voxel_length" );
		voxelSize=voxelSize/1000000.0;
	}
	else{
	    voxelSize=Lz/double(Nz*nprocz);
	}
	if (rank==0)    printf("Voxel Size = %f microns\n",voxelSize*1000000);
    // (BoundaryCondition == 4){
	    if (color_db->keyExists( "fluxReversalFlag" )){
		    fluxReversalFlag = color_db->getScalar<bool>( "fluxReversalFlag" );
		    if (rank==0 && fluxReversalFlag) printf("[In Colour Model], Flux indicators are set to be reversed\n");
	    }  
	    if (color_db->keyExists( "fluxReversalType" )){
		    fluxReversalType = color_db->getScalar<int>( "fluxReversalType" );
	    }    
	    if (color_db->keyExists( "fluxReversalSat" )){
		    fluxReversalSat = color_db->getScalar<double>( "fluxReversalSat" );
	    }    
	    if (rank==0 && fluxReversalFlag && fluxReversalSat>=0) printf("[In Colour Model], Flux reversal is set to occur when saturation reaches %f\n", fluxReversalSat);
	    
	    if (rank==0 && fluxReversalFlag && fluxReversalSat<0) printf("[In Colour Model], Flux reversal is set to occur when saturation stabilises (this may not work, or take a very long time)\n");
	//}
	if (color_db->keyExists( "settlingTolerance" )){
		settlingTolerance = color_db->getScalar<double>( "settlingTolerance" );
		if (rank==0 && fluxReversalFlag) printf("[In Colour Model], The settling tolerance for steady state related switches is %e\n", settlingTolerance);
	}
	if (color_db->keyExists( "affinityRampupFlag" )){
		affinityRampupFlag = color_db->getScalar<bool>( "affinityRampupFlag" );
		if (rank==0 && affinityRampupFlag) printf("[In Colour Model], Affinities are to be set dynamically\n");
	}
	if (color_db->keyExists( "affinityRampSteps" )){
		affinityRampSteps = color_db->getScalar<int>( "affinityRampSteps" );
		if (rank==0 && affinityRampSteps>0) printf("[In Colour Model], Affinities are to be ramped up to their specified values within %d timesteps\n", affinityRampSteps);
	}	
	if (analysis_db->keyExists( "ramp_timesteps" )){
		ramp_timesteps = analysis_db->getScalar<double>( "ramp_timesteps" );
	}
	stabilityCounter = -ramp_timesteps;
	if (color_db->keyExists( "capillary_number" )){
		capillary_number = color_db->getScalar<double>( "capillary_number" );
		SET_CAPILLARY_NUMBER=true;
	}
//	if (BoundaryCondition != 0 && SET_CAPILLARY_NUMBER==true){
//		if (rank == 0) printf("WARINING: capillary number target only supported for BC = 0 \n");
//		SET_CAPILLARY_NUMBER=false;
//	}

	if (analysis_db->keyExists( "tolerance" )){
		tolerance = analysis_db->getScalar<double>( "tolerance" );
	}
	else{
		tolerance = 0.02;
	}
	int analysis_interval = analysis_db->getScalar<int>( "analysis_interval" );
	
	if (analysis_db->keyExists( "raw_visualisation_interval" )){
		visualisation_interval = analysis_db->getScalar<int>( "raw_visualisation_interval" );
	}
	else{
		visualisation_interval = 1e10;
	}
	
	if (rank==0){
		printf("********************************************************\n");
		printf("No. of timesteps: %i \n", timestepMax);
		if (autoMorphFlag){
		    printf("[In Colour Model], Morphological Adaptation is Active. Ramp-up before Morphological Adaptation: %i \n", ramp_timesteps);
		}
		fflush(stdout);
	}
	//.......create and start timer............
	double starttime,stoptime,cputime;
	ScaLBL_DeviceBarrier();
	MPI_Barrier(Dm->Comm);
	starttime = MPI_Wtime();
	//.........................................
	
	//************ MAIN ITERATION LOOP ***************************************/
	//PROFILE_START("Loop");
    //shared_ptr<Database> analysis_db;
	//bool Regular = false;

	//runAnalysis analysis( analysis_db, rank_info, ScaLBL_Comm, Dm, Np, Regular, beta, Map );
	//analysis.createThreads( analysis_method, 4 );
	while (timestep < timestepMax ) {
		//if ( rank==0 ) { printf("Running timestep %i (%i MB)\n",timestep+1,(int)(Utilities::getMemoryUsage()/1048576)); }
		//PROFILE_START("Update");

		// *************ODD TIMESTEP*************
		timestep++;
		// Compute the Phase indicator field
		// Read for Aq, Bq happens in this routine (requires communication)
		ScaLBL_Comm->BiSendD3Q7AA(Aq,Bq); //READ FROM NORMAL
		ScaLBL_D3Q7_AAodd_PhaseField(NeighborList, dvcMap, Aq, Bq, Den, Phi, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
		ScaLBL_Comm->BiRecvD3Q7AA(Aq,Bq); //WRITE INTO OPPOSITE
		ScaLBL_DeviceBarrier();
		ScaLBL_D3Q7_AAodd_PhaseField(NeighborList, dvcMap, Aq, Bq, Den, Phi, 0, ScaLBL_Comm->LastExterior(), Np);

		if (inletA+inletB+outletA+outletB > 0){
			ScaLBL_Comm->Color_BC_z(dvcMap, Phi, Den, inletA, inletB);
			ScaLBL_Comm->Color_BC_Z(dvcMap, Phi, Den, outletA, outletB);
		}
		// Halo exchange for phase field
		ScaLBL_Comm_Regular->SendHalo(Phi);
		// Perform the collision operation
		ScaLBL_Comm->SendD3Q19AA(fq); //READ FROM NORMAL
		ScaLBL_D3Q19_AAodd_Color(NeighborList, dvcMap, fq, Aq, Bq, Den, Phi, Velocity, rhoA, rhoB, tauA, tauB,
				alpha, beta, Fx, Fy, Fz, Nx, Nx*Ny, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
		ScaLBL_Comm_Regular->RecvHalo(Phi);
		ScaLBL_Comm->RecvD3Q19AA(fq); //WRITE INTO OPPOSITE
		ScaLBL_DeviceBarrier();
		// Set BCs
		if (BoundaryCondition == 3){
			ScaLBL_Comm->D3Q19_Pressure_BC_z(NeighborList, fq, din, timestep);
			ScaLBL_Comm->D3Q19_Pressure_BC_Z(NeighborList, fq, dout, timestep);
		}
		if (BoundaryCondition == 4){
			din = ScaLBL_Comm->D3Q19_Flux_BC_z(NeighborList, fq, flux, timestep);
			ScaLBL_Comm->D3Q19_Pressure_BC_Z(NeighborList, fq, dout, timestep);
		}
		ScaLBL_D3Q19_AAodd_Color(NeighborList, dvcMap, fq, Aq, Bq, Den, Phi, Velocity, rhoA, rhoB, tauA, tauB,
				alpha, beta, Fx, Fy, Fz, Nx, Nx*Ny, 0, ScaLBL_Comm->LastExterior(), Np);
		ScaLBL_DeviceBarrier(); 
		MPI_Barrier(Dm->Comm);

		// *************EVEN TIMESTEP*************
		timestep++;
		// Compute the Phase indicator field
		ScaLBL_Comm->BiSendD3Q7AA(Aq,Bq); //READ FROM NORMAL
		ScaLBL_D3Q7_AAeven_PhaseField(dvcMap, Aq, Bq, Den, Phi, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
		ScaLBL_Comm->BiRecvD3Q7AA(Aq,Bq); //WRITE INTO OPPOSITE
		ScaLBL_DeviceBarrier();
		ScaLBL_D3Q7_AAeven_PhaseField(dvcMap, Aq, Bq, Den, Phi, 0, ScaLBL_Comm->LastExterior(), Np);

		// Halo exchange for phase field boundary conditions
		if (inletA+inletB+outletA+outletB > 0){
			ScaLBL_Comm->Color_BC_z(dvcMap, Phi, Den, inletA, inletB);
			ScaLBL_Comm->Color_BC_Z(dvcMap, Phi, Den, outletA, outletB);
		}
		ScaLBL_Comm_Regular->SendHalo(Phi);
		// Perform the collision operation
		ScaLBL_Comm->SendD3Q19AA(fq); //READ FORM NORMAL
		ScaLBL_D3Q19_AAeven_Color(dvcMap, fq, Aq, Bq, Den, Phi, Velocity, rhoA, rhoB, tauA, tauB,
				alpha, beta, Fx, Fy, Fz,  Nx, Nx*Ny, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
		ScaLBL_Comm_Regular->RecvHalo(Phi);
		ScaLBL_Comm->RecvD3Q19AA(fq); //WRITE INTO OPPOSITE
		ScaLBL_DeviceBarrier();
		// Set velocity and pressure boundary conditions
		if (BoundaryCondition == 3){
			ScaLBL_Comm->D3Q19_Pressure_BC_z(NeighborList, fq, din, timestep);
			ScaLBL_Comm->D3Q19_Pressure_BC_Z(NeighborList, fq, dout, timestep);
		}
		else if (BoundaryCondition == 4){
			din = ScaLBL_Comm->D3Q19_Flux_BC_z(NeighborList, fq, flux, timestep);
			ScaLBL_Comm->D3Q19_Pressure_BC_Z(NeighborList, fq, dout, timestep);
		}
		ScaLBL_D3Q19_AAeven_Color(dvcMap, fq, Aq, Bq, Den, Phi, Velocity, rhoA, rhoB, tauA, tauB, alpha, beta, Fx, Fy, Fz, Nx, Nx*Ny, 0, ScaLBL_Comm->LastExterior(), Np);
		ScaLBL_DeviceBarrier(); 
		MPI_Barrier(Dm->Comm);
		//************************************************************************
		
		//PROFILE_STOP("Update");
		//raw visualisation intervals
		if (timestep%visualisation_interval == 0){
		    WriteDebugYDW();
		}
			    // Run the analysis
       
        // analysis intervals
		// allow initial ramp-up to get closer to steady state
		if (timestep%analysis_interval == 0){
		    steadycounter+=analysis_interval;
        	stoptime = MPI_Wtime();
    		cputime = (stoptime - starttime);
		    if (rank==0) {
	            printf("=================TimeStep: %d, Elapsed time = %f=================\n", timestep, cputime);
            }
            //ScaLBL_D3Q19_Pressure(fq,Pressure,Np);
			//ScaLBL_DeviceBarrier(); MPI_Barrier(Dm->Comm);
			// i think these can be done in vector layout?
			ScaLBL_Comm->RegularLayout(Map,&Velocity[0],Velocity_x);
			ScaLBL_Comm->RegularLayout(Map,&Velocity[Np],Velocity_y);
			ScaLBL_Comm->RegularLayout(Map,&Velocity[2*Np],Velocity_z);
			ScaLBL_Comm->RegularLayout(Map,&Den[0],Density_A_Cart);
			ScaLBL_Comm->RegularLayout(Map,&Den[Np],Density_B_Cart);
			
            DoubleArray phase(Nx,Ny,Nz);
        	ScaLBL_CopyToHost(phase.data(), Phi, N*sizeof(double));
        	
        	// initialise the variables needed to aggregate velocity
			//double vax, vay, vaz, vbx, vby, vbz;
            double vA_x;// = Averages->van_global(0);
            double vA_y;// = Averages->van_global(1);
            double vA_z;// = Averages->van_global(2);
            double vB_x;// = Averages->vaw_global(0);
            double vB_y;// = Averages->vaw_global(1);
            double vB_z;// = Averages->vaw_global(2);
			double vax_loc, vay_loc, vaz_loc, vbx_loc, vby_loc, vbz_loc;
			vax_loc = vay_loc = vaz_loc = 0.f;
			vbx_loc = vby_loc = vbz_loc = 0.f;
			
			// initialise the variables needed for hydraulic components 
	        double vF = 0.f;
	        double vS = 0.f;
	        IntArray NWP_blob_label(Nx,Ny,Nz);
	        IntArray WP_blob_label(Nx,Ny,Nz);
            vector<int> connectedNWPBlobs;
            vector<int> connectedWPBlobs;
	        //Array<char> phase_id(Nx,Ny,Nz);
			//double vax, vay, vaz, vbx, vby, vbz;
            double vA_x_H;// = Averages->van_global(0);
            double vA_y_H;// = Averages->van_global(1);
            double vA_z_H;// = Averages->van_global(2);
            double vB_x_H;// = Averages->vaw_global(0);
            double vB_y_H;// = Averages->vaw_global(1);
            double vB_z_H;// = Averages->vaw_global(2);
			double vax_loc_H, vay_loc_H, vaz_loc_H, vbx_loc_H, vby_loc_H, vbz_loc_H;
			vax_loc_H = vay_loc_H = vaz_loc_H = 0.f;
			vbx_loc_H = vby_loc_H = vbz_loc_H = 0.f;
			if (BoundaryCondition == 0){
			    // find the NWP blobs
			    ComputeGlobalBlobIDs(Nx-2,Ny-2,Nz-2,rank_info,phase,Distance,vF,vS,NWP_blob_label,Dm->Comm);
	            MPI_Barrier(Dm->Comm);
	            //find the WP blobs
	            phase.scale(-1);
			    ComputeGlobalBlobIDs(Nx-2,Ny-2,Nz-2,rank_info,phase,Distance,vF,vS,WP_blob_label,Dm->Comm);
	            phase.scale(-1);
	            MPI_Barrier(Dm->Comm);
	            int numLocNWPBlobs=NWP_blob_label.max()+1;
	            int numGlobNWPBlobs;
			    MPI_Allreduce(&numLocNWPBlobs,&numGlobNWPBlobs,1,MPI_INT,MPI_MAX,Dm->Comm);
	            int numLocWPBlobs=WP_blob_label.max()+1;
	            int numGlobWPBlobs;
			    MPI_Allreduce(&numLocWPBlobs,&numGlobWPBlobs,1,MPI_INT,MPI_MAX,Dm->Comm);
	            MPI_Barrier(Dm->Comm);
	            if (rank==0) printf("Phase 1 Bodies: %d, Phase 2 Bodies: %d | ", numGlobNWPBlobs, numGlobWPBlobs);
	            // for each blob, check if it is hydraulically connected
	            // sweep the inlet and outlet ranks to see if a blob exists
	            int kproc = Dm->kproc();
	            vector<int> inletNWPBlobsLoc{-1,-2};
	            //vector<int> inletNWPBlobsGlob;
                int *inletNWPBlobsGlob;
	            vector<int> outletNWPBlobsLoc{-1,-2};
	            //vector<int> outletNWPBlobsGlob;
                int *outletNWPBlobsGlob;
	            vector<int> inletWPBlobsLoc{-1,-2};
	            //vector<int> inletWPBlobsGlob;
                int *inletWPBlobsGlob;
	            vector<int> outletWPBlobsLoc{-1,-2};
	            //vector<int> outletWPBlobsGlob;
                int *outletWPBlobsGlob;
	            int locBlobIDNWP;
	            int locBlobIDWP;
	            // local unique IO blobs
	            if (kproc == 0){
	                // if inlet domain, check which nwp and wp blobs are in inlet
	                // list these blobs per processor and collate globally
	                // check blobs one by one, or check in single sweep?
             		int k=1;
         			for (int j=1;j<Ny-1;j++){
         				for (int i=1;i<Nx-1;i++){
         					int n = k*Nx*Ny+j*Nx+i;
                            locBlobIDNWP = NWP_blob_label(i,j,k);
                            locBlobIDWP = WP_blob_label(i,j,k);
                            if (find(inletNWPBlobsLoc.begin(), inletNWPBlobsLoc.end(), locBlobIDNWP) == inletNWPBlobsLoc.end()){
                                //printf("Rank = %d, New NWP Blob %d in inlet\n",rank, locBlobIDNWP);
                                inletNWPBlobsLoc.push_back(locBlobIDNWP);
                            }
                            if (find(inletWPBlobsLoc.begin(), inletWPBlobsLoc.end(), locBlobIDWP) == inletWPBlobsLoc.end()){
                                //printf("Rank = %d, New WP Blob %d in inlet\n",rank, locBlobIDWP);
                                inletWPBlobsLoc.push_back(locBlobIDWP);
                            }
                            
         				}                    
         			}
	            }
	            //now check outlet domain
	            if (kproc == nprocz-1){
             		int k=Nz-2;
         			for (int j=1;j<Ny-1;j++){
         				for (int i=1;i<Nx-1;i++){
         					int n = k*Nx*Ny+j*Nx+i;
                            locBlobIDNWP = NWP_blob_label(i,j,k);
                            locBlobIDWP = WP_blob_label(i,j,k);
                            if (find(outletNWPBlobsLoc.begin(), outletNWPBlobsLoc.end(), locBlobIDNWP) == outletNWPBlobsLoc.end()){
                                //printf("Rank = %d, New NWP Blob %d in outlet\n",rank, locBlobIDNWP);
                                outletNWPBlobsLoc.push_back(locBlobIDNWP);
                            }
                            if (find(outletWPBlobsLoc.begin(), outletWPBlobsLoc.end(), locBlobIDWP) == outletWPBlobsLoc.end()){
                                //printf("Rank = %d, New WP Blob %d in outlet\n",rank, locBlobIDWP);
                                outletWPBlobsLoc.push_back(locBlobIDWP);
                            }
         				}                    
         			}
	            }
	            MPI_Barrier(Dm->Comm);
	            // collate inlet and outlet blob IDs to global
	            int numInletNWPBlobs = collateBoundaryBlobs(inletNWPBlobsGlob,inletNWPBlobsLoc);
	            int numInletWPBlobs = collateBoundaryBlobs(inletWPBlobsGlob,inletWPBlobsLoc);
	            int numOutletNWPBlobs = collateBoundaryBlobs(outletNWPBlobsGlob,outletNWPBlobsLoc);
	            int numOutletWPBlobs = collateBoundaryBlobs(outletWPBlobsGlob,outletWPBlobsLoc);
	            // check the inlet and outlet blobs to find connected IDs


                sort(inletNWPBlobsGlob, inletNWPBlobsGlob + numInletNWPBlobs); 
                sort(outletNWPBlobsGlob, outletNWPBlobsGlob + numOutletNWPBlobs); 
                vector<int> v(numInletNWPBlobs + numOutletNWPBlobs); 
                vector<int>::iterator it, st; 
                it = set_intersection(inletNWPBlobsGlob, inletNWPBlobsGlob + numInletNWPBlobs, 
                                      outletNWPBlobsGlob, outletNWPBlobsGlob + numOutletNWPBlobs, 
                                      v.begin()); 
              

                for (st = v.begin(); st != it; ++st) 
                    connectedNWPBlobs.push_back(*st); 
                connectedNWPBlobs.erase( unique( connectedNWPBlobs.begin(), connectedNWPBlobs.end() ), connectedNWPBlobs.end() );
                if (rank==0) {
                    printf("Hydraulically connected Phase 1 Blob IDs: "); 
                    for (int i=0;i<connectedNWPBlobs.size();i++){
                        printf("%d ",connectedNWPBlobs[i]);
                    }
                    printf(" | ");
                }
                
                sort(inletWPBlobsGlob, inletWPBlobsGlob + numInletWPBlobs); 
                sort(outletWPBlobsGlob, outletWPBlobsGlob + numOutletWPBlobs); 
                vector<int> v2(numInletWPBlobs + numOutletWPBlobs); 
                vector<int>::iterator it2, st2; 
                it2 = set_intersection(inletWPBlobsGlob, inletWPBlobsGlob + numInletWPBlobs, 
                                      outletWPBlobsGlob, outletWPBlobsGlob + numOutletWPBlobs, 
                                      v2.begin()); 
              
                for (st2 = v2.begin(); st2 != it2; ++st2) 
                    connectedWPBlobs.push_back(*st2); 
                connectedWPBlobs.erase( unique( connectedWPBlobs.begin(), connectedWPBlobs.end() ), connectedWPBlobs.end() );
                if (rank==0) {
                    printf("Hydraulically connected Phase 2 Blob IDs: "); 
                    for (int i=0;i<connectedWPBlobs.size();i++){
                        printf("%d ",connectedWPBlobs[i]);
                    }
                    printf("\n");
                }
            }
            // The settling parameter and phase volumes
            double countA = 0.;// = Averages->Volume_w();
            double countB = 0.;// = Averages->Volume_n();
            double countA_H = 0.;// = Averages->Volume_w();
            double countB_H = 0.;// = Averages->Volume_n();
            double current_saturation;
            double current_saturation_H;
            double settlingParam = 0.;
            double phiDiff = 0.;
			// sum velocities, volumes, differences
			for (int k=1; k<Nz-1; k++){
				for (int j=1; j<Ny-1; j++){
					for (int i=1; i<Nx-1; i++){
						if (Distance(i,j,k) > 0){
	                        phiDiff = phiDiff + (phase(i,j,k) - oldPhase(i,j,k))*(phase(i,j,k) - oldPhase(i,j,k));
	                        oldPhase(i,j,k) = phase(i,j,k);
						    if (phase(i,j,k)>0){
							    vax_loc += Velocity_x(i,j,k);
							    vay_loc += Velocity_y(i,j,k);
							    vaz_loc += Velocity_z(i,j,k);
	                            countA+=1.0;
							    if (find(connectedNWPBlobs.begin(), connectedNWPBlobs.end(), NWP_blob_label(i,j,k)) != connectedNWPBlobs.end()) {
    							    vax_loc_H += Velocity_x(i,j,k);
							        vay_loc_H += Velocity_y(i,j,k);
							        vaz_loc_H += Velocity_z(i,j,k);
	                                countA_H+=1.0;							    
							    }
							} else {
							    vbx_loc += Velocity_x(i,j,k);
							    vby_loc += Velocity_y(i,j,k);
							    vbz_loc += Velocity_z(i,j,k);
	                            countB+=1.0;
							    if (find(connectedWPBlobs.begin(), connectedWPBlobs.end(), WP_blob_label(i,j,k)) != connectedWPBlobs.end()) {
    							    vbx_loc_H += Velocity_x(i,j,k);
							        vby_loc_H += Velocity_y(i,j,k);
							        vbz_loc_H += Velocity_z(i,j,k);
							        countB_H+=1.0;
							    }
							}
						}
					}
				}
			}
        	MPI_Barrier(Dm->Comm);
			MPI_Allreduce(&vax_loc,&vA_x,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vay_loc,&vA_y,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vaz_loc,&vA_z,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vbx_loc,&vB_x,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vby_loc,&vB_y,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vbz_loc,&vB_z,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			
			MPI_Allreduce(&vax_loc_H,&vA_x_H,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vay_loc_H,&vA_y_H,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vaz_loc_H,&vA_z_H,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vbx_loc_H,&vB_x_H,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vby_loc_H,&vB_y_H,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			MPI_Allreduce(&vbz_loc_H,&vB_z_H,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
			
            MPI_Allreduce(&phiDiff,&settlingParam,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	        MPI_Allreduce(&countA,&volA,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	        MPI_Allreduce(&countB,&volB,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	        
	        MPI_Allreduce(&countA_H,&volA_H,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	        MPI_Allreduce(&countB_H,&volB_H,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	        
        	MPI_Barrier(Dm->Comm);
			vA_x /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vA_y /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vA_z /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vB_x /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vB_y /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vB_z /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			
			vA_x_H /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vA_y_H /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vA_z_H /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vB_x_H /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vB_y_H /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
			vB_z_H /= (Nx-2)*(Ny-2)*(Nz-2)*nprocs;
            double muA = rhoA*(tauA-0.5)/3.f;
            double muB = rhoB*(tauB-0.5)/3.f;
            double flow_rate_A = sqrt(vA_x*vA_x + vA_y*vA_y + vA_z*vA_z);
            double flow_rate_B = sqrt(vB_x*vB_x + vB_y*vB_y + vB_z*vB_z);
            double flow_rate_A_H = sqrt(vA_x_H*vA_x_H + vA_y_H*vA_y_H + vA_z_H*vA_z_H);
            double flow_rate_B_H = sqrt(vB_x_H*vB_x_H + vB_y_H*vB_y_H + vB_z_H*vB_z_H);
            double force_magnitude = sqrt(Fx*Fx + Fy*Fy + Fz*Fz);
            //double Ca = fabs((1-current_saturation)*muA*flow_rate_A + current_saturation*muB*flow_rate_B)/(5.796*alpha);
			double gradP=force_magnitude+(din-dout)/((Nz-2)*nprocz)/3;
			double absperm1 = muA*flow_rate_A*9.87e11*voxelSize*voxelSize/gradP;
			double absperm2 = muB*flow_rate_B*9.87e11*voxelSize*voxelSize/gradP;
			double absperm1_H = muA*flow_rate_A_H*9.87e11*voxelSize*voxelSize/gradP;
			double absperm2_H = muB*flow_rate_B_H*9.87e11*voxelSize*voxelSize/gradP;
            //scale the settling parameter by the domain size
            settlingParam = sqrt(settlingParam)/(double((Nx-2)*(Ny-2)*(Nz-2)*nprocs))/poro;
            current_saturation = volB/(volA+volB);
            current_saturation_H = volB_H/(volA_H+volB_H);
            // calculate the hydraulically connected capillary number if necessary...
            double Ca = fabs(volA*muA*flow_rate_A + volB*muB*flow_rate_B)/(alpha*double((Nx-2)*(Ny-2)*(Nz-2)*nprocs));
            Ca_EMA = approxRollingAverage(Ca_EMA, Ca, timestep);
            dCadt = fabs((Ca - Ca_previous)/Ca_previous);
            dCadtEMA = fabs((Ca_EMA - Ca_EMA_previous)/Ca_EMA_previous);
            Ca_previous=Ca;
            Ca_EMA_previous=Ca_EMA;
            absperm1_EMA = approxRollingAverage(absperm1_EMA, absperm1, timestep);
            absperm2_EMA = approxRollingAverage(absperm2_EMA, absperm2, timestep);
            absperm1_H_EMA = approxRollingAverage(absperm1_H_EMA, absperm1_H, timestep);
            absperm2_H_EMA = approxRollingAverage(absperm2_H_EMA, absperm2_H, timestep);
            double MLUPSGlob = 0.;
            double MLUPS = 0.;
		    MLUPS =  double(Np)*timestep/cputime/1000000;
			MPI_Allreduce(&MLUPS,&MLUPSGlob,1,MPI_DOUBLE,MPI_SUM,Dm->Comm); 
            //printf("Rank: %d, MLUPS: %f\n",rank, MLUPS);
			if (rank==0) {
				printf("Phase 1: %f D, Phase 2: %f D, Connected Phase 1: %f D, Connected Phase 2 %f D\n",absperm1, absperm2,absperm1_H, absperm2_H);
				printf("EMA: Phase 1: %f D, Phase 2: %f D, Connected Phase 1: %f D, Connected Phase 2 %f D\n",absperm1_EMA, absperm2_EMA,absperm1_H_EMA, absperm2_H_EMA);
	            printf("MLUPS: %f, Sat = %f, flux = %e, force = %e, gradP = %e, Nca = %e, EMANca = %e, EMAdNca = %e, setPar = %e\n",MLUPSGlob, current_saturation, flux, force_magnitude, gradP, Ca, Ca_EMA, dCadtEMA, settlingParam);
			}

            // rampup the component affinities - functionalise this for better code
            if (affinityRampupFlag && timestep < affinityRampSteps){
            	size_t NLABELS=0;
	            char VALUE=0;
	            double AFFINITY=0.f;

	            auto LabelList = color_db->getVector<char>( "ComponentLabels" );
	            auto AffinityList = color_db->getVector<double>( "ComponentAffinity" );

	            NLABELS=LabelList.size();
	            for (int idx=0; idx < NLABELS; idx++){ //modify the affinities
		            AFFINITY=AffinityList[idx];
		            AffinityList[idx]=AFFINITY/affinityRampSteps*timestep;
	            }
	            int label_count[NLABELS];
	            int label_count_global[NLABELS];
	            // Assign the labels
	            for (int idx=0; idx<NLABELS; idx++) label_count[idx]=0;
	            
	            for (int k=0;k<Nz;k++){
		            for (int j=0;j<Ny;j++){
			            for (int i=0;i<Nx;i++){
				            int n = k*Nx*Ny+j*Nx+i;
				            VALUE=id[n]; //read the geometry
				            for (unsigned int idx=0; idx < NLABELS; idx++){ //find the right component
					            if (VALUE == LabelList[idx]){
						            AFFINITY=AffinityList[idx];
						            label_count[idx]++;
						            idx = NLABELS;//exit the finder loop
    				                AFFINITY=AFFINITY;
				                    phase(i,j,k) = AFFINITY;
					            }
				            }
			            }
		            }
	            }
                //gather summaries and output
	            MPI_Allreduce(&label_count[0],&label_count_global[0],NLABELS,MPI_INT,MPI_SUM,Dm->Comm);
	            if (rank==0){
		            printf("Components labels: %lu \n",NLABELS);
		            for (unsigned int idx=0; idx<NLABELS; idx++){
			            VALUE=LabelList[idx];
			            AFFINITY=AffinityList[idx];
			            double volume_fraction  = double(label_count_global[idx])/double((Nx)*(Ny)*(Nz)*nprocz*nprocy*nprocx);
			            printf(" --label=%i, affinity=%f, volume fraction==%f\n",int(VALUE),AFFINITY,volume_fraction); 
		            }
	            }
	            // 6. copy back to the device
	            ScaLBL_CopyToDevice(Phi,phase.data(),N*sizeof(double));
	            // 7. Re-initialize phase field and density
	            ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq, 0, ScaLBL_Comm->LastExterior(), Np);
	            ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
//	            if (BoundaryCondition > 0 ){
//		            if (Dm->kproc()==0){
//			            ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,0);
//			            ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,1);
//			            ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,2);
//		            }
//		            if (Dm->kproc() == nprocz-1){
//			            ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-1);
//			            ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-2);
//			            ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-3);
//		            }
//	            }
            }
        	MPI_Barrier(Dm->Comm);
            
            // check for flux reversal
	        //printf("Core: %d, setParm: %f\n",rank, settlingParam); 
            if (fluxReversalFlag){
                if (current_saturation < fluxReversalSat || settlingParam < settlingTolerance){
        		    WriteDebugYDW();
                    if (autoMorphFlag) { //if automorph is active, reverse the morph direction, and reverse the force if thats an active flag
                        injectionType = fabs(injectionType-3); //1-2 switch
                        satInit=1.0-satInit;
                        if (rank==0) printf("[AUTOMORPH], Flux reversal conditions have been met, the injection type has been switched to %d\n", injectionType);
                    }
                    if (fluxReversalType==1){
                        inletA=fabs(inletA-1.0); //a is nwp
                        inletB=fabs(inletB-1.0);
                        outletA=fabs(outletA-1.0);
                        outletB=fabs(outletB-1.0);
                        if (rank==0) printf("Flux Reversal Type: Phases flipped\n");
                    } else if (fluxReversalType==2){
                        flux=flux*-1;
                        Fx=Fx*-1;
                        Fy=Fy*-1;
                        Fz=Fz*-1;
                        if (rank==0) printf("Flux Reversal Type: Flow direction flipped\n");
                    }
                    if (rank==0 && current_saturation < fluxReversalSat) printf("Current Saturation of %f is less than the flux reversal saturation of %f. \nFluxes have been reversed. \nThis will not carry on to restart. \nPlease manually change the inputfile at end of simulation.\n", current_saturation, fluxReversalSat);
                    if (rank==0 && settlingParam < settlingTolerance) printf("Current Saturation of %f has reached steady state, so flux reversal has been activated. \nFluxes have been reversed. \nThis will not carry on to restart. \nPlease manually change the inputfile at end of simulation.\n", current_saturation);
                    fluxReversalFlag = false;
                }
            }
            
            
            // adjust the force if capillary number is set
			if (SET_CAPILLARY_NUMBER && !autoMorphAdapt && stabilityCounter <= 0 && !coinjectionFlag && timestep > ramp_timesteps){ // activate if capillary number is specified, and during morph, only after acceleration is done
			    // at each analysis step, 
                if (Ca>0.f){
                    Fx *= capillary_number / Ca;
                    Fy *= capillary_number / Ca;
                    Fz *= capillary_number / Ca;
                    force_magnitude = sqrt(Fx*Fx + Fy*Fy + Fz*Fz);

                    if (force_magnitude > 1e-3){
		                Fx *= 1e-3/force_magnitude;   // impose ceiling for stability
                        Fy *= 1e-3/force_magnitude;
                        Fz *= 1e-3/force_magnitude;
                    }
                    if (force_magnitude < 1e-6 && force_magnitude > 0){
                        Fx *= 1e-6/force_magnitude;   // impose floor
                        Fy *= 1e-6/force_magnitude;
                        Fz *= 1e-6/force_magnitude;
                    }
                    
                    
                    flux *= min(max(0.5, capillary_number / Ca), 2.0); //much more concise than if else bounding
                    if (rank == 0) printf("Adjusting force by factor %f to %e, Nca = %e, Target: %e \n ",min(max(0.5, capillary_number / Ca), 2.0), force_magnitude, Ca, capillary_number);
                    //Averages->SetParams(rhoA,rhoB,tauA,tauB,Fx,Fy,Fz,alpha);
            	}
			}
			
			
			//co-injeciton stabilisation routine
			if (timestep > ramp_timesteps && coinjectionFlag){
			    if (stabilityCounter >= stabilisationRate){ // theres no adaptation phase, so no extra flag here
			        if (dCadtEMA < tolerance ){
            		    WriteDebugYDW();
				        if (rank==0){ //save the data as a rel perm point
					        printf("[CO-INJECTION]: Steady state reached. WRITE STEADY POINT\n");
					        volA /= double((Nx-2)*(Ny-2)*(Nz-2)*nprocs);
					        volB /= double((Nx-2)*(Ny-2)*(Nz-2)*nprocs);// was this supposed to be nprocsz?
					        FILE * kr_log_file = fopen("relperm.csv","a");
					        fprintf(kr_log_file,"%i %.5g %.5g %.5g %.5g %.5g %.5g ",timestep,muA,muB,alpha,Fx,Fy,Fz);
					        fprintf(kr_log_file,"%.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g\n",volA,volB,inletA,inletB,vA_x,vA_y,vA_z,vB_x,vB_y,vB_z,current_saturation,absperm1,absperm2,current_saturation_H,absperm1_H,absperm2_H);
					        fclose(kr_log_file);
				        }
                        MPI_Barrier(Dm->Comm);
				        if (injectionType==1){ //drainage - increase injected oil volumes
                            inletA=inletA+satInc; //a is nwp
                            inletB=inletB-satInc;
                            outletA=outletA+satInc; //a is nwp
                            outletB=outletB-satInc;
		                } else if (injectionType==2){ // imbibiton, reduce injected oil volumes
                            inletA=inletA-satInc; //a is nwp
                            inletB=inletB+satInc;
                            outletA=outletA-satInc; //a is nwp
                            outletB=outletB+satInc;

		                }
        			    if (rank==0) printf("[CO-INJECTION]: Inlet altered to: Phase 1: %f, Phase 2: %f \n",inletA, inletB);
				    } else { //if the system is unstable, continue stabilisation
					    if (rank==0) printf("[CO-INJECTION]: Continuing stabilisation. Inlet: %f, %f. Outlet: %f, %f\n",inletA, inletB, outletA, outletB);
					    stabilityCounter = 1;
				    }
		        }
			    stabilityCounter += analysis_interval;
			}
			
			
            //AUTOMORPH routine
			if (timestep > ramp_timesteps && autoMorphFlag){
//			    if (current_saturation*((injectionType-1)*2-1)<satInit*((injectionType-1)*2-1) && satInit > 0.0){
//			        // initially, use flux conditions to push the system along
//				    if (rank==0) printf("[AUTOMORPH]: Initial Flux injection to target %f (current: %f) \n", satInit, current_saturation);
//                    flux = 0.01*double((Nx-2)*(Ny-2)*(Nz-2)*nprocs)*poro/accelerationRate;
//                    BoundaryCondition = 4;
//			        if (injectionType==1){ //flux morph will temporarily activate flux boundary conditions
//                        inletA=1.0; //a is nwp
//                        inletB=0.0;
//                        outletA=0.0;
//                        outletB=1.0;
//	                } else if (injectionType==2){
//                        inletA=0.0; //a is nwp
//                        inletB=1.0;
//                        outletA=1.0;
//                        outletB=0.0;
//	                }
//	                stabilityCounter = -10*analysis_interval;
//	                accelerationCounter = -analysis_interval;
//			    } else {
//			        BoundaryCondition = 0;
//			        flux = 0;
//			    }
                // once rampup and init flux are done, run morph
			    if (!autoMorphAdapt && stabilityCounter >= stabilisationRate){//if acceleration is currently off, (stabilisation is active)
				    if (dCadtEMA < tolerance ){ //if the capillary number has stabilised, record, adjust, and activate acceleration
	        		    WriteDebugYDW();
					    if (rank==0){ //save the data as a rel perm point
						    printf("[AUTOMORPH]: Steady state reached. WRITE STEADY POINT \n");
						    
						    // here, take the blobmaps - generate new Dm and run singlephase lbm for hydraulically connected phases
						    // take blob maps and get new ID maps
						    // use new DmPh1 and DmPh2 (initialise earlier), populate them appropriately
						    // initialise memory for single phase
						    // run single phase, store the perm values
						    
						    volA /= double((Nx-2)*(Ny-2)*(Nz-2)*nprocs);
						    volB /= double((Nx-2)*(Ny-2)*(Nz-2)*nprocs);// was this supposed to be nprocsz?
						    FILE * kr_log_file = fopen("relperm.csv","a");
						    fprintf(kr_log_file,"%i %.5g %.5g %.5g %.5g %.5g %.5g ",timestep,muA,muB,alpha,Fx,Fy,Fz);
					        fprintf(kr_log_file,"%.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g %.5g\n",volA,volB,inletA,inletB,vA_x,vA_y,vA_z,vB_x,vB_y,vB_z,current_saturation,absperm1,absperm2,current_saturation_H,absperm1_H,absperm2_H);
						    fclose(kr_log_file);
					    }
	                    MPI_Barrier(Dm->Comm);				    
					    if (injectionType==1){
			                targetSaturation = current_saturation - satInc;
		                    shellRadius = 1.0+current_saturation;
		                    if (targetSaturation < 0.05 ){
    						    if (rank==0) printf("[AUTOMORPH]: MorphDrain has reached <0.05 saturation. Terminating simulation. \n");
    						    break; 
						    }
			            } else if (injectionType==2){
			                targetSaturation = current_saturation + satInc;
		                    shellRadius = -0.1;
		                    if (targetSaturation > 0.95 ){
    						    if (rank==0) printf("[AUTOMORPH]: MorphImb has reached >0.95 saturation. Terminating simulation. \n");
    						    break; 
						    }
			            }
			            autoMorphAdapt = true; // turn on acceleration immediately
			            accelerationCounter = accelerationRate+1;
				    } else { //if the system is unstable, continue stabilisation
					    if (rank==0) printf("[AUTOMORPH]: System is unstable, continuing LBM stabilisation.\n");
					    stabilityCounter = 1; //enforce capillary number is not readjusted
				    }
                }
			    stabilityCounter += analysis_interval;
				if (autoMorphAdapt && accelerationCounter >= accelerationRate) { //if the system just stabilised, push forward to the next target
                    // check the state
	                if (current_saturation*((injectionType-1)*2-1)>targetSaturation*((injectionType-1)*2-1)){
		                autoMorphAdapt = false;
    					if (rank==0) printf("[AUTOMORPH]: Target reached, beginning stabilisation\n");
				        stabilityCounter = max(-stabilisationRate,-10*analysis_interval); //give some time to readjust the capillary number
//                        BoundaryCondition = 0;
//                        flux = 0;
                        
	                } else {
    					if (rank==0 && !fluxMorphFlag) printf("[AUTOMORPH]: Automorph executing, beginning interim relaxation\n");
    					if (rank==0 && fluxMorphFlag) printf("[AUTOMORPH]: Fluxmorph still active, continuing...\n");
    					accelerationCounter = 0; //reset the acceleration
	                }
					if (fluxMorphFlag) { //fluxmorph accelearation rate
					    if (rank==0) printf("[AUTOMORPH]: Flux Injection acceleration to target %f (current: %f) \n", targetSaturation, current_saturation);
				        if (injectionType==1){ //flux morph will temporarily activate flux boundary conditions
                            BoundaryCondition = 4;
                            flux = 0.01*double((Nx-2)*(Ny-2)*(Nz-2)*nprocs)*poro/accelerationRate;
		                } else if (injectionType==2){
                            BoundaryCondition = 4;
                            flux = -0.01*double((Nx-2)*(Ny-2)*(Nz-2)*nprocs)*poro/accelerationRate; //inject 1% every accelerationrate intervals
		                }
					} else if (spinoMorphFlag) {
					    if (rank==0) printf("[AUTOMORPH]: Spinodal condensation to target %f (current: %f) \n", targetSaturation, current_saturation);
				        double delta_volume = SpinoInit(satInc*0.5*((injectionType-1)*2-1));//run the adaptation					    
					} else {
					    if (rank==0) printf("[AUTOMORPH]: Morphological acceleration to target %f (current: %f) ", targetSaturation, current_saturation);
				        double delta_volume = MorphInit(beta,shellRadius);//run the adaptation
				        double delta_volume_target = volB - (volA + volB)*targetSaturation; // change in volume to A
                        //shell radius adjustment should be based the change in volume vs the change in saturation
                        if (injectionType==2){
				            if ((delta_volume_target - delta_volume) / delta_volume > 0.f){
					            shellRadius *= 1.01*min((delta_volume_target - delta_volume) / delta_volume, 3.0);
					            if (shellRadius < -1.f) shellRadius = -1.f;
				            }
			            }			            
    				}
    				MPI_Barrier(Dm->Comm);
			    }
				accelerationCounter += analysis_interval; //increment the acceleration
			}
		}
	}
	//analysis.finish();
	//PROFILE_STOP("Loop");
	//PROFILE_SAVE("lbpm_color_simulator",1);
	//************************************************************************
	ScaLBL_DeviceBarrier();
	MPI_Barrier(Dm->Comm);
//	stoptime = MPI_Wtime();
//	if (rank==0) printf("-------------------------------------------------------------------\n");
//	// Compute the walltime per timestep
//	cputime = (stoptime - starttime)/timestep;
//	// Performance obtained from each node
//	double MLUPS = double(Np)/cputime/1000000;

//	if (rank==0) printf("********************************************************\n");
//	if (rank==0) printf("CPU time = %f \n", cputime);
//	if (rank==0) printf("Lattice update rate (per core)= %f MLUPS \n", MLUPS);
//	MLUPS *= nprocs;
//	if (rank==0) printf("Lattice update rate (total)= %f MLUPS \n", MLUPS);
//	if (rank==0) printf("********************************************************\n");

	// ************************************************************************
}

double ScaLBL_ColorModel::SpinoInit(const double delta_sw){
	const RankInfoStruct rank_info(rank,nprocx,nprocy,nprocz);

	double vF = 0.f;
	double vS = 0.f;

	DoubleArray phase(Nx,Ny,Nz);
	IntArray phase_label(Nx,Ny,Nz);;
	DoubleArray phase_distance(Nx,Ny,Nz);
	Array<char> phase_id(Nx,Ny,Nz);

	// Basic algorithm to 
	// 1. Copy phase field Phi to CPU in phase
	ScaLBL_CopyToHost(phase.data(), Phi, N*sizeof(double));
    // compute the current nwp volume
	double count,count_global,volume_initial,volume_final;
	count = 0.f;
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
				if (phase(i,j,k) > 0.f && Distance(i,j,k) > 0.f) count+=1.f;
			}
		}
	}
	MPI_Allreduce(&count,&count_global,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	volume_initial = count_global;
	
	// 2. Loop around the phase, and randomly reassign label based on delta_sw
	// if the delta is positive, add to -1, if negative, add to +1
	
	for (int k=0; k<Nz; k++){
	    for (int j=0; j<Ny; j++){
		    for (int i=0; i<Nx; i++){ 
			    if ( Distance(i,j,k) > 0.f){
				    if (delta_sw < 0.0 && phase(i,j,k) < 0.f) { //if imb, and within nwp
				        float rnd = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				        if (rnd<=fabs(delta_sw)) phase(i,j,k) = 1.0;
				    } else if (delta_sw > 0.0 && phase(i,j,k) > 0.f) { // if drai, and within wp
				        float rnd =  static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
				        if (rnd<=fabs(delta_sw)) phase(i,j,k) = -1.0;
				    }
			    }
												    
		    }
	    }
    }
    // 3. recalculate volume
	count = 0.f;
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
				if (phase(i,j,k) > 0.f && Distance(i,j,k) > 0.f) count+=1.f;
			}
		}
	}
	MPI_Allreduce(&count,&count_global,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	volume_final=count_global;

	double delta_volume = (volume_final-volume_initial);
	if (rank == 0)  printf("SpinoInit: change fluid volume fraction by %f%\n", delta_volume/volume_initial*100);

	// 6. copy back to the device
	//if (rank==0)  printf("MorphInit: copy data  back to device\n");
	ScaLBL_CopyToDevice(Phi,phase.data(),N*sizeof(double));

	// 7. Re-initialize phase field and density
	ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq, 0, ScaLBL_Comm->LastExterior(), Np);
	ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
	
	//compatibility with color boundary conditions, though we should be using the proper inlet outlet values here...
//	if (BoundaryCondition > 0){
//		if (Dm->kproc()==0){
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,0);
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,1);
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,2);
//		}
//		if (Dm->kproc() == nprocz-1){
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-1);
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-2);
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-3);
//		}
//	}
	
	return delta_volume;
}


double ScaLBL_ColorModel::MorphInit(const double beta, const double morph_delta){
	const RankInfoStruct rank_info(rank,nprocx,nprocy,nprocz);

	double vF = 0.f;
	double vS = 0.f;

	DoubleArray phase(Nx,Ny,Nz);
	IntArray phase_label(Nx,Ny,Nz);;
	DoubleArray phase_distance(Nx,Ny,Nz);
	Array<char> phase_id(Nx,Ny,Nz);

	// Basic algorithm to 
	// 1. Copy phase field to CPU
	ScaLBL_CopyToHost(phase.data(), Phi, N*sizeof(double));

	double count,count_global,volume_initial,volume_final;
	count = 0.f;
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
				if (phase(i,j,k) > 0.f && Distance(i,j,k) > 0.f) count+=1.f;
			}
		}
	}
	MPI_Allreduce(&count,&count_global,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	volume_initial = count_global;

	// 2. Identify connected components of phase field -> phase_label
	//BlobIDstruct new_index;
	ComputeGlobalBlobIDs(Nx-2,Ny-2,Nz-2,rank_info,phase,Distance,vF,vS,phase_label,Dm->Comm);
	MPI_Barrier(Dm->Comm);
	// only operate on component "0"
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
				int label = phase_label(i,j,k);
				if (label == 0 )     phase_id(i,j,k) = 0;
				else 		     phase_id(i,j,k) = 1;
			}
		}
	}	
	// 3. Generate a distance map to the largest object -> phase_distance
	CalcDist(phase_distance,phase_id,*Dm);
	if (morph_delta<0.0) { // YDW modification: only do this for imbibition
	    double temp,value;
	    double factor=0.5/beta;
	    for (int k=0; k<Nz; k++){
		    for (int j=0; j<Ny; j++){
			    for (int i=0; i<Nx; i++){
				    if (phase_distance(i,j,k) < 3.f ){ //for voxels near the blob, define the distance using a diffuse interface value
					    value = phase(i,j,k);
					    if (value > 1.f)   value=1.f;
					    if (value < -1.f)  value=-1.f;
					    // temp -- distance based on analytical form McClure, Prins et al, Comp. Phys. Comm.
					    temp = -factor*log((1.0+value)/(1.0-value));
					    /// use this approximation close to the object
					    if (fabs(value) < 0.8 && Distance(i,j,k) > 1.f ){
						    phase_distance(i,j,k) = temp;
					    }
				    }
			    }
		    }
	    }
	    // 4. Apply erosion / dilation operation to phase_distance
	    for (int k=0; k<Nz; k++){
		    for (int j=0; j<Ny; j++){
			    for (int i=0; i<Nx; i++){
				    double walldist=Distance(i,j,k);
				    double wallweight = 1.f / (1+exp(-5.f*(walldist-1.f))); 
				    phase_distance(i,j,k) -= wallweight*morph_delta;
			    }
		    }
	    }
	    // 5. Update phase indicator field based on new distnace
	    for (int k=0; k<Nz; k++){
		    for (int j=0; j<Ny; j++){
			    for (int i=0; i<Nx; i++){
				    int n = k*Nx*Ny + j*Nx + i;
				    double d = phase_distance(i,j,k);
				    if (Distance(i,j,k) > 0.f){
					    // only update phase field in immediate proximity of largest component
					    if (d < 3.f){
						    phase(i,j,k) = (2.f*(exp(-2.f*beta*d))/(1.f+exp(-2.f*beta*d))-1.f);
					    }
				    }
			    } 
		    }
	    }
	} else {// YDW modification: use a simpler and more aggresive method for drainage
	    for (int k=0; k<Nz; k++){
		    for (int j=0; j<Ny; j++){
			    for (int i=0; i<Nx; i++){ // if the distance from the largest blob is within the morph delta, and the wall distance is larger than 1
				    if (phase_distance(i,j,k) < morph_delta && phase_distance(i,j,k) > -morph_delta && Distance(i,j,k) > 1.f){
					    phase(i,j,k) = 1.0*fabs(morph_delta)/morph_delta ; // relax and flip the value	
				    }
													    
			    }
		    }
	    }
	}
	count = 0.f;
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
				if (phase(i,j,k) > 0.f && Distance(i,j,k) > 0.f) count+=1.f;
			}
		}
	}
	MPI_Allreduce(&count,&count_global,1,MPI_DOUBLE,MPI_SUM,Dm->Comm);
	volume_final=count_global;

	double delta_volume = (volume_final-volume_initial);
	if (rank == 0)  printf("MorphInit: change fluid volume fraction by %f% with shell radius %f \n", delta_volume/volume_initial*100,morph_delta);

	// 6. copy back to the device
	//if (rank==0)  printf("MorphInit: copy data  back to device\n");
	ScaLBL_CopyToDevice(Phi,phase.data(),N*sizeof(double));

	// 7. Re-initialize phase field and density
	ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq, 0, ScaLBL_Comm->LastExterior(), Np);
	ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
	
	//compatibility with color boundary conditions, though we should be using the proper inlet outlet values here...
//	if (BoundaryCondition >0 ){
//		if (Dm->kproc()==0){
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,0);
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,1);
//			ScaLBL_SetSlice_z(Phi,1.0,Nx,Ny,Nz,2);
//		}
//		if (Dm->kproc() == nprocz-1){
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-1);
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-2);
//			ScaLBL_SetSlice_z(Phi,-1.0,Nx,Ny,Nz,Nz-3);
//		}
//	}
	
	return delta_volume;
}

void ScaLBL_ColorModel::WriteDebug(){
	// Copy back final phase indicator field and convert to regular layout
	DoubleArray PhaseField(Nx,Ny,Nz);
	//ScaLBL_Comm->RegularLayout(Map,Phi,PhaseField);
	ScaLBL_CopyToHost(PhaseField.data(), Phi, sizeof(double)*N);
	char LocalRankFilename[100];
	FILE *OUTFILE;
	sprintf(LocalRankFilename,"Phase.%05i.raw",rank); //change this file name to include the size
	OUTFILE = fopen(LocalRankFilename,"wb");
	fwrite(PhaseField.data(),8,N,OUTFILE);
	fclose(OUTFILE);
}

void ScaLBL_ColorModel::WriteDebugYDW(){
	//create the folder
	char LocalRankFoldername[100];
	if (rank==0) {
		sprintf(LocalRankFoldername,"./rawVis%d",timestep); 
	    mkdir(LocalRankFoldername, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
	// Copy back final phase indicator field and convert to regular layout
	DoubleArray PhaseField(Nx,Ny,Nz);
	//ScaLBL_Comm->RegularLayout(Map,Phi,PhaseField);
	ScaLBL_CopyToHost(PhaseField.data(), Phi, sizeof(double)*N);
	//create the file
    double temp = 0.0;
	FILE *OUTFILE;
	char LocalRankFilename[100];
	sprintf(LocalRankFilename,"rawVis%d/Part_%d_%d_%d_%d_%d_%d_%d.txt",timestep,rank,Nx,Ny,Nz,nprocx,nprocy,nprocz); //change this file name to include the size
	OUTFILE = fopen(LocalRankFilename,"w");
    //td::fstream ofs(LocalRankFilename, ios::out | ios::binary );
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
			    //fprintf(OUTFILE,"%f\n",PhaseField(i, j, k));
			    temp = PhaseField(i,j,k);
	            fwrite(&temp,sizeof(double),1,OUTFILE);
			}
		}
	}
	fclose(OUTFILE);
	MPI_Barrier(Dm->Comm);
	
	
	FILE *OUTFILEVels;
	char LocalRankFilenameVels[100];
	sprintf(LocalRankFilenameVels,"rawVis%d/Vel_Part_%d_%d_%d_%d_%d_%d_%d.txt",timestep,rank,Nx,Ny,Nz,nprocx,nprocy,nprocz); //change this file name to include the size
	OUTFILEVels = fopen(LocalRankFilenameVels,"w");
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
			    //fprintf(OUTFILEX,"%f\n",Velocity_x(i, j, k));
			    temp = Velocity_x(i,j,k);
	            fwrite(&temp,sizeof(double),1,OUTFILEVels);
			}
		}
	}
	
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
			    //fprintf(OUTFILEY,"%f\n",Velocity_y(i, j, k));
			    temp = Velocity_y(i,j,k);
	            fwrite(&temp,sizeof(double),1,OUTFILEVels);
			}
		}
	}
	
	for (int k=0; k<Nz; k++){
		for (int j=0; j<Ny; j++){
			for (int i=0; i<Nx; i++){
			    //fprintf(OUTFILEZ,"%f\n",Velocity_z(i, j, k));
			    temp = Velocity_z(i,j,k);
	            fwrite(&temp,sizeof(double),1,OUTFILEVels);
			}
		}
	}
	fclose(OUTFILEVels);
	MPI_Barrier(Dm->Comm);
}

double ScaLBL_ColorModel::approxRollingAverage(double avg, double new_sample, int timestep) {
    double counter = min(50.,double(timestep)/1000); 
    avg = avg - avg / counter;
    avg = avg + new_sample / counter;

    return avg;
}

int ScaLBL_ColorModel::collateBoundaryBlobs(int *&inletNWPBlobsGlob, vector<int>inletNWPBlobsLoc) {
    int *recvcounts;
    recvcounts = (int *) malloc( nprocs * sizeof(int));
    inletNWPBlobsLoc.erase(std::remove(inletNWPBlobsLoc.begin(), inletNWPBlobsLoc.end(), -2), inletNWPBlobsLoc.end());
    inletNWPBlobsLoc.erase(std::remove(inletNWPBlobsLoc.begin(), inletNWPBlobsLoc.end(), -1), inletNWPBlobsLoc.end());
    int sizeLocalBlobsList = inletNWPBlobsLoc.size();
    MPI_Allgather(&sizeLocalBlobsList, 1, MPI_INT, recvcounts, 1, MPI_INT, Dm->Comm); // gather the bloblengths
//	        if (rank==0) {
//                for (int i = 0; i < nprocs; i++) {
//	                printf("rank: %d, recvCounts: %d \n",rank, recvcounts[i]);
//                }
//	        }
    int totlen = 0; 
    int *displs;
    displs = new int[sizeLocalBlobsList*sizeof(int)];
    displs[0] = 0; // log up the cumulative values
    totlen += recvcounts[0]; // the total
    for (int i=1; i<nprocs; i++) {
       totlen += recvcounts[i];   
       displs[i] = displs[i-1] + recvcounts[i-1];
    }
//            for (int i = 0; i < nprocs; i++) {
//                printf("rank: %d, totlen: %d, displacements: %d, recvCounts: %d \n",rank, totlen, displs[i], recvcounts[i]);
//            }
    /* allocate string, pre-fill with spaces and null terminator */
    inletNWPBlobsGlob = (int *) malloc( totlen * sizeof(int)) ;          
    MPI_Barrier(Dm->Comm);
    int *temp;
    temp = (int *) malloc( inletNWPBlobsLoc.size() * sizeof(int)) ; 
    for (int i=0;i<inletNWPBlobsLoc.size();i++){
        temp[i] = inletNWPBlobsLoc[i];
    }
    MPI_Allgatherv(temp, sizeLocalBlobsList, MPI_INT, inletNWPBlobsGlob, recvcounts, displs, MPI_INT, Dm->Comm);
//            for (int i = 0; i < totlen; i++) {
//                printf("rank: %d, boundary blobID: %d \n",rank, inletNWPBlobsGlob[i]);
//            }
    MPI_Barrier(Dm->Comm);
    return totlen;
}
