/*
 * Pre-processor to generate signed distance function from segmented data
 * segmented data should be stored in a raw binary file as 1-byte integer (type char)
 * will output distance functions for phases
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "common/Array.h"
#include "common/Domain.h"
#include <inttypes.h>
int main(int argc, char **argv)
{

	int rank=0;

	/*		bool MULTINPUT=false;

		int NWP,SOLID,rank_offset;
		SOLID=atoi(argv[1]);
		NWP=atoi(argv[2]);

		if (rank==0){
			printf("Solid Label: %i \n",SOLID);
			printf("NWP Label: %i \n",NWP);
		}
		if (argc > 3){
			rank_offset = atoi(argv[3]);
		}
		else{
			MULTINPUT=true;
			rank_offset=0;
		}
	 */
	string filename;
	if (argc > 1)
		filename=argv[1];
	else{
		ERROR("lbpm_serial_decomp: no in put database provided \n");
	}
	int rank_offset=0;

	//.......................................................................
	// Reading the domain information file
	//.......................................................................
	int nprocs, nprocx, nprocy, nprocz, nx, ny, nz, nspheres;
	double Lx, Ly, Lz; 
	int64_t Nx,Ny,Nz; //domain counting must be 64bit = indexes commonly exceed 2 billion
	int64_t i,j,k,n;
	int BC=0;
	int64_t xStart,yStart,zStart;
	//  char fluidValue,solidValue;

	xStart=yStart=zStart=0;
	// read the input database 
	auto db = std::make_shared<Database>( filename );
	auto domain_db = db->getDatabase( "Domain" );

	// Read domain parameters
	auto Filename = domain_db->getScalar<std::string>( "Filename" );
	auto L = domain_db->getVector<double>( "L" );
	auto size = domain_db->getVector<int>( "n" );
	auto SIZE = domain_db->getVector<int>( "N" );
	auto nproc = domain_db->getVector<int>( "nproc" );
	if (domain_db->keyExists( "offset" )){
		auto offset = domain_db->getVector<int>( "offset" );
		xStart = offset[0];
		yStart = offset[1];
		zStart = offset[2];
	}
	auto ReadValues = domain_db->getVector<char>( "ReadValues" );
	auto WriteValues = domain_db->getVector<char>( "WriteValues" );
	auto ReadType = domain_db->getScalar<std::string>( "ReadType" );
	if (ReadType == "8bit"){
	}
	else if (ReadType == "16bit"){
	}
	else{
		printf("INPUT ERROR: Valid ReadType are 8bit, 16bit \n");
		ReadType = "8bit";
	}

	nx = size[0];
	ny = size[1];
	nz = size[2];
	nprocx = nproc[0];
	nprocy = nproc[1];
	nprocz = nproc[2];
	Nx = SIZE[0];
	Ny = SIZE[1];
	Nz = SIZE[2];

	printf("Input media: %s\n",Filename.c_str());
	printf("Relabeling %lu values\n",ReadValues.size());
	for (int idx=0; idx<ReadValues.size(); idx++){
		char oldvalue=ReadValues[idx];
		char newvalue=WriteValues[idx];
		printf("oldvalue=%d, newvalue =%d \n",oldvalue,newvalue);
	}

	nprocs=nprocx*nprocy*nprocz;

	char *SegData = NULL;
	// Rank=0 reads the entire segmented data and distributes to worker processes
	if (rank==0){
		printf("Dimensions of segmented image: %ld x %ld x %ld \n",Nx,Ny,Nz);
		int64_t SIZE = Nx*Ny*Nz;
		SegData = new char[SIZE];
		if (ReadType == "8bit"){
			printf("Reading 8-bit input data \n");
			FILE *SEGDAT = fopen(Filename.c_str(),"rb");
			if (SEGDAT==NULL) ERROR("Error reading segmented data");
			size_t ReadSeg;
			ReadSeg=fread(SegData,1,SIZE,SEGDAT);
			if (ReadSeg != size_t(SIZE)) printf("lbpm_segmented_decomp: Error reading segmented data (rank=%i)\n",rank);
			fclose(SEGDAT);
		}
		else if (ReadType == "16bit"){
			printf("Reading 16-bit input data \n");
			short int *InputData;
			InputData = new short int[SIZE];
			FILE *SEGDAT = fopen(Filename.c_str(),"rb");
			if (SEGDAT==NULL) ERROR("Error reading segmented data");
			size_t ReadSeg;
			ReadSeg=fread(InputData,2,SIZE,SEGDAT);
			if (ReadSeg != size_t(SIZE)) printf("lbpm_segmented_decomp: Error reading segmented data (rank=%i)\n",rank);
			fclose(SEGDAT);
			for (int n=0; n<SIZE; n++){
				SegData[n] = char(InputData[n]);
			}
		}
		printf("Read segmented data from %s \n",Filename.c_str());
	}

	// Get the rank info
	int64_t N = (nx+2)*(ny+2)*(nz+2);

	// number of sites to use for periodic boundary condition transition zone
	int64_t z_transition_size = (nprocz*nz - (Nz - zStart))/2;
	if (z_transition_size < 0) z_transition_size=0;

	char LocalRankFilename[40];
	char *loc_id;
	loc_id = new char [(nx+2)*(ny+2)*(nz+2)];

	std::vector<int64_t> LabelCount(ReadValues.size(),0);
	// Set up the sub-domains
	if (rank==0){
		printf("Distributing subdomains across %i processors \n",nprocs);
		printf("Process grid: %i x %i x %i \n",nprocx,nprocy,nprocz);
		printf("Subdomain size: %i x %i x %i \n",nx,ny,nz);
		printf("Size of transition region: %ld \n", z_transition_size);

		for (int kp=0; kp<nprocz; kp++){
   ////cout<<"kp="<<kp<<endl;
			for (int jp=0; jp<nprocy; jp++){
      ////cout<<"jp="<<jp<<endl;
				for (int ip=0; ip<nprocx; ip++){
        ////cout<<"ip="<<ip<<endl;
					// rank of the process that gets this subdomain
					int rnk = kp*nprocx*nprocy + jp*nprocx + ip;
					// Pack and send the subdomain for rnk
					for (k=0;k<nz+2;k++){
      ////cout<<"kpack="<<k<<endl;
						for (j=0;j<ny+2;j++){
                                       ////cout<<"jpack="<<j<<endl;
							for (i=0;i<nx+2;i++){
                                             ////cout<<"ipack="<<i<<endl;
                                             ////cout<<"xs="<<xStart<<endl;
                                             ////cout<<"ys="<<yStart<<endl;
                                             ////cout<<"zs="<<zStart<<endl;
								int64_t x = xStart + ip*nx + i-1;
								int64_t y = yStart + jp*ny + j-1;
								// int64_t z = zStart + kp*nz + k-1;
								int64_t z = zStart + kp*nz + k-1 - z_transition_size;
								if (x<xStart) 	x=xStart;
								if (!(x<Nx))	x=Nx-1;
								if (y<yStart) 	y=yStart;
								if (!(y<Ny))	y=Ny-1;
								if (z<zStart) 	z=zStart;
								if (!(z<Nz))	z=Nz-1;
  							    int64_t nlocal = k*(nx+2)*(ny+2) + j*(nx+2) + i;
								int64_t nglobal = z*Nx*Ny+y*Nx+x;
                                  ////cout<<"nloc="<<nlocal<<endl;
                                    ////cout<<"nglob="<<nglobal<<endl;
								loc_id[nlocal] = SegData[nglobal];
							}
						}
					}
					// relabel the data
					for (k=0;k<nz+2;k++){
              ////cout<<"klabel="<<k<<endl;
						for (j=0;j<ny+2;j++){
                       ////cout<<"jlabel="<<j<<endl;          
							for (i=0;i<nx+2;i++){
                                                          ////cout<<"ilabel="<<i<<endl;
								n = k*(nx+2)*(ny+2) + j*(nx+2) + i;;
								char locval = loc_id[n];
								for (int idx=0; idx<ReadValues.size(); idx++){
									char oldvalue=ReadValues[idx];
									char newvalue=WriteValues[idx];
									if (locval == oldvalue){
										loc_id[n] = newvalue;
										LabelCount[idx]++;
										idx = ReadValues.size();
									}
								}
								//if (loc_id[n]==char(SOLID))     loc_id[n] = 0;
								//else if (loc_id[n]==char(NWP))  loc_id[n] = 1;
								//else                     loc_id[n] = 2;

							}
						}
					}

					// Write the data for this rank data 
					sprintf(LocalRankFilename,"ID.%05i",rnk+rank_offset);
					FILE *ID = fopen(LocalRankFilename,"wb");
					fwrite(loc_id,1,(nx+2)*(ny+2)*(nz+2),ID);
					fclose(ID);
				}
			}
		}
	}
	for (int idx=0; idx<ReadValues.size(); idx++){
		char label=ReadValues[idx];
		int64_t count=LabelCount[idx];
		printf("Label=%d, Count=%" PRId64 "\n",label,count);
	}

}
