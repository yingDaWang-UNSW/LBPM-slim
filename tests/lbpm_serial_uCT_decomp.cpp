/*
 * Pre-processor to decompose 16 bit raw domains
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "common/Array.h"
#include "common/Domain.h"

int main(int argc, char **argv)
{
    int rank=0;
    string filename;
    if (argc > 1)
        filename=argv[1];
    else{
        ERROR("lbpm_serial_decomp: no in put database provided \n");
    }

    //.......................................................................
    // Reading the domain information file
    //.......................................................................
    int nprocs, nprocx, nprocy, nprocz, nx, ny, nz, nspheres;
    double Lx, Ly, Lz;
    int64_t Nx,Ny,Nz;
    int64_t i,j,k,n;
    int BC=0;

    // read the input database 
    auto db = std::make_shared<Database>( filename );
    auto domain_db = db->getDatabase( "Domain" );
    // Read domain parameters
    auto Filename = domain_db->getScalar<std::string>( "Filename" );
    auto L = domain_db->getVector<double>( "L" );
    auto size = domain_db->getVector<int>( "n" );
    auto SIZE = domain_db->getVector<int>( "N" );
    auto nproc = domain_db->getVector<int>( "nproc" );

    nx = size[0];
    ny = size[1];
    nz = size[2];
    nprocx = nproc[0];
    nprocy = nproc[1];
    nprocz = nproc[2];
    Nx = SIZE[0];
    Ny = SIZE[1];
    Nz = SIZE[2];

    nprocs=nprocx*nprocy*nprocz;
    auto uct_db = db->getDatabase( "uCT" );
    auto nlm_depth    = uct_db->getScalar<int>( "nlm_depth" );    
    short int *SegData = NULL;
    // Rank=0 reads the entire data and distributes to worker processes
    if (rank==0){
        printf("Dimensions of image: %ld x %ld x %ld \n",Nx,Ny,Nz);
        int64_t SIZE = Nx*Ny*Nz;
        SegData = new short int[SIZE];
        printf("Reading 16-bit input data \n");
        FILE *SEGDAT = fopen(Filename.c_str(),"rb");
        if (SEGDAT==NULL) ERROR("Error reading  data");
        size_t ReadSeg;
        ReadSeg=fread(SegData,sizeof(short),SIZE,SEGDAT);
        //if (ReadSeg != size_t(SIZE)) printf("Error reading data (rank=%i)\n",rank);
        fclose(SEGDAT);
        printf("Read segmented data from %s \n",Filename.c_str());
    }

    char LocalRankFilename[40];
    short int *loc_id;
    loc_id = new short int [(nx+nlm_depth*2)*(ny+nlm_depth*2)*(nz+nlm_depth*2)];
    // Set up the sub-domains
    if (rank==0){
        printf("Distributing subdomains across %i processors \n",nprocs);
        printf("Process grid: %i x %i x %i \n",nprocx,nprocy,nprocz);
        printf("Subdomain size: %i x %i x %i \n",nx,ny,nz);
        for (int kp=0; kp<nprocz; kp++){
            for (int jp=0; jp<nprocy; jp++){
                for (int ip=0; ip<nprocx; ip++){
                    // rank of the process that gets this subdomain
                    int rnk = kp*nprocx*nprocy + jp*nprocx + ip;
                    //printf("Distributing rank %i\n",rnk);
                    // Pack and send the subdomain for rnk
                    for (k=0;k<nz+nlm_depth*2;k++){
                        for (j=0;j<ny+nlm_depth*2;j++){
                            for (i=0;i<nx+nlm_depth*2;i++){
                            
                                int64_t x = ip*nx + i-nlm_depth;
                                int64_t y = jp*ny + j-nlm_depth;
                                int64_t z = kp*nz + k-nlm_depth;
                                
                                if (x<0)     x=0;
                                if (!(x<Nx))    x=Nx-1;
                                if (y<0)     y=0;
                                if (!(y<Ny))    y=Ny-1;
                                if (z<0)     z=0;
                                if (!(z<Nz))    z=Nz-1;
                                
                                int64_t nlocal = k*(nx+nlm_depth*2)*(ny+nlm_depth*2) + j*(nx+nlm_depth*2) + i;
                                int64_t nglobal = z*Nx*Ny+y*Nx+x;
                                loc_id[nlocal] = SegData[nglobal];
                            }
                        }
                    }
                    // Write the data for this rank data 
                    sprintf(LocalRankFilename,"ID.%05i",rnk);
                    FILE *ID = fopen(LocalRankFilename,"wb");
                    fwrite(loc_id,sizeof(short int),(nx+nlm_depth*2)*(ny+nlm_depth*2)*(nz+nlm_depth*2),ID);
                    fclose(ID);
                }
            }
        }
    }
}
