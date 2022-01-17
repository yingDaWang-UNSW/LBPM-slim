// Sequential blob analysis 
// Reads parallel simulation data and performs connectivity analysis
// and averaging on a blob-by-blob basis
// James E. McClure 2014

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>

#include "common/Array.h"
#include "common/Domain.h"
#include "common/Communication.h"
#include "common/MPI_Helpers.h"
//#include "IO/MeshDatabase.h"
//#include "IO/Mesh.h"
//#include "IO/Writer.h"
//#include "IO/netcdf.h"
#include "analysis/analysis.h"
#include "analysis/filters.h"
#include "analysis/uCT.h"
#include "analysis/distance.h"
//#include "analysis/Minkowski.h"

//#include "PROFILErApp.h"

int main(int argc, char **argv)
{
    // Initialize MPI
    int rank, nprocs;
    MPI_Init(&argc,&argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&nprocs);
    {
        Utilities::setErrorHandlers();
        //PROFILE_START("Main");
        //std::vector<std::string> filenames;
        if ( argc<2 ) {
            if ( rank == 0 ){
                printf("At least one filename must be specified\n");
            }
            return 1;
        }
        std::string filename = std::string(argv[1]);
        if ( rank == 0 ){
            printf("Input data file: %s\n",filename.c_str());
        }

        bool FILTER_CONNECTED_COMPONENTS = true;
        auto db = std::make_shared<Database>( filename );
        auto domain_db = db->getDatabase( "Domain" );
        auto uct_db = db->getDatabase( "uCT" );
        auto analysis_db = db->getDatabase( "Analysis" );

        // Read domain values
        auto L = domain_db->getVector<double>( "L" );
        auto size = domain_db->getVector<int>( "n" );
        auto nproc = domain_db->getVector<int>( "nproc" );
        //int BoundaryCondition = domain_db->getScalar<int>( "BC" );
        int nx = size[0];
        int ny = size[1];
        int nz = size[2];
        double Lx = L[0];
        double Ly = L[1];
        double Lz = L[2];
        int nprocx = nproc[0];
        int nprocy = nproc[1];
        int nprocz = nproc[2];

        auto nlm_sigsq    = uct_db->getScalar<float>( "nlm_sigsq" );    

        // Check that the number of processors >= the number of ranks
        if ( rank==0 ) {
            printf("Number of MPI ranks required: %i \n", nprocx*nprocy*nprocz);
            printf("Number of MPI ranks used: %i \n", nprocs);
            printf("Full domain size: %i x %i x %i  \n",nx*nprocx,ny*nprocy,nz*nprocz);
            printf("NLMF Strength = %f \n",nlm_sigsq);
        }
        if ( nprocs < nprocx*nprocy*nprocz ){
            ERROR("Insufficient number of processors");
        }

        // Determine the maximum number of levels for the desired coarsen ratio
        // YDWs dirty modification for 1 level
        int ratio[3] = {1,1,1};
        //std::vector<size_t> ratio = {4,4,4};
        // need to set up databases for each level of the mesh
        std::vector<std::shared_ptr<Database>> multidomain_db(1,domain_db);
        std::vector<int> Nx(1,nx), Ny(1,ny), Nz(1,nz);

        Nx.push_back( Nx.back()/ratio[0] );
        Ny.push_back( Ny.back()/ratio[1] );
        Nz.push_back( Nz.back()/ratio[2] );
        // clone the domain and create coarse version based on Nx,Ny,Nz
        auto db2 = domain_db->cloneDatabase();
        db2->putVector<int>( "n", { Nx.back(), Ny.back(), Nz.back() } );
        multidomain_db.push_back(db2);
        int N_levels = Nx.size();

        // Initialize the domain
        std::vector<std::shared_ptr<Domain>> Dm(N_levels);
        for (int i=0; i<N_levels; i++) {
            // This line is no good -- will create identical Domain structures instead of
            // Need a way to define a coarse structure for the coarse domain (see above)
            Dm[i].reset( new Domain(multidomain_db[i], comm) );
            int N = (Nx[i]+2)*(Ny[i]+2)*(Nz[i]+2);
            for (int n=0; n<N; n++){
                Dm[i]->id[n] = 1;
            }
            Dm[i]->CommInit();
        }

        // array containing a distance mask
        Array<float> MASK(Nx[0]+2,Ny[0]+2,Nz[0]+2);
        MASK.fill(0);

        // Create the level data
        std::vector<Array<char>>  ID(N_levels);
        std::vector<Array<float>> LOCVOL(N_levels);
        std::vector<Array<float>> Dist(N_levels);
        std::vector<Array<float>> MultiScaleSmooth(N_levels);
        std::vector<Array<float>> Mean(N_levels);
        std::vector<Array<float>> NonLocalMean(N_levels);
        std::vector<std::shared_ptr<fillHalo<double>>> fillDouble(N_levels);
        std::vector<std::shared_ptr<fillHalo<float>>>  fillFloat(N_levels);
        std::vector<std::shared_ptr<fillHalo<char>>>   fillChar(N_levels);
        for (int i=0; i<N_levels; i++) {
            ID[i] = Array<char>(Nx[i]+2,Ny[i]+2,Nz[i]+2);
            LOCVOL[i] = Array<float>(Nx[i]+2,Ny[i]+2,Nz[i]+2);
            Dist[i] = Array<float>(Nx[i]+2,Ny[i]+2,Nz[i]+2);
            MultiScaleSmooth[i] = Array<float>(Nx[i]+2,Ny[i]+2,Nz[i]+2);
            Mean[i] = Array<float>(Nx[i]+2,Ny[i]+2,Nz[i]+2);
            NonLocalMean[i] = Array<float>(Nx[i]+2,Ny[i]+2,Nz[i]+2);
            ID[i].fill(0);
            LOCVOL[i].fill(0);
            Dist[i].fill(0);
            MultiScaleSmooth[i].fill(0);
            Mean[i].fill(0);
            NonLocalMean[i].fill(0);
            fillDouble[i].reset(new fillHalo<double>(Dm[i]->Comm,Dm[i]->rank_info,{Nx[i],Ny[i],Nz[i]},{1,1,1},0,1) );
            fillFloat[i].reset(new fillHalo<float>(Dm[i]->Comm,Dm[i]->rank_info,{Nx[i],Ny[i],Nz[i]},{1,1,1},0,1) );
            fillChar[i].reset(new fillHalo<char>(Dm[i]->Comm,Dm[i]->rank_info,{Nx[i],Ny[i],Nz[i]},{1,1,1},0,1) );
        }

        Array<short> readVol(Nx[0]+2,Ny[0]+2,Nz[0]+2);
        size_t readID;
        char LocalRankString[8];
        char LocalRankFilename[40];
        if (rank == 0)    printf("Read input media... \n");
        sprintf(LocalRankString,"%05d",rank);
        sprintf(LocalRankFilename,"%s%s","ID.",LocalRankString);
        if (rank==0) printf("Initialize from decomposed data ID.%05i\n",rank);
        sprintf(LocalRankFilename,"ID.%05i",rank);
        FILE *IDFILE = fopen(LocalRankFilename,"rb");
        if (IDFILE==NULL) ERROR("Domain::ReadIDs --  Error opening file: ID.xxxxx");
        short int id;
        for (int k=0;k<Nz[0]+2;k++) {
            for (int j=0;j<Ny[0]+2;j++) {
                for (int i=0;i<Nx[0]+2;i++) {
                    fread(&id,sizeof(short int),1,IDFILE);
                    //if (i>0 || j>0 ||k>0 ||i<Nx[0]+1||j<Ny[0]+1||k<Nz[0]+1) {
                        readVol(i,j,k)=id;
                    //}
                }
            }
        }       
        fclose(IDFILE);
        LOCVOL[0].fill(0);
        fillFloat[0]->copy( readVol, LOCVOL[0] );
        fillFloat[0]->fill( LOCVOL[0] );
        
        MPI_Barrier(comm);
        //PROFILE_STOP("ReadVolume");
        printf("Read complete\nFiltering NLMF rank: %d\n",rank);

        //PROFILE_STOP("CoarsenMesh");

        // Initialize the coarse level
        //PROFILE_START("Solve coarse mesh");
        //if (rank==0) ("Filtering NLMF\n");
            
        //NLM3D( const Array<float> &Input, Array<float> &Mean, const Array<float> &Distance, Array<float> &Output, const int d, const float h)
    	int nlm_count = NLM3D(LOCVOL[0], LOCVOL[0], LOCVOL[0], NonLocalMean[0], 1e8, nlm_sigsq);
        //PROFILE_STOP("Refine distance");
        MPI_Barrier(comm);    
       

            if (rank==0) printf("Dumping entire visualization structure \n");
            char LocalRankFoldername[100];
            if (rank==0) {
                sprintf(LocalRankFoldername,"./segResult"); 
                mkdir(LocalRankFoldername, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            }
            MPI_Barrier(comm);
            
            
            for (size_t n=0; n<N_levels; n++) {
            //create the file
                FILE *OUTFILE;
                char LocalRankFilename[100];
                sprintf(LocalRankFilename,"segResult/nlmf_%d_%d_%d_%d_%d_%d_%d_%d.raw",rank,n,Nx[n]+2,Ny[n]+2,Nz[n]+2,nprocx,nprocy,nprocz); //change this file name to include the size
                OUTFILE = fopen(LocalRankFilename,"wb");

                float temp;
                for (int k=0;k<Nz[n]+2;k++) {
                    for (int j=0;j<Ny[n]+2;j++) {
                        for (int i=0;i<Nx[n]+2;i++) {
                            temp = NonLocalMean[n](i,j,k);
                            //printf("%f\n",temp);
                            fwrite(&temp,sizeof(float),1,OUTFILE);
                        }
                    }
                }
                fclose(OUTFILE);
                MPI_Barrier(comm);
            }
           
            printf("Dump complete for rank %i\n",rank);


        MPI_Barrier(comm);
        if (rank==0) printf("Filter Complete\n");
    }
    MPI_Finalize();
    return 0;
}

