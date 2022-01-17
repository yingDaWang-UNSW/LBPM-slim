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

        auto target       = uct_db->getScalar<float>("target");
        auto background   = uct_db->getScalar<float>("background");
        auto rough_cutoff = uct_db->getScalar<float>( "rough_cutoff" );    
        auto lamda        = uct_db->getScalar<float>( "lamda" );    
        auto nlm_sigsq    = uct_db->getScalar<float>( "nlm_sigsq" );    
        auto nlm_depth    = uct_db->getScalar<int>( "nlm_depth" );    
        auto center       = uct_db->getVector<int>( "center" );
        auto CylRad       = uct_db->getScalar<float>( "cylinder_radius" );
        auto maxLevels    = uct_db->getScalar<int>( "max_levels" );
        bool debugDump       = uct_db->getScalar<bool>( "debugDump" );
        std::vector<int> offset( 3, 0 );
        if ( uct_db->keyExists( "offset" ) )
            offset = uct_db->getVector<int>( "offset" );

        if ( uct_db->keyExists( "filter_connected_components" ) )
            FILTER_CONNECTED_COMPONENTS = uct_db->getScalar<bool>( "filter_connected_components" );

        // Check that the number of processors >= the number of ranks
        if ( rank==0 ) {
            printf("Number of MPI ranks required: %i \n", nprocx*nprocy*nprocz);
            printf("Number of MPI ranks used: %i \n", nprocs);
            printf("Full domain size: %i x %i x %i  \n",nx*nprocx,ny*nprocy,nz*nprocz);
            printf("target value = %f \n",target);
            printf("background value = %f \n",background);
            printf("cylinder center = %i, %i, %i \n",center[0],center[1],center[2]);
            printf("cylinder radius = %f \n",CylRad);
        }
        if ( nprocs < nprocx*nprocy*nprocz ){
            ERROR("Insufficient number of processors");
        }

        // Determine the maximum number of levels for the desired coarsen ratio
        int ratio[3] = {2,2,2};
        //std::vector<size_t> ratio = {4,4,4};
        // need to set up databases for each level of the mesh
        std::vector<std::shared_ptr<Database>> multidomain_db(1,domain_db);
        std::vector<int> Nx(1,nx), Ny(1,ny), Nz(1,nz);
        while ( Nx.back()%ratio[0]==0 && Nx.back()>8 &&
                Ny.back()%ratio[1]==0 && Ny.back()>8 &&
                Nz.back()%ratio[2]==0 && Nz.back()>8 &&
                (int) Nx.size() < maxLevels )
        {
            Nx.push_back( Nx.back()/ratio[0] );
            Ny.push_back( Ny.back()/ratio[1] );
            Nz.push_back( Nz.back()/ratio[2] );
            // clone the domain and create coarse version based on Nx,Ny,Nz
            auto db2 = domain_db->cloneDatabase();
            db2->putVector<int>( "n", { Nx.back(), Ny.back(), Nz.back() } );
            multidomain_db.push_back(db2);
        }
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
        if (rank==0) printf("Read complete\n");

        // Filter the original data
        filter_src( *Dm[0], LOCVOL[0] );

        // Set up the mask to be distance to cylinder (crop outside cylinder)
        if (rank==0) printf("Cropping with cylinder: %i, %i, %i, radius=%f \n",Dm[0]->nprocx()*Nx[0],Dm[0]->nprocy()*Ny[0],Dm[0]->nprocz()*Nz[0],CylRad);
        for (int k=0;k<Nz[0]+2;k++) {
            for (int j=0;j<Ny[0]+2;j++) {
                for (int i=0;i<Nx[0]+2;i++) {
                  float x= float(Dm[0]->iproc()*Nx[0]+i-1);
                  float y= float (Dm[0]->jproc()*Ny[0]+j-1);
                  float z= float(Dm[0]->kproc()*Nz[0]+k-1);
                  float cx = float(center[0] - offset[0]);
                  float cy = float(center[1] - offset[1]);
                  float cz = float(center[2] - offset[2]);
                    // distance from the center line 
                    MASK(i,j,k) = sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
                    //MASK(i,j,k) = sqrt((z-cz)*(z-cz) + (y-cy)*(y-cy));
                    //if (sqrt(((z-cz)*(z-cz) + (y-cy)*(y-cy)) ) > CylRad) LOCVOL[0](i,j,k)=background;
                }
            }
        }

        // Compute the means for the high/low regions
        // (should use automated mixture model to approximate histograms)
        //float THRESHOLD = 0.05*maxReduce( Dm[0]->Comm, std::max( LOCVOL[0].max(), fabs(LOCVOL[0].min()) ) );
        float THRESHOLD=0.5*(target+background);
        float mean_plus=0;
        float mean_minus=0;
        float min_value = LOCVOL[0](0);
        float max_value = LOCVOL[0](0);
        int count_plus=0;
        int count_minus=0;
        for (int k=1;k<Nz[0]+1;k++) {
            for (int j=1;j<Ny[0]+1;j++) {
                for (int i=1;i<Nx[0]+1;i++) {
                  //LOCVOL[0](i,j,k) = MASK(i,j,k);
                 if (MASK(i,j,k) < CylRad ){
                      float tmp = LOCVOL[0](i,j,k);
                        /*                        if ((tmp-background)*(tmp-target) > 0){
                            // direction to background / target is the same
                            if (fabs(tmp-target) > fabs(tmp-background)) tmp=background; // tmp closer to background
                            else                                          tmp=target;     // tmp closer to target
                        }
                         */
                        if ( tmp > THRESHOLD ) {
                            mean_plus += tmp;
                            count_plus++;
                        } 
                        else {
                            mean_minus += tmp;
                            count_minus++;
                        }
                        if (tmp < min_value) min_value = tmp;
                        if (tmp > max_value) max_value = tmp;
                    }
                }
            }
        }
        count_plus=sumReduce( Dm[0]->Comm, count_plus);
        count_minus=sumReduce( Dm[0]->Comm, count_minus);
        if (rank==0) printf("minimum value=%f, max value=%f \n",min_value,max_value);
        if (rank==0) printf("plus=%i, minus=%i \n",count_plus,count_minus);
        ASSERT( count_plus > 0 && count_minus > 0 );
        MPI_Barrier(comm);
        mean_plus = sumReduce( Dm[0]->Comm, mean_plus ) / count_plus;
        mean_minus = sumReduce( Dm[0]->Comm, mean_minus ) / count_minus;
        MPI_Barrier(comm);
        if (rank==0) printf("    Region 1 mean (+): %f, Region 2 mean (-): %f \n",mean_plus, mean_minus);

        //if (rank==0) printf("Scale the input data (size = %i) \n",LOCVOL[0].length());
        for (size_t i=0; i<LOCVOL[0].length(); i++) {
            if ( MASK(i) > CylRad ){
              LOCVOL[0](i)=background;
            }
            if ( LOCVOL[0](i) >= THRESHOLD ) {
                auto tmp = LOCVOL[0](i)/ mean_plus;
                LOCVOL[0](i) = std::min( tmp, 1.0f );
            } 
            else {
                auto tmp = -LOCVOL[0](i)/mean_minus;
                LOCVOL[0](i) = std::max( tmp, -1.0f );
            }
            //LOCVOL[0](i) = MASK(i);
        }

        // Fill the source data for the coarse meshes
        if (rank==0) printf("Coarsen the grid for N_levels=%i \n",N_levels);
        MPI_Barrier(comm); 
        //PROFILE_START("CoarsenMesh");
        for (int i=1; i<N_levels; i++) {
            Array<float> filter(ratio[0],ratio[1],ratio[2]);
            filter.fill(1.0f/filter.length());
            Array<float> tmp(Nx[i-1],Ny[i-1],Nz[i-1]);
            fillFloat[i-1]->copy( LOCVOL[i-1], tmp );
            Array<float> coarse = tmp.coarsen( filter );
            fillFloat[i]->copy( coarse, LOCVOL[i] );
            fillFloat[i]->fill( LOCVOL[i] );
            if (rank==0){
                printf("Coarsen level %i \n",i);
                printf("   Nx=%i, Ny=%i, Nz=%i \n",int(tmp.size(0)),int(tmp.size(1)),int(tmp.size(2))  );
                printf("   filter_x=%i, filter_y=%i, filter_z=%i \n",int(filter.size(0)),int(filter.size(1)),int(filter.size(2))  );
            }
            MPI_Barrier(comm);
        }
        //PROFILE_STOP("CoarsenMesh");

        // Initialize the coarse level
        //PROFILE_START("Solve coarse mesh");
        if (rank==0) ("Processing Coarsest Grid\n");
            
        solve( LOCVOL.back(), Mean.back(), ID.back(), Dist.back(), MultiScaleSmooth.back(),
                NonLocalMean.back(), *fillFloat.back(), *Dm.back(), nprocx, 
                rough_cutoff, lamda, nlm_sigsq, nlm_depth);
        //PROFILE_STOP("Solve coarse mesh");
        MPI_Barrier(comm);

        // Refine the solution
        //PROFILE_START("Refine distance");
        if (rank==0)
            printf("Refining Grid\n");
        for (int i=N_levels-2; i>=0; i--) {
            if (rank==0)
                printf("   Refining to level %i\n",i);
            refine( Dist[i+1], LOCVOL[i], Mean[i], ID[i], Dist[i], MultiScaleSmooth[i],
                    NonLocalMean[i], *fillFloat[i], *Dm[i], nprocx, i, 
                rough_cutoff, lamda, nlm_sigsq, nlm_depth);
        }
        //PROFILE_STOP("Refine distance");
        MPI_Barrier(comm);    

        // Perform a final filter
        //PROFILE_START("Filtering final domains");
        if (FILTER_CONNECTED_COMPONENTS){
            if (rank==0)
                printf("Filtering final domains\n");
            Array<float> filter_Mean, filter_Dist1, filter_Dist2;
            filter_final( ID[0], Dist[0], *fillFloat[0], *Dm[0], filter_Mean, filter_Dist1, filter_Dist2 );
            //PROFILE_STOP("Filtering final domains");
        }
        
        
        
        
        // SLIM: dump results as raw, reconstruct outside lbpm
        if (rank==0) printf("Saving segmented subdomains \n");
        char LocalRankFoldername[100];
        if (rank==0) {
            sprintf(LocalRankFoldername,"./segmented"); 
            mkdir(LocalRankFoldername, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        }
        MPI_Barrier(comm);
        
        int n=0;
        //create the file
        FILE *OUTFILE;
        char LocalRankFilenameSeg[100];
        sprintf(LocalRankFilenameSeg,"segmented/seg_%d_%d_%d_%d_%d_%d_%d_%d.raw",rank,n,Nx[n]+2,Ny[n]+2,Nz[n]+2,nprocx,nprocy,nprocz); //change this file name to include the size
        OUTFILE = fopen(LocalRankFilenameSeg,"wb");

        char temp;
        for (int k=0;k<Nz[n]+2;k++) {
            for (int j=0;j<Ny[n]+2;j++) {
                for (int i=0;i<Nx[n]+2;i++) {
                    temp = ID[n](i,j,k);
                    //printf("%f\n",temp);
                    fwrite(&temp,sizeof(char),1,OUTFILE);
                }
            }
        }
        fclose(OUTFILE);
        MPI_Barrier(comm);
        
        if (debugDump) {
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
                sprintf(LocalRankFilename,"segResult/seg_%d_%d_%d_%d_%d_%d_%d_%d.raw",rank,n,Nx[n]+2,Ny[n]+2,Nz[n]+2,nprocx,nprocy,nprocz); //change this file name to include the size
                OUTFILE = fopen(LocalRankFilename,"wb");

                char temp;
                for (int k=0;k<Nz[n]+2;k++) {
                    for (int j=0;j<Ny[n]+2;j++) {
                        for (int i=0;i<Nx[n]+2;i++) {
                            temp = ID[n](i,j,k);
                            //printf("%f\n",temp);
                            fwrite(&temp,sizeof(char),1,OUTFILE);
                        }
                    }
                }
                fclose(OUTFILE);
                MPI_Barrier(comm);
            }
            
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
            
            for (size_t n=0; n<N_levels; n++) {
            //create the file
                FILE *OUTFILE;
                char LocalRankFilename[100];
                sprintf(LocalRankFilename,"segResult/dist_%d_%d_%d_%d_%d_%d_%d_%d.raw",rank,n,Nx[n]+2,Ny[n]+2,Nz[n]+2,nprocx,nprocy,nprocz); //change this file name to include the size
                OUTFILE = fopen(LocalRankFilename,"wb");

                float temp;
                for (int k=0;k<Nz[n]+2;k++) {
                    for (int j=0;j<Ny[n]+2;j++) {
                        for (int i=0;i<Nx[n]+2;i++) {
                            temp = Dist[n](i,j,k);
                            //printf("%f\n",temp);
                            fwrite(&temp,sizeof(float),1,OUTFILE);
                        }
                    }
                }
                fclose(OUTFILE);
                MPI_Barrier(comm);
            }
            
            for (size_t n=0; n<N_levels; n++) {
            //create the file
                FILE *OUTFILE;
                char LocalRankFilename[100];
                sprintf(LocalRankFilename,"segResult/smooth_%d_%d_%d_%d_%d_%d_%d_%d.raw",rank,n,Nx[n]+2,Ny[n]+2,Nz[n]+2,nprocx,nprocy,nprocz); //change this file name to include the size
                OUTFILE = fopen(LocalRankFilename,"wb");

                float temp;
                for (int k=0;k<Nz[n]+2;k++) {
                    for (int j=0;j<Ny[n]+2;j++) {
                        for (int i=0;i<Nx[n]+2;i++) {
                            temp = MultiScaleSmooth[n](i,j,k);
                            //printf("%f\n",temp);
                            fwrite(&temp,sizeof(float),1,OUTFILE);
                        }
                    }
                }
                fclose(OUTFILE);
                MPI_Barrier(comm);
            }
            for (size_t n=0; n<N_levels; n++) {
            //create the file
                FILE *OUTFILE;
                char LocalRankFilename[100];
                sprintf(LocalRankFilename,"segResult/locvol_%d_%d_%d_%d_%d_%d_%d_%d.raw",rank,n,Nx[n]+2,Ny[n]+2,Nz[n]+2,nprocx,nprocy,nprocz); //change this file name to include the size
                OUTFILE = fopen(LocalRankFilename,"wb");

                float temp;
                for (int k=0;k<Nz[n]+2;k++) {
                    for (int j=0;j<Ny[n]+2;j++) {
                        for (int i=0;i<Nx[n]+2;i++) {
                            temp = LOCVOL[n](i,j,k);
                            //printf("%f\n",temp);
                            fwrite(&temp,sizeof(float),1,OUTFILE);
                        }
                    }
                }
                fclose(OUTFILE);
                MPI_Barrier(comm);
            }
            printf("Dump complete for rank %i\n",rank);
        }
        
        
                //create the folder
        if (rank==0) {
            printf("Stitching segmented domain \n");
            // create seg domain
            Array<char> segFull(Nx[0]*nprocx,Ny[0]*nprocy,Nz[0]*nprocz);
            segFull.fill(0);
            // read the files in a loop
            char LocalRankFoldername[100];
            sprintf(LocalRankFoldername,"./segmented"); 
            for (int r=0;r<nprocs;r++) {
                int n=0;
                //read the file
                FILE *OUTFILE;
                char LocalRankFilenameSeg2[100];
                sprintf(LocalRankFilenameSeg2,"segmented/seg_%d_%d_%d_%d_%d_%d_%d_%d.raw",r,n,Nx[n]+2,Ny[n]+2,Nz[n]+2,nprocx,nprocy,nprocz); //change this file name to include the size
                OUTFILE = fopen(LocalRankFilenameSeg2,"r");
                char temp;
                Array<char> segTemp(Nx[0]+2,Ny[0]+2,Nz[0]+2);
                segTemp.fill(0);
                for (int k=0;k<Nz[n]+2;k++) {
                    for (int j=0;j<Ny[n]+2;j++) {
                        for (int i=0;i<Nx[n]+2;i++) {
                            //printf("%f\n",temp);
                            fread(&temp,sizeof(char),1,OUTFILE);
                            segTemp(i,j,k) = temp;
                        }
                    }
                }
                fclose(OUTFILE);
                
                // copy into sfull seg
            	int ix = r%nprocx;
	            int jy = (r/nprocx)%nprocy;
	            int kz = r/(nprocx*nprocy);
                int originX = ix*Nx[n];
                int originY = jy*Ny[n];
                int originZ = kz*Nz[n];
                for (int k=0;k<Nz[n];k++) {
                    for (int j=0;j<Ny[n];j++) {
                        for (int i=0;i<Nx[n];i++) {
                            segFull(originX+i,originY+j,originZ+k)=segTemp(i+1,j+1,k+1);
                        }
                    }
                }
            }
            //save full seg
            FILE *OUTFILE;
            char GlobalRankFilenameSeg[100];
            sprintf(GlobalRankFilenameSeg,"segmented/fullseg_%d_%d_%d.raw",Nx[n]*nprocx,Ny[n]*nprocy,Nz[n]*nprocz); //change this file name to include the size
            OUTFILE = fopen(GlobalRankFilenameSeg,"wb");
            char temp;
            for (int k=0;k<Nz[n]*nprocz;k++) {
                for (int j=0;j<Ny[n]*nprocy;j++) {
                    for (int i=0;i<Nx[n]*nprocx;i++) {
                            temp = segFull(i,j,k);
                            //printf("%f\n",temp);
                            fwrite(&temp,sizeof(char),1,OUTFILE);
                    }
                }
            }
            fclose(OUTFILE);
            printf("Segmentation Complete\n");
            return 0;
        }
        MPI_Barrier(comm);
        if (rank==0) printf("Segmentation Complete\n");
    }
    MPI_Finalize();
    return 0;
}

