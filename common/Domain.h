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
#ifndef Domain_INC
#define Domain_INC

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
#include <exception>
#include <stdexcept>

#include "common/Array.h"
#include "common/Utilities.h"
#include "common/MPI_Helpers.h"
#include "common/Communication.h"
#include "common/Database.h"


class Domain;

enum class DataLocation { CPU, DEVICE };


//! Class to hold domain info
class Domain{
public:
    //! Default constructor
    Domain( std::shared_ptr<Database> db, MPI_Comm Communicator);

   //! Empty constructor
    Domain() = delete;

    //! Copy constructor
    Domain( const Domain& ) = delete;

    //! Assignment operator
    Domain& operator=( const Domain& ) = delete;

    //! Destructor
    ~Domain();
    
    //! Get the database
    inline std::shared_ptr<const Database> getDatabase() const { return d_db; }


private:

    void initialize( std::shared_ptr<Database> db );

    std::shared_ptr<Database> d_db;


public: // Public variables (need to create accessors instead)

    double Lx,Ly,Lz,Volume;
    int Nx,Ny,Nz,N;
    RankInfoStruct rank_info; // call from communication cpp

    MPI_Comm Comm;        // MPI Communicator for this domain

    int BoundaryCondition;

    MPI_Group Group;    // Group of processors associated with this domain

    //**********************************
    // MPI ranks for all 18 neighbors
    //**********************************
    inline int iproc() const { return rank_info.ix; }
    inline int jproc() const { return rank_info.jy; }
    inline int kproc() const { return rank_info.kz; }
    inline int nprocx() const { return rank_info.nx; }
    inline int nprocy() const { return rank_info.ny; }
    inline int nprocz() const { return rank_info.nz; }
    inline int rank() const { return rank_info.rank[1][1][1]; }
    inline int rank_X() const { return rank_info.rank[2][1][1]; }
    inline int rank_x() const { return rank_info.rank[0][1][1]; }
    inline int rank_Y() const { return rank_info.rank[1][2][1]; }
    inline int rank_y() const { return rank_info.rank[1][0][1]; }
    inline int rank_Z() const { return rank_info.rank[1][1][2]; }
    inline int rank_z() const { return rank_info.rank[1][1][0]; }
    inline int rank_XY() const { return rank_info.rank[2][2][1]; }
    inline int rank_xy() const { return rank_info.rank[0][0][1]; }
    inline int rank_Xy() const { return rank_info.rank[2][0][1]; }
    inline int rank_xY() const { return rank_info.rank[0][2][1]; }
    inline int rank_XZ() const { return rank_info.rank[2][1][2]; }
    inline int rank_xz() const { return rank_info.rank[0][1][0]; }
    inline int rank_Xz() const { return rank_info.rank[2][1][0]; }
    inline int rank_xZ() const { return rank_info.rank[0][1][2]; }
    inline int rank_YZ() const { return rank_info.rank[1][2][2]; }
    inline int rank_yz() const { return rank_info.rank[1][0][0]; }
    inline int rank_Yz() const { return rank_info.rank[1][2][0]; }
    inline int rank_yZ() const { return rank_info.rank[1][0][2]; }

    //**********************************
    //......................................................................................
    // Get the actual D3Q19 communication counts (based on location of solid phase)
    // Discrete velocity set symmetry implies the sendcount = recvcount
    //......................................................................................
    int sendCount_x, sendCount_y, sendCount_z, sendCount_X, sendCount_Y, sendCount_Z;
    int sendCount_xy, sendCount_yz, sendCount_xz, sendCount_Xy, sendCount_Yz, sendCount_xZ;
    int sendCount_xY, sendCount_yZ, sendCount_Xz, sendCount_XY, sendCount_YZ, sendCount_XZ;
    //......................................................................................
    int *sendList_x, *sendList_y, *sendList_z, *sendList_X, *sendList_Y, *sendList_Z;
    int *sendList_xy, *sendList_yz, *sendList_xz, *sendList_Xy, *sendList_Yz, *sendList_xZ;
    int *sendList_xY, *sendList_yZ, *sendList_Xz, *sendList_XY, *sendList_YZ, *sendList_XZ;
    //......................................................................................
    int recvCount_x, recvCount_y, recvCount_z, recvCount_X, recvCount_Y, recvCount_Z;
    int recvCount_xy, recvCount_yz, recvCount_xz, recvCount_Xy, recvCount_Yz, recvCount_xZ;
    int recvCount_xY, recvCount_yZ, recvCount_Xz, recvCount_XY, recvCount_YZ, recvCount_XZ;
    //......................................................................................
    int *recvList_x, *recvList_y, *recvList_z, *recvList_X, *recvList_Y, *recvList_Z;
    int *recvList_xy, *recvList_yz, *recvList_xz, *recvList_Xy, *recvList_Yz, *recvList_xZ;
    int *recvList_xY, *recvList_yZ, *recvList_Xz, *recvList_XY, *recvList_YZ, *recvList_XZ;
    //......................................................................................    
    // Solid indicator function
    char *id;

    void ReadIDs();
    //void CommunicateMeshHalo(DoubleArray &Mesh);
    void CommInit(); 
    int PoreCount();
    double Porosity();
private:
    // packing and unpacking is defined on a case-by-case basis
/*    void PackID(int *list, int count, char *sendbuf, char *ID);*/
/*    void UnpackID(int *list, int count, char *recvbuf, char *ID);*/
/*    void CommHaloIDs();*/
    
	//......................................................................................
	MPI_Request req1[18], req2[18];
	MPI_Status stat1[18],stat2[18];

    int *sendBuf_x, *sendBuf_y, *sendBuf_z, *sendBuf_X, *sendBuf_Y, *sendBuf_Z;
    int *sendBuf_xy, *sendBuf_yz, *sendBuf_xz, *sendBuf_Xy, *sendBuf_Yz, *sendBuf_xZ;
    int *sendBuf_xY, *sendBuf_yZ, *sendBuf_Xz, *sendBuf_XY, *sendBuf_YZ, *sendBuf_XZ;
    //......................................................................................
    int *recvBuf_x, *recvBuf_y, *recvBuf_z, *recvBuf_X, *recvBuf_Y, *recvBuf_Z;
    int *recvBuf_xy, *recvBuf_yz, *recvBuf_xz, *recvBuf_Xy, *recvBuf_Yz, *recvBuf_xZ;
    int *recvBuf_xY, *recvBuf_yZ, *recvBuf_Xz, *recvBuf_XY, *recvBuf_YZ, *recvBuf_XZ;
    //......................................................................................
    double *sendData_x, *sendData_y, *sendData_z, *sendData_X, *sendData_Y, *sendData_Z;
    double *sendData_xy, *sendData_yz, *sendData_xz, *sendData_Xy, *sendData_Yz, *sendData_xZ;
    double *sendData_xY, *sendData_yZ, *sendData_Xz, *sendData_XY, *sendData_YZ, *sendData_XZ;
    double *recvData_x, *recvData_y, *recvData_z, *recvData_X, *recvData_Y, *recvData_Z;
    double *recvData_xy, *recvData_yz, *recvData_xz, *recvData_Xy, *recvData_Yz, *recvData_xZ;
    double *recvData_xY, *recvData_yZ, *recvData_Xz, *recvData_XY, *recvData_YZ, *recvData_XZ;
};

#endif
