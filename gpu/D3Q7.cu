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
// GPU Functions for D3Q7 Lattice Boltzmann Methods

#define NBLOCKS 560
#define NTHREADS 128

__global__  void dvc_ScaLBL_Scalar_Pack(int *list, int count, double *sendbuf, double *Data, int N){
	//....................................................................................
	// Pack distribution q into the send buffer for the listed lattice sites
	// dist may be even or odd distributions stored by stream layout
	//....................................................................................
	int idx,n;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<count){
		n = list[idx];
		sendbuf[idx] = Data[n];
	}
}
__global__  void dvc_ScaLBL_Scalar_Unpack(int *list, int count, double *recvbuf, double *Data, int N){
	//....................................................................................
	// Pack distribution q into the send buffer for the listed lattice sites
	// dist may be even or odd distributions stored by stream layout
	//....................................................................................
	int idx,n;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<count){
		n = list[idx];
		Data[n] = recvbuf[idx];
	}
}

__global__  void dvc_ScaLBL_PackDenD3Q7(int *list, int count, double *sendbuf, int number, double *Data, int N){
	//....................................................................................
	// Pack distribution into the send buffer for the listed lattice sites
	//....................................................................................
	int idx,n,component;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<count){
		for (component=0; component<number; component++){
			n = list[idx];
			sendbuf[idx*number+component] = Data[number*n+component];
			Data[number*n+component] = 0.0;	// Set the data value to zero once it's in the buffer!
		}
	}
}


__global__ void dvc_ScaLBL_UnpackDenD3Q7(int *list, int count, double *recvbuf, int number, double *Data, int N){
	//....................................................................................
	// Unack distribution from the recv buffer
	// Sum to the existing density value
	//....................................................................................
	int idx,n,component;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<count){
			for (component=0; component<number; component++){
			n = list[idx];
			Data[number*n+component] += recvbuf[idx*number+component];
		}
	}
}

__global__ void dvc_ScaLBL_D3Q7_Unpack(int q,  int *list,  int start, int count,
		double *recvbuf, double *dist, int N){
	//....................................................................................
	// Unpack distribution from the recv buffer
	// Distribution q matche Cqx, Cqy, Cqz
	// swap rule means that the distributions in recvbuf are OPPOSITE of q
	// dist may be even or odd distributions stored by stream layout
	//....................................................................................
	int n,idx;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<count){
		// Get the value from the list -- note that n is the index is from the send (non-local) process
		n = list[idx];
		// unpack the distribution to the proper location
		if (!(n<0)) { dist[q*N+n] = recvbuf[start+idx];
		//printf("%f \n",,dist[q*N+n]);
		}
	}
}


extern "C" void ScaLBL_D3Q7_Unpack(int q, int *list,  int start, int count, double *recvbuf, double *dist, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q7_Unpack <<<GRID,512 >>>(q, list, start, count, recvbuf, dist, N);
}

extern "C" void ScaLBL_Scalar_Pack(int *list, int count, double *sendbuf, double *Data, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_Scalar_Pack <<<GRID,512 >>>(list, count, sendbuf, Data, N);
}

extern "C" void ScaLBL_Scalar_Unpack(int *list, int count, double *recvbuf, double *Data, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_Scalar_Unpack <<<GRID,512 >>>(list, count, recvbuf, Data, N);
}
extern "C" void ScaLBL_PackDenD3Q7(int *list, int count, double *sendbuf, int number, double *Data, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_PackDenD3Q7 <<<GRID,512 >>>(list, count, sendbuf, number, Data, N);
}

extern "C" void ScaLBL_UnpackDenD3Q7(int *list, int count, double *recvbuf, int number, double *Data, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_UnpackDenD3Q7 <<<GRID,512 >>>(list, count, recvbuf, number, Data, N);
}

