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
// Basic cuda functions callable from C/C++ code
#include <cuda.h>
#include <stdio.h>

extern "C" int ScaLBL_SetDevice(int rank){
	int n_devices; 
	//int local_rank = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
	cudaGetDeviceCount(&n_devices); 
	//int device = local_rank % n_devices; 
	int device = rank % n_devices; 
	cudaSetDevice(device); 
 	printf("MPI rank=%i will use GPU ID %i / %i \n",rank,device,n_devices);
	return device;
}

extern "C" void ScaLBL_AllocateDeviceMemory(void** address, size_t size){
	cudaMalloc(address,size);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("Error in cudaMalloc: %s \n",cudaGetErrorString(err));
	}	
}

extern "C" void ScaLBL_FreeDeviceMemory(void* pointer){
       cudaFree(pointer);
}

extern "C" void ScaLBL_CopyToDevice(void* dest, const void* source, size_t size){
	cudaMemcpy(dest,source,size,cudaMemcpyHostToDevice);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
	   printf("Error in cudaMemcpy (host->device): %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_AllocateZeroCopy(void** address, size_t size){
	//cudaMallocHost(address,size);
	cudaMalloc(address,size);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("Error in cudaMallocHost: %s \n",cudaGetErrorString(err));
	}
}

// Host-pinned allocation for MPI staging buffers (used by SendD3Q7AA /
// RecvD3Q7AA in common/ScaLBL.cpp). Plain host memory + the cudaMemcpy
// staging avoids handing MPI a device pointer, which side-steps the
// mca_btl_self crash on large self-loopback halos.
extern "C" void ScaLBL_AllocateHostPinned(void** address, size_t size){
	// Use the cudaMallocHost return value directly rather than cudaGetLastError,
	// which would surface any *prior* uncaught CUDA error (eg from a kernel
	// launched earlier) and falsely blame this allocation.
	cudaError_t err = cudaMallocHost(address, size);
	if (cudaSuccess != err){
		printf("Error in cudaMallocHost (ScaLBL_AllocateHostPinned, size=%zu): %s\n", size, cudaGetErrorString(err));
		*address = nullptr;
	}
}

extern "C" void ScaLBL_FreeHostPinned(void* pointer){
	if (pointer) cudaFreeHost(pointer);
}

// Sync + check; used to localize silent kernel failures.
extern "C" int ScaLBL_SyncAndCheck(const char *where){
	cudaError_t derr = cudaDeviceSynchronize();
	cudaError_t serr = cudaGetLastError();
	cudaError_t err = (derr != cudaSuccess) ? derr : serr;
	if (err != cudaSuccess) {
		printf("ScaLBL_SyncAndCheck FAIL at %s: %s\n", where, cudaGetErrorString(err));
		return -1;
	}
	return 0;
}

extern "C" void ScaLBL_CopyToZeroCopy(void* dest, const void* source, size_t size){
        cudaMemcpy(dest,source,size,cudaMemcpyHostToDevice);
        cudaError_t err = cudaGetLastError();
        //memcpy(dest, source, size);

}

extern "C" void ScaLBL_CopyToHost(void* dest, const void* source, size_t size){
	cudaMemcpy(dest,source,size,cudaMemcpyDeviceToHost);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
	   printf("Error in cudaMemcpy (device->host): %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_DeviceBarrier(){
	cudaDeviceSynchronize();
}
