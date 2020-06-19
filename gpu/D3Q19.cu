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
#include <stdio.h>
#include <cooperative_groups.h>
#include <cuda.h>
#define NBLOCKS 1024
#define NTHREADS 256

/*
1. constants that are known at compile time should be defined using preprocessor macros (e.g. #define) or via C/C++ const variables at global/file scope.
2. Usage of __constant__ memory may be beneficial for programs who use certain values that don't change for the duration of the kernel and for which certain access patterns are present (e.g. all threads access the same value at the same time). This is not better or faster than constants that satisfy the requirements of item 1 above.
3. If the number of choices to be made by a program are relatively small in number, and these choices affect kernel execution, one possible approach for additional compile-time optimization would be to use templated code/kernels
 */

__constant__ __device__ double mrt_V1=0.05263157894736842;
__constant__ __device__ double mrt_V2=0.012531328320802;
__constant__ __device__ double mrt_V3=0.04761904761904762;
__constant__ __device__ double mrt_V4=0.004594820384294068;
__constant__ __device__ double mrt_V5=0.01587301587301587;
__constant__ __device__ double mrt_V6=0.0555555555555555555555555;
__constant__ __device__ double mrt_V7=0.02777777777777778;
__constant__ __device__ double mrt_V8=0.08333333333333333;
__constant__ __device__ double mrt_V9=0.003341687552213868;
__constant__ __device__ double mrt_V10=0.003968253968253968;
__constant__ __device__ double mrt_V11=0.01388888888888889;
__constant__ __device__ double mrt_V12=0.04166666666666666;


// functionality for parallel reduction in Flux BC routines -- probably should be re-factored to another location
// functions copied from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/

//__shared__ double Transform[722]=
//	   {};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) { 
   unsigned long long int* address_as_ull = (unsigned long long int*)address;
   unsigned long long int old = *address_as_ull, assumed;

   do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
   } while (assumed != old);
   return __longlong_as_double(old);
}
#endif

using namespace cooperative_groups;
__device__ double reduce_sum(thread_group g, double *temp, double val)
{
    int lane = g.thread_rank();

    // Each iteration halves the number of active threads
    // Each thread adds its partial sum[i] to sum[lane+i]
    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); // wait for all threads to store
        if(lane<i) val += temp[lane + i];
        g.sync(); // wait for all threads to load
    }
    return val; // note: only thread 0 will return full sum
}

__device__ double thread_sum(double *input, double n) 
{
    double sum = 0;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n / 4; 
        i += blockDim.x * gridDim.x)
    {
        int4 in = ((int4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

__global__ void sum_kernel_block(double *sum, double *input, int n)
{
	double my_sum = thread_sum(input, n);

    extern __shared__ double temp[];
    thread_group g = this_thread_block();
    double block_sum = reduce_sum(g, temp, my_sum);

    if (g.thread_rank() == 0) atomicAdd(sum, block_sum);
}

__inline__ __device__
double warpReduceSum(double val) {
	for (int offset = warpSize/2; offset > 0; offset /= 2)
		val += __shfl_down_sync(0xFFFFFFFF, val, offset, 32);
	return val;
}

__inline__ __device__
double blockReduceSum(double val) {

	static __shared__ double shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

	return val;
}

__global__ void deviceReduceKernel(double *in, double* out, int N) {
	double sum = 0;
	//reduce multiple elements per thread
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
			i < N;
			i += blockDim.x * gridDim.x) {
		sum += in[i];
	}
	sum = blockReduceSum(sum);
	if (threadIdx.x==0)
		out[blockIdx.x]=sum;
}

__global__ void dvc_ScaLBL_D3Q19_Pack(int q, int *list, int start, int count, double *sendbuf, double *dist, int N){
	//....................................................................................
	// Pack distribution q into the send buffer for the listed lattice sites
	// dist may be even or odd distributions stored by stream layout
	//....................................................................................
	int idx,n;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx<count){
		n = list[idx];
		sendbuf[start+idx] = dist[q*N+n];
		//printf("%f \n",dist[q*N+n]);
	}

}

__global__ void dvc_ScaLBL_D3Q19_Unpack(int q,  int *list,  int start, int count,
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
		n = list[start+idx];
		// unpack the distribution to the proper location
		if (!(n<0)) { dist[q*N+n] = recvbuf[start+idx];
		//printf("%f \n",,dist[q*N+n]);
		}
	}
}

__global__ void dvc_ScaLBL_D3Q19_Init(double *dist, int Np)
{
	int n;
	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n = S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x;
		if (n<Np ){
			dist[n] = 0.3333333333333333;
			dist[Np+n] = 0.055555555555555555;		//double(100*n)+1.f;
			dist[2*Np+n] = 0.055555555555555555;	//double(100*n)+2.f;
			dist[3*Np+n] = 0.055555555555555555;	//double(100*n)+3.f;
			dist[4*Np+n] = 0.055555555555555555;	//double(100*n)+4.f;
			dist[5*Np+n] = 0.055555555555555555;	//double(100*n)+5.f;
			dist[6*Np+n] = 0.055555555555555555;	//double(100*n)+6.f;
			dist[7*Np+n] = 0.0277777777777778;   //double(100*n)+7.f;
			dist[8*Np+n] = 0.0277777777777778;   //double(100*n)+8.f;
			dist[9*Np+n] = 0.0277777777777778;   //double(100*n)+9.f;
			dist[10*Np+n] = 0.0277777777777778;  //double(100*n)+10.f;
			dist[11*Np+n] = 0.0277777777777778;  //double(100*n)+11.f;
			dist[12*Np+n] = 0.0277777777777778;  //double(100*n)+12.f;
			dist[13*Np+n] = 0.0277777777777778;  //double(100*n)+13.f;
			dist[14*Np+n] = 0.0277777777777778;  //double(100*n)+14.f;
			dist[15*Np+n] = 0.0277777777777778;  //double(100*n)+15.f;
			dist[16*Np+n] = 0.0277777777777778;  //double(100*n)+16.f;
			dist[17*Np+n] = 0.0277777777777778;  //double(100*n)+17.f;
			dist[18*Np+n] = 0.0277777777777778;  //double(100*n)+18.f;
		}
	}
}


__global__ void 
dvc_ScaLBL_AAodd_MRT(int *neighborList, double *dist, int start, int finish, int Np, double rlx_setA, double rlx_setB, double Fx, double Fy, double Fz) {

	int n;
	double fq;
	// conserved momemnts
	double rho,jx,jy,jz;
	// non-conserved moments
	double m1,m2,m4,m6,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18;

	int nread;
	int S = Np/NBLOCKS/NTHREADS+1;

	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
			// q=0
			fq = dist[n];
			rho = fq;
			m1  = -30.0*fq;
			m2  = 12.0*fq;

			// q=1
			nread = neighborList[n]; // neighbor 2 ( > 10Np => odd part of dist)
			fq = dist[nread]; // reading the f1 data into register fq
			//fp = dist[10*Np+n];
			rho += fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jx = fq;
			m4 = -4.0*fq;
			m9 = 2.0*fq;
			m10 = -4.0*fq;

			// f2 = dist[10*Np+n];
			nread = neighborList[n+Np]; // neighbor 1 ( < 10Np => even part of dist)
			fq = dist[nread];  // reading the f2 data into register fq
			//fq = dist[Np+n];
			rho += fq;
			m1 -= 11.0*(fq);
			m2 -= 4.0*(fq);
			jx -= fq;
			m4 += 4.0*(fq);
			m9 += 2.0*(fq);
			m10 -= 4.0*(fq);

			// q=3
			nread = neighborList[n+2*Np]; // neighbor 4
			fq = dist[nread];
			//fq = dist[11*Np+n];
			rho += fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jy = fq;
			m6 = -4.0*fq;
			m9 -= fq;
			m10 += 2.0*fq;
			m11 = fq;
			m12 = -2.0*fq;

			// q = 4
			nread = neighborList[n+3*Np]; // neighbor 3
			fq = dist[nread];
			//fq = dist[2*Np+n];
			rho+= fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jy -= fq;
			m6 += 4.0*fq;
			m9 -= fq;
			m10 += 2.0*fq;
			m11 += fq;
			m12 -= 2.0*fq;

			// q=5
			nread = neighborList[n+4*Np];
			fq = dist[nread];
			//fq = dist[12*Np+n];
			rho += fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jz = fq;
			m8 = -4.0*fq;
			m9 -= fq;
			m10 += 2.0*fq;
			m11 -= fq;
			m12 += 2.0*fq;


			// q = 6
			nread = neighborList[n+5*Np];
			fq = dist[nread];
			//fq = dist[3*Np+n];
			rho+= fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jz -= fq;
			m8 += 4.0*fq;
			m9 -= fq;
			m10 += 2.0*fq;
			m11 -= fq;
			m12 += 2.0*fq;

			// q=7
			nread = neighborList[n+6*Np];
			fq = dist[nread];
			//fq = dist[13*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx += fq;
			m4 += fq;
			jy += fq;
			m6 += fq;
			m9  += fq;
			m10 += fq;
			m11 += fq;
			m12 += fq;
			m13 = fq;
			m16 = fq;
			m17 = -fq;

			// q = 8
			nread = neighborList[n+7*Np];
			fq = dist[nread];
			//fq = dist[4*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx -= fq;
			m4 -= fq;
			jy -= fq;
			m6 -= fq;
			m9 += fq;
			m10 += fq;
			m11 += fq;
			m12 += fq;
			m13 += fq;
			m16 -= fq;
			m17 += fq;

			// q=9
			nread = neighborList[n+8*Np];
			fq = dist[nread];
			//fq = dist[14*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx += fq;
			m4 += fq;
			jy -= fq;
			m6 -= fq;
			m9 += fq;
			m10 += fq;
			m11 += fq;
			m12 += fq;
			m13 -= fq;
			m16 += fq;
			m17 += fq;

			// q = 10
			nread = neighborList[n+9*Np];
			fq = dist[nread];
			//fq = dist[5*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx -= fq;
			m4 -= fq;
			jy += fq;
			m6 += fq;
			m9 += fq;
			m10 += fq;
			m11 += fq;
			m12 += fq;
			m13 -= fq;
			m16 -= fq;
			m17 -= fq;

			// q=11
			nread = neighborList[n+10*Np];
			fq = dist[nread];
			//fq = dist[15*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx += fq;
			m4 += fq;
			jz += fq;
			m8 += fq;
			m9 += fq;
			m10 += fq;
			m11 -= fq;
			m12 -= fq;
			m15 = fq;
			m16 -= fq;
			m18 = fq;

			// q=12
			nread = neighborList[n+11*Np];
			fq = dist[nread];
			//fq = dist[6*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx -= fq;
			m4 -= fq;
			jz -= fq;
			m8 -= fq;
			m9 += fq;
			m10 += fq;
			m11 -= fq;
			m12 -= fq;
			m15 += fq;
			m16 += fq;
			m18 -= fq;

			// q=13
			nread = neighborList[n+12*Np];
			fq = dist[nread];
			//fq = dist[16*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx += fq;
			m4 += fq;
			jz -= fq;
			m8 -= fq;
			m9 += fq;
			m10 += fq;
			m11 -= fq;
			m12 -= fq;
			m15 -= fq;
			m16 -= fq;
			m18 -= fq;

			// q=14
			nread = neighborList[n+13*Np];
			fq = dist[nread];
			//fq = dist[7*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx -= fq;
			m4 -= fq;
			jz += fq;
			m8 += fq;
			m9 += fq;
			m10 += fq;
			m11 -= fq;
			m12 -= fq;
			m15 -= fq;
			m16 += fq;
			m18 += fq;

			// q=15
			nread = neighborList[n+14*Np];
			fq = dist[nread];
			//fq = dist[17*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jy += fq;
			m6 += fq;
			jz += fq;
			m8 += fq;
			m9 -= 2.0*fq;
			m10 -= 2.0*fq;
			m14 = fq;
			m17 += fq;
			m18 -= fq;

			// q=16
			nread = neighborList[n+15*Np];
			fq = dist[nread];
			//fq = dist[8*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jy -= fq;
			m6 -= fq;
			jz -= fq;
			m8 -= fq;
			m9 -= 2.0*fq;
			m10 -= 2.0*fq;
			m14 += fq;
			m17 -= fq;
			m18 += fq;

			// q=17
			//fq = dist[18*Np+n];
			nread = neighborList[n+16*Np];
			fq = dist[nread];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jy += fq;
			m6 += fq;
			jz -= fq;
			m8 -= fq;
			m9 -= 2.0*fq;
			m10 -= 2.0*fq;
			m14 -= fq;
			m17 += fq;
			m18 += fq;

			// q=18
			nread = neighborList[n+17*Np];
			fq = dist[nread];
			//fq = dist[9*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jy -= fq;
			m6 -= fq;
			jz += fq;
			m8 += fq;
			m9 -= 2.0*fq;
			m10 -= 2.0*fq;
			m14 -= fq;
			m17 -= fq;
			m18 -= fq;

			//..............incorporate external force................................................
			//..............carry out relaxation process...............................................
			m1 = m1 + rlx_setA*((19*(jx*jx+jy*jy+jz*jz)/rho - 11*rho) - m1);
			m2 = m2 + rlx_setA*((3*rho - 5.5*(jx*jx+jy*jy+jz*jz)/rho) - m2);
			m4 = m4 + rlx_setB*((-0.6666666666666666*jx) - m4);
			m6 = m6 + rlx_setB*((-0.6666666666666666*jy) - m6);
			m8 = m8 + rlx_setB*((-0.6666666666666666*jz) - m8);
			m9 = m9 + rlx_setA*(((2*jx*jx-jy*jy-jz*jz)/rho) - m9);
			m10 = m10 + rlx_setA*(-0.5*((2*jx*jx-jy*jy-jz*jz)/rho) - m10);
			m11 = m11 + rlx_setA*(((jy*jy-jz*jz)/rho) - m11);
			m12 = m12 + rlx_setA*(-0.5*((jy*jy-jz*jz)/rho) - m12);
			m13 = m13 + rlx_setA*((jx*jy/rho) - m13);
			m14 = m14 + rlx_setA*((jy*jz/rho) - m14);
			m15 = m15 + rlx_setA*((jx*jz/rho) - m15);
			m16 = m16 + rlx_setB*( - m16);
			m17 = m17 + rlx_setB*( - m17);
			m18 = m18 + rlx_setB*( - m18);
			//.......................................................................................................
			//.................inverse transformation......................................................

			// q=0
			fq = mrt_V1*rho-mrt_V2*m1+mrt_V3*m2;
			dist[n] = fq;

			// q = 1
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jx-m4)+mrt_V6*(m9-m10)+0.16666666*Fx;
			nread = neighborList[n+Np];
			dist[nread] = fq;

			// q=2
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m4-jx)+mrt_V6*(m9-m10) -  0.16666666*Fx;
			nread = neighborList[n];
			dist[nread] = fq;

			// q = 3
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jy-m6)+mrt_V7*(m10-m9)+mrt_V8*(m11-m12) + 0.16666666*Fy;
			nread = neighborList[n+3*Np];
			dist[nread] = fq;

			// q = 4
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m6-jy)+mrt_V7*(m10-m9)+mrt_V8*(m11-m12) - 0.16666666*Fy;
			nread = neighborList[n+2*Np];
			dist[nread] = fq;

			// q = 5
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jz-m8)+mrt_V7*(m10-m9)+mrt_V8*(m12-m11) + 0.16666666*Fz;
			nread = neighborList[n+5*Np];
			dist[nread] = fq;

			// q = 6
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m8-jz)+mrt_V7*(m10-m9)+mrt_V8*(m12-m11) - 0.16666666*Fz;
			nread = neighborList[n+4*Np];
			dist[nread] = fq;

			// q = 7
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jx+jy)+0.025*(m4+m6)+mrt_V7*m9+mrt_V11*m10+
					mrt_V8*m11+mrt_V12*m12+0.25*m13+0.125*(m16-m17) + 0.08333333333*(Fx+Fy);
			
			nread = neighborList[n+7*Np];
			dist[nread] = fq;

			// q = 8
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2-0.1*(jx+jy)-0.025*(m4+m6) +mrt_V7*m9+mrt_V11*m10+mrt_V8*m11
					+mrt_V12*m12+0.25*m13+0.125*(m17-m16) - 0.08333333333*(Fx+Fy);
			nread = neighborList[n+6*Np];
			dist[nread] = fq;

			// q = 9
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jx-jy)+0.025*(m4-m6)+mrt_V7*m9+mrt_V11*m10+
					mrt_V8*m11+mrt_V12*m12-0.25*m13+0.125*(m16+m17) + 0.08333333333*(Fx-Fy);
			nread = neighborList[n+9*Np];
			dist[nread] = fq;

			// q = 10
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jy-jx)+0.025*(m6-m4)+mrt_V7*m9+mrt_V11*m10+
					mrt_V8*m11+mrt_V12*m12-0.25*m13-0.125*(m16+m17)- 0.08333333333*(Fx-Fy);
			nread = neighborList[n+8*Np];
			dist[nread] = fq;

			// q = 11
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jx+jz)+0.025*(m4+m8)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12+0.25*m15+0.125*(m18-m16) + 0.08333333333*(Fx+Fz);
			nread = neighborList[n+11*Np];
			dist[nread] = fq;

			// q = 12
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2-0.1*(jx+jz)-0.025*(m4+m8)+
					mrt_V7*m9+mrt_V11*m10-mrt_V8*m11-mrt_V12*m12+0.25*m15+0.125*(m16-m18) - 0.08333333333*(Fx+Fz);
			nread = neighborList[n+10*Np];
			dist[nread]= fq;

			// q = 13
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jx-jz)+0.025*(m4-m8)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12-0.25*m15-0.125*(m16+m18) + 0.08333333333*(Fx-Fz);
			nread = neighborList[n+13*Np];
			dist[nread] = fq;

			// q= 14
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jz-jx)+0.025*(m8-m4)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12-0.25*m15+0.125*(m16+m18) - 0.08333333333*(Fx-Fz);
			nread = neighborList[n+12*Np];
			dist[nread] = fq;


			// q = 15
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jy+jz)+0.025*(m6+m8)
					-mrt_V6*m9-mrt_V7*m10+0.25*m14+0.125*(m17-m18) + 0.08333333333*(Fy+Fz);
			nread = neighborList[n+15*Np];
			dist[nread] = fq;

			// q = 16
			fq =  mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2-0.1*(jy+jz)-0.025*(m6+m8)
					-mrt_V6*m9-mrt_V7*m10+0.25*m14+0.125*(m18-m17)- 0.08333333333*(Fy+Fz);
			nread = neighborList[n+14*Np];
			dist[nread] = fq;


			// q = 17
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jy-jz)+0.025*(m6-m8)
					-mrt_V6*m9-mrt_V7*m10-0.25*m14+0.125*(m17+m18) + 0.08333333333*(Fy-Fz);
			nread = neighborList[n+17*Np];
			dist[nread] = fq;

			// q = 18
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jz-jy)+0.025*(m8-m6)
					-mrt_V6*m9-mrt_V7*m10-0.25*m14-0.125*(m17+m18) - 0.08333333333*(Fy-Fz);
			nread = neighborList[n+16*Np];
			dist[nread] = fq;

		}
	}
}


//__launch_bounds__(512,1)
__global__ void 
dvc_ScaLBL_AAeven_MRT(double *dist, int start, int finish, int Np, double rlx_setA, double rlx_setB, double Fx, double Fy, double Fz) {

	int n;
	double fq;
	// conserved momemnts
	double rho,jx,jy,jz;
	// non-conserved moments
	double m1,m2,m4,m6,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18;
	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n = S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;

		if ( n<finish ){

			// q=0
			fq = dist[n];
			rho = fq;
			m1  = -30.0*fq;
			m2  = 12.0*fq;

			// q=1
			fq = dist[2*Np+n];
			rho += fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jx = fq;
			m4 = -4.0*fq;
			m9 = 2.0*fq;
			m10 = -4.0*fq;

			// q=2
			fq = dist[1*Np+n];
			rho += fq;
			m1 -= 11.0*(fq);
			m2 -= 4.0*(fq);
			jx -= fq;
			m4 += 4.0*(fq);
			m9 += 2.0*(fq);
			m10 -= 4.0*(fq);

			// q=3
			fq = dist[4*Np+n];
			rho += fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jy = fq;
			m6 = -4.0*fq;
			m9 -= fq;
			m10 += 2.0*fq;
			m11 = fq;
			m12 = -2.0*fq;

			// q = 4
			fq = dist[3*Np+n];
			rho+= fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jy -= fq;
			m6 += 4.0*fq;
			m9 -= fq;
			m10 += 2.0*fq;
			m11 += fq;
			m12 -= 2.0*fq;

			// q=5
			fq = dist[6*Np+n];
			rho += fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jz = fq;
			m8 = -4.0*fq;
			m9 -= fq;
			m10 += 2.0*fq;
			m11 -= fq;
			m12 += 2.0*fq;

			// q = 6
			fq = dist[5*Np+n];
			rho+= fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jz -= fq;
			m8 += 4.0*fq;
			m9 -= fq;
			m10 += 2.0*fq;
			m11 -= fq;
			m12 += 2.0*fq;

			// q=7
			fq = dist[8*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx += fq;
			m4 += fq;
			jy += fq;
			m6 += fq;
			m9  += fq;
			m10 += fq;
			m11 += fq;
			m12 += fq;
			m13 = fq;
			m16 = fq;
			m17 = -fq;

			// q = 8
			fq = dist[7*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx -= fq;
			m4 -= fq;
			jy -= fq;
			m6 -= fq;
			m9 += fq;
			m10 += fq;
			m11 += fq;
			m12 += fq;
			m13 += fq;
			m16 -= fq;
			m17 += fq;

			// q=9
			fq = dist[10*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx += fq;
			m4 += fq;
			jy -= fq;
			m6 -= fq;
			m9 += fq;
			m10 += fq;
			m11 += fq;
			m12 += fq;
			m13 -= fq;
			m16 += fq;
			m17 += fq;

			// q = 10
			fq = dist[9*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx -= fq;
			m4 -= fq;
			jy += fq;
			m6 += fq;
			m9 += fq;
			m10 += fq;
			m11 += fq;
			m12 += fq;
			m13 -= fq;
			m16 -= fq;
			m17 -= fq;

			// q=11
			fq = dist[12*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx += fq;
			m4 += fq;
			jz += fq;
			m8 += fq;
			m9 += fq;
			m10 += fq;
			m11 -= fq;
			m12 -= fq;
			m15 = fq;
			m16 -= fq;
			m18 = fq;

			// q=12
			fq = dist[11*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx -= fq;
			m4 -= fq;
			jz -= fq;
			m8 -= fq;
			m9 += fq;
			m10 += fq;
			m11 -= fq;
			m12 -= fq;
			m15 += fq;
			m16 += fq;
			m18 -= fq;

			// q=13
			fq = dist[14*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx += fq;
			m4 += fq;
			jz -= fq;
			m8 -= fq;
			m9 += fq;
			m10 += fq;
			m11 -= fq;
			m12 -= fq;
			m15 -= fq;
			m16 -= fq;
			m18 -= fq;

			// q=14
			fq = dist[13*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jx -= fq;
			m4 -= fq;
			jz += fq;
			m8 += fq;
			m9 += fq;
			m10 += fq;
			m11 -= fq;
			m12 -= fq;
			m15 -= fq;
			m16 += fq;
			m18 += fq;

			// q=15
			fq = dist[16*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jy += fq;
			m6 += fq;
			jz += fq;
			m8 += fq;
			m9 -= 2.0*fq;
			m10 -= 2.0*fq;
			m14 = fq;
			m17 += fq;
			m18 -= fq;

			// q=16
			fq = dist[15*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jy -= fq;
			m6 -= fq;
			jz -= fq;
			m8 -= fq;
			m9 -= 2.0*fq;
			m10 -= 2.0*fq;
			m14 += fq;
			m17 -= fq;
			m18 += fq;

			// q=17
			fq = dist[18*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jy += fq;
			m6 += fq;
			jz -= fq;
			m8 -= fq;
			m9 -= 2.0*fq;
			m10 -= 2.0*fq;
			m14 -= fq;
			m17 += fq;
			m18 += fq;

			// q=18
			fq = dist[17*Np+n];
			rho += fq;
			m1 += 8.0*fq;
			m2 += fq;
			jy -= fq;
			m6 -= fq;
			jz += fq;
			m8 += fq;
			m9 -= 2.0*fq;
			m10 -= 2.0*fq;
			m14 -= fq;
			m17 -= fq;
			m18 -= fq;

			//........................................................................
			//					READ THE DISTRIBUTIONS
			//		(read from opposite array due to previous swap operation)
			//........................................................................

			//..............incorporate external force................................................
			//..............carry out relaxation process...............................................
			m1 = m1 + rlx_setA*((19*(jx*jx+jy*jy+jz*jz)/rho - 11*rho) - m1);
			m2 = m2 + rlx_setA*((3*rho - 5.5*(jx*jx+jy*jy+jz*jz)/rho) - m2);
			m4 = m4 + rlx_setB*((-0.6666666666666666*jx) - m4);
			m6 = m6 + rlx_setB*((-0.6666666666666666*jy) - m6);
			m8 = m8 + rlx_setB*((-0.6666666666666666*jz) - m8);
			m9 = m9 + rlx_setA*(((2*jx*jx-jy*jy-jz*jz)/rho) - m9);
			m10 = m10 + rlx_setA*(-0.5*((2*jx*jx-jy*jy-jz*jz)/rho) - m10);
			m11 = m11 + rlx_setA*(((jy*jy-jz*jz)/rho) - m11);
			m12 = m12 + rlx_setA*(-0.5*((jy*jy-jz*jz)/rho) - m12);
			m13 = m13 + rlx_setA*((jx*jy/rho) - m13);
			m14 = m14 + rlx_setA*((jy*jz/rho) - m14);
			m15 = m15 + rlx_setA*((jx*jz/rho) - m15);
			m16 = m16 + rlx_setB*( - m16);
			m17 = m17 + rlx_setB*( - m17);
			m18 = m18 + rlx_setB*( - m18);
			//.......................................................................................................
			//.................inverse transformation......................................................

			// q=0
			fq = mrt_V1*rho-mrt_V2*m1+mrt_V3*m2;
			dist[n] = fq;

			// q = 1
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jx-m4)+mrt_V6*(m9-m10) + 0.16666666*Fx;
			dist[1*Np+n] = fq;

			// q=2
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m4-jx)+mrt_V6*(m9-m10) -  0.16666666*Fx;
			dist[2*Np+n] = fq;

			// q = 3
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jy-m6)+mrt_V7*(m10-m9)+mrt_V8*(m11-m12) + 0.16666666*Fy;
			dist[3*Np+n] = fq;

			// q = 4
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m6-jy)+mrt_V7*(m10-m9)+mrt_V8*(m11-m12) - 0.16666666*Fy;
			dist[4*Np+n] = fq;

			// q = 5
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jz-m8)+mrt_V7*(m10-m9)+mrt_V8*(m12-m11) + 0.16666666*Fz;
			dist[5*Np+n] = fq;

			// q = 6
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m8-jz)+mrt_V7*(m10-m9)+mrt_V8*(m12-m11) - 0.16666666*Fz;
			dist[6*Np+n] = fq;

			// q = 7
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jx+jy)+0.025*(m4+m6)+
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12+0.25*m13+0.125*(m16-m17) + 
					0.08333333333*(Fx+Fy);
			dist[7*Np+n] = fq;


			// q = 8
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2-0.1*(jx+jy)-0.025*(m4+m6) +mrt_V7*m9+mrt_V11*m10+mrt_V8*m11
					+mrt_V12*m12+0.25*m13+0.125*(m17-m16) - 0.08333333333*(Fx+Fy);
			dist[8*Np+n] = fq;

			// q = 9
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jx-jy)+0.025*(m4-m6)+
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12-0.25*m13+0.125*(m16+m17)+
					0.08333333333*(Fx-Fy);
			dist[9*Np+n] = fq;

			// q = 10
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jy-jx)+0.025*(m6-m4)+
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12-0.25*m13-0.125*(m16+m17)-
					0.08333333333*(Fx-Fy);
			dist[10*Np+n] = fq;


			// q = 11
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jx+jz)+0.025*(m4+m8)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12+0.25*m15+0.125*(m18-m16) + 0.08333333333*(Fx+Fz);
			dist[11*Np+n] = fq;

			// q = 12
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2-0.1*(jx+jz)-0.025*(m4+m8)+
					mrt_V7*m9+mrt_V11*m10-mrt_V8*m11-mrt_V12*m12+0.25*m15+0.125*(m16-m18)-
					0.08333333333*(Fx+Fz);
			dist[12*Np+n] = fq;

			// q = 13
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jx-jz)+0.025*(m4-m8)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12-0.25*m15-0.125*(m16+m18) + 0.08333333333*(Fx-Fz);
			dist[13*Np+n] = fq;

			// q= 14
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jz-jx)+0.025*(m8-m4)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12-0.25*m15+0.125*(m16+m18) - 0.08333333333*(Fx-Fz);

			dist[14*Np+n] = fq;

			// q = 15
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jy+jz)+0.025*(m6+m8)
					-mrt_V6*m9-mrt_V7*m10+0.25*m14+0.125*(m17-m18) + 0.08333333333*(Fy+Fz);
			dist[15*Np+n] = fq;

			// q = 16
			fq =  mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2-0.1*(jy+jz)-0.025*(m6+m8)
					-mrt_V6*m9-mrt_V7*m10+0.25*m14+0.125*(m18-m17)- 0.08333333333*(Fy+Fz);
			dist[16*Np+n] = fq;


			// q = 17
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jy-jz)+0.025*(m6-m8)
					-mrt_V6*m9-mrt_V7*m10-0.25*m14+0.125*(m17+m18) + 0.08333333333*(Fy-Fz);
			dist[17*Np+n] = fq;

			// q = 18
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jz-jy)+0.025*(m8-m6)
					-mrt_V6*m9-mrt_V7*m10-0.25*m14-0.125*(m17+m18) - 0.08333333333*(Fy-Fz);
			dist[18*Np+n] = fq;
			//........................................................................
		}
	}
}

__global__  void dvc_ScaLBL_D3Q19_Momentum(double *dist, double *vel, int N)
{
	int n;
	// distributions
	double f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;
	double f10,f11,f12,f13,f14,f15,f16,f17,f18;
	double vx,vy,vz;
	char id;

	int S = N/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n = S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x;
		if (n<N){
			f0 = dist[n];
			f2 = dist[2*N+n];
			f4 = dist[4*N+n];
			f6 = dist[6*N+n];
			f8 = dist[8*N+n];
			f10 = dist[10*N+n];
			f12 = dist[12*N+n];
			f14 = dist[14*N+n];
			f16 = dist[16*N+n];
			f18 = dist[18*N+n];
			//........................................................................
			f1 = dist[N+n];
			f3 = dist[3*N+n];
			f5 = dist[5*N+n];
			f7 = dist[7*N+n];
			f9 = dist[9*N+n];
			f11 = dist[11*N+n];
			f13 = dist[13*N+n];
			f15 = dist[15*N+n];
			f17 = dist[17*N+n];			

			//.................Compute the velocity...................................
			vx = f1-f2+f7-f8+f9-f10+f11-f12+f13-f14;
			vy = f3-f4+f7-f8-f9+f10+f15-f16+f17-f18;
			vz = f5-f6+f11-f12-f13+f14+f15-f16-f17+f18;\
		
			//..................Write the velocity.....................................
			vel[n] = vx;
		    vel[N+n] = vy;
			vel[2*N+n] = vz;
			//........................................................................
		}
	}
}

__global__  void dvc_ScaLBL_D3Q19_Pressure(const double *dist, double *Pressure, int N)
{
	int n;
	// distributions
	double f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;
	double f10,f11,f12,f13,f14,f15,f16,f17,f18;

	int S = N/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n = S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x;
		if (n<N){				//.......................................................................
			// Registers to store the distributions
			//........................................................................
			//........................................................................
			// Registers to store the distributions
			//........................................................................
			f0 = dist[n];
			f2 = dist[2*N+n];
			f4 = dist[4*N+n];
			f6 = dist[6*N+n];
			f8 = dist[8*N+n];
			f10 = dist[10*N+n];
			f12 = dist[12*N+n];
			f14 = dist[14*N+n];
			f16 = dist[16*N+n];
			f18 = dist[18*N+n];
			//........................................................................
			f1 = dist[N+n];
			f3 = dist[3*N+n];
			f5 = dist[5*N+n];
			f7 = dist[7*N+n];
			f9 = dist[9*N+n];
			f11 = dist[11*N+n];
			f13 = dist[13*N+n];
			f15 = dist[15*N+n];
			f17 = dist[17*N+n];
			//.................Compute the velocity...................................
			Pressure[n] = 0.3333333333333333*(f0+f2+f1+f4+f3+f6+f5+f8+f7+f10+
					f9+f12+f11+f14+f13+f16+f15+f18+f17);
		}
	}
}

__global__  void dvc_ScaLBL_D3Q19_AAeven_Pressure_BC_z(int *list, double *dist, double din, int count, int Np)
{
	int idx, n;
	// distributions
	double f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;
	double f10,f11,f12,f13,f14,f15,f16,f17,f18;
	double ux,uy,uz,Cyz,Cxz;
	ux = uy = 0.0;

	idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < count){

		n = list[idx];
		f0 = dist[n];
		f1 = dist[2*Np+n];
		f2 = dist[1*Np+n];
		f3 = dist[4*Np+n];
		f4 = dist[3*Np+n];
		f6 = dist[5*Np+n];
		f7 = dist[8*Np+n];
		f8 = dist[7*Np+n];
		f9 = dist[10*Np+n];
		f10 = dist[9*Np+n];
		f12 = dist[11*Np+n];
		f13 = dist[14*Np+n];
		f16 = dist[15*Np+n];
		f17 = dist[18*Np+n];
		//...................................................
		// Determine the inlet flow velocity
		//ux = (f1-f2+f7-f8+f9-f10+f11-f12+f13-f14);
		//uy = (f3-f4+f7-f8-f9+f10+f15-f16+f17-f18);
		uz = din - (f0+f1+f2+f3+f4+f7+f8+f9+f10 + 2*(f6+f12+f13+f16+f17));

		Cxz = 0.5*(f1+f7+f9-f2-f10-f8) - 0.3333333333333333*ux;
		Cyz = 0.5*(f3+f7+f10-f4-f9-f8) - 0.3333333333333333*uy;

		f5 = f6 + 0.33333333333333338*uz;
		f11 = f12 + 0.16666666666666678*(uz+ux)-Cxz;
		f14 = f13 + 0.16666666666666678*(uz-ux)+Cxz;
		f15 = f16 + 0.16666666666666678*(uy+uz)-Cyz;
		f18 = f17 + 0.16666666666666678*(uz-uy)+Cyz;
		//........Store in "opposite" memory location..........
		dist[6*Np+n] = f5;
		dist[12*Np+n] = f11;
		dist[13*Np+n] = f14;
		dist[16*Np+n] = f15;
		dist[17*Np+n] = f18;
		/*
		if (idx == count-1) {
		    printf("Site=%i\n",n);
		    printf("ux=%f, uy=%f, uz=%f\n",ux,uy,uz);
		    printf("Cxz=%f, Cyz=%f\n",Cxz,Cyz);
		}
		*/

	}
}

__global__  void dvc_ScaLBL_D3Q19_AAeven_Pressure_BC_Z(int *list, double *dist, double dout, int count, int Np)
{
	int idx,n;
	// distributions
	double f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;
	double f10,f11,f12,f13,f14,f15,f16,f17,f18;
	double ux,uy,uz,Cyz,Cxz;
	ux = uy = 0.0;

	idx = blockIdx.x*blockDim.x + threadIdx.x;

	// Loop over the boundary - threadblocks delineated by start...finish
	if ( idx < count ){

		n = list[idx];
		//........................................................................
		// Read distributions 
		//........................................................................
		f0 = dist[n];
		f1 = dist[2*Np+n];
		f2 = dist[1*Np+n];
		f3 = dist[4*Np+n];
		f4 = dist[3*Np+n];
		f5 = dist[6*Np+n];
		f7 = dist[8*Np+n];
		f8 = dist[7*Np+n];
		f9 = dist[10*Np+n];
		f10 = dist[9*Np+n];
		f11 = dist[12*Np+n];
		f14 = dist[13*Np+n];
		f15 = dist[16*Np+n];
		f18 = dist[17*Np+n];
		
		// Determine the outlet flow velocity
		//ux = f1-f2+f7-f8+f9-f10+f11-f12+f13-f14;
		//uy = f3-f4+f7-f8-f9+f10+f15-f16+f17-f18;
		uz = -dout + (f0+f1+f2+f3+f4+f7+f8+f9+f10 + 2*(f5+f11+f14+f15+f18));

		Cxz = 0.5*(f1+f7+f9-f2-f10-f8) - 0.3333333333333333*ux;
		Cyz = 0.5*(f3+f7+f10-f4-f9-f8) - 0.3333333333333333*uy;

		f6 = f5 - 0.33333333333333338*uz;
		f12 = f11 - 0.16666666666666678*(uz+ux)+Cxz;
		f13 = f14 - 0.16666666666666678*(uz-ux)-Cxz;
		f16 = f15 - 0.16666666666666678*(uy+uz)+Cyz;
		f17 = f18 - 0.16666666666666678*(uz-uy)-Cyz;

		dist[5*Np+n] = f6;
		dist[11*Np+n] = f12;
		dist[14*Np+n] = f13;
		dist[15*Np+n] = f16;
		dist[18*Np+n] = f17;
		//...................................................
	}
}

__global__  void dvc_ScaLBL_D3Q19_AAodd_Pressure_BC_z(int *d_neighborList, int *list, double *dist, double din, int count, int Np)
{
	int idx, n;
	int nread;
	int nr5,nr11,nr14,nr15,nr18;
	// distributions
	double f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;
	double f10,f11,f12,f13,f14,f15,f16,f17,f18;
	double ux,uy,uz,Cyz,Cxz;
	ux = uy = 0.0;

	idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < count){
		
		n = list[idx];
		f0 = dist[n];
				
		nread = d_neighborList[n];
		f1 = dist[nread];

		nread = d_neighborList[n+2*Np];
		f3 = dist[nread];

		nread = d_neighborList[n+6*Np];
		f7 = dist[nread];

		nread = d_neighborList[n+8*Np];
		f9 = dist[nread];

		nread = d_neighborList[n+12*Np];
		f13 = dist[nread];

		nread = d_neighborList[n+16*Np];
		f17 = dist[nread];

		nread = d_neighborList[n+Np];
		f2 = dist[nread];

		nread = d_neighborList[n+3*Np];
		f4 = dist[nread];

		nread = d_neighborList[n+5*Np];
		f6 = dist[nread];

		nread = d_neighborList[n+7*Np];
		f8 = dist[nread];

		nread = d_neighborList[n+9*Np];
		f10 = dist[nread];

		nread = d_neighborList[n+11*Np];
		f12 = dist[nread];

		nread = d_neighborList[n+15*Np];
		f16 = dist[nread];

		// Unknown distributions
		nr5 = d_neighborList[n+4*Np];
		nr11 = d_neighborList[n+10*Np];
		nr15 = d_neighborList[n+14*Np];
		nr14 = d_neighborList[n+13*Np];
		nr18 = d_neighborList[n+17*Np];
		
		//...................................................
		//........Determine the inlet flow velocity.........
		//ux = (f1-f2+f7-f8+f9-f10+f11-f12+f13-f14);
		//uy = (f3-f4+f7-f8-f9+f10+f15-f16+f17-f18);
		uz = din - (f0+f1+f2+f3+f4+f7+f8+f9+f10 + 2*(f6+f12+f13+f16+f17));

		Cxz = 0.5*(f1+f7+f9-f2-f10-f8) - 0.3333333333333333*ux;
		Cyz = 0.5*(f3+f7+f10-f4-f9-f8) - 0.3333333333333333*uy;

		f5 = f6 + 0.33333333333333338*uz;
		f11 = f12 + 0.16666666666666678*(uz+ux)-Cxz;
		f14 = f13 + 0.16666666666666678*(uz-ux)+Cxz;
		f15 = f16 + 0.16666666666666678*(uy+uz)-Cyz;
		f18 = f17 + 0.16666666666666678*(uz-uy)+Cyz;
		//........Store in "opposite" memory location..........
		dist[nr5] = f5;
		dist[nr11] = f11;
		dist[nr14] = f14;
		dist[nr15] = f15;
		dist[nr18] = f18;
	}
}

__global__  void dvc_ScaLBL_D3Q19_AAodd_Pressure_BC_Z(int *d_neighborList, int *list, double *dist, double dout, int count, int Np)
{
	int idx,n,nread;
	int nr6,nr12,nr13,nr16,nr17;
	// distributions
	double f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;
	double f10,f11,f12,f13,f14,f15,f16,f17,f18;
	double ux,uy,uz,Cyz,Cxz;
	ux = uy = 0.0;

	idx = blockIdx.x*blockDim.x + threadIdx.x;

	// Loop over the boundary - threadblocks delineated by start...finish
	if ( idx < count ){

		n = list[idx];
		//........................................................................
		// Read distributions 
		//........................................................................
		f0 = dist[n];

		nread = d_neighborList[n];
		f1 = dist[nread];

		nread = d_neighborList[n+2*Np];
		f3 = dist[nread];

		nread = d_neighborList[n+4*Np];
		f5 = dist[nread];

		nread = d_neighborList[n+6*Np];
		f7 = dist[nread];

		nread = d_neighborList[n+8*Np];
		f9 = dist[nread];

		nread = d_neighborList[n+10*Np];
		f11 = dist[nread];

		nread = d_neighborList[n+14*Np];
		f15 = dist[nread];


		nread = d_neighborList[n+Np];
		f2 = dist[nread];

		nread = d_neighborList[n+3*Np];
		f4 = dist[nread];

		nread = d_neighborList[n+7*Np];
		f8 = dist[nread];

		nread = d_neighborList[n+9*Np];
		f10 = dist[nread];

		nread = d_neighborList[n+13*Np];
		f14 = dist[nread];

		nread = d_neighborList[n+17*Np];
		f18 = dist[nread];
		
		// unknown distributions
		nr6 = d_neighborList[n+5*Np];
		nr12 = d_neighborList[n+11*Np];
		nr16 = d_neighborList[n+15*Np];
		nr17 = d_neighborList[n+16*Np];
		nr13 = d_neighborList[n+12*Np];

		
		//........Determine the outlet flow velocity.........
		//ux = f1-f2+f7-f8+f9-f10+f11-f12+f13-f14;
		//uy = f3-f4+f7-f8-f9+f10+f15-f16+f17-f18;
		uz = -dout + (f0+f1+f2+f3+f4+f7+f8+f9+f10 + 2*(f5+f11+f14+f15+f18));

		Cxz = 0.5*(f1+f7+f9-f2-f10-f8) - 0.3333333333333333*ux;
		Cyz = 0.5*(f3+f7+f10-f4-f9-f8) - 0.3333333333333333*uy;

		f6 = f5 - 0.33333333333333338*uz;
		f12 = f11 - 0.16666666666666678*(uz+ux)+Cxz;
		f13 = f14 - 0.16666666666666678*(uz-ux)-Cxz;
		f16 = f15 - 0.16666666666666678*(uy+uz)+Cyz;
		f17 = f18 - 0.16666666666666678*(uz-uy)-Cyz;

		//........Store in "opposite" memory location..........
		dist[nr6] = f6;
		dist[nr12] = f12;
		dist[nr13] = f13;
		dist[nr16] = f16;
		dist[nr17] = f17;
		//...................................................
	}
}


__global__  void dvc_ScaLBL_D3Q19_AAeven_Flux_BC_z(int *list, double *dist, double flux, double Area, 
		double *dvcsum, int count, int Np)
{
	int idx, n;
	// distributions
	double f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;
	double f10,f11,f12,f13,f14,f15,f16,f17,f18;
	double factor = 1.f/(Area);
	double sum = 0.f;

	idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < count){
		
		n = list[idx];
		f0 = dist[n];
		f1 = dist[2*Np+n];
		f2 = dist[1*Np+n];
		f3 = dist[4*Np+n];
		f4 = dist[3*Np+n];
		f6 = dist[5*Np+n];
		f7 = dist[8*Np+n];
		f8 = dist[7*Np+n];
		f9 = dist[10*Np+n];
		f10 = dist[9*Np+n];
		f12 = dist[11*Np+n];
		f13 = dist[14*Np+n];
		f16 = dist[15*Np+n];
		f17 = dist[18*Np+n];
		sum = factor*(f0+f1+f2+f3+f4+f7+f8+f9+f10 + 2*(f6+f12+f13+f16+f17));
	}

	//sum = blockReduceSum(sum);
	//if (threadIdx.x==0)
	//   atomicAdd(dvcsum, sum);
	
    extern __shared__ double temp[];
    thread_group g = this_thread_block();
    double block_sum = reduce_sum(g, temp, sum);

    if (g.thread_rank() == 0) atomicAdd(dvcsum, block_sum);
}


__global__  void dvc_ScaLBL_D3Q19_AAodd_Flux_BC_z(int *d_neighborList, int *list, double *dist, double flux, 
		double Area, double *dvcsum, int count, int Np)
{
	int idx, n;
	int nread;

	// distributions
	double f0,f1,f2,f3,f4,f5,f6,f7,f8,f9;
	double f10,f11,f12,f13,f14,f15,f16,f17,f18;
	double factor = 1.f/(Area);
	double sum = 0.f;

	idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < count){
		
		n = list[idx];
				
		f0 = dist[n];
		
		nread = d_neighborList[n];
		f1 = dist[nread];

		nread = d_neighborList[n+2*Np];
		f3 = dist[nread];

		nread = d_neighborList[n+6*Np];
		f7 = dist[nread];

		nread = d_neighborList[n+8*Np];
		f9 = dist[nread];

		nread = d_neighborList[n+12*Np];
		f13 = dist[nread];

		nread = d_neighborList[n+16*Np];
		f17 = dist[nread];

		nread = d_neighborList[n+Np];
		f2 = dist[nread];

		nread = d_neighborList[n+3*Np];
		f4 = dist[nread];

		nread = d_neighborList[n+5*Np];
		f6 = dist[nread];

		nread = d_neighborList[n+7*Np];
		f8 = dist[nread];

		nread = d_neighborList[n+9*Np];
		f10 = dist[nread];

		nread = d_neighborList[n+11*Np];
		f12 = dist[nread];

		nread = d_neighborList[n+15*Np];
		f16 = dist[nread];

		sum = factor*(f0+f1+f2+f3+f4+f7+f8+f9+f10 + 2*(f6+f12+f13+f16+f17));

	}

	//sum = blockReduceSum(sum);
	//if (threadIdx.x==0)
	//   atomicAdd(dvcsum, sum);
	
    extern __shared__ double temp[];
    thread_group g = this_thread_block();
    double block_sum = reduce_sum(g, temp, sum);

    if (g.thread_rank() == 0) atomicAdd(dvcsum, block_sum);
}

//*************************************************************************

//extern "C" void ScaLBL_D3Q19_MapRecv(int q, int Cqx, int Cqy, int Cqz, int *list,  int start, int count,
//			int *d3q19_recvlist, int Nx, int Ny, int Nz){
//	int GRID = count / 512 + 1;
//	dvc_ScaLBL_D3Q19_Unpack <<<GRID,512 >>>(q, Cqx, Cqy, Cqz, list, start, count, d3q19_recvlist, Nx, Ny, Nz);
//}

extern "C" void ScaLBL_D3Q19_Pack(int q, int *list, int start, int count, double *sendbuf, double *dist, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q19_Pack <<<GRID,512 >>>(q, list, start, count, sendbuf, dist, N);
}

extern "C" void ScaLBL_D3Q19_Unpack(int q, int *list,  int start, int count, double *recvbuf, double *dist, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q19_Unpack <<<GRID,512 >>>(q, list, start, count, recvbuf, dist, N);
}
//*************************************************************************


extern "C" void ScaLBL_D3Q19_Init(double *dist, int Np){
	dvc_ScaLBL_D3Q19_Init<<<NBLOCKS,NTHREADS >>>(dist, Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q19_AA_Init: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q19_Momentum(double *dist, double *vel, int Np){

	dvc_ScaLBL_D3Q19_Momentum<<<NBLOCKS,NTHREADS >>>(dist, vel, Np);

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q19_Velocity: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q19_Pressure(double *fq, double *Pressure, int Np){
	dvc_ScaLBL_D3Q19_Pressure<<< NBLOCKS,NTHREADS >>>(fq, Pressure, Np);
}

extern "C" void ScaLBL_D3Q19_AAeven_Pressure_BC_z(int *list, double *dist, double din, int count, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q19_AAeven_Pressure_BC_z<<<GRID,512>>>(list, dist, din, count, N);
}

extern "C" void ScaLBL_D3Q19_AAeven_Pressure_BC_Z(int *list, double *dist, double dout, int count, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q19_AAeven_Pressure_BC_Z<<<GRID,512>>>(list, dist, dout, count, N);
}

extern "C" void ScaLBL_D3Q19_AAodd_Pressure_BC_z(int *neighborList, int *list, double *dist, double din, int count, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q19_AAodd_Pressure_BC_z<<<GRID,512>>>(neighborList, list, dist, din, count, N);
}

extern "C" void ScaLBL_D3Q19_AAodd_Pressure_BC_Z(int *neighborList, int *list, double *dist, double dout, int count, int N){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q19_AAodd_Pressure_BC_Z<<<GRID,512>>>(neighborList, list, dist, dout, count, N);
}


extern "C" double ScaLBL_D3Q19_AAeven_Flux_BC_z(int *list, double *dist, double flux, double area, 
		 int count, int N){

	int GRID = count / 512 + 1;

	// IMPORTANT -- this routine may fail if Nx*Ny > 512*512
	if (count > 512*512){
		printf("WARNING (ScaLBL_D3Q19_Flux_BC_Z): CUDA reduction operation may fail if count > 512*512");
	}

	// Allocate memory to store the sums
	double din;
	double sum[1];
 	double *dvcsum;
	cudaMalloc((void **)&dvcsum,sizeof(double)*count);
	cudaMemset(dvcsum,0,sizeof(double)*count);
	int sharedBytes = 512*sizeof(double);

	// compute the local flux and store the result
	dvc_ScaLBL_D3Q19_AAeven_Flux_BC_z<<<GRID,512,sharedBytes>>>(list, dist, flux, area, dvcsum, count, N);

	// Now read the total flux
	cudaMemcpy(&sum[0],dvcsum,sizeof(double),cudaMemcpyDeviceToHost);
	din=sum[0];

	// free the memory needed for reduction
	cudaFree(dvcsum);

	return din;
}

extern "C" double ScaLBL_D3Q19_AAodd_Flux_BC_z(int *neighborList, int *list, double *dist, double flux, 
		double area, int count, int N){

	int GRID = count / 512 + 1;

	// IMPORTANT -- this routine may fail if Nx*Ny > 512*512
	if (count > 512*512){
		printf("WARNING (ScaLBL_D3Q19_Flux_BC_Z): CUDA reduction operation may fail if count > 512*512");
	}

	// Allocate memory to store the sums
	double din;
	double sum[1];
 	double *dvcsum;
	cudaMalloc((void **)&dvcsum,sizeof(double)*count);
	cudaMemset(dvcsum,0,sizeof(double)*count);
	int sharedBytes = 512*sizeof(double);

	// compute the local flux and store the result
	dvc_ScaLBL_D3Q19_AAodd_Flux_BC_z<<<GRID,512,sharedBytes>>>(neighborList, list, dist, flux, area, dvcsum, count, N);

	// Now read the total flux
	cudaMemcpy(&sum[0],dvcsum,sizeof(double),cudaMemcpyDeviceToHost);
	din=sum[0];

	// free the memory needed for reduction
	cudaFree(dvcsum);

	return din;
}


extern "C" double deviceReduce(double *in, double* out, int N) {
	int threads = 512;
	int blocks = min((N + threads - 1) / threads, 1024);

	double sum = 0.f;
	deviceReduceKernel<<<blocks, threads>>>(in, out, N);
	deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
	return sum;
}

//
//extern "C" void ScaLBL_D3Q19_Pressure_BC_Z(int *list, double *dist, double dout, int count, int Np){
//	int GRID = count / 512 + 1;
//	dvc_ScaLBL_D3Q19_Pressure_BC_Z<<<GRID,512>>>(disteven, distodd, dout, Nx, Ny, Nz, outlet);
//}

extern "C" void ScaLBL_D3Q19_AAeven_MRT(double *dist, int start, int finish, int Np, double rlx_setA, double rlx_setB, double Fx,
       double Fy, double Fz){
       
       dvc_ScaLBL_AAeven_MRT<<<NBLOCKS,NTHREADS >>>(dist,start,finish,Np,rlx_setA,rlx_setB,Fx,Fy,Fz);

       cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q19_AAeven_MRT: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q19_AAodd_MRT(int *neighborlist, double *dist, int start, int finish, int Np, double rlx_setA, double rlx_setB, double Fx,
       double Fy, double Fz){
       
       dvc_ScaLBL_AAodd_MRT<<<NBLOCKS,NTHREADS >>>(neighborlist,dist,start,finish,Np,rlx_setA,rlx_setB,Fx,Fy,Fz);

       cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q19_AAeven_MRT: %s \n",cudaGetErrorString(err));
	}
}

