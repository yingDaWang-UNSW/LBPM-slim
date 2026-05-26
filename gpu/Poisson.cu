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
  Poisson.cu -- D3Q7 lattice Boltzmann Poisson / Laplace kernels.

  Solves   div(epsilon_r grad psi) = -rho_e   on fluid voxels, with Dirichlet
  potential BCs on the +z/-z domain faces and implicit no-flux (bounce-back)
  on solid voxels. Pass a zeroed ChargeDensity array to recover pure Laplace
  -- this is what the tortuosity simulator (lbpm_tortuosity_simulator) does.

  Ported from gaslbm/cuda/Poisson.cu + gaslbm/cuda/D3Q7BC.cu (D3Q7 sections
  only; the D3Q19 variants, charge-coupled bulk variants, slipping-velocity
  BC, and zeta-potential helpers are intentionally omitted).
*/
#include <stdio.h>
#include <math.h>

#define NBLOCKS 1024
#define NTHREADS 256

// -------------------------------------------------------------------------
// Bulk kernels: compute Psi from streamed distributions (no source term)
// -------------------------------------------------------------------------
__global__  void dvc_ScaLBL_D3Q7_AAodd_Poisson_ElectricPotential(int *neighborList,int *Map, double *dist, double *Psi, int start, int finish, int Np){
	int n;
	double psi;//electric potential
	double fq;
	int nread;
    int idx;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
            // q=0
            fq = dist[n];
            psi = fq;
            // q=1
            nread = neighborList[n];
            fq = dist[nread];
            psi += fq;
            // q=2
            nread = neighborList[n+Np];
            fq = dist[nread];
            psi += fq;
            // q=3
            nread = neighborList[n+2*Np];
            fq = dist[nread];
            psi += fq;
            // q = 4
            nread = neighborList[n+3*Np];
            fq = dist[nread];
            psi += fq;
            // q=5
            nread = neighborList[n+4*Np];
            fq = dist[nread];
            psi += fq;
            // q = 6
            nread = neighborList[n+5*Np];
            fq = dist[nread];
            psi += fq;

            idx=Map[n];
            Psi[idx] = psi;
		}
	}
}

__global__  void dvc_ScaLBL_D3Q7_AAeven_Poisson_ElectricPotential(int *Map, double *dist, double *Psi, int start, int finish, int Np){
	int n;
	double psi;
	double fq;
    int idx;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
            fq = dist[n];        psi = fq;
            fq = dist[2*Np+n];   psi += fq;
            fq = dist[1*Np+n];   psi += fq;
            fq = dist[4*Np+n];   psi += fq;
            fq = dist[3*Np+n];   psi += fq;
            fq = dist[6*Np+n];   psi += fq;
            fq = dist[5*Np+n];   psi += fq;
            idx=Map[n];
            Psi[idx] = psi;
		}
	}
}

// -------------------------------------------------------------------------
// Collision kernels: relax distributions toward equilibrium f^eq = w_q*Psi
// plus a charge-density source term. With Den_charge == 0 these reduce to
// the pure Laplace collide step.
// -------------------------------------------------------------------------
__global__  void dvc_ScaLBL_D3Q7_AAodd_Poisson(int *neighborList, int *Map, double *dist, double *Den_charge, double *Psi, double *ElectricField, double tau, double epsilon_LB,bool EnforceElectroneutrality,int start, int finish, int Np){
	int n;
	double psi;
    double Ex,Ey,Ez;
    double rho_e;
	double f0,f1,f2,f3,f4,f5,f6;
	int nr1,nr2,nr3,nr4,nr5,nr6;
    double rlx=1.0/tau;
    int idx;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
            rho_e = (EnforceElectroneutrality==1) ? 0.0 : Den_charge[n] / epsilon_LB;
            idx=Map[n];
            psi = Psi[idx];

            f0 = dist[n];
            nr1 = neighborList[n];        f1 = dist[nr1];
            nr2 = neighborList[n+Np];     f2 = dist[nr2];
            nr3 = neighborList[n+2*Np];   f3 = dist[nr3];
            nr4 = neighborList[n+3*Np];   f4 = dist[nr4];
            nr5 = neighborList[n+4*Np];   f5 = dist[nr5];
            nr6 = neighborList[n+5*Np];   f6 = dist[nr6];

            Ex = (f1-f2)*rlx*4.0;
            Ey = (f3-f4)*rlx*4.0;
            Ez = (f5-f6)*rlx*4.0;
            ElectricField[n+0*Np] = Ex;
            ElectricField[n+1*Np] = Ey;
            ElectricField[n+2*Np] = Ez;

            dist[n]   = f0*(1.0-rlx) + 0.25 *(rlx*psi+rho_e);
            dist[nr2] = f1*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[nr1] = f2*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[nr4] = f3*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[nr3] = f4*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[nr6] = f5*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[nr5] = f6*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
		}
	}
}

__global__  void dvc_ScaLBL_D3Q7_AAeven_Poisson(int *Map, double *dist, double *Den_charge, double *Psi, double *ElectricField, double tau, double epsilon_LB,bool EnforceElectroneutrality,int start, int finish, int Np){
	int n;
	double psi;
    double Ex,Ey,Ez;
    double rho_e;
	double f0,f1,f2,f3,f4,f5,f6;
    double rlx=1.0/tau;
    int idx;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
            rho_e = (EnforceElectroneutrality==1) ? 0.0 : Den_charge[n] / epsilon_LB;
            idx=Map[n];
            psi = Psi[idx];

            f0 = dist[n];
            f1 = dist[2*Np+n];
            f2 = dist[1*Np+n];
            f3 = dist[4*Np+n];
            f4 = dist[3*Np+n];
            f5 = dist[6*Np+n];
            f6 = dist[5*Np+n];

            Ex = (f1-f2)*rlx*4.0;
            Ey = (f3-f4)*rlx*4.0;
            Ez = (f5-f6)*rlx*4.0;
            ElectricField[n+0*Np] = Ex;
            ElectricField[n+1*Np] = Ey;
            ElectricField[n+2*Np] = Ez;

            dist[n]        = f0*(1.0-rlx) + 0.25 *(rlx*psi+rho_e);
            dist[1*Np+n]   = f1*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[2*Np+n]   = f2*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[3*Np+n]   = f3*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[4*Np+n]   = f4*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[5*Np+n]   = f5*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
            dist[6*Np+n]   = f6*(1.0-rlx) + 0.125*(rlx*psi+rho_e);
		}
	}
}

__global__  void dvc_ScaLBL_D3Q7_Poisson_Init(int *Map, double *dist, double *Psi, int start, int finish, int Np){
	int n;
    int ijk;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
            ijk = Map[n];
            dist[0*Np+n] = 0.25*Psi[ijk];
            dist[1*Np+n] = 0.125*Psi[ijk];
            dist[2*Np+n] = 0.125*Psi[ijk];
            dist[3*Np+n] = 0.125*Psi[ijk];
            dist[4*Np+n] = 0.125*Psi[ijk];
            dist[5*Np+n] = 0.125*Psi[ijk];
            dist[6*Np+n] = 0.125*Psi[ijk];
		}
	}
}

// -------------------------------------------------------------------------
// Boundary-condition kernels: Dirichlet potential on inlet (-z, "BC_z") and
// outlet (+z, "BC_Z") faces. The unknown distribution leaving the boundary
// node into the domain is set to enforce sum_q f_q = V_in (or V_out).
// -------------------------------------------------------------------------
__global__ void dvc_ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_z(int *list, double *dist, double Vin, int count, int Np)
{
    int idx,n;
	double f0,f1,f2,f3,f4,f5,f6;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < count){
		n = list[idx];
		f0 = dist[n];
		f1 = dist[2*Np+n];
		f2 = dist[1*Np+n];
		f3 = dist[4*Np+n];
		f4 = dist[3*Np+n];
		f6 = dist[5*Np+n];
		f5 = Vin - (f0+f1+f2+f3+f4+f6);
		dist[6*Np+n] = f5;
	}
}

__global__ void dvc_ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_Z(int *list, double *dist, double Vout, int count, int Np)
{
    int idx,n;
	double f0,f1,f2,f3,f4,f5,f6;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < count){
		n = list[idx];
		f0 = dist[n];
		f1 = dist[2*Np+n];
		f2 = dist[1*Np+n];
		f3 = dist[4*Np+n];
		f4 = dist[3*Np+n];
		f5 = dist[6*Np+n];
		f6 = Vout - (f0+f1+f2+f3+f4+f5);
		dist[5*Np+n] = f6;
	}
}

__global__ void dvc_ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_z(int *d_neighborList, int *list, double *dist, double Vin, int count, int Np)
{
	int idx, n;
    int nread,nr5;
	double f0,f1,f2,f3,f4,f5,f6;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < count){
		n = list[idx];
		f0 = dist[n];
		nread = d_neighborList[n];        f1 = dist[nread];
		nread = d_neighborList[n+2*Np];   f3 = dist[nread];
		nread = d_neighborList[n+Np];     f2 = dist[nread];
		nread = d_neighborList[n+3*Np];   f4 = dist[nread];
		nread = d_neighborList[n+5*Np];   f6 = dist[nread];
		nr5   = d_neighborList[n+4*Np];
		f5 = Vin - (f0+f1+f2+f3+f4+f6);
		dist[nr5] = f5;
	}
}

__global__ void dvc_ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_Z(int *d_neighborList, int *list, double *dist, double Vout, int count, int Np)
{
	int idx, n;
    int nread,nr6;
	double f0,f1,f2,f3,f4,f5,f6;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < count){
		n = list[idx];
		f0 = dist[n];
		nread = d_neighborList[n];        f1 = dist[nread];
		nread = d_neighborList[n+2*Np];   f3 = dist[nread];
		nread = d_neighborList[n+4*Np];   f5 = dist[nread];
		nread = d_neighborList[n+Np];     f2 = dist[nread];
		nread = d_neighborList[n+3*Np];   f4 = dist[nread];
		nr6   = d_neighborList[n+5*Np];
		f6 = Vout - (f0+f1+f2+f3+f4+f5);
		dist[nr6] = f6;
	}
}

__global__ void dvc_ScaLBL_Poisson_D3Q7_BC_z(int *list, int *Map, double *Psi, double Vin, int count)
{
	int idx,n,nm;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < count){
		n = list[idx];
		nm = Map[n];
		Psi[nm] = Vin;
	}
}

__global__ void dvc_ScaLBL_Poisson_D3Q7_BC_Z(int *list, int *Map, double *Psi, double Vout, int count)
{
	int idx,n,nm;
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < count){
		n = list[idx];
		nm = Map[n];
		Psi[nm] = Vout;
	}
}

// =========================================================================
// C wrappers (host)
// =========================================================================
extern "C" void ScaLBL_D3Q7_AAodd_Poisson_ElectricPotential(int *neighborList,int *Map, double *dist, double *Psi, int start, int finish, int Np){
	dvc_ScaLBL_D3Q7_AAodd_Poisson_ElectricPotential<<<NBLOCKS,NTHREADS >>>(neighborList,Map,dist,Psi,start,finish,Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAodd_Poisson_ElectricPotential: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q7_AAeven_Poisson_ElectricPotential(int *Map, double *dist, double *Psi, int start, int finish, int Np){
	dvc_ScaLBL_D3Q7_AAeven_Poisson_ElectricPotential<<<NBLOCKS,NTHREADS >>>(Map,dist,Psi,start,finish,Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAeven_Poisson_ElectricPotential: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q7_AAodd_Poisson(int *neighborList, int *Map, double *dist, double *Den_charge, double *Psi, double *ElectricField, double tau, double epsilon_LB,bool EnforceElectroneutrality,int start, int finish, int Np){
	dvc_ScaLBL_D3Q7_AAodd_Poisson<<<NBLOCKS,NTHREADS >>>(neighborList,Map,dist,Den_charge,Psi,ElectricField,tau,epsilon_LB,EnforceElectroneutrality,start,finish,Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAodd_Poisson: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q7_AAeven_Poisson(int *Map, double *dist, double *Den_charge, double *Psi, double *ElectricField, double tau, double epsilon_LB,bool EnforceElectroneutrality,int start, int finish, int Np){
	dvc_ScaLBL_D3Q7_AAeven_Poisson<<<NBLOCKS,NTHREADS >>>(Map,dist,Den_charge,Psi,ElectricField,tau,epsilon_LB,EnforceElectroneutrality,start,finish,Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAeven_Poisson: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q7_Poisson_Init(int *Map, double *dist, double *Psi, int start, int finish, int Np){
	dvc_ScaLBL_D3Q7_Poisson_Init<<<NBLOCKS,NTHREADS >>>(Map,dist,Psi,start,finish,Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_Poisson_Init: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_z(int *list, double *dist, double Vin, int count, int Np){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_z<<<GRID,512>>>(list, dist, Vin, count, Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_z: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_Z(int *list, double *dist, double Vout, int count, int Np){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_Z<<<GRID,512>>>(list, dist, Vout, count, Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_Z: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_z(int *d_neighborList, int *list, double *dist, double Vin, int count, int Np){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_z<<<GRID,512>>>(d_neighborList, list, dist, Vin, count, Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_z: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_Z(int *d_neighborList, int *list, double *dist, double Vout, int count, int Np){
	int GRID = count / 512 + 1;
	dvc_ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_Z<<<GRID,512>>>(d_neighborList, list, dist, Vout, count, Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_Z: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_Poisson_D3Q7_BC_z(int *list, int *Map, double *Psi, double Vin, int count){
	int GRID = count / 512 + 1;
    dvc_ScaLBL_Poisson_D3Q7_BC_z<<<GRID,512>>>(list, Map, Psi, Vin, count);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_Poisson_D3Q7_BC_z: %s \n",cudaGetErrorString(err));
	}
}

extern "C" void ScaLBL_Poisson_D3Q7_BC_Z(int *list, int *Map, double *Psi, double Vout, int count){
	int GRID = count / 512 + 1;
    dvc_ScaLBL_Poisson_D3Q7_BC_Z<<<GRID,512>>>(list, Map, Psi, Vout, count);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_Poisson_D3Q7_BC_Z: %s \n",cudaGetErrorString(err));
	}
}
