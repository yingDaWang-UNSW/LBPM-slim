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
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

#define NBLOCKS 1024
#define NTHREADS 256

__global__  void dvc_ScaLBL_Color_Init(char *ID, double *Den, double *Phi, double das, double dbs, int Nx, int Ny, int Nz)
{
	//int i,j,k;
	int n,N;
	char id;
	N = Nx*Ny*Nz;

	int S = N/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n = S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x;
		if (n<N){
		
  		id=ID[n];	
 		//.......Back out the 3-D indices for node n..............
		//k = n/(Nx*Ny);
		//j = (n-Nx*Ny*k)/Nx;
		//i = n-Nx*Ny*k-Nx*j;

		if ( id == 1){
			Den[n] = 1.0;
			Den[N+n] = 0.0;
			Phi[n] = 1.0;
		}
		else if ( id == 2){
			Den[n] = 0.0;
			Den[N+n] = 1.0;
			Phi[n] = -1.0;
		}
		else{
			Den[n] = das;
			Den[N+n] = dbs;
			Phi[n] = (das-dbs)/(das+dbs);
		}
		}
	}
}

__global__  void dvc_ScaLBL_Color_BC(int *list, int *Map, double *Phi, double *Den, double vA, double vB, int count, int Np)
{
	int idx,n,nm;
	// Fill the outlet with component b
	idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < count){
		n = list[idx];
		Den[n] = vA;
		Den[Np+n] = vB;
		
		nm = Map[n];
		Phi[nm] = (vA-vB)/(vA+vB);


	}
}

//*************************************************************************


__global__  void dvc_ScaLBL_SetSlice_z(double *Phi, double value, int Nx, int Ny, int Nz, int Slice)
{
	int n = Slice*Nx*Ny +  blockIdx.x*blockDim.x + threadIdx.x;
	if (n < (Slice+1)*Nx*Ny){
		Phi[n] = value;
	}
}

__global__  void dvc_ScaLBL_CopySlice_z(double *Phi, int Nx, int Ny, int Nz, int Source, int Dest){
	double value;
	int n =  blockIdx.x*blockDim.x + threadIdx.x;
	if (n < Nx*Ny){
		value = Phi[Source*Nx*Ny+n];
		Phi[Dest*Nx*Ny+n] = value;
	}
}

__global__  void dvc_ScaLBL_D3Q19_AAeven_Color(int *Map, double *dist, double *Aq, double *Bq, double *Den, double *Phi,
		double *Velocity, double rhoA, double rhoB, double tauA, double tauB, double alpha, double beta,
		double Fx, double Fy, double Fz, int strideY, int strideZ, int start, int finish, int Np){
	int ijk,nn,n;
	double fq;
	// conserved momemnts
	double rho,jx,jy,jz;
	// non-conserved moments
	double m1,m2,m4,m6,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18;
	double m3,m5,m7;
	double nA,nB; // number density
	double a1,b1,a2,b2,nAB,delta;
	double C,nx,ny,nz; //color gradient magnitude and direction
	double ux,uy,uz;
	double phi,tau,rho0,rlx_setA,rlx_setB;

	const double mrt_V1=0.05263157894736842;
	const double mrt_V2=0.012531328320802;
	const double mrt_V3=0.04761904761904762;
	const double mrt_V4=0.004594820384294068;
	const double mrt_V5=0.01587301587301587;
	const double mrt_V6=0.0555555555555555555555555;
	const double mrt_V7=0.02777777777777778;
	const double mrt_V8=0.08333333333333333;
	const double mrt_V9=0.003341687552213868;
	const double mrt_V10=0.003968253968253968;
	const double mrt_V11=0.01388888888888889;
	const double mrt_V12=0.04166666666666666;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {

			// read the component number densities
			nA = Den[n];
			nB = Den[Np + n];

			// compute phase indicator field
			phi=(nA-nB)/(nA+nB);

			// local density
			rho0=rhoA + 0.5*(1.0-phi)*(rhoB-rhoA);
			// local relaxation time
			tau=tauA + 0.5*(1.0-phi)*(tauB-tauA);
			rlx_setA = 1.f/tau;
			rlx_setB = 8.f*(2.f-rlx_setA)/(8.f-rlx_setA);

			// Get the 1D index based on regular data layout
			ijk = Map[n];
			//					COMPUTE THE COLOR GRADIENT
			//........................................................................
			//.................Read Phase Indicator Values............................
			//........................................................................
			nn = ijk-1;							// neighbor index (get convention)
			m1 = Phi[nn];						// get neighbor for phi - 1
			//........................................................................
			nn = ijk+1;							// neighbor index (get convention)
			m2 = Phi[nn];						// get neighbor for phi - 2
			//........................................................................
			nn = ijk-strideY;							// neighbor index (get convention)
			m3 = Phi[nn];					// get neighbor for phi - 3
			//........................................................................
			nn = ijk+strideY;							// neighbor index (get convention)
			m4 = Phi[nn];					// get neighbor for phi - 4
			//........................................................................
			nn = ijk-strideZ;						// neighbor index (get convention)
			m5 = Phi[nn];					// get neighbor for phi - 5
			//........................................................................
			nn = ijk+strideZ;						// neighbor index (get convention)
			m6 = Phi[nn];					// get neighbor for phi - 6
			//........................................................................
			nn = ijk-strideY-1;						// neighbor index (get convention)
			m7 = Phi[nn];					// get neighbor for phi - 7
			//........................................................................
			nn = ijk+strideY+1;						// neighbor index (get convention)
			m8 = Phi[nn];					// get neighbor for phi - 8
			//........................................................................
			nn = ijk+strideY-1;						// neighbor index (get convention)
			m9 = Phi[nn];					// get neighbor for phi - 9
			//........................................................................
			nn = ijk-strideY+1;						// neighbor index (get convention)
			m10 = Phi[nn];					// get neighbor for phi - 10
			//........................................................................
			nn = ijk-strideZ-1;						// neighbor index (get convention)
			m11 = Phi[nn];					// get neighbor for phi - 11
			//........................................................................
			nn = ijk+strideZ+1;						// neighbor index (get convention)
			m12 = Phi[nn];					// get neighbor for phi - 12
			//........................................................................
			nn = ijk+strideZ-1;						// neighbor index (get convention)
			m13 = Phi[nn];					// get neighbor for phi - 13
			//........................................................................
			nn = ijk-strideZ+1;						// neighbor index (get convention)
			m14 = Phi[nn];					// get neighbor for phi - 14
			//........................................................................
			nn = ijk-strideZ-strideY;					// neighbor index (get convention)
			m15 = Phi[nn];					// get neighbor for phi - 15
			//........................................................................
			nn = ijk+strideZ+strideY;					// neighbor index (get convention)
			m16 = Phi[nn];					// get neighbor for phi - 16
			//........................................................................
			nn = ijk+strideZ-strideY;					// neighbor index (get convention)
			m17 = Phi[nn];					// get neighbor for phi - 17
			//........................................................................
			nn = ijk-strideZ+strideY;					// neighbor index (get convention)
			m18 = Phi[nn];					// get neighbor for phi - 18
			//............Compute the Color Gradient...................................
			nx = -(m1-m2+0.5*(m7-m8+m9-m10+m11-m12+m13-m14));
			ny = -(m3-m4+0.5*(m7-m8-m9+m10+m15-m16+m17-m18));
			nz = -(m5-m6+0.5*(m11-m12-m13+m14+m15-m16-m17+m18));

			//...........Normalize the Color Gradient.................................
			C = sqrt(nx*nx+ny*ny+nz*nz);
			double ColorMag = C;
			if (C==0.0) ColorMag=1.0;
			nx = nx/ColorMag;
			ny = ny/ColorMag;
			nz = nz/ColorMag;		

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

			// f2 = dist[10*Np+n];
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
			//..............carry out relaxation process..............................
			//..........Toelke, Fruediger et. al. 2006................................
			if (C == 0.0)	nx = ny = nz = 0.0;
			m1 = m1 + rlx_setA*((19*(jx*jx+jy*jy+jz*jz)/rho0 - 11*rho) -19*alpha*C - m1);
			m2 = m2 + rlx_setA*((3*rho - 5.5*(jx*jx+jy*jy+jz*jz)/rho0)- m2);
			m4 = m4 + rlx_setB*((-0.6666666666666666*jx)- m4);
			m6 = m6 + rlx_setB*((-0.6666666666666666*jy)- m6);
			m8 = m8 + rlx_setB*((-0.6666666666666666*jz)- m8);
			m9 = m9 + rlx_setA*(((2*jx*jx-jy*jy-jz*jz)/rho0) + 0.5*alpha*C*(2*nx*nx-ny*ny-nz*nz) - m9);
			m10 = m10 + rlx_setA*( - m10);
			m11 = m11 + rlx_setA*(((jy*jy-jz*jz)/rho0) + 0.5*alpha*C*(ny*ny-nz*nz)- m11);
			m12 = m12 + rlx_setA*( - m12);
			m13 = m13 + rlx_setA*( (jx*jy/rho0) + 0.5*alpha*C*nx*ny - m13);
			m14 = m14 + rlx_setA*( (jy*jz/rho0) + 0.5*alpha*C*ny*nz - m14);
			m15 = m15 + rlx_setA*( (jx*jz/rho0) + 0.5*alpha*C*nx*nz - m15);
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
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12+0.25*m13+0.125*(m16-m17) + 0.08333333333*(Fx+Fy);
			dist[7*Np+n] = fq;


			// q = 8
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2-0.1*(jx+jy)-0.025*(m4+m6) +mrt_V7*m9+mrt_V11*m10+mrt_V8*m11
					+mrt_V12*m12+0.25*m13+0.125*(m17-m16) - 0.08333333333*(Fx+Fy);
			dist[8*Np+n] = fq;

			// q = 9
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jx-jy)+0.025*(m4-m6)+
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12-0.25*m13+0.125*(m16+m17) + 0.08333333333*(Fx-Fy);
			dist[9*Np+n] = fq;

			// q = 10
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jy-jx)+0.025*(m6-m4)+
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12-0.25*m13-0.125*(m16+m17)- 0.08333333333*(Fx-Fy);
			dist[10*Np+n] = fq;


			// q = 11
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jx+jz)+0.025*(m4+m8)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12+0.25*m15+0.125*(m18-m16) + 0.08333333333*(Fx+Fz);
			dist[11*Np+n] = fq;

			// q = 12
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2-0.1*(jx+jz)-0.025*(m4+m8)+
					mrt_V7*m9+mrt_V11*m10-mrt_V8*m11-mrt_V12*m12+0.25*m15+0.125*(m16-m18)-0.08333333333*(Fx+Fz);
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

			// write the velocity 
			ux = jx / rho0;
			uy = jy / rho0;
			uz = jz / rho0;
			Velocity[n] = ux;
			Velocity[Np+n] = uy;
			Velocity[2*Np+n] = uz;

			// Instantiate mass transport distributions
			// Stationary value - distribution 0

			nAB = 1.0/(nA+nB);
			Aq[n] = 0.3333333333333333*nA;
			Bq[n] = 0.3333333333333333*nB;

			//...............................................
			// q = 0,2,4
			// Cq = {1,0,0}, {0,1,0}, {0,0,1}
			delta = beta*nA*nB*nAB*0.1111111111111111*nx;
			if (!(nA*nB*nAB>0)) delta=0;
			a1 = nA*(0.1111111111111111*(1+4.5*ux))+delta;
			b1 = nB*(0.1111111111111111*(1+4.5*ux))-delta;
			a2 = nA*(0.1111111111111111*(1-4.5*ux))-delta;
			b2 = nB*(0.1111111111111111*(1-4.5*ux))+delta;

			Aq[1*Np+n] = a1;
			Bq[1*Np+n] = b1;
			Aq[2*Np+n] = a2;
			Bq[2*Np+n] = b2;

			//...............................................
			// q = 2
			// Cq = {0,1,0}
			delta = beta*nA*nB*nAB*0.1111111111111111*ny;
			if (!(nA*nB*nAB>0)) delta=0;
			a1 = nA*(0.1111111111111111*(1+4.5*uy))+delta;
			b1 = nB*(0.1111111111111111*(1+4.5*uy))-delta;
			a2 = nA*(0.1111111111111111*(1-4.5*uy))-delta;
			b2 = nB*(0.1111111111111111*(1-4.5*uy))+delta;

			Aq[3*Np+n] = a1;
			Bq[3*Np+n] = b1;
			Aq[4*Np+n] = a2;
			Bq[4*Np+n] = b2;
			//...............................................
			// q = 4
			// Cq = {0,0,1}
			delta = beta*nA*nB*nAB*0.1111111111111111*nz;
			if (!(nA*nB*nAB>0)) delta=0;
			a1 = nA*(0.1111111111111111*(1+4.5*uz))+delta;
			b1 = nB*(0.1111111111111111*(1+4.5*uz))-delta;
			a2 = nA*(0.1111111111111111*(1-4.5*uz))-delta;
			b2 = nB*(0.1111111111111111*(1-4.5*uz))+delta;

			Aq[5*Np+n] = a1;
			Bq[5*Np+n] = b1;
			Aq[6*Np+n] = a2;
			Bq[6*Np+n] = b2;
			//...............................................

		}
	}
}


__global__ void dvc_ScaLBL_D3Q19_AAodd_Color(int *neighborList, int *Map, double *dist, double *Aq, double *Bq, double *Den,
		 double *Phi, double *Velocity, double rhoA, double rhoB, double tauA, double tauB, double alpha, double beta,
		double Fx, double Fy, double Fz, int strideY, int strideZ, int start, int finish, int Np){

	int n,nn,ijk,nread;
	int nr1,nr2,nr3,nr4,nr5,nr6;
	int nr7,nr8,nr9,nr10;
	int nr11,nr12,nr13,nr14;
	//int nr15,nr16,nr17,nr18;
	double fq;
	// conserved momemnts
	double rho,jx,jy,jz;
	// non-conserved moments
	double m1,m2,m4,m6,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18;
	double m3,m5,m7;
	double nA,nB; // number density
	double a1,b1,a2,b2,nAB,delta;
	double C,nx,ny,nz; //color gradient magnitude and direction
	double ux,uy,uz;
	double phi,tau,rho0,rlx_setA,rlx_setB;

	const double mrt_V1=0.05263157894736842;
	const double mrt_V2=0.012531328320802;
	const double mrt_V3=0.04761904761904762;
	const double mrt_V4=0.004594820384294068;
	const double mrt_V5=0.01587301587301587;
	const double mrt_V6=0.0555555555555555555555555;
	const double mrt_V7=0.02777777777777778;
	const double mrt_V8=0.08333333333333333;
	const double mrt_V9=0.003341687552213868;
	const double mrt_V10=0.003968253968253968;
	const double mrt_V11=0.01388888888888889;
	const double mrt_V12=0.04166666666666666;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
			// read the component number densities
			nA = Den[n];
			nB = Den[Np + n];

			// compute phase indicator field
			phi=(nA-nB)/(nA+nB);

			// local density
			rho0=rhoA + 0.5*(1.0-phi)*(rhoB-rhoA);
			// local relaxation time
			tau=tauA + 0.5*(1.0-phi)*(tauB-tauA);
			rlx_setA = 1.f/tau;
			rlx_setB = 8.f*(2.f-rlx_setA)/(8.f-rlx_setA);
			
			// Get the 1D index based on regular data layout
			ijk = Map[n];
			//					COMPUTE THE COLOR GRADIENT
			//........................................................................
			//.................Read Phase Indicator Values............................
			//........................................................................
			nn = ijk-1;							// neighbor index (get convention)
			m1 = Phi[nn];						// get neighbor for phi - 1
			//........................................................................
			nn = ijk+1;							// neighbor index (get convention)
			m2 = Phi[nn];						// get neighbor for phi - 2
			//........................................................................
			nn = ijk-strideY;							// neighbor index (get convention)
			m3 = Phi[nn];					// get neighbor for phi - 3
			//........................................................................
			nn = ijk+strideY;							// neighbor index (get convention)
			m4 = Phi[nn];					// get neighbor for phi - 4
			//........................................................................
			nn = ijk-strideZ;						// neighbor index (get convention)
			m5 = Phi[nn];					// get neighbor for phi - 5
			//........................................................................
			nn = ijk+strideZ;						// neighbor index (get convention)
			m6 = Phi[nn];					// get neighbor for phi - 6
			//........................................................................
			nn = ijk-strideY-1;						// neighbor index (get convention)
			m7 = Phi[nn];					// get neighbor for phi - 7
			//........................................................................
			nn = ijk+strideY+1;						// neighbor index (get convention)
			m8 = Phi[nn];					// get neighbor for phi - 8
			//........................................................................
			nn = ijk+strideY-1;						// neighbor index (get convention)
			m9 = Phi[nn];					// get neighbor for phi - 9
			//........................................................................
			nn = ijk-strideY+1;						// neighbor index (get convention)
			m10 = Phi[nn];					// get neighbor for phi - 10
			//........................................................................
			nn = ijk-strideZ-1;						// neighbor index (get convention)
			m11 = Phi[nn];					// get neighbor for phi - 11
			//........................................................................
			nn = ijk+strideZ+1;						// neighbor index (get convention)
			m12 = Phi[nn];					// get neighbor for phi - 12
			//........................................................................
			nn = ijk+strideZ-1;						// neighbor index (get convention)
			m13 = Phi[nn];					// get neighbor for phi - 13
			//........................................................................
			nn = ijk-strideZ+1;						// neighbor index (get convention)
			m14 = Phi[nn];					// get neighbor for phi - 14
			//........................................................................
			nn = ijk-strideZ-strideY;					// neighbor index (get convention)
			m15 = Phi[nn];					// get neighbor for phi - 15
			//........................................................................
			nn = ijk+strideZ+strideY;					// neighbor index (get convention)
			m16 = Phi[nn];					// get neighbor for phi - 16
			//........................................................................
			nn = ijk+strideZ-strideY;					// neighbor index (get convention)
			m17 = Phi[nn];					// get neighbor for phi - 17
			//........................................................................
			nn = ijk-strideZ+strideY;					// neighbor index (get convention)
			m18 = Phi[nn];					// get neighbor for phi - 18
			//............Compute the Color Gradient...................................
			nx = -(m1-m2+0.5*(m7-m8+m9-m10+m11-m12+m13-m14));
			ny = -(m3-m4+0.5*(m7-m8-m9+m10+m15-m16+m17-m18));
			nz = -(m5-m6+0.5*(m11-m12-m13+m14+m15-m16-m17+m18));

			//...........Normalize the Color Gradient.................................
			C = sqrt(nx*nx+ny*ny+nz*nz);
			if (C==0.0) C=1.0;
			nx = nx/C;
			ny = ny/C;
			nz = nz/C;		

			// q=0
			fq = dist[n];
			rho = fq;
			m1  = -30.0*fq;
			m2  = 12.0*fq;

			// q=1
			//nread = neighborList[n]; // neighbor 2 
			//fq = dist[nread]; // reading the f1 data into register fq		
			nr1 = neighborList[n]; 
			fq = dist[nr1]; // reading the f1 data into register fq
			rho += fq;
			m1 -= 11.0*fq;
			m2 -= 4.0*fq;
			jx = fq;
			m4 = -4.0*fq;
			m9 = 2.0*fq;
			m10 = -4.0*fq;

			// f2 = dist[10*Np+n];
			//nread = neighborList[n+Np]; // neighbor 1 ( < 10Np => even part of dist)
			//fq = dist[nread];  // reading the f2 data into register fq
			nr2 = neighborList[n+Np]; // neighbor 1 ( < 10Np => even part of dist)
			fq = dist[nr2];  // reading the f2 data into register fq
			rho += fq;
			m1 -= 11.0*(fq);
			m2 -= 4.0*(fq);
			jx -= fq;
			m4 += 4.0*(fq);
			m9 += 2.0*(fq);
			m10 -= 4.0*(fq);

			// q=3
			//nread = neighborList[n+2*Np]; // neighbor 4
			//fq = dist[nread];
			nr3 = neighborList[n+2*Np]; // neighbor 4
			fq = dist[nr3];
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
			//nread = neighborList[n+3*Np]; // neighbor 3
			//fq = dist[nread];
			nr4 = neighborList[n+3*Np]; // neighbor 3
			fq = dist[nr4];
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
			//nread = neighborList[n+4*Np];
			//fq = dist[nread];
			nr5 = neighborList[n+4*Np];
			fq = dist[nr5];
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
			//nread = neighborList[n+5*Np];
			//fq = dist[nread];
			nr6 = neighborList[n+5*Np];
			fq = dist[nr6];
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
			//nread = neighborList[n+6*Np];
			//fq = dist[nread];
			nr7 = neighborList[n+6*Np];
			fq = dist[nr7];
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
			//nread = neighborList[n+7*Np];
			//fq = dist[nread];
			nr8 = neighborList[n+7*Np];
			fq = dist[nr8];
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
			//nread = neighborList[n+8*Np];
			//fq = dist[nread];
			nr9 = neighborList[n+8*Np];
			fq = dist[nr9];
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
			//nread = neighborList[n+9*Np];
			//fq = dist[nread];
			nr10 = neighborList[n+9*Np];
			fq = dist[nr10];
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
			//nread = neighborList[n+10*Np];
			//fq = dist[nread];
			nr11 = neighborList[n+10*Np];
			fq = dist[nr11];
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
			//nread = neighborList[n+11*Np];
			//fq = dist[nread];
			nr12 = neighborList[n+11*Np];
			fq = dist[nr12];
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
			//nread = neighborList[n+12*Np];
			//fq = dist[nread];
			nr13 = neighborList[n+12*Np];
			fq = dist[nr13];
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
			//nread = neighborList[n+13*Np];
			//fq = dist[nread];
			nr14 = neighborList[n+13*Np];
			fq = dist[nr14];
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
			
			//........................................................................
			//..............carry out relaxation process..............................
			//..........Toelke, Fruediger et. al. 2006................................
			if (C == 0.0)	nx = ny = nz = 0.0;
			m1 = m1 + rlx_setA*((19*(jx*jx+jy*jy+jz*jz)/rho0 - 11*rho) -19*alpha*C - m1);
			m2 = m2 + rlx_setA*((3*rho - 5.5*(jx*jx+jy*jy+jz*jz)/rho0)- m2);
			m4 = m4 + rlx_setB*((-0.6666666666666666*jx)- m4);
			m6 = m6 + rlx_setB*((-0.6666666666666666*jy)- m6);
			m8 = m8 + rlx_setB*((-0.6666666666666666*jz)- m8);
			m9 = m9 + rlx_setA*(((2*jx*jx-jy*jy-jz*jz)/rho0) + 0.5*alpha*C*(2*nx*nx-ny*ny-nz*nz) - m9);
			m10 = m10 + rlx_setA*( - m10);
			m11 = m11 + rlx_setA*(((jy*jy-jz*jz)/rho0) + 0.5*alpha*C*(ny*ny-nz*nz)- m11);
			m12 = m12 + rlx_setA*( - m12);
			m13 = m13 + rlx_setA*( (jx*jy/rho0) + 0.5*alpha*C*nx*ny - m13);
			m14 = m14 + rlx_setA*( (jy*jz/rho0) + 0.5*alpha*C*ny*nz - m14);
			m15 = m15 + rlx_setA*( (jx*jz/rho0) + 0.5*alpha*C*nx*nz - m15);
			m16 = m16 + rlx_setB*( - m16);
			m17 = m17 + rlx_setB*( - m17);
			m18 = m18 + rlx_setB*( - m18);
			//.................inverse transformation......................................................

			// q=0
			fq = mrt_V1*rho-mrt_V2*m1+mrt_V3*m2;
			dist[n] = fq;

			// q = 1
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jx-m4)+mrt_V6*(m9-m10)+0.16666666*Fx;
			//nread = neighborList[n+Np];
			dist[nr2] = fq;

			// q=2
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m4-jx)+mrt_V6*(m9-m10) -  0.16666666*Fx;
			//nread = neighborList[n];
			dist[nr1] = fq;

			// q = 3
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jy-m6)+mrt_V7*(m10-m9)+mrt_V8*(m11-m12) + 0.16666666*Fy;
			//nread = neighborList[n+3*Np];
			dist[nr4] = fq;

			// q = 4
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m6-jy)+mrt_V7*(m10-m9)+mrt_V8*(m11-m12) - 0.16666666*Fy;
			//nread = neighborList[n+2*Np];
			dist[nr3] = fq;

			// q = 5
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(jz-m8)+mrt_V7*(m10-m9)+mrt_V8*(m12-m11) + 0.16666666*Fz;
			//nread = neighborList[n+5*Np];
			dist[nr6] = fq;

			// q = 6
			fq = mrt_V1*rho-mrt_V4*m1-mrt_V5*m2+0.1*(m8-jz)+mrt_V7*(m10-m9)+mrt_V8*(m12-m11) - 0.16666666*Fz;
			//nread = neighborList[n+4*Np];
			dist[nr5] = fq;

			// q = 7
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jx+jy)+0.025*(m4+m6)+
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12+0.25*m13+0.125*(m16-m17) + 0.08333333333*(Fx+Fy);
			//nread = neighborList[n+7*Np];
			dist[nr8] = fq;

			// q = 8
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2-0.1*(jx+jy)-0.025*(m4+m6) +mrt_V7*m9+mrt_V11*m10+mrt_V8*m11
					+mrt_V12*m12+0.25*m13+0.125*(m17-m16) - 0.08333333333*(Fx+Fy);
			//nread = neighborList[n+6*Np];
			dist[nr7] = fq;

			// q = 9
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jx-jy)+0.025*(m4-m6)+
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12-0.25*m13+0.125*(m16+m17) + 0.08333333333*(Fx-Fy);
			//nread = neighborList[n+9*Np];
			dist[nr10] = fq;

			// q = 10
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2+0.1*(jy-jx)+0.025*(m6-m4)+
					mrt_V7*m9+mrt_V11*m10+mrt_V8*m11+mrt_V12*m12-0.25*m13-0.125*(m16+m17)- 0.08333333333*(Fx-Fy);
			//nread = neighborList[n+8*Np];
			dist[nr9] = fq;

			// q = 11
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jx+jz)+0.025*(m4+m8)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12+0.25*m15+0.125*(m18-m16) + 0.08333333333*(Fx+Fz);
			//nread = neighborList[n+11*Np];
			dist[nr12] = fq;

			// q = 12
			fq = mrt_V1*rho+mrt_V9*m1+mrt_V10*m2-0.1*(jx+jz)-0.025*(m4+m8)+
					mrt_V7*m9+mrt_V11*m10-mrt_V8*m11-mrt_V12*m12+0.25*m15+0.125*(m16-m18) - 0.08333333333*(Fx+Fz);
			//nread = neighborList[n+10*Np];
			dist[nr11]= fq;

			// q = 13
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jx-jz)+0.025*(m4-m8)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12-0.25*m15-0.125*(m16+m18) + 0.08333333333*(Fx-Fz);
			//nread = neighborList[n+13*Np];
			dist[nr14] = fq;

			// q= 14
			fq = mrt_V1*rho+mrt_V9*m1
					+mrt_V10*m2+0.1*(jz-jx)+0.025*(m8-m4)
					+mrt_V7*m9+mrt_V11*m10-mrt_V8*m11
					-mrt_V12*m12-0.25*m15+0.125*(m16+m18) - 0.08333333333*(Fx-Fz);
			//nread = neighborList[n+12*Np];
			dist[nr13] = fq;


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

			// write the velocity 
			ux = jx / rho0;
			uy = jy / rho0;
			uz = jz / rho0;
			Velocity[n] = ux;
			Velocity[Np+n] = uy;
			Velocity[2*Np+n] = uz;

			// Instantiate mass transport distributions
			// Stationary value - distribution 0
			nAB = 1.0/(nA+nB);
			Aq[n] = 0.3333333333333333*nA;
			Bq[n] = 0.3333333333333333*nB;

			//...............................................
			// q = 0,2,4
			// Cq = {1,0,0}, {0,1,0}, {0,0,1}
			delta = beta*nA*nB*nAB*0.1111111111111111*nx;
			if (!(nA*nB*nAB>0)) delta=0;
			a1 = nA*(0.1111111111111111*(1+4.5*ux))+delta;
			b1 = nB*(0.1111111111111111*(1+4.5*ux))-delta;
			a2 = nA*(0.1111111111111111*(1-4.5*ux))-delta;
			b2 = nB*(0.1111111111111111*(1-4.5*ux))+delta;

			// q = 1
			//nread = neighborList[n+Np];
			Aq[nr2] = a1;
			Bq[nr2] = b1;
			// q=2
			//nread = neighborList[n];
			Aq[nr1] = a2;
			Bq[nr1] = b2;

			//...............................................
			// Cq = {0,1,0}
			delta = beta*nA*nB*nAB*0.1111111111111111*ny;
			if (!(nA*nB*nAB>0)) delta=0;
			a1 = nA*(0.1111111111111111*(1+4.5*uy))+delta;
			b1 = nB*(0.1111111111111111*(1+4.5*uy))-delta;
			a2 = nA*(0.1111111111111111*(1-4.5*uy))-delta;
			b2 = nB*(0.1111111111111111*(1-4.5*uy))+delta;

			// q = 3
			//nread = neighborList[n+3*Np];
			Aq[nr4] = a1;
			Bq[nr4] = b1;
			// q = 4
			//nread = neighborList[n+2*Np];
			Aq[nr3] = a2;
			Bq[nr3] = b2;

			//...............................................
			// q = 4
			// Cq = {0,0,1}
			delta = beta*nA*nB*nAB*0.1111111111111111*nz;
			if (!(nA*nB*nAB>0)) delta=0;
			a1 = nA*(0.1111111111111111*(1+4.5*uz))+delta;
			b1 = nB*(0.1111111111111111*(1+4.5*uz))-delta;
			a2 = nA*(0.1111111111111111*(1-4.5*uz))-delta;
			b2 = nB*(0.1111111111111111*(1-4.5*uz))+delta;

			// q = 5
			//nread = neighborList[n+5*Np];
			Aq[nr6] = a1;
			Bq[nr6] = b1;
			// q = 6
			//nread = neighborList[n+4*Np];
			Aq[nr5] = a2;
			Bq[nr5] = b2;
			//...............................................
		}
	}
}

__global__  void dvc_ScaLBL_D3Q7_AAodd_PhaseField(int *neighborList, int *Map, double *Aq, double *Bq, 
		double *Den, double *Phi, int start, int finish, int Np){
	int idx,n,nread;
	double fq,nA,nB;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
			//..........Compute the number density for each component ............
			// q=0
			fq = Aq[n];
			nA = fq;
			fq = Bq[n];
			nB = fq;
			
			// q=1
			nread = neighborList[n]; 
			fq = Aq[nread];
			nA += fq;
			fq = Bq[nread]; 
			nB += fq;
			
			// q=2
			nread = neighborList[n+Np]; 
			fq = Aq[nread];  
			nA += fq;
			fq = Bq[nread]; 
			nB += fq;
			
			// q=3
			nread = neighborList[n+2*Np]; 
			fq = Aq[nread];
			nA += fq;
			fq = Bq[nread];
			nB += fq;
			
			// q = 4
			nread = neighborList[n+3*Np]; 
			fq = Aq[nread];
			nA += fq;
			fq = Bq[nread];
			nB += fq;

			// q=5
			nread = neighborList[n+4*Np];
			fq = Aq[nread];
			nA += fq;
			fq = Bq[nread];
			nB += fq;
			
			// q = 6
			nread = neighborList[n+5*Np];
			fq = Aq[nread];
			nA += fq;
			fq = Bq[nread];
			nB += fq;

			// save the number densities
			Den[n] = nA;
			Den[Np+n] = nB;

			// save the phase indicator field
			idx = Map[n];
			Phi[idx] = (nA-nB)/(nA+nB); 
		}
	}
}

__global__  void dvc_ScaLBL_D3Q7_AAeven_PhaseField(int *Map, double *Aq, double *Bq, double *Den, double *Phi, 
		int start, int finish, int Np){
	int idx,n;
	double fq,nA,nB;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		n =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (n<finish) {
			// compute number density for each component
			// q=0
			fq = Aq[n];
			nA = fq;
			fq = Bq[n];
			nB = fq;
			
			// q=1
			fq = Aq[2*Np+n];
			nA += fq;
			fq = Bq[2*Np+n];
			nB += fq;

			// q=2
			fq = Aq[1*Np+n];
			nA += fq;
			fq = Bq[1*Np+n];
			nB += fq;

			// q=3
			fq = Aq[4*Np+n];
			nA += fq;
			fq = Bq[4*Np+n];
			nB += fq;

			// q = 4
			fq = Aq[3*Np+n];
			nA += fq;
			fq = Bq[3*Np+n];
			nB += fq;
			
			// q=5
			fq = Aq[6*Np+n];
			nA += fq;
			fq = Bq[6*Np+n];
			nB += fq;
			
			// q = 6
			fq = Aq[5*Np+n];
			nA += fq;
			fq = Bq[5*Np+n];
			nB += fq;

			// save the number densities
			Den[n] = nA;
			Den[Np+n] = nB;

			// save the phase indicator field
			idx = Map[n];
			Phi[idx] = (nA-nB)/(nA+nB); 	
		}
	}
}

__global__ void dvc_ScaLBL_PhaseField_Init(int *Map, double *Phi, double *Den, double *Aq, double *Bq, int start, int finish, int Np){
	int idx,n;
	double phi,nA,nB;

	int S = Np/NBLOCKS/NTHREADS + 1;
	for (int s=0; s<S; s++){
		//........Get 1-D index for this thread....................
		idx =  S*blockIdx.x*blockDim.x + s*blockDim.x + threadIdx.x + start;
		if (idx<finish) {

			n = Map[idx];
			phi = Phi[n];
            if (phi > 1.f){
                    nA = 1.0; nB = 0.f;
            }
            else if (phi < -1.f){
                    nB = 1.0; nA = 0.f;
            }
            else{
                    nA=0.5*(phi+1.f);
                    nB=0.5*(1.f-phi);
            }
			Den[idx] = nA;
			Den[Np+idx] = nB;

			Aq[idx]=0.3333333333333333*nA;
			Aq[Np+idx]=0.1111111111111111*nA;
			Aq[2*Np+idx]=0.1111111111111111*nA;
			Aq[3*Np+idx]=0.1111111111111111*nA;
			Aq[4*Np+idx]=0.1111111111111111*nA;
			Aq[5*Np+idx]=0.1111111111111111*nA;
			Aq[6*Np+idx]=0.1111111111111111*nA;

			Bq[idx]=0.3333333333333333*nB;
			Bq[Np+idx]=0.1111111111111111*nB;
			Bq[2*Np+idx]=0.1111111111111111*nB;
			Bq[3*Np+idx]=0.1111111111111111*nB;
			Bq[4*Np+idx]=0.1111111111111111*nB;
			Bq[5*Np+idx]=0.1111111111111111*nB;
			Bq[6*Np+idx]=0.1111111111111111*nB;
		}
	}
}

extern "C" void ScaLBL_SetSlice_z(double *Phi, double value, int Nx, int Ny, int Nz, int Slice){
	int GRID = Nx*Ny / 512 + 1;
	dvc_ScaLBL_SetSlice_z<<<GRID,512>>>(Phi,value,Nx,Ny,Nz,Slice);
}
extern "C" void ScaLBL_CopySlice_z(double *Phi, int Nx, int Ny, int Nz, int Source, int Dest){
	int GRID = Nx*Ny / 512 + 1;
	dvc_ScaLBL_CopySlice_z<<<GRID,512>>>(Phi,Nx,Ny,Nz,Source,Dest);
}
extern "C" void ScaLBL_Color_BC(int *list, int *Map, double *Phi, double *Den, double vA, double vB, int count, int Np){
    int GRID = count / 512 + 1;
    dvc_ScaLBL_Color_BC<<<GRID,512>>>(list, Map, Phi, Den, vA, vB, count, Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_Color_BC: %s \n",cudaGetErrorString(err));
	}
}
// Pressure Boundary Conditions Functions

extern "C" void ScaLBL_D3Q19_AAeven_Color(int *Map, double *dist, double *Aq, double *Bq, double *Den, double *Phi,
		double *Vel, double rhoA, double rhoB, double tauA, double tauB, double alpha, double beta,
		double Fx, double Fy, double Fz, int strideY, int strideZ, int start, int finish, int Np){

	cudaProfilerStart();
	cudaFuncSetCacheConfig(dvc_ScaLBL_D3Q19_AAeven_Color, cudaFuncCachePreferL1);

	dvc_ScaLBL_D3Q19_AAeven_Color<<<NBLOCKS,NTHREADS >>>(Map, dist, Aq, Bq, Den, Phi, Vel, rhoA, rhoB, tauA, tauB, 
			alpha, beta, Fx, Fy, Fz, strideY, strideZ, start, finish, Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q19_AAeven_Color: %s \n",cudaGetErrorString(err));
	}
	cudaProfilerStop();

}

extern "C" void ScaLBL_D3Q19_AAodd_Color(int *d_neighborList, int *Map, double *dist, double *Aq, double *Bq, double *Den, 
		double *Phi, double *Vel, double rhoA, double rhoB, double tauA, double tauB, double alpha, double beta,
		double Fx, double Fy, double Fz, int strideY, int strideZ, int start, int finish, int Np){

	cudaProfilerStart();
	cudaFuncSetCacheConfig(dvc_ScaLBL_D3Q19_AAodd_Color, cudaFuncCachePreferL1);
	
	dvc_ScaLBL_D3Q19_AAodd_Color<<<NBLOCKS,NTHREADS >>>(d_neighborList, Map, dist, Aq, Bq, Den, Phi, Vel, 
			rhoA, rhoB, tauA, tauB, alpha, beta, Fx, Fy, Fz, strideY, strideZ, start, finish, Np);

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q19_AAodd_Color: %s \n",cudaGetErrorString(err));
	}
	cudaProfilerStop();
}

extern "C" void ScaLBL_D3Q7_AAodd_PhaseField(int *NeighborList, int *Map, double *Aq, double *Bq, 
		double *Den, double *Phi, int start, int finish, int Np){

	cudaProfilerStart();
	dvc_ScaLBL_D3Q7_AAodd_PhaseField<<<NBLOCKS,NTHREADS >>>(NeighborList, Map, Aq, Bq, Den, Phi, start, finish, Np);

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAodd_PhaseField: %s \n",cudaGetErrorString(err));
	}
	cudaProfilerStop();
}

extern "C" void ScaLBL_D3Q7_AAeven_PhaseField(int *Map, double *Aq, double *Bq, double *Den, double *Phi, 
		int start, int finish, int Np){

	cudaProfilerStart();
	dvc_ScaLBL_D3Q7_AAeven_PhaseField<<<NBLOCKS,NTHREADS >>>(Map, Aq, Bq, Den, Phi, start, finish, Np);
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_D3Q7_AAeven_PhaseField: %s \n",cudaGetErrorString(err));
	}
	cudaProfilerStop();

}

extern "C" void ScaLBL_PhaseField_Init(int *Map, double *Phi, double *Den, double *Aq, double *Bq, int start, int finish, int Np){
	dvc_ScaLBL_PhaseField_Init<<<NBLOCKS,NTHREADS >>>(Map, Phi, Den, Aq, Bq, start, finish, Np); 
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err){
		printf("CUDA error in ScaLBL_PhaseField_Init: %s \n",cudaGetErrorString(err));
	}
}


