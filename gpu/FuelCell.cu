/*
  FuelCell.cu — GPU kernels for the Fuel Cell model

  D3Q7 weights: w0=1/4, w1..6=1/8, cs^2=1/4
  AA streaming: even reads with anti-direction swap, odd writes with pair swap.

  Kernels:
    1. Shan-Chen pseudopotential force (Carnahan-Starling EoS)
    2. D3Q7-BGK species transport with source terms
    3. D3Q7-BGK Laplace potential solver
    4. D3Q7-BGK thermal transport with source terms
    5. Butler-Volmer reaction source computation
    6. Phase change source computation
*/
#include <stdio.h>
#include <math.h>

#define NBLOCKS 1024
#define NTHREADS 256

// =========================================================================
//  Carnahan-Starling EoS: p = rho*R*T*(1+eta+eta^2-eta^3)/(1-eta)^3 - a*rho^2
// =========================================================================
__device__ double CS_EoS_Pressure(double rho, double a, double b, double T_R)
{
    double eta = b * rho * 0.25;
    double denom = (1.0 - eta);
    double num = 1.0 + eta + eta * eta - eta * eta * eta;
    return rho * T_R * num / (denom * denom * denom) - a * rho * rho;
}

__device__ double SC_EffectiveMass(double rho, double G, double a, double b, double T_R)
{
    double p = CS_EoS_Pressure(rho, a, b, T_R);
    double cs2 = 1.0 / 3.0;
    double arg = 2.0 * (p - rho * cs2) / (G * cs2);
    if (arg < 0.0) arg = 0.0;
    return sqrt(arg);
}

// =========================================================================
//  Kernel 1: Shan-Chen pseudopotential force
//  F_SC = -G * psi(x) * sum_q w_q * psi(x+e_q) * e_q
//  Operates on Cartesian Phi field → stores per-voxel force in ForceX/Y/Z
// =========================================================================
__global__ void dvc_ScaLBL_D3Q19_AAeven_ShanChen_Force(
    int *Map, double *Den, double *Phi,
    double *ForceX, double *ForceY, double *ForceZ,
    double G, double cs_a, double cs_b, double cs_T,
    double rhoA, double rhoB,
    int strideY, int strideZ, int start, int finish, int Np)
{
    int n, ijk;
    double nA, nB, phi, rho_local, psi;
    double Fsc_x, Fsc_y, Fsc_z;
    const double w_face = 1.0 / 18.0;
    const double w_edge = 1.0 / 36.0;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            nA = Den[n];
            nB = Den[Np + n];
            phi = (nA - nB) / (nA + nB + 1.0e-15);
            phi = fmin(1.0, fmax(-1.0, phi));
            rho_local = rhoA + 0.5 * (1.0 - phi) * (rhoB - rhoA);
            ijk = Map[n];
            psi = SC_EffectiveMass(rho_local, G, cs_a, cs_b, cs_T);

            // Read Phi neighbors from Cartesian layout
            double phi_px = Phi[ijk + 1];
            double phi_mx = Phi[ijk - 1];
            double phi_py = Phi[ijk + strideY];
            double phi_my = Phi[ijk - strideY];
            double phi_pz = Phi[ijk + strideZ];
            double phi_mz = Phi[ijk - strideZ];
            double phi_pxpy = Phi[ijk + 1 + strideY];
            double phi_mxmy = Phi[ijk - 1 - strideY];
            double phi_pxmy = Phi[ijk + 1 - strideY];
            double phi_mxpy = Phi[ijk - 1 + strideY];
            double phi_pxpz = Phi[ijk + 1 + strideZ];
            double phi_mxmz = Phi[ijk - 1 - strideZ];
            double phi_pxmz = Phi[ijk + 1 - strideZ];
            double phi_mxpz = Phi[ijk - 1 + strideZ];
            double phi_pypz = Phi[ijk + strideY + strideZ];
            double phi_mymz = Phi[ijk - strideY - strideZ];
            double phi_pymz = Phi[ijk + strideY - strideZ];
            double phi_mypz = Phi[ijk - strideY + strideZ];

            auto phi2rho = [&](double p) -> double {
                return rhoA + 0.5 * (1.0 - p) * (rhoB - rhoA);
            };
            auto psi_of_phi = [&](double p) -> double {
                return SC_EffectiveMass(phi2rho(p), G, cs_a, cs_b, cs_T);
            };

            double psi_px = psi_of_phi(phi_px), psi_mx = psi_of_phi(phi_mx);
            double psi_py = psi_of_phi(phi_py), psi_my = psi_of_phi(phi_my);
            double psi_pz = psi_of_phi(phi_pz), psi_mz = psi_of_phi(phi_mz);
            double psi_pxpy = psi_of_phi(phi_pxpy), psi_mxmy = psi_of_phi(phi_mxmy);
            double psi_pxmy = psi_of_phi(phi_pxmy), psi_mxpy = psi_of_phi(phi_mxpy);
            double psi_pxpz = psi_of_phi(phi_pxpz), psi_mxmz = psi_of_phi(phi_mxmz);
            double psi_pxmz = psi_of_phi(phi_pxmz), psi_mxpz = psi_of_phi(phi_mxpz);
            double psi_pypz = psi_of_phi(phi_pypz), psi_mymz = psi_of_phi(phi_mymz);
            double psi_pymz = psi_of_phi(phi_pymz), psi_mypz = psi_of_phi(phi_mypz);

            Fsc_x = -G * psi * (
                w_face * (psi_px - psi_mx) +
                w_edge * (psi_pxpy - psi_mxmy + psi_pxmy - psi_mxpy +
                          psi_pxpz - psi_mxmz + psi_pxmz - psi_mxpz));
            Fsc_y = -G * psi * (
                w_face * (psi_py - psi_my) +
                w_edge * (psi_pxpy - psi_mxmy - psi_pxmy + psi_mxpy +
                          psi_pypz - psi_mymz + psi_pymz - psi_mypz));
            Fsc_z = -G * psi * (
                w_face * (psi_pz - psi_mz) +
                w_edge * (psi_pxpz - psi_mxmz - psi_pxmz + psi_mxpz +
                          psi_pypz - psi_mymz - psi_pymz + psi_mypz));

            ForceX[n] = Fsc_x;
            ForceY[n] = Fsc_y;
            ForceZ[n] = Fsc_z;
        }
    }
}

__global__ void dvc_ScaLBL_D3Q19_AAodd_ShanChen_Force(
    int *neighborList, int *Map, double *Den, double *Phi,
    double *ForceX, double *ForceY, double *ForceZ,
    double G, double cs_a, double cs_b, double cs_T,
    double rhoA, double rhoB,
    int strideY, int strideZ, int start, int finish, int Np)
{
    int n, ijk;
    double nA, nB, phi, rho_local, psi;
    double Fsc_x, Fsc_y, Fsc_z;
    const double w_face = 1.0 / 18.0;
    const double w_edge = 1.0 / 36.0;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            nA = Den[n];
            nB = Den[Np + n];
            phi = (nA - nB) / (nA + nB + 1.0e-15);
            phi = fmin(1.0, fmax(-1.0, phi));
            rho_local = rhoA + 0.5 * (1.0 - phi) * (rhoB - rhoA);
            ijk = Map[n];
            psi = SC_EffectiveMass(rho_local, G, cs_a, cs_b, cs_T);

            double phi_px = Phi[ijk + 1];
            double phi_mx = Phi[ijk - 1];
            double phi_py = Phi[ijk + strideY];
            double phi_my = Phi[ijk - strideY];
            double phi_pz = Phi[ijk + strideZ];
            double phi_mz = Phi[ijk - strideZ];
            double phi_pxpy = Phi[ijk + 1 + strideY];
            double phi_mxmy = Phi[ijk - 1 - strideY];
            double phi_pxmy = Phi[ijk + 1 - strideY];
            double phi_mxpy = Phi[ijk - 1 + strideY];
            double phi_pxpz = Phi[ijk + 1 + strideZ];
            double phi_mxmz = Phi[ijk - 1 - strideZ];
            double phi_pxmz = Phi[ijk + 1 - strideZ];
            double phi_mxpz = Phi[ijk - 1 + strideZ];
            double phi_pypz = Phi[ijk + strideY + strideZ];
            double phi_mymz = Phi[ijk - strideY - strideZ];
            double phi_pymz = Phi[ijk + strideY - strideZ];
            double phi_mypz = Phi[ijk - strideY + strideZ];

            auto phi2rho = [&](double p) -> double {
                return rhoA + 0.5 * (1.0 - p) * (rhoB - rhoA);
            };
            auto psi_of_phi = [&](double p) -> double {
                return SC_EffectiveMass(phi2rho(p), G, cs_a, cs_b, cs_T);
            };

            double psi_px = psi_of_phi(phi_px), psi_mx = psi_of_phi(phi_mx);
            double psi_py = psi_of_phi(phi_py), psi_my = psi_of_phi(phi_my);
            double psi_pz = psi_of_phi(phi_pz), psi_mz = psi_of_phi(phi_mz);
            double psi_pxpy = psi_of_phi(phi_pxpy), psi_mxmy = psi_of_phi(phi_mxmy);
            double psi_pxmy = psi_of_phi(phi_pxmy), psi_mxpy = psi_of_phi(phi_mxpy);
            double psi_pxpz = psi_of_phi(phi_pxpz), psi_mxmz = psi_of_phi(phi_mxmz);
            double psi_pxmz = psi_of_phi(phi_pxmz), psi_mxpz = psi_of_phi(phi_mxpz);
            double psi_pypz = psi_of_phi(phi_pypz), psi_mymz = psi_of_phi(phi_mymz);
            double psi_pymz = psi_of_phi(phi_pymz), psi_mypz = psi_of_phi(phi_mypz);

            Fsc_x = -G * psi * (
                w_face * (psi_px - psi_mx) +
                w_edge * (psi_pxpy - psi_mxmy + psi_pxmy - psi_mxpy +
                          psi_pxpz - psi_mxmz + psi_pxmz - psi_mxpz));
            Fsc_y = -G * psi * (
                w_face * (psi_py - psi_my) +
                w_edge * (psi_pxpy - psi_mxmy - psi_pxmy + psi_mxpy +
                          psi_pypz - psi_mymz + psi_pymz - psi_mypz));
            Fsc_z = -G * psi * (
                w_face * (psi_pz - psi_mz) +
                w_edge * (psi_pxpz - psi_mxmz - psi_pxmz + psi_mxpz +
                          psi_pypz - psi_mymz - psi_pymz + psi_mypz));

            ForceX[n] = Fsc_x;
            ForceY[n] = Fsc_y;
            ForceZ[n] = Fsc_z;
        }
    }
}


// =========================================================================
//  Kernel 2: D3Q7 Species Transport (advection-diffusion + source)
//  Weights: w0=1/4, w1..6=1/8  →  cs^2=1/4  →  coeff=4.0
//  tau_sp = 0.5 + 4.0 * D
// =========================================================================
__global__ void dvc_ScaLBL_D3Q7_AAeven_FuelCell_Species(
    int *Map, double *Cq, double *Conc, double *Velocity,
    double *Phi, double *Poros, double *DiffCoeff, double *SourceTerm,
    double tau_sp, int start, int finish, int Np)
{
    int n;
    double f0, f1, f2, f3, f4, f5, f6;
    double C, ux, uy, uz, porosity, diff_eff, source, phi;
    const double w0 = 0.25;
    const double w1 = 0.125;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            porosity = Poros[n];
            diff_eff = DiffCoeff[n];
            source = SourceTerm[n];
            ux = Velocity[n];
            uy = Velocity[Np + n];
            uz = Velocity[2 * Np + n];
            phi = Phi[Map[n]];
            double gas_mask = (phi < 0.5) ? 1.0 : 0.0;

            // Even step: read with anti-direction swap
            f0 = Cq[n];
            f1 = Cq[2 * Np + n];   // +x reads from -x slot
            f2 = Cq[1 * Np + n];   // -x reads from +x slot
            f3 = Cq[4 * Np + n];   // +y reads from -y slot
            f4 = Cq[3 * Np + n];   // -y reads from +y slot
            f5 = Cq[6 * Np + n];   // +z reads from -z slot
            f6 = Cq[5 * Np + n];   // -z reads from +z slot

            C = f0 + f1 + f2 + f3 + f4 + f5 + f6;

            double tau_eff = 0.5 + (tau_sp - 0.5) * porosity * diff_eff;
            if (tau_eff < 0.501) tau_eff = 0.501;
            double rlx = 1.0 / tau_eff;

            // BGK with f_eq = w*C*(1 + 4*e·u)
            f0 = f0 * (1.0 - rlx) + rlx * w0 * C + w0 * source * gas_mask;
            f1 = f1 * (1.0 - rlx) + rlx * w1 * C * (1.0 + 4.0 * ux) + w1 * source * gas_mask;
            f2 = f2 * (1.0 - rlx) + rlx * w1 * C * (1.0 - 4.0 * ux) + w1 * source * gas_mask;
            f3 = f3 * (1.0 - rlx) + rlx * w1 * C * (1.0 + 4.0 * uy) + w1 * source * gas_mask;
            f4 = f4 * (1.0 - rlx) + rlx * w1 * C * (1.0 - 4.0 * uy) + w1 * source * gas_mask;
            f5 = f5 * (1.0 - rlx) + rlx * w1 * C * (1.0 + 4.0 * uz) + w1 * source * gas_mask;
            f6 = f6 * (1.0 - rlx) + rlx * w1 * C * (1.0 - 4.0 * uz) + w1 * source * gas_mask;

            // Write at natural positions
            Cq[n] = f0;
            Cq[1 * Np + n] = f1;
            Cq[2 * Np + n] = f2;
            Cq[3 * Np + n] = f3;
            Cq[4 * Np + n] = f4;
            Cq[5 * Np + n] = f5;
            Cq[6 * Np + n] = f6;

            Conc[n] = C;
        }
    }
}

__global__ void dvc_ScaLBL_D3Q7_AAodd_FuelCell_Species(
    int *neighborList, int *Map, double *Cq, double *Conc, double *Velocity,
    double *Phi, double *Poros, double *DiffCoeff, double *SourceTerm,
    double tau_sp, int start, int finish, int Np)
{
    int n, nr1, nr2, nr3, nr4, nr5, nr6;
    double f0, f1, f2, f3, f4, f5, f6;
    double C, ux, uy, uz, porosity, diff_eff, source, phi;
    const double w0 = 0.25;
    const double w1 = 0.125;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            porosity = Poros[n];
            diff_eff = DiffCoeff[n];
            source = SourceTerm[n];
            ux = Velocity[n];
            uy = Velocity[Np + n];
            uz = Velocity[2 * Np + n];
            phi = Phi[Map[n]];
            double gas_mask = (phi < 0.5) ? 1.0 : 0.0;

            // Read neighbor addresses
            nr1 = neighborList[n];
            nr2 = neighborList[n + Np];
            nr3 = neighborList[n + 2 * Np];
            nr4 = neighborList[n + 3 * Np];
            nr5 = neighborList[n + 4 * Np];
            nr6 = neighborList[n + 5 * Np];

            // Odd step: read from neighbor positions
            f0 = Cq[n];
            f1 = Cq[nr1];
            f2 = Cq[nr2];
            f3 = Cq[nr3];
            f4 = Cq[nr4];
            f5 = Cq[nr5];
            f6 = Cq[nr6];

            C = f0 + f1 + f2 + f3 + f4 + f5 + f6;

            double tau_eff = 0.5 + (tau_sp - 0.5) * porosity * diff_eff;
            if (tau_eff < 0.501) tau_eff = 0.501;
            double rlx = 1.0 / tau_eff;

            f0 = f0 * (1.0 - rlx) + rlx * w0 * C + w0 * source * gas_mask;
            f1 = f1 * (1.0 - rlx) + rlx * w1 * C * (1.0 + 4.0 * ux) + w1 * source * gas_mask;
            f2 = f2 * (1.0 - rlx) + rlx * w1 * C * (1.0 - 4.0 * ux) + w1 * source * gas_mask;
            f3 = f3 * (1.0 - rlx) + rlx * w1 * C * (1.0 + 4.0 * uy) + w1 * source * gas_mask;
            f4 = f4 * (1.0 - rlx) + rlx * w1 * C * (1.0 - 4.0 * uy) + w1 * source * gas_mask;
            f5 = f5 * (1.0 - rlx) + rlx * w1 * C * (1.0 + 4.0 * uz) + w1 * source * gas_mask;
            f6 = f6 * (1.0 - rlx) + rlx * w1 * C * (1.0 - 4.0 * uz) + w1 * source * gas_mask;

            // Odd write: swap direction pairs
            Cq[n]   = f0;
            Cq[nr2] = f1;   // +x → nread for -x
            Cq[nr1] = f2;   // -x → nread for +x
            Cq[nr4] = f3;   // +y → nread for -y
            Cq[nr3] = f4;   // -y → nread for +y
            Cq[nr6] = f5;   // +z → nread for -z
            Cq[nr5] = f6;   // -z → nread for +z

            Conc[n] = C;
        }
    }
}


// =========================================================================
//  Kernel 3: D3Q7 Laplace Potential Solver (steady-state diffusion)
//  tau_pot = 0.5 + 4.0*sigma
// =========================================================================
__global__ void dvc_ScaLBL_D3Q7_AAeven_FuelCell_Potential(
    int *Map, double *Pq, double *PotentialField, double *Conductivity,
    int start, int finish, int Np)
{
    int n;
    double f0, f1, f2, f3, f4, f5, f6;
    double pot, sigma_local;
    const double w0 = 0.25;
    const double w1 = 0.125;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            sigma_local = Conductivity[n];
            double tau_pot = 0.5 + 4.0 * sigma_local;
            if (tau_pot < 0.501) tau_pot = 0.501;
            double rlx = 1.0 / tau_pot;

            // Even: anti-direction swap read
            f0 = Pq[n];
            f1 = Pq[2 * Np + n];
            f2 = Pq[1 * Np + n];
            f3 = Pq[4 * Np + n];
            f4 = Pq[3 * Np + n];
            f5 = Pq[6 * Np + n];
            f6 = Pq[5 * Np + n];

            pot = f0 + f1 + f2 + f3 + f4 + f5 + f6;

            // BGK (no advection)
            f0 = f0 * (1.0 - rlx) + rlx * w0 * pot;
            f1 = f1 * (1.0 - rlx) + rlx * w1 * pot;
            f2 = f2 * (1.0 - rlx) + rlx * w1 * pot;
            f3 = f3 * (1.0 - rlx) + rlx * w1 * pot;
            f4 = f4 * (1.0 - rlx) + rlx * w1 * pot;
            f5 = f5 * (1.0 - rlx) + rlx * w1 * pot;
            f6 = f6 * (1.0 - rlx) + rlx * w1 * pot;

            // Write at natural positions
            Pq[n] = f0;
            Pq[1 * Np + n] = f1;
            Pq[2 * Np + n] = f2;
            Pq[3 * Np + n] = f3;
            Pq[4 * Np + n] = f4;
            Pq[5 * Np + n] = f5;
            Pq[6 * Np + n] = f6;

            PotentialField[n] = pot;
        }
    }
}

__global__ void dvc_ScaLBL_D3Q7_AAodd_FuelCell_Potential(
    int *neighborList, int *Map, double *Pq, double *PotentialField,
    double *Conductivity, int start, int finish, int Np)
{
    int n, nr1, nr2, nr3, nr4, nr5, nr6;
    double f0, f1, f2, f3, f4, f5, f6;
    double pot, sigma_local;
    const double w0 = 0.25;
    const double w1 = 0.125;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            sigma_local = Conductivity[n];
            double tau_pot = 0.5 + 4.0 * sigma_local;
            if (tau_pot < 0.501) tau_pot = 0.501;
            double rlx = 1.0 / tau_pot;

            nr1 = neighborList[n];
            nr2 = neighborList[n + Np];
            nr3 = neighborList[n + 2 * Np];
            nr4 = neighborList[n + 3 * Np];
            nr5 = neighborList[n + 4 * Np];
            nr6 = neighborList[n + 5 * Np];

            f0 = Pq[n];
            f1 = Pq[nr1];
            f2 = Pq[nr2];
            f3 = Pq[nr3];
            f4 = Pq[nr4];
            f5 = Pq[nr5];
            f6 = Pq[nr6];

            pot = f0 + f1 + f2 + f3 + f4 + f5 + f6;

            f0 = f0 * (1.0 - rlx) + rlx * w0 * pot;
            f1 = f1 * (1.0 - rlx) + rlx * w1 * pot;
            f2 = f2 * (1.0 - rlx) + rlx * w1 * pot;
            f3 = f3 * (1.0 - rlx) + rlx * w1 * pot;
            f4 = f4 * (1.0 - rlx) + rlx * w1 * pot;
            f5 = f5 * (1.0 - rlx) + rlx * w1 * pot;
            f6 = f6 * (1.0 - rlx) + rlx * w1 * pot;

            // Swap pairs for odd write
            Pq[n]   = f0;
            Pq[nr2] = f1;
            Pq[nr1] = f2;
            Pq[nr4] = f3;
            Pq[nr3] = f4;
            Pq[nr6] = f5;
            Pq[nr5] = f6;

            PotentialField[n] = pot;
        }
    }
}


// =========================================================================
//  Kernel 4: D3Q7 Thermal Transport (advection-diffusion + source)
//  tau_eff = 0.5 + 4.0 * k_thermal
// =========================================================================
__global__ void dvc_ScaLBL_D3Q7_AAeven_FuelCell_Thermal(
    int *Map, double *Tq, double *Temperature, double *Velocity,
    double *ThermalCond, double *SourceThermal,
    double tau_thermal, int start, int finish, int Np)
{
    int n;
    double f0, f1, f2, f3, f4, f5, f6;
    double T, ux, uy, uz, kth, source;
    const double w0 = 0.25;
    const double w1 = 0.125;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            kth = ThermalCond[n];
            source = SourceThermal[n];
            ux = Velocity[n];
            uy = Velocity[Np + n];
            uz = Velocity[2 * Np + n];

            double tau_eff = 0.5 + 4.0 * kth;
            if (tau_eff < 0.501) tau_eff = 0.501;
            double rlx = 1.0 / tau_eff;

            // Even: anti-direction swap read
            f0 = Tq[n];
            f1 = Tq[2 * Np + n];
            f2 = Tq[1 * Np + n];
            f3 = Tq[4 * Np + n];
            f4 = Tq[3 * Np + n];
            f5 = Tq[6 * Np + n];
            f6 = Tq[5 * Np + n];

            T = f0 + f1 + f2 + f3 + f4 + f5 + f6;

            f0 = f0 * (1.0 - rlx) + rlx * w0 * T + w0 * source;
            f1 = f1 * (1.0 - rlx) + rlx * w1 * T * (1.0 + 4.0 * ux) + w1 * source;
            f2 = f2 * (1.0 - rlx) + rlx * w1 * T * (1.0 - 4.0 * ux) + w1 * source;
            f3 = f3 * (1.0 - rlx) + rlx * w1 * T * (1.0 + 4.0 * uy) + w1 * source;
            f4 = f4 * (1.0 - rlx) + rlx * w1 * T * (1.0 - 4.0 * uy) + w1 * source;
            f5 = f5 * (1.0 - rlx) + rlx * w1 * T * (1.0 + 4.0 * uz) + w1 * source;
            f6 = f6 * (1.0 - rlx) + rlx * w1 * T * (1.0 - 4.0 * uz) + w1 * source;

            Tq[n] = f0;
            Tq[1 * Np + n] = f1;
            Tq[2 * Np + n] = f2;
            Tq[3 * Np + n] = f3;
            Tq[4 * Np + n] = f4;
            Tq[5 * Np + n] = f5;
            Tq[6 * Np + n] = f6;

            Temperature[n] = T;
        }
    }
}

__global__ void dvc_ScaLBL_D3Q7_AAodd_FuelCell_Thermal(
    int *neighborList, int *Map, double *Tq, double *Temperature,
    double *Velocity, double *ThermalCond, double *SourceThermal,
    double tau_thermal, int start, int finish, int Np)
{
    int n, nr1, nr2, nr3, nr4, nr5, nr6;
    double f0, f1, f2, f3, f4, f5, f6;
    double T, ux, uy, uz, kth, source;
    const double w0 = 0.25;
    const double w1 = 0.125;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            kth = ThermalCond[n];
            source = SourceThermal[n];
            ux = Velocity[n];
            uy = Velocity[Np + n];
            uz = Velocity[2 * Np + n];

            double tau_eff = 0.5 + 4.0 * kth;
            if (tau_eff < 0.501) tau_eff = 0.501;
            double rlx = 1.0 / tau_eff;

            nr1 = neighborList[n];
            nr2 = neighborList[n + Np];
            nr3 = neighborList[n + 2 * Np];
            nr4 = neighborList[n + 3 * Np];
            nr5 = neighborList[n + 4 * Np];
            nr6 = neighborList[n + 5 * Np];

            f0 = Tq[n];
            f1 = Tq[nr1];
            f2 = Tq[nr2];
            f3 = Tq[nr3];
            f4 = Tq[nr4];
            f5 = Tq[nr5];
            f6 = Tq[nr6];

            T = f0 + f1 + f2 + f3 + f4 + f5 + f6;

            f0 = f0 * (1.0 - rlx) + rlx * w0 * T + w0 * source;
            f1 = f1 * (1.0 - rlx) + rlx * w1 * T * (1.0 + 4.0 * ux) + w1 * source;
            f2 = f2 * (1.0 - rlx) + rlx * w1 * T * (1.0 - 4.0 * ux) + w1 * source;
            f3 = f3 * (1.0 - rlx) + rlx * w1 * T * (1.0 + 4.0 * uy) + w1 * source;
            f4 = f4 * (1.0 - rlx) + rlx * w1 * T * (1.0 - 4.0 * uy) + w1 * source;
            f5 = f5 * (1.0 - rlx) + rlx * w1 * T * (1.0 + 4.0 * uz) + w1 * source;
            f6 = f6 * (1.0 - rlx) + rlx * w1 * T * (1.0 - 4.0 * uz) + w1 * source;

            Tq[n]   = f0;
            Tq[nr2] = f1;
            Tq[nr1] = f2;
            Tq[nr4] = f3;
            Tq[nr3] = f4;
            Tq[nr6] = f5;
            Tq[nr5] = f6;

            Temperature[n] = T;
        }
    }
}


// =========================================================================
//  Kernel 5: Butler-Volmer reaction source terms at CL voxels
// =========================================================================
__global__ void dvc_ScaLBL_FuelCell_ButlerVolmer(
    double *PhiS, double *PhiE, double *Temperature,
    double *RegionID, double *ReactionRate,
    double *SourceO2, double *SourceN2, double *SourceH2, double *SourceH2O,
    double i0_cathode, double i0_anode,
    double alpha_a_c, double alpha_c_c,
    double alpha_a_a, double alpha_c_a,
    double E_eq, double F_const, double R_gas, double T_ref,
    int start, int finish, int Np)
{
    int n;
    double phiS, phiE, T_local, T_dim, eta, j, i0;
    double region;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            region = RegionID[n];
            phiS = PhiS[n];
            phiE = PhiE[n];
            T_local = Temperature[n];
            if (T_local < 0.1) T_local = 1.0;
            T_dim = T_local * T_ref;  // convert to dimensional [K]

            if (fabs(region - 3.0) < 0.5) {
                // Cathode ORR: O2 + 4H+ + 4e- -> 2H2O
                eta = phiS - phiE - E_eq;
                i0 = i0_cathode;
                j = i0 * (exp(alpha_a_c * F_const * eta / (R_gas * T_dim))
                        - exp(-alpha_c_c * F_const * eta / (R_gas * T_dim)));
                j = fmin(fmax(j, -1.0e4), 1.0e4);

                ReactionRate[n] = j;
                // j<0 cathodic: O2 consumed (negative source), H2O produced (positive)
                SourceO2[n]  = j / (4.0 * F_const);
                SourceN2[n]  = 0.0;
                SourceH2[n]  = 0.0;
                SourceH2O[n] = -j / (2.0 * F_const);
            } else {
                ReactionRate[n] = 0.0;
                SourceO2[n]  = 0.0;
                SourceN2[n]  = 0.0;
                SourceH2[n]  = 0.0;
                SourceH2O[n] = 0.0;
            }
        }
    }
}


// =========================================================================
//  Kernel 6: Phase change sources (Clausius-Clapeyron saturation)
// =========================================================================
__global__ void dvc_ScaLBL_FuelCell_PhaseChange(
    double *Phi, double *Temperature, double *ConcH2O,
    double *SourcePhaseField, double *SourceThermal,
    double h_fg, double R_gas, double T_ref, double P_ref,
    double k_evap, double k_cond,
    int *Map, int start, int finish, int Np)
{
    int n, ijk;
    double T, c_h2o, phi, P_h2o, P_sat, dm;
    const double M_H2O = 0.018;

    int S = Np / NBLOCKS / NTHREADS + 1;
    for (int s = 0; s < S; s++) {
        n = S * blockIdx.x * blockDim.x + s * blockDim.x + threadIdx.x + start;
        if (n < finish) {
            ijk = Map[n];
            T = Temperature[n];
            if (T < 0.1) T = 1.0;
            c_h2o = ConcH2O[n];
            phi = Phi[ijk];

            P_h2o = c_h2o / 3.0;
            P_sat = P_ref * exp(h_fg * M_H2O / R_gas * (1.0 / T_ref - 1.0 / T));

            if (P_h2o > P_sat && phi < 0.3) {
                dm = k_cond * (P_h2o - P_sat);
            } else if (P_h2o < P_sat && phi > -0.3) {
                dm = -k_evap * (P_sat - P_h2o);
            } else {
                dm = 0.0;
            }

            SourcePhaseField[n] = dm;
            SourceThermal[n] = -dm * h_fg;
        }
    }
}


// =========================================================================
//  Extern "C" wrappers
// =========================================================================

extern "C" void ScaLBL_D3Q19_AAeven_ShanChen_Force(
    int *Map, double *Den, double *Phi,
    double *ForceX, double *ForceY, double *ForceZ,
    double G, double cs_a, double cs_b, double cs_T,
    double rhoA, double rhoB,
    int strideY, int strideZ, int start, int finish, int Np)
{
    dvc_ScaLBL_D3Q19_AAeven_ShanChen_Force<<<NBLOCKS, NTHREADS>>>(
        Map, Den, Phi, ForceX, ForceY, ForceZ,
        G, cs_a, cs_b, cs_T, rhoA, rhoB,
        strideY, strideZ, start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in AAeven_ShanChen_Force: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_D3Q19_AAodd_ShanChen_Force(
    int *neighborList, int *Map, double *Den, double *Phi,
    double *ForceX, double *ForceY, double *ForceZ,
    double G, double cs_a, double cs_b, double cs_T,
    double rhoA, double rhoB,
    int strideY, int strideZ, int start, int finish, int Np)
{
    dvc_ScaLBL_D3Q19_AAodd_ShanChen_Force<<<NBLOCKS, NTHREADS>>>(
        neighborList, Map, Den, Phi, ForceX, ForceY, ForceZ,
        G, cs_a, cs_b, cs_T, rhoA, rhoB,
        strideY, strideZ, start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in AAodd_ShanChen_Force: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_D3Q7_AAeven_FuelCell_Species(
    int *Map, double *Cq, double *Conc, double *Velocity,
    double *Phi, double *Poros, double *DiffCoeff, double *SourceTerm,
    double tau_sp, int start, int finish, int Np)
{
    dvc_ScaLBL_D3Q7_AAeven_FuelCell_Species<<<NBLOCKS, NTHREADS>>>(
        Map, Cq, Conc, Velocity, Phi, Poros, DiffCoeff, SourceTerm,
        tau_sp, start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in AAeven_FuelCell_Species: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_D3Q7_AAodd_FuelCell_Species(
    int *neighborList, int *Map, double *Cq, double *Conc, double *Velocity,
    double *Phi, double *Poros, double *DiffCoeff, double *SourceTerm,
    double tau_sp, int start, int finish, int Np)
{
    dvc_ScaLBL_D3Q7_AAodd_FuelCell_Species<<<NBLOCKS, NTHREADS>>>(
        neighborList, Map, Cq, Conc, Velocity, Phi, Poros, DiffCoeff, SourceTerm,
        tau_sp, start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in AAodd_FuelCell_Species: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_D3Q7_AAeven_FuelCell_Potential(
    int *Map, double *Pq, double *PotentialField, double *Conductivity,
    int start, int finish, int Np)
{
    dvc_ScaLBL_D3Q7_AAeven_FuelCell_Potential<<<NBLOCKS, NTHREADS>>>(
        Map, Pq, PotentialField, Conductivity, start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in AAeven_FuelCell_Potential: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_D3Q7_AAodd_FuelCell_Potential(
    int *neighborList, int *Map, double *Pq, double *PotentialField,
    double *Conductivity, int start, int finish, int Np)
{
    dvc_ScaLBL_D3Q7_AAodd_FuelCell_Potential<<<NBLOCKS, NTHREADS>>>(
        neighborList, Map, Pq, PotentialField, Conductivity,
        start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in AAodd_FuelCell_Potential: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_D3Q7_AAeven_FuelCell_Thermal(
    int *Map, double *Tq, double *Temperature, double *Velocity,
    double *ThermalCond, double *SourceThermal,
    double tau_thermal, int start, int finish, int Np)
{
    dvc_ScaLBL_D3Q7_AAeven_FuelCell_Thermal<<<NBLOCKS, NTHREADS>>>(
        Map, Tq, Temperature, Velocity, ThermalCond, SourceThermal,
        tau_thermal, start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in AAeven_FuelCell_Thermal: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_D3Q7_AAodd_FuelCell_Thermal(
    int *neighborList, int *Map, double *Tq, double *Temperature,
    double *Velocity, double *ThermalCond, double *SourceThermal,
    double tau_thermal, int start, int finish, int Np)
{
    dvc_ScaLBL_D3Q7_AAodd_FuelCell_Thermal<<<NBLOCKS, NTHREADS>>>(
        neighborList, Map, Tq, Temperature, Velocity,
        ThermalCond, SourceThermal, tau_thermal, start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in AAodd_FuelCell_Thermal: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_FuelCell_ButlerVolmer(
    double *PhiS, double *PhiE, double *Temperature,
    double *RegionID, double *ReactionRate,
    double *SourceO2, double *SourceN2, double *SourceH2, double *SourceH2O,
    double i0_cathode, double i0_anode,
    double alpha_a_c, double alpha_c_c,
    double alpha_a_a, double alpha_c_a,
    double E_eq, double F_const, double R_gas, double T_ref,
    int start, int finish, int Np)
{
    dvc_ScaLBL_FuelCell_ButlerVolmer<<<NBLOCKS, NTHREADS>>>(
        PhiS, PhiE, Temperature, RegionID, ReactionRate,
        SourceO2, SourceN2, SourceH2, SourceH2O,
        i0_cathode, i0_anode,
        alpha_a_c, alpha_c_c, alpha_a_a, alpha_c_a,
        E_eq, F_const, R_gas, T_ref,
        start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in FuelCell_ButlerVolmer: %s\n", cudaGetErrorString(err));
}

extern "C" void ScaLBL_FuelCell_PhaseChange(
    double *Phi, double *Temperature, double *ConcH2O,
    double *SourcePhaseField, double *SourceThermal,
    double h_fg, double R_gas, double T_ref, double P_ref,
    double k_evap, double k_cond,
    int *Map, int start, int finish, int Np)
{
    dvc_ScaLBL_FuelCell_PhaseChange<<<NBLOCKS, NTHREADS>>>(
        Phi, Temperature, ConcH2O,
        SourcePhaseField, SourceThermal,
        h_fg, R_gas, T_ref, P_ref, k_evap, k_cond,
        Map, start, finish, Np);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) printf("CUDA error in FuelCell_PhaseChange: %s\n", cudaGetErrorString(err));
}
