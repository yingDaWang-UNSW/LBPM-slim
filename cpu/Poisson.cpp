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
  Poisson.cpp -- CPU D3Q7 lattice Boltzmann Poisson / Laplace kernels.
  CPU mirror of gpu/Poisson.cu (see that file for details).
*/
#include <math.h>

extern "C" void ScaLBL_D3Q7_AAodd_Poisson_ElectricPotential(int *neighborList, int *Map,
                                            double *dist, double *Psi,
                                            int start, int finish, int Np) {
    int n, nread, idx;
    double psi, fq;
    for (n = start; n < finish; n++) {
        fq = dist[n];                            psi = fq;
        nread = neighborList[n];                 fq = dist[nread]; psi += fq;
        nread = neighborList[n + Np];            fq = dist[nread]; psi += fq;
        nread = neighborList[n + 2*Np];          fq = dist[nread]; psi += fq;
        nread = neighborList[n + 3*Np];          fq = dist[nread]; psi += fq;
        nread = neighborList[n + 4*Np];          fq = dist[nread]; psi += fq;
        nread = neighborList[n + 5*Np];          fq = dist[nread]; psi += fq;
        idx = Map[n];
        Psi[idx] = psi;
    }
}

extern "C" void ScaLBL_D3Q7_AAeven_Poisson_ElectricPotential(
    int *Map, double *dist, double *Psi, int start, int finish, int Np) {
    int n, idx;
    double psi, fq;
    for (n = start; n < finish; n++) {
        fq = dist[n];          psi  = fq;
        fq = dist[2*Np+n];     psi += fq;
        fq = dist[1*Np+n];     psi += fq;
        fq = dist[4*Np+n];     psi += fq;
        fq = dist[3*Np+n];     psi += fq;
        fq = dist[6*Np+n];     psi += fq;
        fq = dist[5*Np+n];     psi += fq;
        idx = Map[n];
        Psi[idx] = psi;
    }
}

extern "C" void ScaLBL_D3Q7_AAodd_Poisson(int *neighborList, int *Map,
                                          double *dist, double *Den_charge,
                                          double *Psi, double *ElectricField,
                                          double tau, double epsilon_LB, bool EnforceElectroneutrality,
                                          int start, int finish, int Np) {
    int n, idx;
    int nr1, nr2, nr3, nr4, nr5, nr6;
    double psi, rho_e, Ex, Ey, Ez;
    double f0, f1, f2, f3, f4, f5, f6;
    double rlx = 1.0 / tau;
    for (n = start; n < finish; n++) {
        rho_e = (EnforceElectroneutrality==1) ? 0.0 : Den_charge[n] / epsilon_LB;
        idx = Map[n]; psi = Psi[idx];
        f0  = dist[n];
        nr1 = neighborList[n];        f1 = dist[nr1];
        nr2 = neighborList[n + Np];   f2 = dist[nr2];
        nr3 = neighborList[n + 2*Np]; f3 = dist[nr3];
        nr4 = neighborList[n + 3*Np]; f4 = dist[nr4];
        nr5 = neighborList[n + 4*Np]; f5 = dist[nr5];
        nr6 = neighborList[n + 5*Np]; f6 = dist[nr6];
        Ex = (f1 - f2) * rlx * 4.0;
        Ey = (f3 - f4) * rlx * 4.0;
        Ez = (f5 - f6) * rlx * 4.0;
        ElectricField[n + 0*Np] = Ex;
        ElectricField[n + 1*Np] = Ey;
        ElectricField[n + 2*Np] = Ez;
        dist[n]   = f0 * (1.0 - rlx) + 0.25  * (rlx * psi + rho_e);
        dist[nr2] = f1 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[nr1] = f2 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[nr4] = f3 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[nr3] = f4 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[nr6] = f5 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[nr5] = f6 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
    }
}

extern "C" void ScaLBL_D3Q7_AAeven_Poisson(int *Map, double *dist,
                                           double *Den_charge, double *Psi,
                                           double *ElectricField, double tau,
                                           double epsilon_LB, bool EnforceElectroneutrality,
                                           int start, int finish, int Np) {
    int n, idx;
    double psi, rho_e, Ex, Ey, Ez;
    double f0, f1, f2, f3, f4, f5, f6;
    double rlx = 1.0 / tau;
    for (n = start; n < finish; n++) {
        rho_e = (EnforceElectroneutrality==1) ? 0.0 : Den_charge[n] / epsilon_LB;
        idx = Map[n]; psi = Psi[idx];
        f0 = dist[n];
        f1 = dist[2*Np + n];
        f2 = dist[1*Np + n];
        f3 = dist[4*Np + n];
        f4 = dist[3*Np + n];
        f5 = dist[6*Np + n];
        f6 = dist[5*Np + n];
        Ex = (f1 - f2) * rlx * 4.0;
        Ey = (f3 - f4) * rlx * 4.0;
        Ez = (f5 - f6) * rlx * 4.0;
        ElectricField[n + 0*Np] = Ex;
        ElectricField[n + 1*Np] = Ey;
        ElectricField[n + 2*Np] = Ez;
        dist[n]        = f0 * (1.0 - rlx) + 0.25  * (rlx * psi + rho_e);
        dist[1*Np + n] = f1 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[2*Np + n] = f2 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[3*Np + n] = f3 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[4*Np + n] = f4 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[5*Np + n] = f5 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
        dist[6*Np + n] = f6 * (1.0 - rlx) + 0.125 * (rlx * psi + rho_e);
    }
}

extern "C" void ScaLBL_D3Q7_Poisson_Init(int *Map, double *dist, double *Psi,
                                         int start, int finish, int Np) {
    int n, ijk;
    for (n = start; n < finish; n++) {
        ijk = Map[n];
        dist[0*Np + n] = 0.25  * Psi[ijk];
        dist[1*Np + n] = 0.125 * Psi[ijk];
        dist[2*Np + n] = 0.125 * Psi[ijk];
        dist[3*Np + n] = 0.125 * Psi[ijk];
        dist[4*Np + n] = 0.125 * Psi[ijk];
        dist[5*Np + n] = 0.125 * Psi[ijk];
        dist[6*Np + n] = 0.125 * Psi[ijk];
    }
}

// -------------------------------------------------------------------------
// Dirichlet potential BCs on inlet (-z) and outlet (+z) faces.
// -------------------------------------------------------------------------
extern "C" void ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_z(int *list,
                                                          double *dist,
                                                          double Vin, int count,
                                                          int Np) {
    for (int idx = 0; idx < count; idx++) {
        int n = list[idx];
        double f0 = dist[n];
        double f1 = dist[2*Np + n];
        double f2 = dist[1*Np + n];
        double f3 = dist[4*Np + n];
        double f4 = dist[3*Np + n];
        double f6 = dist[5*Np + n];
        double f5 = Vin - (f0 + f1 + f2 + f3 + f4 + f6);
        dist[6*Np + n] = f5;
    }
}

extern "C" void ScaLBL_D3Q7_AAeven_Poisson_Potential_BC_Z(int *list,
                                                          double *dist,
                                                          double Vout, int count,
                                                          int Np) {
    for (int idx = 0; idx < count; idx++) {
        int n = list[idx];
        double f0 = dist[n];
        double f1 = dist[2*Np + n];
        double f2 = dist[1*Np + n];
        double f3 = dist[4*Np + n];
        double f4 = dist[3*Np + n];
        double f5 = dist[6*Np + n];
        double f6 = Vout - (f0 + f1 + f2 + f3 + f4 + f5);
        dist[5*Np + n] = f6;
    }
}

extern "C" void ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_z(int *d_neighborList,
                                                         int *list,
                                                         double *dist,
                                                         double Vin, int count,
                                                         int Np) {
    int nread, nr5;
    for (int idx = 0; idx < count; idx++) {
        int n = list[idx];
        double f0 = dist[n];
        nread = d_neighborList[n];        double f1 = dist[nread];
        nread = d_neighborList[n + 2*Np]; double f3 = dist[nread];
        nread = d_neighborList[n + Np];   double f2 = dist[nread];
        nread = d_neighborList[n + 3*Np]; double f4 = dist[nread];
        nread = d_neighborList[n + 5*Np]; double f6 = dist[nread];
        nr5 = d_neighborList[n + 4*Np];
        double f5 = Vin - (f0 + f1 + f2 + f3 + f4 + f6);
        dist[nr5] = f5;
    }
}

extern "C" void ScaLBL_D3Q7_AAodd_Poisson_Potential_BC_Z(int *d_neighborList,
                                                         int *list,
                                                         double *dist,
                                                         double Vout, int count,
                                                         int Np) {
    int nread, nr6;
    for (int idx = 0; idx < count; idx++) {
        int n = list[idx];
        double f0 = dist[n];
        nread = d_neighborList[n];        double f1 = dist[nread];
        nread = d_neighborList[n + 2*Np]; double f3 = dist[nread];
        nread = d_neighborList[n + 4*Np]; double f5 = dist[nread];
        nread = d_neighborList[n + Np];   double f2 = dist[nread];
        nread = d_neighborList[n + 3*Np]; double f4 = dist[nread];
        nr6 = d_neighborList[n + 5*Np];
        double f6 = Vout - (f0 + f1 + f2 + f3 + f4 + f5);
        dist[nr6] = f6;
    }
}

extern "C" void ScaLBL_Poisson_D3Q7_BC_z(int *list, int *Map, double *Psi,
                                         double Vin, int count) {
    for (int idx = 0; idx < count; idx++) {
        int n  = list[idx];
        int nm = Map[n];
        Psi[nm] = Vin;
    }
}

extern "C" void ScaLBL_Poisson_D3Q7_BC_Z(int *list, int *Map, double *Psi,
                                         double Vout, int count) {
    for (int idx = 0; idx < count; idx++) {
        int n  = list[idx];
        int nm = Map[n];
        Psi[nm] = Vout;
    }
}
