/*
  FuelCell.cpp — CPU fallback kernels for the Fuel Cell model
  Mirrors gpu/FuelCell.cu with serial for-loops.
  D3Q7 weights: w0=1/4, w1..6=1/8, cs^2=1/4
*/
#include <stdio.h>
#include <math.h>
#include <algorithm>

static double CS_EoS_Pressure_CPU(double rho, double a, double b, double T_R) {
    double eta = b * rho * 0.25;
    double denom = (1.0 - eta);
    double num = 1.0 + eta + eta * eta - eta * eta * eta;
    return rho * T_R * num / (denom * denom * denom) - a * rho * rho;
}

static double SC_EffectiveMass_CPU(double rho, double G, double a, double b, double T_R) {
    double p = CS_EoS_Pressure_CPU(rho, a, b, T_R);
    double cs2 = 1.0 / 3.0;
    double arg = 2.0 * (p - rho * cs2) / (G * cs2);
    if (arg < 0.0) arg = 0.0;
    return sqrt(arg);
}

static double phi2rho_CPU(double p, double rhoA, double rhoB) {
    return rhoA + 0.5 * (1.0 - p) * (rhoB - rhoA);
}

// Shan-Chen force (even)
extern "C" void ScaLBL_D3Q19_AAeven_ShanChen_Force(
    int *Map, double *Den, double *Phi,
    double *ForceX, double *ForceY, double *ForceZ,
    double G, double cs_a, double cs_b, double cs_T,
    double rhoA, double rhoB,
    int strideY, int strideZ, int start, int finish, int Np)
{
    const double w_face = 1.0 / 18.0;
    const double w_edge = 1.0 / 36.0;
    for (int n = start; n < finish; n++) {
        double nA = Den[n], nB = Den[Np + n];
        double phi = (nA - nB) / (nA + nB + 1.0e-15);
        phi = std::min(1.0, std::max(-1.0, phi));
        double rho_local = phi2rho_CPU(phi, rhoA, rhoB);
        int ijk = Map[n];
        double psi = SC_EffectiveMass_CPU(rho_local, G, cs_a, cs_b, cs_T);

        auto psi_of = [&](int off) -> double {
            double p = Phi[ijk + off];
            return SC_EffectiveMass_CPU(phi2rho_CPU(p, rhoA, rhoB), G, cs_a, cs_b, cs_T);
        };
        double px = psi_of(1), mx = psi_of(-1);
        double py = psi_of(strideY), my = psi_of(-strideY);
        double pz = psi_of(strideZ), mz = psi_of(-strideZ);
        double pxpy = psi_of(1+strideY), mxmy = psi_of(-1-strideY);
        double pxmy = psi_of(1-strideY), mxpy = psi_of(-1+strideY);
        double pxpz = psi_of(1+strideZ), mxmz = psi_of(-1-strideZ);
        double pxmz = psi_of(1-strideZ), mxpz = psi_of(-1+strideZ);
        double pypz = psi_of(strideY+strideZ), mymz = psi_of(-strideY-strideZ);
        double pymz = psi_of(strideY-strideZ), mypz = psi_of(-strideY+strideZ);

        ForceX[n] = -G*psi*(w_face*(px-mx)+w_edge*(pxpy-mxmy+pxmy-mxpy+pxpz-mxmz+pxmz-mxpz));
        ForceY[n] = -G*psi*(w_face*(py-my)+w_edge*(pxpy-mxmy-pxmy+mxpy+pypz-mymz+pymz-mypz));
        ForceZ[n] = -G*psi*(w_face*(pz-mz)+w_edge*(pxpz-mxmz-pxmz+mxpz+pypz-mymz-pymz+mypz));
    }
}

extern "C" void ScaLBL_D3Q19_AAodd_ShanChen_Force(
    int *neighborList, int *Map, double *Den, double *Phi,
    double *ForceX, double *ForceY, double *ForceZ,
    double G, double cs_a, double cs_b, double cs_T,
    double rhoA, double rhoB,
    int strideY, int strideZ, int start, int finish, int Np)
{
    ScaLBL_D3Q19_AAeven_ShanChen_Force(Map, Den, Phi, ForceX, ForceY, ForceZ,
        G, cs_a, cs_b, cs_T, rhoA, rhoB, strideY, strideZ, start, finish, Np);
}

// Species D3Q7 (even)
extern "C" void ScaLBL_D3Q7_AAeven_FuelCell_Species(
    int *Map, double *Cq, double *Conc, double *Velocity,
    double *Phi, double *Poros, double *DiffCoeff, double *SourceTerm,
    double tau_sp, int start, int finish, int Np)
{
    const double w0 = 0.25, w1 = 0.125;
    for (int n = start; n < finish; n++) {
        double porosity = Poros[n], diff_eff = DiffCoeff[n], source = SourceTerm[n];
        double ux = Velocity[n], uy = Velocity[Np+n], uz = Velocity[2*Np+n];
        double phi = Phi[Map[n]];
        double gas_mask = (phi < 0.5) ? 1.0 : 0.0;

        double f0 = Cq[n];
        double f1 = Cq[2*Np+n], f2 = Cq[1*Np+n];
        double f3 = Cq[4*Np+n], f4 = Cq[3*Np+n];
        double f5 = Cq[6*Np+n], f6 = Cq[5*Np+n];
        double C = f0+f1+f2+f3+f4+f5+f6;

        double tau_eff = 0.5 + (tau_sp-0.5)*porosity*diff_eff;
        if (tau_eff < 0.501) tau_eff = 0.501;
        double rlx = 1.0/tau_eff;

        f0 = f0*(1.0-rlx) + rlx*w0*C + w0*source*gas_mask;
        f1 = f1*(1.0-rlx) + rlx*w1*C*(1.0+4.0*ux) + w1*source*gas_mask;
        f2 = f2*(1.0-rlx) + rlx*w1*C*(1.0-4.0*ux) + w1*source*gas_mask;
        f3 = f3*(1.0-rlx) + rlx*w1*C*(1.0+4.0*uy) + w1*source*gas_mask;
        f4 = f4*(1.0-rlx) + rlx*w1*C*(1.0-4.0*uy) + w1*source*gas_mask;
        f5 = f5*(1.0-rlx) + rlx*w1*C*(1.0+4.0*uz) + w1*source*gas_mask;
        f6 = f6*(1.0-rlx) + rlx*w1*C*(1.0-4.0*uz) + w1*source*gas_mask;

        Cq[n]=f0; Cq[1*Np+n]=f1; Cq[2*Np+n]=f2;
        Cq[3*Np+n]=f3; Cq[4*Np+n]=f4; Cq[5*Np+n]=f5; Cq[6*Np+n]=f6;
        Conc[n] = C;
    }
}

// Species D3Q7 (odd)
extern "C" void ScaLBL_D3Q7_AAodd_FuelCell_Species(
    int *neighborList, int *Map, double *Cq, double *Conc, double *Velocity,
    double *Phi, double *Poros, double *DiffCoeff, double *SourceTerm,
    double tau_sp, int start, int finish, int Np)
{
    const double w0 = 0.25, w1 = 0.125;
    for (int n = start; n < finish; n++) {
        double porosity = Poros[n], diff_eff = DiffCoeff[n], source = SourceTerm[n];
        double ux = Velocity[n], uy = Velocity[Np+n], uz = Velocity[2*Np+n];
        double phi = Phi[Map[n]];
        double gas_mask = (phi < 0.5) ? 1.0 : 0.0;

        int nr1 = neighborList[n], nr2 = neighborList[n+Np];
        int nr3 = neighborList[n+2*Np], nr4 = neighborList[n+3*Np];
        int nr5 = neighborList[n+4*Np], nr6 = neighborList[n+5*Np];

        double f0 = Cq[n];
        double f1 = Cq[nr1], f2 = Cq[nr2], f3 = Cq[nr3];
        double f4 = Cq[nr4], f5 = Cq[nr5], f6 = Cq[nr6];
        double C = f0+f1+f2+f3+f4+f5+f6;

        double tau_eff = 0.5 + (tau_sp-0.5)*porosity*diff_eff;
        if (tau_eff < 0.501) tau_eff = 0.501;
        double rlx = 1.0/tau_eff;

        f0 = f0*(1.0-rlx) + rlx*w0*C + w0*source*gas_mask;
        f1 = f1*(1.0-rlx) + rlx*w1*C*(1.0+4.0*ux) + w1*source*gas_mask;
        f2 = f2*(1.0-rlx) + rlx*w1*C*(1.0-4.0*ux) + w1*source*gas_mask;
        f3 = f3*(1.0-rlx) + rlx*w1*C*(1.0+4.0*uy) + w1*source*gas_mask;
        f4 = f4*(1.0-rlx) + rlx*w1*C*(1.0-4.0*uy) + w1*source*gas_mask;
        f5 = f5*(1.0-rlx) + rlx*w1*C*(1.0+4.0*uz) + w1*source*gas_mask;
        f6 = f6*(1.0-rlx) + rlx*w1*C*(1.0-4.0*uz) + w1*source*gas_mask;

        Cq[n]=f0; Cq[nr2]=f1; Cq[nr1]=f2;
        Cq[nr4]=f3; Cq[nr3]=f4; Cq[nr6]=f5; Cq[nr5]=f6;
        Conc[n] = C;
    }
}

// Potential D3Q7 (even)
extern "C" void ScaLBL_D3Q7_AAeven_FuelCell_Potential(
    int *Map, double *Pq, double *PotentialField, double *Conductivity,
    int start, int finish, int Np)
{
    const double w0 = 0.25, w1 = 0.125;
    for (int n = start; n < finish; n++) {
        double sigma = Conductivity[n];
        double tau = 0.5 + 4.0*sigma;
        if (tau < 0.501) tau = 0.501;
        double rlx = 1.0/tau;

        double f0 = Pq[n];
        double f1 = Pq[2*Np+n], f2 = Pq[1*Np+n];
        double f3 = Pq[4*Np+n], f4 = Pq[3*Np+n];
        double f5 = Pq[6*Np+n], f6 = Pq[5*Np+n];
        double pot = f0+f1+f2+f3+f4+f5+f6;

        f0 = f0*(1.0-rlx) + rlx*w0*pot;
        f1 = f1*(1.0-rlx) + rlx*w1*pot;
        f2 = f2*(1.0-rlx) + rlx*w1*pot;
        f3 = f3*(1.0-rlx) + rlx*w1*pot;
        f4 = f4*(1.0-rlx) + rlx*w1*pot;
        f5 = f5*(1.0-rlx) + rlx*w1*pot;
        f6 = f6*(1.0-rlx) + rlx*w1*pot;

        Pq[n]=f0; Pq[1*Np+n]=f1; Pq[2*Np+n]=f2;
        Pq[3*Np+n]=f3; Pq[4*Np+n]=f4; Pq[5*Np+n]=f5; Pq[6*Np+n]=f6;
        PotentialField[n] = pot;
    }
}

// Potential D3Q7 (odd)
extern "C" void ScaLBL_D3Q7_AAodd_FuelCell_Potential(
    int *neighborList, int *Map, double *Pq, double *PotentialField,
    double *Conductivity, int start, int finish, int Np)
{
    const double w0 = 0.25, w1 = 0.125;
    for (int n = start; n < finish; n++) {
        double sigma = Conductivity[n];
        double tau = 0.5 + 4.0*sigma;
        if (tau < 0.501) tau = 0.501;
        double rlx = 1.0/tau;

        int nr1 = neighborList[n], nr2 = neighborList[n+Np];
        int nr3 = neighborList[n+2*Np], nr4 = neighborList[n+3*Np];
        int nr5 = neighborList[n+4*Np], nr6 = neighborList[n+5*Np];

        double f0 = Pq[n];
        double f1 = Pq[nr1], f2 = Pq[nr2], f3 = Pq[nr3];
        double f4 = Pq[nr4], f5 = Pq[nr5], f6 = Pq[nr6];
        double pot = f0+f1+f2+f3+f4+f5+f6;

        f0 = f0*(1.0-rlx) + rlx*w0*pot;
        f1 = f1*(1.0-rlx) + rlx*w1*pot;
        f2 = f2*(1.0-rlx) + rlx*w1*pot;
        f3 = f3*(1.0-rlx) + rlx*w1*pot;
        f4 = f4*(1.0-rlx) + rlx*w1*pot;
        f5 = f5*(1.0-rlx) + rlx*w1*pot;
        f6 = f6*(1.0-rlx) + rlx*w1*pot;

        Pq[n]=f0; Pq[nr2]=f1; Pq[nr1]=f2;
        Pq[nr4]=f3; Pq[nr3]=f4; Pq[nr6]=f5; Pq[nr5]=f6;
        PotentialField[n] = pot;
    }
}

// Thermal D3Q7 (even)
extern "C" void ScaLBL_D3Q7_AAeven_FuelCell_Thermal(
    int *Map, double *Tq, double *Temperature, double *Velocity,
    double *ThermalCond, double *SourceThermal,
    double tau_thermal, int start, int finish, int Np)
{
    const double w0 = 0.25, w1 = 0.125;
    for (int n = start; n < finish; n++) {
        double kth = ThermalCond[n], source = SourceThermal[n];
        double ux = Velocity[n], uy = Velocity[Np+n], uz = Velocity[2*Np+n];
        double tau_eff = 0.5 + 4.0*kth;
        if (tau_eff < 0.501) tau_eff = 0.501;
        double rlx = 1.0/tau_eff;

        double f0 = Tq[n];
        double f1 = Tq[2*Np+n], f2 = Tq[1*Np+n];
        double f3 = Tq[4*Np+n], f4 = Tq[3*Np+n];
        double f5 = Tq[6*Np+n], f6 = Tq[5*Np+n];
        double T = f0+f1+f2+f3+f4+f5+f6;

        f0 = f0*(1.0-rlx) + rlx*w0*T + w0*source;
        f1 = f1*(1.0-rlx) + rlx*w1*T*(1.0+4.0*ux) + w1*source;
        f2 = f2*(1.0-rlx) + rlx*w1*T*(1.0-4.0*ux) + w1*source;
        f3 = f3*(1.0-rlx) + rlx*w1*T*(1.0+4.0*uy) + w1*source;
        f4 = f4*(1.0-rlx) + rlx*w1*T*(1.0-4.0*uy) + w1*source;
        f5 = f5*(1.0-rlx) + rlx*w1*T*(1.0+4.0*uz) + w1*source;
        f6 = f6*(1.0-rlx) + rlx*w1*T*(1.0-4.0*uz) + w1*source;

        Tq[n]=f0; Tq[1*Np+n]=f1; Tq[2*Np+n]=f2;
        Tq[3*Np+n]=f3; Tq[4*Np+n]=f4; Tq[5*Np+n]=f5; Tq[6*Np+n]=f6;
        Temperature[n] = T;
    }
}

// Thermal D3Q7 (odd)
extern "C" void ScaLBL_D3Q7_AAodd_FuelCell_Thermal(
    int *neighborList, int *Map, double *Tq, double *Temperature,
    double *Velocity, double *ThermalCond, double *SourceThermal,
    double tau_thermal, int start, int finish, int Np)
{
    const double w0 = 0.25, w1 = 0.125;
    for (int n = start; n < finish; n++) {
        double kth = ThermalCond[n], source = SourceThermal[n];
        double ux = Velocity[n], uy = Velocity[Np+n], uz = Velocity[2*Np+n];
        double tau_eff = 0.5 + 4.0*kth;
        if (tau_eff < 0.501) tau_eff = 0.501;
        double rlx = 1.0/tau_eff;

        int nr1 = neighborList[n], nr2 = neighborList[n+Np];
        int nr3 = neighborList[n+2*Np], nr4 = neighborList[n+3*Np];
        int nr5 = neighborList[n+4*Np], nr6 = neighborList[n+5*Np];

        double f0 = Tq[n];
        double f1 = Tq[nr1], f2 = Tq[nr2], f3 = Tq[nr3];
        double f4 = Tq[nr4], f5 = Tq[nr5], f6 = Tq[nr6];
        double T = f0+f1+f2+f3+f4+f5+f6;

        f0 = f0*(1.0-rlx) + rlx*w0*T + w0*source;
        f1 = f1*(1.0-rlx) + rlx*w1*T*(1.0+4.0*ux) + w1*source;
        f2 = f2*(1.0-rlx) + rlx*w1*T*(1.0-4.0*ux) + w1*source;
        f3 = f3*(1.0-rlx) + rlx*w1*T*(1.0+4.0*uy) + w1*source;
        f4 = f4*(1.0-rlx) + rlx*w1*T*(1.0-4.0*uy) + w1*source;
        f5 = f5*(1.0-rlx) + rlx*w1*T*(1.0+4.0*uz) + w1*source;
        f6 = f6*(1.0-rlx) + rlx*w1*T*(1.0-4.0*uz) + w1*source;

        Tq[n]=f0; Tq[nr2]=f1; Tq[nr1]=f2;
        Tq[nr4]=f3; Tq[nr3]=f4; Tq[nr6]=f5; Tq[nr5]=f6;
        Temperature[n] = T;
    }
}

// Butler-Volmer
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
    for (int n = start; n < finish; n++) {
        double region = RegionID[n];
        double phiS = PhiS[n], phiE = PhiE[n], T_local = Temperature[n];
        if (T_local < 0.1) T_local = 1.0;
        double T_dim = T_local * T_ref;
        if (fabs(region - 3.0) < 0.5) {
            double eta = phiS - phiE - E_eq;
            double j = i0_cathode * (exp(alpha_a_c*F_const*eta/(R_gas*T_dim))
                     - exp(-alpha_c_c*F_const*eta/(R_gas*T_dim)));
            j = std::min(std::max(j, -1.0e4), 1.0e4);
            ReactionRate[n] = j;
            SourceO2[n]  = j / (4.0*F_const);
            SourceN2[n]  = 0.0;
            SourceH2[n]  = 0.0;
            SourceH2O[n] = -j / (2.0*F_const);
        } else {
            ReactionRate[n] = 0.0;
            SourceO2[n] = SourceN2[n] = SourceH2[n] = SourceH2O[n] = 0.0;
        }
    }
}

// Phase change
extern "C" void ScaLBL_FuelCell_PhaseChange(
    double *Phi, double *Temperature, double *ConcH2O,
    double *SourcePhaseField, double *SourceThermal,
    double h_fg, double R_gas, double T_ref, double P_ref,
    double k_evap, double k_cond,
    int *Map, int start, int finish, int Np)
{
    const double M_H2O = 0.018;
    for (int n = start; n < finish; n++) {
        double T = Temperature[n]; if (T < 0.1) T = 1.0;
        double c_h2o = ConcH2O[n];
        double phi = Phi[Map[n]];
        double P_h2o = c_h2o / 3.0;
        double P_sat = P_ref * exp(h_fg*M_H2O/R_gas*(1.0/T_ref - 1.0/T));
        double dm = 0.0;
        if (P_h2o > P_sat && phi < 0.3) dm = k_cond*(P_h2o - P_sat);
        else if (P_h2o < P_sat && phi > -0.3) dm = -k_evap*(P_sat - P_h2o);
        SourcePhaseField[n] = dm;
        SourceThermal[n] = -dm * h_fg;
    }
}
