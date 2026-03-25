/*
  ColorModelSI: SI-unit interface for the Color Lattice Boltzmann Model
  
  This class wraps ScaLBL_ColorModel and provides an interface where all
  physical inputs are specified in SI units. The conversion to lattice
  units is performed automatically.

  NOTE ON BODY FORCE:
    In the Color model the body force F is a uniform pressure gradient
    applied to BOTH phases equally (not gravity). There is no buoyancy
    or Bond number. The force drives flow like an imposed dP/dx.
  
  =========================================================================
  MODE 1: SI Unit Input  (ReadParams)
  =========================================================================
  SI Input Parameters (in the "Color" section of the input file):
    viscosity_A        - kinematic viscosity of phase A (phi=+1)  [m²/s]
    viscosity_B        - kinematic viscosity of phase B (phi=-1)  [m²/s]
    density_A          - density of phase A                       [kg/m³]
    density_B          - density of phase B                       [kg/m³]
    surface_tension    - interfacial tension                      [N/m]
    pressure_gradient  - imposed dP/dx vector {x, y, z}           [Pa/m]
    inlet_pressure     - inlet gauge pressure                     [Pa]
    outlet_pressure    - outlet gauge pressure                    [Pa]
    flow_rate          - volumetric flow rate (for BC=4)           [m³/s]
    target_tau         - stability parameter for more viscous phase (default 1.0)
    beta               - interface width parameter (dimensionless)
    timestepMax        - number of timesteps
    Restart            - restart flag
    inletA/B           - inlet phase fractions (dimensionless)
    outletA/B          - outlet phase fractions (dimensionless)

  Domain section must include:
    voxel_length     - voxel size [µm]
    
  Conversion approach:
    dx = voxel_length * 1e-6 (metres)
    dt is derived from target_tau and the more viscous phase:
      nu_LB = (target_tau - 0.5) / 3
      dt = nu_LB * dx² / nu_max
    rho_ref = density_A (so rhoA_LB = 1.0)
    
    tau_A = 3 * viscosity_A * dt / dx² + 0.5
    tau_B = 3 * viscosity_B * dt / dx² + 0.5
    rhoA  = 1.0
    rhoB  = density_B / density_A
    alpha = surface_tension * dt² / (K_sigma * rho_ref * dx³)
    Fx    = (dP/dx)_SI * dt² / (rho_ref * dx)      [pressure gradient -> lattice force]
    din   = 1 + 1.5 * (P_in - P_out) * dt² / (rho_ref * dx²)
    dout  = 1 - 1.5 * (P_in - P_out) * dt² / (rho_ref * dx²)
    flux  = flow_rate * dt / dx³

  =========================================================================
  MODE 2: Dimensionless Number Matching  (ReadParamsDimensionless)
  =========================================================================
  Input Parameters (in the "Color" section):
    capillary_number   - Ca = mu * u / sigma
    reynolds_number    - Re = u * L / nu   (based on domain length)
    viscosity_ratio    - M  = nu_A / nu_B
    density_ratio      - Lambda = rho_A / rho_B  (typically ~1 for color model)
    target_tau         - tau for more viscous phase (default 1.0)
    beta               - interface width parameter (dimensionless)
    timestepMax, Restart, inletA/B, outletA/B, BC - as usual

  The body force F is the pressure gradient that drives flow. It is
  computed from Re to produce the characteristic velocity:
    u_LB  = Re * nu_LB / L_LB
    F_LB  = 8 * nu_LB * u_LB / L_LB²   (Poiseuille scaling)
  
  The code auto-selects stable lattice parameters that reproduce these
  dimensionless groups exactly. The characteristic length L is the domain
  extent in the z-direction.
*/
#ifndef COLORMODELSI_H
#define COLORMODELSI_H

#include "models/ColorModel.h"

class ScaLBL_ColorModelSI : public ScaLBL_ColorModel {
public:
    ScaLBL_ColorModelSI(int RANK, int NP, MPI_Comm COMM);
    ~ScaLBL_ColorModelSI();

    // Mode 1: SI unit input
    void ReadParams(string filename);

    // Mode 2: Dimensionless number matching
    void ReadParamsDimensionless(string filename);

    // Stability check (called automatically, can also be called manually)
    // Returns true if all parameters are in stable range, false otherwise.
    // If abort_on_failure is true, calls MPI_Abort on critical instability.
    bool CheckStability(bool abort_on_failure = true);

    // SI input parameters (Mode 1)
    double viscosity_A;     // [m²/s]
    double viscosity_B;     // [m²/s]
    double density_A;       // [kg/m³]
    double density_B;       // [kg/m³]
    double surface_tension; // [N/m]
    double target_tau;      // dimensionless stability parameter
    double inlet_pressure;  // [Pa]
    double outlet_pressure; // [Pa]
    double flow_rate_SI;    // [m³/s]

    // Dimensionless input parameters (Mode 2)
    double capillary_number;
    double reynolds_number;
    double viscosity_ratio;
    double density_ratio;

    void PrintConversionInfo();

    // Surface tension calibration factor K_sigma(tau).
    // The D3Q19 MRT color-gradient kernel produces sigma_eff = K * alpha.
    // K ~ 7.28 for tau >= 0.7, with a small exponential correction at low tau.
    static double computeKSigma(double tau);
};

#endif
