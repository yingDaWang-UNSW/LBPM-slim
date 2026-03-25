/*
  ColorModelSI: SI-unit interface for the Color Lattice Boltzmann Model
*/
#include "models/ColorModelSI.h"
#include <algorithm>
#include <cmath>

using namespace std;

ScaLBL_ColorModelSI::ScaLBL_ColorModelSI(int RANK, int NP, MPI_Comm COMM)
    : ScaLBL_ColorModel(RANK, NP, COMM),
      viscosity_A(0), viscosity_B(0), density_A(0), density_B(0),
      surface_tension(0), target_tau(1.0),
      inlet_pressure(0), outlet_pressure(0), flow_rate_SI(0),
      capillary_number(0), reynolds_number(0), viscosity_ratio(1.0),
      density_ratio(1.0)
{
}

ScaLBL_ColorModelSI::~ScaLBL_ColorModelSI() {}

// =========================================================================
// Surface tension calibration factor K_sigma(tau)
//
// The D3Q19 MRT color-gradient kernel uses the CSS (Continuum Surface
// Stress) formulation where only deviatoric stress moments (m9-m15)
// carry the capillary stress tensor:
//     ΔΠ_αβ^dev = (α/2)·C·(n_α·n_β − δ_αβ/3)
//
// The CSS tensor is TRACELESS, so the energy moment (m1) is NOT modified.
// This is the theoretically correct formulation: the capillary stress
// from the Kirkwood-Buff formula gives σ = (α/2)·∫C dz, with the
// D3Q19 gradient stencil providing a factor of 6× over the continuum
// gradient plus higher-order CE corrections.
//
// Calibrated via static Laplace pressure tests (sphere R=20 in 80^3
// domain, equal viscosities and densities, β=0.95):
//     K_sigma = 7.9583 − 2.3995 × exp(−25.839 × (tau − 0.5))
// Converges to K_inf ≈ 7.958 for tau >= 0.70 (0.23% spread).
// =========================================================================
double ScaLBL_ColorModelSI::computeKSigma(double tau) {
    const double K_inf = 7.9583;
    const double A     = 2.3995;
    const double B     = 25.839;
    return K_inf - A * exp(-B * (tau - 0.5));
}

void ScaLBL_ColorModelSI::ReadParams(string filename) {
    // Read the input database
    db = make_shared<Database>(filename);
    domain_db = db->getDatabase("Domain");
    color_db = db->getDatabase("Color");
    analysis_db = db->getDatabase("Analysis");

    // ==========================================
    // Read SI parameters from Color section
    // ==========================================
    viscosity_A = color_db->getScalar<double>("viscosity_A");
    viscosity_B = color_db->getScalar<double>("viscosity_B");
    density_A = color_db->getScalar<double>("density_A");
    density_B = color_db->getScalar<double>("density_B");
    surface_tension = color_db->getScalar<double>("surface_tension");

    // Pressure gradient [Pa/m] — this is a uniform dP/dx applied to both phases
    double dPdx_x = 0.0, dPdx_y = 0.0, dPdx_z = 0.0;
    if (color_db->keyExists("pressure_gradient")) {
        auto dPdx_SI = color_db->getVector<double>("pressure_gradient");
        dPdx_x = dPdx_SI[0];
        dPdx_y = dPdx_SI[1];
        dPdx_z = dPdx_SI[2];
    }

    if (color_db->keyExists("target_tau"))
        target_tau = color_db->getScalar<double>("target_tau");

    beta = color_db->getScalar<double>("beta");
    timestepMax = color_db->getScalar<int>("timestepMax");
    Restart = color_db->getScalar<bool>("Restart");

    // Pressure BCs and flow rate — optional (only relevant for BC=3 or BC=4)
    if (color_db->keyExists("inlet_pressure"))
        inlet_pressure = color_db->getScalar<double>("inlet_pressure");
    if (color_db->keyExists("outlet_pressure"))
        outlet_pressure = color_db->getScalar<double>("outlet_pressure");
    if (color_db->keyExists("flow_rate"))
        flow_rate_SI = color_db->getScalar<double>("flow_rate");

    // Phase fraction parameters (dimensionless, same as lattice model)
    inletA = 1.0; inletB = 0.0; outletA = 0.0; outletB = 1.0;
    if (color_db->keyExists("inletA")) inletA = color_db->getScalar<double>("inletA");
    if (color_db->keyExists("inletB")) inletB = color_db->getScalar<double>("inletB");
    if (color_db->keyExists("outletA")) outletA = color_db->getScalar<double>("outletA");
    if (color_db->keyExists("outletB")) outletB = color_db->getScalar<double>("outletB");

    // ==========================================
    // Read domain parameters
    // ==========================================
    auto L = domain_db->getVector<double>("L");
    auto size = domain_db->getVector<int>("n");
    auto nproc = domain_db->getVector<int>("nproc");
    BoundaryCondition = domain_db->getScalar<int>("BC");
    Nx = size[0]; Ny = size[1]; Nz = size[2];
    Lx = L[0]; Ly = L[1]; Lz = L[2];
    nprocx = nproc[0]; nprocy = nproc[1]; nprocz = nproc[2];

    // ==========================================
    // SI -> Lattice unit conversion
    // ==========================================

    // 1. Lattice spacing from voxel geometry
    if (domain_db->keyExists("voxel_length")) {
        dx_si = domain_db->getScalar<double>("voxel_length") * 1.0e-6; // um -> m
    } else {
        dx_si = Lz / double(Nz * nprocz);
    }

    // 2. Time step from target tau and the more viscous phase
    //    nu_LB = (tau - 0.5) / 3
    //    dt = nu_LB * dx^2 / nu_physical
    double nu_max = max(viscosity_A, viscosity_B);
    double nu_LB_target = (target_tau - 0.5) / 3.0;
    dt_si = nu_LB_target * dx_si * dx_si / nu_max;

    // 3. Relaxation parameters
    //    tau = 3 * nu_LB + 0.5 = 3 * (nu_SI * dt / dx^2) + 0.5
    tauA = 3.0 * viscosity_A * dt_si / (dx_si * dx_si) + 0.5;
    tauB = 3.0 * viscosity_B * dt_si / (dx_si * dx_si) + 0.5;

    // 4. Density (reference = density_A, so rhoA_LB = 1.0)
    rho_ref = density_A;
    rhoA = 1.0;
    rhoB = density_B / density_A;

    // 5. Surface tension
    //    The D3Q19 MRT color-gradient model produces an effective surface
    //    tension sigma_eff = K_sigma * alpha, where K_sigma is an empirical
    //    calibration factor determined from Laplace pressure tests.
    //    K_sigma ~ 7.28 for tau >= 0.7, with a small correction at low tau.
    //    We invert this:  alpha = sigma / K_sigma * dt^2 / (rho_ref * dx^3)
    double K_sigma = computeKSigma(target_tau);
    alpha = surface_tension * dt_si * dt_si / (K_sigma * rho_ref * dx_si * dx_si * dx_si);

    // 6. Pressure gradient -> lattice body force
    //    The LBM body force is a force per unit mass (acceleration).
    //    F_LB = (dP/dx)_SI / rho_ref * dt^2 / dx
    Fx = dPdx_x * dt_si * dt_si / (rho_ref * dx_si);
    Fy = dPdx_y * dt_si * dt_si / (rho_ref * dx_si);
    Fz = dPdx_z * dt_si * dt_si / (rho_ref * dx_si);

    // 7. Phase-weighted reference densities at boundaries
    double rho_inlet_ref = rhoA;
    if (inletA + inletB > 0)
        rho_inlet_ref = (rhoA * inletA + rhoB * inletB) / (inletA + inletB);
    double rho_outlet_ref = rhoA;
    if (outletA + outletB > 0)
        rho_outlet_ref = (rhoA * outletA + rhoB * outletB) / (outletA + outletB);

    //    Pressure BCs: centre around phase-weighted density
    //    dp_LB = (P_in - P_out) * dt^2 / (rho_ref * dx^2)
    double dp_SI = inlet_pressure - outlet_pressure;
    double dp_LB = dp_SI * dt_si * dt_si / (rho_ref * dx_si * dx_si);
    din  = rho_inlet_ref  + 1.5 * dp_LB;
    dout = rho_outlet_ref - 1.5 * dp_LB;

    // 8. Flux (volumetric flow rate -> lattice flux)
    //    flux_LB = Q_SI * dt / dx^3
    flux = flow_rate_SI * dt_si / (dx_si * dx_si * dx_si);
    if (BoundaryCondition == 4) flux *= rho_inlet_ref; // mass flux with inlet phase density

    // Print conversion summary and check stability
    PrintConversionInfo();
    CheckStability(true);

    // Write SI conversion metadata for post-processing
    if (rank == 0 && dt_si > 0 && dx_si > 0) {
        FILE *meta = fopen("si_conversion.db", "w");
        fprintf(meta, "dx_si = %.15e\n", dx_si);
        fprintf(meta, "dt_si = %.15e\n", dt_si);
        fprintf(meta, "rho_ref = %.15e\n", rho_ref);
        fprintf(meta, "viscosity_A = %.15e\n", viscosity_A);
        fprintf(meta, "viscosity_B = %.15e\n", viscosity_B);
        fprintf(meta, "density_A = %.15e\n", density_A);
        fprintf(meta, "density_B = %.15e\n", density_B);
        fprintf(meta, "surface_tension = %.15e\n", surface_tension);
        fclose(meta);
    }
}

// =========================================================================
// Mode 2: Dimensionless Number Matching
// =========================================================================
void ScaLBL_ColorModelSI::ReadParamsDimensionless(string filename) {
    // Read the input database
    db = make_shared<Database>(filename);
    domain_db = db->getDatabase("Domain");
    color_db = db->getDatabase("Color");
    analysis_db = db->getDatabase("Analysis");

    // ==========================================
    // Read dimensionless parameters
    // Use "target_" prefix to avoid triggering parent's adaptive controller
    // ==========================================
    if (color_db->keyExists("target_capillary_number"))
        capillary_number = color_db->getScalar<double>("target_capillary_number");
    else
        capillary_number = color_db->getScalar<double>("capillary_number");

    if (color_db->keyExists("target_reynolds_number"))
        reynolds_number = color_db->getScalar<double>("target_reynolds_number");
    else
        reynolds_number = color_db->getScalar<double>("reynolds_number");
    viscosity_ratio  = color_db->getScalar<double>("viscosity_ratio");  // M = nu_A / nu_B
    density_ratio    = color_db->getScalar<double>("density_ratio");    // Lambda = rho_A / rho_B

    if (color_db->keyExists("target_tau"))
        target_tau = color_db->getScalar<double>("target_tau");

    beta = color_db->getScalar<double>("beta");
    timestepMax = color_db->getScalar<int>("timestepMax");
    Restart = color_db->getScalar<bool>("Restart");

    // Phase fraction parameters (dimensionless)
    inletA = 1.0; inletB = 0.0; outletA = 0.0; outletB = 1.0;
    if (color_db->keyExists("inletA")) inletA = color_db->getScalar<double>("inletA");
    if (color_db->keyExists("inletB")) inletB = color_db->getScalar<double>("inletB");
    if (color_db->keyExists("outletA")) outletA = color_db->getScalar<double>("outletA");
    if (color_db->keyExists("outletB")) outletB = color_db->getScalar<double>("outletB");

    // ==========================================
    // Read domain parameters
    // ==========================================
    auto L = domain_db->getVector<double>("L");
    auto size = domain_db->getVector<int>("n");
    auto nproc = domain_db->getVector<int>("nproc");
    BoundaryCondition = domain_db->getScalar<int>("BC");
    Nx = size[0]; Ny = size[1]; Nz = size[2];
    Lx = L[0]; Ly = L[1]; Lz = L[2];
    nprocx = nproc[0]; nprocy = nproc[1]; nprocz = nproc[2];

    // ==========================================
    // Auto-compute stable lattice parameters
    // ==========================================
    // Characteristic length in lattice units = domain z-extent
    double L_LB = double(Nz * nprocz);

    // The more viscous phase gets target_tau
    // nu_LB_max = (target_tau - 0.5) / 3
    double nu_LB_max = (target_tau - 0.5) / 3.0;

    // Determine which phase is more viscous based on viscosity_ratio
    // M = nu_A / nu_B
    double nu_LB_A, nu_LB_B;
    if (viscosity_ratio >= 1.0) {
        // Phase A is more viscous
        nu_LB_A = nu_LB_max;
        nu_LB_B = nu_LB_max / viscosity_ratio;
    } else {
        // Phase B is more viscous
        nu_LB_B = nu_LB_max;
        nu_LB_A = nu_LB_max * viscosity_ratio;
    }

    tauA = 3.0 * nu_LB_A + 0.5;
    tauB = 3.0 * nu_LB_B + 0.5;

    // Density ratio
    rhoA = 1.0;
    rhoB = 1.0 / density_ratio;  // Lambda = rho_A / rho_B

    // Characteristic velocity from Re:
    //   Re = u * L / nu_A  =>  u_LB = Re * nu_LB_A / L_LB
    double u_LB = reynolds_number * nu_LB_A / L_LB;

    // Surface tension (alpha) from Ca:
    //   Ca = mu * u / sigma = rho * nu * u / sigma_eff
    //   In LBM: sigma_eff = K_sigma * alpha, so
    //   Ca = rhoA * nu_LB_A * u_LB / (K_sigma * alpha)
    //   => alpha = rhoA * nu_LB_A * u_LB / (Ca * K_sigma)
    double K_sigma = computeKSigma(target_tau);
    if (capillary_number > 0.0) {
        alpha = rhoA * nu_LB_A * u_LB / (capillary_number * K_sigma);
    } else {
        alpha = 1.0e-3; // default small value
    }

    // Body force (pressure gradient) from Re:
    //   The body force IS the pressure gradient driving flow (applied to both phases).
    //   Using Poiseuille scaling: u ~ F * L^2 / (8 * nu)
    //   => F_LB = 8 * nu_LB_A * u_LB / L_LB^2
    Fx = 0.0; Fy = 0.0;
    Fz = 8.0 * nu_LB_A * u_LB / (L_LB * L_LB);

    // Phase-weighted reference densities at boundaries
    // When inlet/outlet phase composition is specified, the pressure BC must
    // target the correct phase density, not rhoA=1.  Without this, setting
    // e.g. outletB=1 with rhoB=6.67 creates a permanent conflict between
    // the collision (wants rho=rhoB) and the pressure BC (forces rho=1).
    double rho_inlet_ref = rhoA;
    if (inletA + inletB > 0)
        rho_inlet_ref = (rhoA * inletA + rhoB * inletB) / (inletA + inletB);
    double rho_outlet_ref = rhoA;
    if (outletA + outletB > 0)
        rho_outlet_ref = (rhoA * outletA + rhoB * outletB) / (outletA + outletB);

    // Pressure BCs from Reynolds number:
    //   For pressure-driven flow: dp_LB ~ rho * nu * u / L (Poiseuille scaling)
    //   din = rho_ref_in + 1.5*dp,  dout = rho_ref_out - 1.5*dp
    //   where dp = rhoA * nu_LB_A * u_LB / (L_LB * c_s^2), c_s^2 = 1/3
    double dp_LB = rhoA * nu_LB_A * u_LB / L_LB;  // pressure gradient scale * L
    din  = rho_inlet_ref  + 1.5 * dp_LB;
    dout = rho_outlet_ref - 1.5 * dp_LB;

    // Flux from characteristic velocity — only relevant for BC=4 (flux BC)
    //   Q_LB = u_LB * A_cross * rho_phase (mass flux)
    if (BoundaryCondition == 4) {
        double A_cross = double(Nx * nprocx) * double(Ny * nprocy);
        flux = u_LB * A_cross * rho_inlet_ref;
    } else {
        flux = 0.0;
    }

    // Set dx_si and dt_si from voxel_length if available (for output scaling)
    if (domain_db->keyExists("voxel_length")) {
        dx_si = domain_db->getScalar<double>("voxel_length") * 1.0e-6;
        // dt from nu matching: dt = nu_LB_A * dx^2 / nu_SI (if SI viscosity known)
        // In dimensionless mode we don't necessarily have SI viscosity,
        // but if the user provides it for output scaling:
        if (color_db->keyExists("viscosity_A")) {
            viscosity_A = color_db->getScalar<double>("viscosity_A");
            dt_si = nu_LB_A * dx_si * dx_si / viscosity_A;
        } else {
            dt_si = 0.0; // unknown physical time scale
        }
    } else {
        dx_si = 0.0;
        dt_si = 0.0;
    }
    rho_ref = 0.0; // not applicable in dimensionless mode unless density_A given
    if (color_db->keyExists("density_A")) {
        density_A = color_db->getScalar<double>("density_A");
        rho_ref = density_A;
    }
    if (color_db->keyExists("density_B"))
        density_B = color_db->getScalar<double>("density_B");
    if (color_db->keyExists("viscosity_B"))
        viscosity_B = color_db->getScalar<double>("viscosity_B");
    if (color_db->keyExists("surface_tension"))
        surface_tension = color_db->getScalar<double>("surface_tension");

    // Override body force with user-specified pressure gradient [Pa/m] if provided
    if (color_db->keyExists("pressure_gradient") && dx_si > 0 && dt_si > 0 && rho_ref > 0) {
        auto dPdx_SI = color_db->getVector<double>("pressure_gradient");
        Fx = dPdx_SI[0] * dt_si * dt_si / (rho_ref * dx_si);
        Fy = dPdx_SI[1] * dt_si * dt_si / (rho_ref * dx_si);
        Fz = dPdx_SI[2] * dt_si * dt_si / (rho_ref * dx_si);
        if (rank == 0)
            printf("  Body force overridden by pressure_gradient: (%.4e, %.4e, %.4e) Pa/m -> F_LB = (%.4e, %.4e, %.4e)\n",
                   dPdx_SI[0], dPdx_SI[1], dPdx_SI[2], Fx, Fy, Fz);
    }

    // Print summary
    if (rank == 0) {
        printf("================================================================\n");
        printf("  ColorModelSI: Dimensionless Number Matching Mode\n");
        printf("================================================================\n");
        printf("Dimensionless Inputs:\n");
        printf("  Capillary number (Ca): %.6e\n", capillary_number);
        printf("  Reynolds number  (Re): %.6e\n", reynolds_number);
        printf("  Viscosity ratio  (M = nuA/nuB): %.6f\n", viscosity_ratio);
        printf("  Density ratio    (Lambda = rhoA/rhoB): %.6f\n", density_ratio);
        printf("  Target tau:       %.4f\n", target_tau);
        printf("Derived Lattice Parameters:\n");
        printf("  tauA: %.6f   tauB: %.6f\n", tauA, tauB);
        printf("  rhoA: %.6f   rhoB: %.6f\n", rhoA, rhoB);
        printf("  nu_LB_A: %.6e   nu_LB_B: %.6e\n", nu_LB_A, nu_LB_B);
        printf("  alpha (sfc. tension): %.6e\n", alpha);
        printf("  beta  (intfc. width): %.6f\n", beta);
        printf("  u_LB (characteristic): %.6e\n", u_LB);
        printf("  Fz (pressure gradient): %.6e  (Poiseuille: F=8*nu*u/L^2)\n", Fz);
        printf("  din: %.8f   dout: %.8f\n", din, dout);
        printf("  flux: %.6e\n", flux);
        printf("  BC:   %d\n", BoundaryCondition);
        printf("  L_LB (domain z): %.0f\n", L_LB);
        printf("Verification of dimensionless groups:\n");
        double Ca_check = rhoA * nu_LB_A * u_LB / alpha;
        double Re_check = u_LB * L_LB / nu_LB_A;
        printf("  Ca_actual: %.6e  (target: %.6e)\n", Ca_check, capillary_number);
        printf("  Re_actual: %.6e  (target: %.6e)\n", Re_check, reynolds_number);
        if (dx_si > 0.0 && dt_si > 0.0) {
            printf("Physical Scales (from voxel_length + viscosity_A):\n");
            printf("  dx: %.6e m  (%.2f um)\n", dx_si, dx_si * 1e6);
            printf("  dt: %.6e s\n", dt_si);
            printf("  u_LB=0.1 => %.6e m/s\n", 0.1 * dx_si / dt_si);
            printf("  Physical time for %d steps: %.6e s\n", timestepMax, timestepMax * dt_si);
        }
        printf("================================================================\n");
    }

    CheckStability(true);

    // Write SI conversion metadata for post-processing
    if (rank == 0 && dt_si > 0 && dx_si > 0) {
        FILE *meta = fopen("si_conversion.db", "w");
        fprintf(meta, "dx_si = %.15e\n", dx_si);
        fprintf(meta, "dt_si = %.15e\n", dt_si);
        fprintf(meta, "rho_ref = %.15e\n", rho_ref);
        fprintf(meta, "viscosity_A = %.15e\n", viscosity_A);
        fprintf(meta, "viscosity_B = %.15e\n", viscosity_B);
        fprintf(meta, "density_A = %.15e\n", density_A);
        fprintf(meta, "density_B = %.15e\n", density_B);
        fprintf(meta, "surface_tension = %.15e\n", surface_tension);
        fclose(meta);
    }
}

// =========================================================================
// Stability Check
// =========================================================================
bool ScaLBL_ColorModelSI::CheckStability(bool abort_on_failure) {
    bool stable = true;
    int error_count = 0;
    int warn_count = 0;

    double tau_min = min(tauA, tauB);
    double tau_max = max(tauA, tauB);
    double rho_ratio = (rhoB > 1.0e-15) ? rhoA / rhoB : 1.0e15;
    if (rho_ratio < 1.0) rho_ratio = 1.0 / rho_ratio;

    if (rank == 0) {
        printf("================================================================\n");
        printf("  Stability Analysis\n");
        printf("================================================================\n");
    }

    // --- CRITICAL: tau below 0.5 is unconditionally unstable ---
    if (tau_min < 0.5) {
        if (rank == 0) {
            printf("  CRITICAL: tau_min = %f < 0.5 -- UNCONDITIONALLY UNSTABLE\n", tau_min);
            printf("            The simulation CANNOT run with this configuration.\n");
        }
        stable = false;
        error_count++;
    }
    // --- ERROR: tau very close to 0.5 ---
    else if (tau_min < 0.505) {
        if (rank == 0) {
            printf("  ERROR: tau_min = %f is extremely close to 0.5\n", tau_min);
            printf("         Effective viscosity near zero. Severe numerical artifacts expected.\n");
            printf("         Remedy: increase target_tau or reduce the viscosity ratio.\n");
        }
        stable = false;
        error_count++;
    }
    // --- WARNING: tau marginally stable ---
    else if (tau_min < 0.55) {
        if (rank == 0) {
            printf("  WARNING: tau_min = %f is in the marginal zone [0.505, 0.55)\n", tau_min);
            printf("           Results may have significant numerical diffusion artifacts.\n");
        }
        warn_count++;
    }
    // --- OK ---
    else if (tau_min >= 0.55 && tau_min <= 2.0) {
        if (rank == 0) printf("  OK: tau_min = %f (in stable range [0.55, 2.0])\n", tau_min);
    }

    // --- WARNING: high tau ---
    if (tau_max > 2.0) {
        if (rank == 0) {
            printf("  WARNING: tau_max = %f > 2.0 -- accuracy degrades at high tau\n", tau_max);
            printf("           Consider lowering target_tau.\n");
        }
        warn_count++;
    }

    // --- Density ratio ---
    if (rho_ratio > 50.0) {
        if (rank == 0) {
            printf("  ERROR: density ratio = %.1f:1 exceeds color model capability (~50:1 max)\n", rho_ratio);
            printf("         The color gradient method cannot handle this contrast.\n");
            printf("         Remedy: use dimensionless matching mode with a capped density ratio.\n");
        }
        stable = false;
        error_count++;
    } else if (rho_ratio > 10.0) {
        if (rank == 0) {
            printf("  WARNING: density ratio = %.1f:1 is high for the color model\n", rho_ratio);
            printf("           Recommend keeping density ratio <= 10:1 for robust results.\n");
        }
        warn_count++;
    } else {
        if (rank == 0) printf("  OK: density ratio = %.2f:1\n", rho_ratio);
    }

    // --- Surface tension parameter alpha ---
    if (alpha < 1.0e-6) {
        if (rank == 0) {
            printf("  WARNING: alpha = %.6e is very small -- interface may not be maintained\n", alpha);
            printf("           Surface tension force may be negligible relative to numerical noise.\n");
        }
        warn_count++;
    } else if (alpha > 10.0) {
        if (rank == 0) {
            printf("  ERROR: alpha = %.6e is dangerously large (limit: 10)\n", alpha);
            printf("         The surface tension cannot be mapped to stable lattice parameters.\n");
            printf("         Laplace number too high for this voxel size / fluid combination.\n");
            printf("         Remedy: use dimensionless mode (--dimless) to match Ca/Re instead.\n");
        }
        stable = false;
        error_count++;
    } else if (alpha > 0.1) {
        if (rank == 0) {
            printf("  WARNING: alpha = %.6e is large -- may cause lattice velocity spikes\n", alpha);
            printf("           Consider using dimensionless mode (--dimless) if instability occurs.\n");
        }
        warn_count++;
    } else {
        if (rank == 0) printf("  OK: alpha = %.6e (in typical range [1e-6, 0.1])\n", alpha);
    }

    // --- Lattice velocity (Mach number) check ---
    // Open-channel Poiseuille estimate: u_max ~ F * L^2 / (8*nu)
    // NOTE: In porous media (BC=0) actual velocities are much lower due to solid resistance.
    //       This estimate is conservative; runtime NaN detection will catch real instability.
    double F_mag = sqrt(Fx*Fx + Fy*Fy + Fz*Fz);
    double nu_min_LB = (tau_min - 0.5) / 3.0;
    if (F_mag > 0 && nu_min_LB > 0) {
        double L_domain = max(double(Nx), max(double(Ny), double(Nz)));
        double u_est = F_mag * L_domain * L_domain / (8.0 * nu_min_LB);
        double Ma_est = u_est * sqrt(3.0); // Ma = u / c_s, c_s = 1/sqrt(3)
        if (Ma_est > 0.3 && BoundaryCondition != 0) {
            // Open-channel flow: Poiseuille estimate is realistic -> ERROR
            if (rank == 0) {
                printf("  ERROR: estimated Ma = %.4f > 0.3 -- compressibility errors dominate\n", Ma_est);
                printf("         Estimated max velocity = %.6e (lattice units)\n", u_est);
                printf("         Remedy: reduce body force or increase viscosity.\n");
            }
            stable = false;
            error_count++;
        } else if (Ma_est > 0.3) {
            // BC=0 (porous media): Poiseuille overestimates -> WARNING only
            if (rank == 0) {
                printf("  WARNING: open-channel Ma estimate = %.4f (BC=0: actual velocity will be much lower in porous media)\n", Ma_est);
                printf("           u_est = %.6e (Poiseuille, no solid resistance). F_LB = %.4e\n", u_est, F_mag);
            }
            warn_count++;
        } else if (Ma_est > 0.1) {
            if (rank == 0) {
                printf("  WARNING: estimated Ma = %.4f -- approaching compressibility limit\n", Ma_est);
                printf("           Recommend Ma < 0.1 for accuracy. u_est = %.6e\n", u_est);
            }
            warn_count++;
        } else if (u_est > 0) {
            if (rank == 0) printf("  OK: estimated Ma = %.4f (u_est = %.6e)\n", Ma_est, u_est);
        }
    }

    // --- Pressure BC check ---
    // Compare din/dout to the expected phase-weighted reference density,
    // not to rho=1.  With density_ratio != 1, dout = rhoB is correct.
    double rho_in_ref = rhoA;
    if (inletA + inletB > 0)
        rho_in_ref = (rhoA * inletA + rhoB * inletB) / (inletA + inletB);
    double rho_out_ref = rhoA;
    if (outletA + outletB > 0)
        rho_out_ref = (rhoA * outletA + rhoB * outletB) / (outletA + outletB);
    double din_dev  = (rho_in_ref  > 1e-10) ? fabs(din  - rho_in_ref)  / rho_in_ref  : 0.0;
    double dout_dev = (rho_out_ref > 1e-10) ? fabs(dout - rho_out_ref) / rho_out_ref : 0.0;
    if (din < 0.1 * rho_in_ref || dout < 0.1 * rho_out_ref) {
        if (rank == 0) {
            printf("  ERROR: din=%.6f or dout=%.6f far below phase reference (%.4f, %.4f)\n",
                   din, dout, rho_in_ref, rho_out_ref);
            printf("         The applied pressure difference is far too large.\n");
        }
        stable = false;
        error_count++;
    } else if (din_dev > 0.5 || dout_dev > 0.5) {
        if (rank == 0) {
            printf("  WARNING: din=%.6f (ref=%.4f, dev=%.1f%%), dout=%.6f (ref=%.4f, dev=%.1f%%)\n",
                   din, rho_in_ref, din_dev*100, dout, rho_out_ref, dout_dev*100);
            printf("           Large pressure deviation from phase equilibrium density.\n");
        }
        warn_count++;
    } else {
        if (rank == 0) printf("  OK: din = %.6f, dout = %.6f\n", din, dout);
    }

    // --- Summary ---
    if (rank == 0) {
        printf("----------------------------------------------------------------\n");
        if (error_count > 0)
            printf("  RESULT: %d ERROR(s), %d WARNING(s) -- UNSTABLE CONFIGURATION\n", error_count, warn_count);
        else if (warn_count > 0)
            printf("  RESULT: 0 errors, %d WARNING(s) -- proceed with caution\n", warn_count);
        else
            printf("  RESULT: All checks passed -- configuration is stable\n");
        printf("================================================================\n");
    }

    if (!stable && abort_on_failure) {
        if (rank == 0) {
            printf("\nAborting due to critical stability errors.\n");
            printf("Fix the input parameters or set abort_on_failure=false to override.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return stable;
}

void ScaLBL_ColorModelSI::PrintConversionInfo() {
    if (rank != 0) return;

    printf("================================================================\n");
    printf("  ColorModelSI: SI -> Lattice Unit Conversion Summary\n");
    printf("================================================================\n");
    printf("SI Inputs:\n");
    printf("  Viscosity A:      %.6e m^2/s\n", viscosity_A);
    printf("  Viscosity B:      %.6e m^2/s\n", viscosity_B);
    printf("  Density A:        %.4f kg/m^3\n", density_A);
    printf("  Density B:        %.4f kg/m^3\n", density_B);
    printf("  Surface tension:  %.6e N/m\n", surface_tension);
    if (dt_si > 0.0 && dx_si > 0.0 && rho_ref > 0.0) {
        printf("  Pressure grad:    (%.6e, %.6e, %.6e) Pa/m\n",
               Fx * rho_ref * dx_si / (dt_si * dt_si),
               Fy * rho_ref * dx_si / (dt_si * dt_si),
               Fz * rho_ref * dx_si / (dt_si * dt_si));
    }
    printf("  Inlet pressure:   %.4f Pa\n", inlet_pressure);
    printf("  Outlet pressure:  %.4f Pa\n", outlet_pressure);
    printf("  Flow rate:        %.6e m^3/s\n", flow_rate_SI);
    printf("  Target tau:       %.4f\n", target_tau);
    printf("Conversion Factors:\n");
    printf("  dx (voxel size):  %.6e m  (%.2f um)\n", dx_si, dx_si * 1e6);
    printf("  dt (time step):   %.6e s\n", dt_si);
    printf("  rho_ref:          %.4f kg/m^3\n", rho_ref);
    if (dt_si > 0.0 && dx_si > 0.0)
        printf("  c_s (lattice):    %.6e m/s\n", dx_si / (dt_si * sqrt(3.0)));
    printf("Lattice Parameters:\n");
    printf("  tauA: %.6f   tauB: %.6f\n", tauA, tauB);
    printf("  rhoA: %.6f   rhoB: %.6f\n", rhoA, rhoB);
    printf("  alpha (sfc. tension): %.6e\n", alpha);
    printf("  K_sigma (calibration): %.4f  (sigma_eff = K*alpha)\n", computeKSigma(target_tau));
    printf("  beta  (intfc. width): %.6f\n", beta);
    printf("  Fx: %.6e   Fy: %.6e   Fz: %.6e\n", Fx, Fy, Fz);
    printf("  din: %.8f   dout: %.8f\n", din, dout);
    printf("  flux: %.6e\n", flux);
    printf("  BC:   %d\n", BoundaryCondition);

    // Physical time and velocity scales
    if (dt_si > 0.0 && dx_si > 0.0) {
        double u_max_LB = 0.1;
        double u_max_SI = u_max_LB * dx_si / dt_si;
        printf("Derived Scales:\n");
        printf("  Physical time for %d steps: %.6e s\n", timestepMax, timestepMax * dt_si);
        printf("  Lattice velocity 0.1 corresponds to: %.6e m/s\n", u_max_SI);
    }
    if (viscosity_B > 0.0)
        printf("  Viscosity ratio (nuA/nuB): %.4f\n", viscosity_A / viscosity_B);
    if (density_B > 0.0)
        printf("  Density ratio   (rhoA/rhoB): %.4f\n", density_A / density_B);

    // ---- Compute and print dimensionless numbers ----
    double nu_LB_A = (tauA - 0.5) / 3.0;
    double nu_LB_B = (tauB - 0.5) / 3.0;
    double F_mag = sqrt(Fx*Fx + Fy*Fy + Fz*Fz);
    double L_LB = double(Nz * nprocz);

    // Estimate characteristic velocity from Poiseuille: u ~ F*L^2 / (8*nu)
    double u_LB = 0.0;
    if (F_mag > 0 && nu_LB_A > 0)
        u_LB = F_mag * L_LB * L_LB / (8.0 * nu_LB_A);

    double Ca_est = 0.0, Re_est = 0.0;
    if (alpha > 0 && u_LB > 0)
        Ca_est = rhoA * nu_LB_A * u_LB / alpha;
    if (nu_LB_A > 0 && u_LB > 0)
        Re_est = u_LB * L_LB / nu_LB_A;

    double M_val = (viscosity_B > 0.0) ? viscosity_A / viscosity_B : 1.0;
    double Lambda_val = (density_B > 0.0) ? density_A / density_B : 1.0;

    printf("Dimensionless Numbers (from SI inputs):\n");
    printf("  Ca  = %.6e    (capillary number)\n", Ca_est);
    printf("  Re  = %.6e    (Reynolds number, Poiseuille est.)\n", Re_est);
    printf("  M   = %.6f     (viscosity ratio nuA/nuB)\n", M_val);
    printf("  Lambda = %.6f  (density ratio rhoA/rhoB)\n", Lambda_val);
    printf("  u_LB (estimated) = %.6e\n", u_LB);

    printf("\n--- Equivalent dimensionless-mode input (paste into db file) ---\n");
    printf("  target_capillary_number = %.6e\n", Ca_est);
    printf("  target_reynolds_number = %.6e\n", Re_est);
    printf("  viscosity_ratio = %.6e\n", M_val);
    printf("  density_ratio = %.6e\n", Lambda_val);
    printf("  target_tau = 1.0\n");
    printf("--- end ---\n");
    printf("================================================================\n");
}
