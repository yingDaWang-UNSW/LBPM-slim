/*
  ColorModelSI: SI-unit interface for the Color Lattice Boltzmann Model
*/
#include "models/ColorModelSI.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

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

// =========================================================================
// preloadPorosity: read the raw geometry file (rank 0 only) and count
// fluid voxels over the simulation domain to compute porosity.
// Must be called AFTER db/domain_db are set but BEFORE SetDomain().
// =========================================================================
double ScaLBL_ColorModelSI::preloadPorosity() {
    double porosity = 0.0;

    // Read domain parameters needed for raw file reading
    auto Filename = domain_db->getScalar<std::string>("Filename");
    auto SIZE = domain_db->getVector<int>("N");        // raw file dimensions
    auto ReadValues = domain_db->getVector<char>("ReadValues");
    auto WriteValues = domain_db->getVector<char>("WriteValues");
    std::string ReadType = "8bit";
    if (domain_db->keyExists("ReadType"))
        ReadType = domain_db->getScalar<std::string>("ReadType");

    int64_t NX = SIZE[0], NY = SIZE[1], NZ = SIZE[2]; // raw file dimensions

    // Offset into the raw file
    int64_t xStart = 0, yStart = 0, zStart = 0;
    if (domain_db->keyExists("offset")) {
        auto offset = domain_db->getVector<int>("offset");
        xStart = offset[0]; yStart = offset[1]; zStart = offset[2];
    }

    // Simulation domain extent in voxels
    int64_t simNx = int64_t(Nx) * nprocx;
    int64_t simNy = int64_t(Ny) * nprocy;
    int64_t simNz = int64_t(Nz) * nprocz;

    if (rank == 0) {
        int64_t TOTAL = NX * NY * NZ;
        char *SegData = new char[TOTAL];

        // Read the raw file
        FILE *fp = fopen(Filename.c_str(), "rb");
        if (fp == NULL) {
            printf("WARNING: Cannot open geometry file '%s' for porosity preload\n", Filename.c_str());
            delete[] SegData;
            return 0.0;
        }

        if (ReadType == "16bit") {
            short int *tmp = new short int[TOTAL];
            size_t nread = fread(tmp, 2, TOTAL, fp);
            fclose(fp);
            if (nread != size_t(TOTAL)) {
                printf("WARNING: Short read of geometry file (16bit)\n");
                delete[] tmp; delete[] SegData;
                return 0.0;
            }
            for (int64_t n = 0; n < TOTAL; n++) SegData[n] = char(tmp[n]);
            delete[] tmp;
        } else {
            size_t nread = fread(SegData, 1, TOTAL, fp);
            fclose(fp);
            if (nread != size_t(TOTAL)) {
                printf("WARNING: Short read of geometry file (8bit)\n");
                delete[] SegData;
                return 0.0;
            }
        }

        // Count fluid voxels over the simulation domain region
        int64_t fluid_count = 0;
        int64_t total_count = 0;
        for (int64_t kk = 0; kk < simNz; kk++) {
            for (int64_t jj = 0; jj < simNy; jj++) {
                for (int64_t ii = 0; ii < simNx; ii++) {
                    int64_t x = xStart + ii;
                    int64_t y = yStart + jj;
                    int64_t z = zStart + kk;
                    // Clamp to raw file bounds
                    if (x < 0) x = 0; if (x >= NX) x = NX - 1;
                    if (y < 0) y = 0; if (y >= NY) y = NY - 1;
                    if (z < 0) z = 0; if (z >= NZ) z = NZ - 1;
                    int64_t idx = z * NX * NY + y * NX + x;
                    char raw_val = SegData[idx];
                    // Apply ReadValues -> WriteValues mapping
                    char mapped_val = raw_val;
                    for (size_t m = 0; m < ReadValues.size(); m++) {
                        if (raw_val == ReadValues[m]) {
                            mapped_val = WriteValues[m];
                            break;
                        }
                    }
                    total_count++;
                    if (mapped_val > 0) fluid_count++;
                }
            }
        }
        delete[] SegData;

        if (total_count > 0)
            porosity = double(fluid_count) / double(total_count);
        printf("Geometry preload: %s (%ldx%ldx%ld)\n", Filename.c_str(), simNx, simNy, simNz);
        printf("  Fluid voxels: %ld / %ld  ->  porosity = %.6f\n", fluid_count, total_count, porosity);
    }

    // Broadcast porosity to all ranks
    MPI_Bcast(&porosity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return porosity;
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
    if (color_db->keyExists("viscosity_B"))
        viscosity_B = color_db->getScalar<double>("viscosity_B");
    else if (color_db->keyExists("viscosity_ratio"))
        viscosity_B = viscosity_A / color_db->getScalar<double>("viscosity_ratio");
    else
        ERROR("Must specify either viscosity_B or viscosity_ratio");
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

    // 7. Pressure BCs: centre around rho=1 (the D3Q19 reference density).
    //    The phase composition at boundaries is handled separately by Color_BC
    //    which sets Den (nA,nB) and Phi.  The D3Q19 density (sum of fi) stays
    //    near 1.0 regardless of the phase — rho0 (from phase) appears only in
    //    the collision equilibrium.  Setting dout=rhoB would create a massive
    //    Zou-He velocity: uz = -rhoB + sum(fi) ≈ -(rhoB-1).  
    //    dp_LB = (P_in - P_out) * dt^2 / (rho_ref * dx^2)
    double dp_SI = inlet_pressure - outlet_pressure;
    double dp_LB = dp_SI * dt_si * dt_si / (rho_ref * dx_si * dx_si);
    din  = 1.0 + 1.5 * dp_LB;
    dout = 1.0 - 1.5 * dp_LB;

    // 8. Flux (volumetric flow rate -> lattice flux)
    //    flux_LB = Q_SI * dt / dx^3.  With rho ~= 1, mass flux = volumetric flux.
    flux = flow_rate_SI * dt_si / (dx_si * dx_si * dx_si);

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
//
// Accepts EITHER:
//   (a) capillary_number + reynolds_number  (pure dimensionless)
//   (b) pressure_gradient [Pa/m] + permeability [mD]  (SI-driven)
//   (c) flow_rate [m³/s]                              (SI-driven)
//
// EXPERIMENTAL CONVENTION: Ca and Re are defined with the Darcy
// (superficial) velocity  v_D = Q/A  — the quantity an experimentalist
// measures at the core face without seeing into the sample:
//     Ca = mu * v_D / sigma       Re = v_D * sqrt(k) / nu
// The characteristic length in Re is sqrt(permeability) — an intrinsic
// property of the medium — so the dimensionless numbers are unique
// regardless of sample or domain size.  Permeability is REQUIRED.
// Porosity is still preloaded for informational output.
// =========================================================================
void ScaLBL_ColorModelSI::ReadParamsDimensionless(string filename) {
    // Read the input database
    db = make_shared<Database>(filename);
    domain_db = db->getDatabase("Domain");
    color_db = db->getDatabase("Color");
    analysis_db = db->getDatabase("Analysis");

    // ==========================================
    // Read common parameters
    // ==========================================
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
    // Determine input mode
    // ==========================================
    bool has_Ca_Re = (color_db->keyExists("capillary_number") || color_db->keyExists("target_capillary_number"))
                  && (color_db->keyExists("reynolds_number")  || color_db->keyExists("target_reynolds_number"));
    bool has_dP    = color_db->keyExists("pressure_gradient");
    bool has_flux  = color_db->keyExists("flow_rate");
    bool has_perm  = color_db->keyExists("permeability");

    // Read SI fluid properties (needed for SI-driven modes, optional for Ca/Re mode)
    if (color_db->keyExists("viscosity_A"))
        viscosity_A = color_db->getScalar<double>("viscosity_A");
    if (color_db->keyExists("viscosity_B"))
        viscosity_B = color_db->getScalar<double>("viscosity_B");
    if (color_db->keyExists("density_A"))
        density_A = color_db->getScalar<double>("density_A");
    if (color_db->keyExists("density_B"))
        density_B = color_db->getScalar<double>("density_B");
    if (color_db->keyExists("surface_tension"))
        surface_tension = color_db->getScalar<double>("surface_tension");

    // ==========================================
    // Auto-compute lattice parameters (tau, rho, nu)
    // ==========================================
    double L_LB = double(Nz * nprocz);
    double nu_LB_max = (target_tau - 0.5) / 3.0;

    double nu_LB_A, nu_LB_B;
    if (viscosity_ratio >= 1.0) {
        nu_LB_A = nu_LB_max;
        nu_LB_B = nu_LB_max / viscosity_ratio;
    } else {
        nu_LB_B = nu_LB_max;
        nu_LB_A = nu_LB_max * viscosity_ratio;
    }

    tauA = 3.0 * nu_LB_A + 0.5;
    tauB = 3.0 * nu_LB_B + 0.5;
    rhoA = 1.0;
    rhoB = 1.0 / density_ratio;

    // ==========================================
    // Compute dx_si, dt_si from voxel_length + viscosity_A
    // ==========================================
    dx_si = 0.0; dt_si = 0.0; rho_ref = 0.0;
    if (domain_db->keyExists("voxel_length")) {
        dx_si = domain_db->getScalar<double>("voxel_length") * 1.0e-6;
        if (viscosity_A > 0.0)
            dt_si = nu_LB_A * dx_si * dx_si / viscosity_A;
    }
    if (density_A > 0.0) rho_ref = density_A;

    // ==========================================
    // Determine characteristic velocity u_LB
    // u_LB represents the DARCY (superficial) velocity in lattice units.
    // This matches the experimental convention: v_D = Q/A.
    // ==========================================
    double u_LB = 0.0;
    double porosity_preload = 0.0;
    double perm_SI = 0.0;  // permeability in m²
    double dPdz_SI = 0.0;  // pressure gradient magnitude [Pa/m]
    double Q_SI = 0.0;     // volumetric flow rate [m³/s]

    // Read permeability (REQUIRED for Re = v_D * sqrt(k) / nu)
    if (has_perm) {
        perm_SI = color_db->getScalar<double>("permeability") * 9.869233e-16;  // mD -> m²
    } else {
        if (rank == 0)
            printf("ERROR: 'permeability' [mD] is required (Re = v_D * sqrt(k) / nu)\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double k_LB = perm_SI / (dx_si > 0 ? dx_si * dx_si : 1.0);  // lattice permeability
    double sqrt_k_LB = sqrt(k_LB);

    if (!has_Ca_Re && (has_dP || has_flux)) {
        // ----- SI-driven mode: derive Ca & Re from physical inputs -----
        // Validate required SI properties
        if (viscosity_A <= 0.0 || density_A <= 0.0 || surface_tension <= 0.0 || dx_si <= 0.0) {
            if (rank == 0)
                printf("ERROR: SI-driven dimless mode requires viscosity_A, density_A, surface_tension, voxel_length\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Preload geometry porosity (informational; not required for Ca/Re)
        porosity_preload = preloadPorosity();

        double A_total = double(Nx * nprocx) * double(Ny * nprocy);  // cross-section in voxels²
        double A_SI = A_total * dx_si * dx_si;                       // cross-section in m²

        double mu_A = viscosity_A * density_A;  // dynamic viscosity [Pa·s]
        double u_Darcy = 0.0;

        if (has_dP) {
            // Mode (b): pressure gradient + permeability -> Darcy velocity
            auto dPdx_SI = color_db->getVector<double>("pressure_gradient");
            dPdz_SI = dPdx_SI[2];  // z-component (flow direction)
            if (perm_SI > 0.0) {
                // Darcy: u_Darcy = (k / mu) * |dP/dz|  [superficial velocity]
                u_Darcy = perm_SI * fabs(dPdz_SI) / mu_A;
            } else {
                if (rank == 0) printf("ERROR: pressure_gradient mode requires 'permeability' [mD]\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            // Mode (c): flow rate -> Darcy velocity
            Q_SI = color_db->getScalar<double>("flow_rate");
            u_Darcy = Q_SI / A_SI;
            // Back-derive dP from Darcy if permeability given (for body force)
            if (perm_SI > 0.0)
                dPdz_SI = mu_A * u_Darcy / perm_SI;
        }

        if (has_dP) Q_SI = u_Darcy * A_SI;

        // Convert Darcy velocity to lattice units
        u_LB = u_Darcy * dt_si / dx_si;

        // Derive dimensionless numbers from Darcy velocity
        // Ca = mu * v_D / sigma,  Re = v_D * sqrt(k) / nu
        capillary_number = mu_A * u_Darcy / surface_tension;
        reynolds_number  = u_Darcy * sqrt(perm_SI) / viscosity_A;

        if (rank == 0) {
            printf("================================================================\n");
            printf("  ColorModelSI: SI-Driven Dimensionless Mode\n");
            printf("  (Experimental convention: Ca & Re use Darcy velocity)\n");
            printf("================================================================\n");
            printf("SI Inputs:\n");
            if (has_dP) printf("  Pressure gradient (z): %.4e Pa/m\n", dPdz_SI);
            if (has_flux) printf("  Flow rate: %.4e m3/s\n", Q_SI);
            printf("  Permeability: %.4f mD  (%.4e m2)  sqrt(k) = %.4e m\n", perm_SI/9.869233e-16, perm_SI, sqrt(perm_SI));
            if (porosity_preload > 0.0) printf("  Porosity (preloaded): %.4f\n", porosity_preload);
            printf("  mu_A = %.4e Pa.s  |  sigma = %.4e N/m\n", mu_A, surface_tension);
            printf("  v_Darcy = %.4e m/s  |  Q = %.4e m3/s\n", u_Darcy, Q_SI);
            if (porosity_preload > 0.0)
                printf("  v_pore  = %.4e m/s  (= v_Darcy / phi, for reference)\n", u_Darcy / porosity_preload);
            printf("Derived Dimensionless Numbers (Darcy-velocity, sqrt(k) length):\n");
            printf("  Ca = mu*v_D/sigma      = %.6e\n", capillary_number);
            printf("  Re = v_D*sqrt(k)/nu    = %.6e\n", reynolds_number);
        }
    } else {
        // ----- Mode (a): pure Ca/Re input -----
        if (color_db->keyExists("target_capillary_number"))
            capillary_number = color_db->getScalar<double>("target_capillary_number");
        else
            capillary_number = color_db->getScalar<double>("capillary_number");

        if (color_db->keyExists("target_reynolds_number"))
            reynolds_number = color_db->getScalar<double>("target_reynolds_number");
        else
            reynolds_number = color_db->getScalar<double>("reynolds_number");

        // u_LB from Re:  Re = v_D * sqrt(k) / nu  =>  u_LB = Re * nu_LB / sqrt(k_LB)
        u_LB = reynolds_number * nu_LB_A / sqrt_k_LB;

        if (rank == 0) {
            printf("================================================================\n");
            printf("  ColorModelSI: Dimensionless Number Matching Mode\n");
            printf("================================================================\n");
            printf("Dimensionless Inputs:\n");
            printf("  Capillary number (Ca = mu*v_D/sigma):   %.6e\n", capillary_number);
            printf("  Reynolds number  (Re = v_D*sqrt(k)/nu): %.6e\n", reynolds_number);
            printf("  Permeability: %.4f mD  (k_LB = %.4e, sqrt(k_LB) = %.4e)\n",
                   perm_SI/9.869233e-16, k_LB, sqrt_k_LB);
        }
    }

    // ==========================================
    // Surface tension (alpha) from Ca
    // ==========================================
    double K_sigma = computeKSigma(target_tau);
    if (capillary_number > 0.0) {
        alpha = rhoA * nu_LB_A * u_LB / (capillary_number * K_sigma);
    } else {
        alpha = 1.0e-3;
    }

    // ==========================================
    // Body force / din-dout / flux  (BC-dependent)
    // ==========================================
    // Darcy:  F = nu * u / k_LB  (permeability is mandatory)
    Fx = 0.0; Fy = 0.0; Fz = 0.0;
    if (BoundaryCondition == 0) {
        Fz = nu_LB_A * u_LB / k_LB;  // Darcy: F = nu * u / k
    }

    // Pressure BCs (BC=3)
    din = 1.0; dout = 1.0;
    if (BoundaryCondition == 3) {
        double dp_LB = nu_LB_A * u_LB * L_LB / k_LB;  // dp = F * L = (nu*u/k)*L
        din  = 1.0 + 1.5 * dp_LB;
        dout = 1.0 - 1.5 * dp_LB;
    }

    // Flux (BC=4)
    // u_LB is Darcy velocity => total volumetric flux = v_D * A_total
    flux = 0.0;
    if (BoundaryCondition == 4) {
        double A_cross = double(Nx * nprocx) * double(Ny * nprocy);
        flux = u_LB * A_cross;
    }

    // ==========================================
    // Print summary
    // ==========================================
    if (rank == 0) {
        printf("  Viscosity ratio  (M = nuA/nuB): %.6f\n", viscosity_ratio);
        printf("  Density ratio    (Lambda = rhoA/rhoB): %.6f\n", density_ratio);
        printf("  Target tau:       %.4f\n", target_tau);
        printf("Derived Lattice Parameters:\n");
        printf("  tauA: %.6f   tauB: %.6f\n", tauA, tauB);
        printf("  rhoA: %.6f   rhoB: %.6f\n", rhoA, rhoB);
        printf("  nu_LB_A: %.6e   nu_LB_B: %.6e\n", nu_LB_A, nu_LB_B);
        printf("  alpha (sfc. tension): %.6e\n", alpha);
        printf("  beta  (intfc. width): %.6f\n", beta);
        printf("  u_LB (Darcy vel, lattice): %.6e\n", u_LB);
        printf("  Fz (Darcy: nu*u/k): %.6e  (k_LB = %.4e)\n", Fz, k_LB);
        printf("  din: %.8f   dout: %.8f\n", din, dout);
        printf("  flux: %.6e\n", flux);
        printf("  BC:   %d\n", BoundaryCondition);
        printf("  L_LB (domain z): %.0f\n", L_LB);
        printf("Verification of dimensionless groups:\n");
        double Ca_check = (alpha > 0) ? rhoA * nu_LB_A * u_LB / (alpha * K_sigma) : 0.0;
        double Re_check = u_LB * sqrt_k_LB / nu_LB_A;
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
        if (perm_SI > 0.0) fprintf(meta, "permeability_mD = %.15e\n", perm_SI/9.869233e-16);
        if (porosity_preload > 0.0) fprintf(meta, "porosity = %.15e\n", porosity_preload);
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
    // din/dout should be close to 1.0 (the D3Q19 reference density).
    // The phase composition at boundaries is independent of din/dout.
    if (din < 0.5 || din > 2.0 || dout < 0.5 || dout > 2.0) {
        if (rank == 0) {
            printf("  ERROR: din=%.6f or dout=%.6f far from reference density 1.0\n", din, dout);
            printf("         Zou-He BC requires din,dout close to 1.0 (D3Q19 density).\n");
        }
        stable = false;
        error_count++;
    } else if (fabs(din - 1.0) > 0.1 || fabs(dout - 1.0) > 0.1) {
        if (rank == 0) {
            printf("  WARNING: din=%.6f, dout=%.6f -- large deviation from rho=1\n", din, dout);
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
