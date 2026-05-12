/*
  FuelCellModel: Full MEA fuel cell operando simulator

  Hybrid Color-Gradient + Shan-Chen pseudopotential + Greyscale LBM for
  multi-component multiphase flow with thermodynamic EoS in a multiscale
  domain (resolved void + greyscale MPL/CL + membrane diffusion), coupled
  with Butler-Volmer electrochemistry and non-isothermal energy transport.

  DOMAIN REGIONS (from micro-CT segmentation label map):
    0 = solid (bounce-back)
    1 = resolved void (porosity=1)
    2 = MPL greyscale
    3 = catalyst layer (CL) greyscale + reaction source
    4 = membrane (Springer model)

  LATTICE SYSTEM (per active voxel):
    D3Q19       fq          Momentum (liquid water + gas mixture)
    D3Q7 x2     Aq, Bq      Phase A (liquid), Phase B (gas)
    D3Q7 x4     Cq[0..3]    Species: O2, N2, H2, H2O_vapor
    D3Q7 x2     PhiSq,PhiEq Electronic / protonic potential (Laplace)
    D3Q7         Tq          Temperature
*/
#ifndef FUELCELLMODEL_H
#define FUELCELLMODEL_H

#include "common/ScaLBL.h"
#include "common/Communication.h"
#include "common/MPI_Helpers.h"
#include "analysis/distance.h"

#include <string>
#include <vector>
#include <memory>

enum SpeciesIndex { SP_O2 = 0, SP_N2 = 1, SP_H2 = 2, SP_H2OV = 3, NUM_SPECIES = 4 };

enum RegionLabel {
    REGION_SOLID    = 0,
    REGION_VOID     = 1,
    REGION_MPL      = 2,
    REGION_CL       = 3,
    REGION_MEMBRANE = 4
};

class ScaLBL_FuelCellModel {
public:
    ScaLBL_FuelCellModel(int RANK, int NP, MPI_Comm COMM);
    ~ScaLBL_FuelCellModel();

    // ---- Lifecycle ----
    void ReadParams(std::string filename);
    void SetDomain();
    void ReadInput();
    void Create();
    void Initialize();
    void Run();
    void WriteDebug();

    // ---- Two-phase parameters ----
    bool Restart;
    int timestep, timestepMax;
    int BoundaryCondition;
    double tauA, tauB;
    double rhoA, rhoB;
    double alpha, beta;
    double Fx, Fy, Fz, flux;
    double din, dout;
    double inletA, inletB, outletA, outletB;

    // ---- SI conversion ----
    double dx_si, dt_si, rho_ref;

    // ---- Shan-Chen / Carnahan-Starling EoS ----
    double sc_G, cs_a, cs_b, cs_T;

    // ---- Greyscale ----
    double tauA_eff, tauB_eff;

    // ---- Species transport ----
    double D_O2, D_N2, D_H2, D_H2Ov;
    double c_O2_in, c_N2_in, c_H2_in, c_H2Ov_in;

    // ---- Electrochemistry ----
    double i0_cathode, i0_anode;
    double alpha_a_c, alpha_c_c, alpha_a_a, alpha_c_a;
    double E_eq, F_const, R_gas;
    double sigma_s, sigma_e;
    double V_cell;
    int    electro_interval;

    // ---- Thermal ----
    double T_ref;
    double k_thermal_gas, k_thermal_liquid, k_thermal_GDL, k_thermal_membrane;
    double h_fg, cp_ref;

    // ---- Membrane (Springer) ----
    double lambda_membrane, n_drag;

    // ---- Domain ----
    int Nx, Ny, Nz, N, Np;
    int rank, nprocx, nprocy, nprocz, nprocs;
    double Lx, Ly, Lz;

    // ---- Infrastructure ----
    std::shared_ptr<Domain> Dm;
    std::shared_ptr<Domain> Mask;
    std::shared_ptr<ScaLBL_Communicator> ScaLBL_Comm;
    std::shared_ptr<ScaLBL_Communicator> ScaLBL_Comm_Regular;

    std::shared_ptr<Database> db;
    std::shared_ptr<Database> domain_db;
    std::shared_ptr<Database> color_db;
    std::shared_ptr<Database> fuelcell_db;
    std::shared_ptr<Database> analysis_db;

    IntArray Map;
    char *id;

    // ---- Device arrays: momentum (D3Q19) ----
    int *NeighborList;
    int *dvcMap;
    double *fq;
    double *Aq, *Bq;
    double *Den;
    double *Phi;
    double *ColorGrad;
    double *Velocity;
    double *Pressure;

    // ---- Device arrays: Shan-Chen force ----
    double *ForceX, *ForceY, *ForceZ;

    // ---- Device arrays: greyscale ----
    double *Poros, *Perm, *GreySolidGrad;

    // ---- Device arrays: species (D3Q7, 7*Np each) ----
    double *Cq[NUM_SPECIES];
    double *ConcentrationDev[NUM_SPECIES];

    // ---- Device arrays: species source terms ----
    double *SourceO2, *SourceN2, *SourceH2, *SourceH2O;

    // ---- Device arrays: temperature ----
    double *Tq;
    double *Temperature;
    double *SourceThermal;

    // ---- Device arrays: potentials (D3Q7, 7*Np each) ----
    double *PhiSq, *PhiEq;       // D3Q7 distributions
    double *PhiS, *PhiE;         // scalar potential fields

    // ---- Device arrays: phase change ----
    double *SourcePhaseField;

    // ---- Device arrays: per-voxel material ----
    double *RegionID;
    double *DiffCoeff, *ThermalCond, *ElecCond, *IonCond;
    double *ReactionRate;

    // ---- Cartesian arrays (output) ----
    DoubleArray Distance, PhaseField, Pressure_Cart;
    DoubleArray Velocity_x, Velocity_y, Velocity_z;
    DoubleArray Temperature_Cart;
    DoubleArray Concentration_Cart[NUM_SPECIES];
    DoubleArray PhiS_Cart, PhiE_Cart;

private:
    MPI_Comm comm;
    size_t dist_mem_size;
    size_t neighborSize;
    char LocalRankString[8];
    char LocalRankFilename[40];
    char LocalRestartFile[40];

    void LoadParams(std::shared_ptr<Database> db0);
    void AssignComponentLabels();
    void AssignMaterialProperties();
    void SolveElectricPotentials();
    void ComputeButlerVolmerSources();
    void ComputePhaseChangeSources();
};

#endif
