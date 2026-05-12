/*
  FuelCellModel.cpp — Full MEA fuel cell operando simulator

  Hybrid Color-Gradient + Shan-Chen + Greyscale LBM with Carnahan-Starling EoS,
  Butler-Volmer electrochemistry, Springer membrane model, and thermal transport.
*/
#include "models/FuelCellModel.h"
#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>

using namespace std;

// =========================================================================
//  Constructor / Destructor
// =========================================================================
ScaLBL_FuelCellModel::ScaLBL_FuelCellModel(int RANK, int NP, MPI_Comm COMM)
    : Restart(false), timestep(0), timestepMax(0),
      BoundaryCondition(0),
      tauA(1.0), tauB(1.0), rhoA(1.0), rhoB(1.0),
      alpha(0), beta(0.95),
      Fx(0), Fy(0), Fz(0), flux(0),
      din(1.0), dout(1.0),
      inletA(1.0), inletB(0.0), outletA(0.0), outletB(1.0),
      dx_si(0), dt_si(0), rho_ref(0),
      sc_G(-1.0), cs_a(0.4963), cs_b(4.0), cs_T(0.7),
      tauA_eff(1.0), tauB_eff(1.0),
      D_O2(0.1), D_N2(0.1), D_H2(0.1), D_H2Ov(0.1),
      c_O2_in(0.0), c_N2_in(0.0), c_H2_in(0.0), c_H2Ov_in(0.0),
      i0_cathode(1.0e-3), i0_anode(1.0),
      alpha_a_c(0.5), alpha_c_c(0.5), alpha_a_a(0.5), alpha_c_a(0.5),
      E_eq(1.23), F_const(96485.3329), R_gas(8.314462),
      sigma_s(300.0), sigma_e(10.0), V_cell(0.6),
      electro_interval(100),
      T_ref(353.15),
      k_thermal_gas(0.03), k_thermal_liquid(0.6),
      k_thermal_GDL(1.7), k_thermal_membrane(0.2),
      h_fg(2.26e6), cp_ref(4180.0),
      lambda_membrane(14.0), n_drag(1.0),
      Nx(0), Ny(0), Nz(0), N(0), Np(0),
      rank(RANK), nprocx(0), nprocy(0), nprocz(0), nprocs(NP),
      Lx(0), Ly(0), Lz(0),
      id(nullptr), NeighborList(nullptr), dvcMap(nullptr),
      fq(nullptr), Aq(nullptr), Bq(nullptr),
      Den(nullptr), Phi(nullptr), ColorGrad(nullptr),
      Velocity(nullptr), Pressure(nullptr),
      ForceX(nullptr), ForceY(nullptr), ForceZ(nullptr),
      Poros(nullptr), Perm(nullptr), GreySolidGrad(nullptr),
      SourceO2(nullptr), SourceN2(nullptr), SourceH2(nullptr), SourceH2O(nullptr),
      Tq(nullptr), Temperature(nullptr), SourceThermal(nullptr),
      PhiSq(nullptr), PhiEq(nullptr), PhiS(nullptr), PhiE(nullptr),
      SourcePhaseField(nullptr),
      RegionID(nullptr), DiffCoeff(nullptr), ThermalCond(nullptr),
      ElecCond(nullptr), IonCond(nullptr), ReactionRate(nullptr),
      comm(COMM), dist_mem_size(0), neighborSize(0)
{
    for (int s = 0; s < NUM_SPECIES; s++) {
        Cq[s] = nullptr;
        ConcentrationDev[s] = nullptr;
    }
}

ScaLBL_FuelCellModel::~ScaLBL_FuelCellModel()
{
    delete[] id;
    ScaLBL_FreeDeviceMemory(NeighborList);
    ScaLBL_FreeDeviceMemory(dvcMap);
    ScaLBL_FreeDeviceMemory(fq);
    ScaLBL_FreeDeviceMemory(Aq);
    ScaLBL_FreeDeviceMemory(Bq);
    ScaLBL_FreeDeviceMemory(Den);
    ScaLBL_FreeDeviceMemory(Phi);
    ScaLBL_FreeDeviceMemory(ColorGrad);
    ScaLBL_FreeDeviceMemory(Velocity);
    ScaLBL_FreeDeviceMemory(Pressure);
    ScaLBL_FreeDeviceMemory(ForceX);
    ScaLBL_FreeDeviceMemory(ForceY);
    ScaLBL_FreeDeviceMemory(ForceZ);
    ScaLBL_FreeDeviceMemory(Poros);
    ScaLBL_FreeDeviceMemory(Perm);
    ScaLBL_FreeDeviceMemory(GreySolidGrad);
    for (int s = 0; s < NUM_SPECIES; s++) {
        ScaLBL_FreeDeviceMemory(Cq[s]);
        ScaLBL_FreeDeviceMemory(ConcentrationDev[s]);
    }
    ScaLBL_FreeDeviceMemory(SourceO2);
    ScaLBL_FreeDeviceMemory(SourceN2);
    ScaLBL_FreeDeviceMemory(SourceH2);
    ScaLBL_FreeDeviceMemory(SourceH2O);
    ScaLBL_FreeDeviceMemory(Tq);
    ScaLBL_FreeDeviceMemory(Temperature);
    ScaLBL_FreeDeviceMemory(SourceThermal);
    ScaLBL_FreeDeviceMemory(PhiSq);
    ScaLBL_FreeDeviceMemory(PhiEq);
    ScaLBL_FreeDeviceMemory(PhiS);
    ScaLBL_FreeDeviceMemory(PhiE);
    ScaLBL_FreeDeviceMemory(SourcePhaseField);
    ScaLBL_FreeDeviceMemory(RegionID);
    ScaLBL_FreeDeviceMemory(DiffCoeff);
    ScaLBL_FreeDeviceMemory(ThermalCond);
    ScaLBL_FreeDeviceMemory(ElecCond);
    ScaLBL_FreeDeviceMemory(IonCond);
    ScaLBL_FreeDeviceMemory(ReactionRate);
}

// =========================================================================
//  ReadParams
// =========================================================================
void ScaLBL_FuelCellModel::ReadParams(string filename)
{
    db = make_shared<Database>(filename);
    domain_db  = db->getDatabase("Domain");
    color_db   = db->getDatabase("Color");
    fuelcell_db = db->getDatabase("FuelCell");
    analysis_db = db->getDatabase("Analysis");

    auto L     = domain_db->getVector<double>("L");
    auto size  = domain_db->getVector<int>("n");
    auto nproc = domain_db->getVector<int>("nproc");
    BoundaryCondition = domain_db->getScalar<int>("BC");
    Nx = size[0]; Ny = size[1]; Nz = size[2];
    Lx = L[0];   Ly = L[1];   Lz = L[2];
    nprocx = nproc[0]; nprocy = nproc[1]; nprocz = nproc[2];

    timestepMax = color_db->getScalar<int>("timestepMax");
    tauA   = color_db->getScalar<double>("tauA");
    tauB   = color_db->getScalar<double>("tauB");
    rhoA   = color_db->getScalar<double>("rhoA");
    rhoB   = color_db->getScalar<double>("rhoB");
    alpha  = color_db->getScalar<double>("alpha");
    beta   = color_db->getScalar<double>("beta");
    Restart = color_db->getScalar<bool>("Restart");

    auto F = color_db->getVector<double>("F");
    Fx = F[0]; Fy = F[1]; Fz = F[2];

    din  = color_db->getWithDefault<double>("din", 1.0);
    dout = color_db->getWithDefault<double>("dout", 1.0);
    flux = color_db->getWithDefault<double>("flux", 0.0);
    inletA  = color_db->getWithDefault<double>("inletA", 1.0);
    inletB  = color_db->getWithDefault<double>("inletB", 0.0);
    outletA = color_db->getWithDefault<double>("outletA", 0.0);
    outletB = color_db->getWithDefault<double>("outletB", 1.0);

    sc_G = fuelcell_db->getWithDefault<double>("sc_G", -1.0);
    cs_a = fuelcell_db->getWithDefault<double>("cs_a", 0.4963);
    cs_b = fuelcell_db->getWithDefault<double>("cs_b", 4.0);
    cs_T = fuelcell_db->getWithDefault<double>("cs_T", 0.7);
    tauA_eff = fuelcell_db->getWithDefault<double>("tauA_eff", tauA);
    tauB_eff = fuelcell_db->getWithDefault<double>("tauB_eff", tauB);
    D_O2   = fuelcell_db->getWithDefault<double>("D_O2",   0.1);
    D_N2   = fuelcell_db->getWithDefault<double>("D_N2",   0.08);
    D_H2   = fuelcell_db->getWithDefault<double>("D_H2",   0.15);
    D_H2Ov = fuelcell_db->getWithDefault<double>("D_H2Ov", 0.12);
    c_O2_in   = fuelcell_db->getWithDefault<double>("c_O2_inlet",   0.21);
    c_N2_in   = fuelcell_db->getWithDefault<double>("c_N2_inlet",   0.79);
    c_H2_in   = fuelcell_db->getWithDefault<double>("c_H2_inlet",   1.0);
    c_H2Ov_in = fuelcell_db->getWithDefault<double>("c_H2Ov_inlet", 0.0);
    i0_cathode = fuelcell_db->getWithDefault<double>("i0_cathode", 1.0e-3);
    i0_anode   = fuelcell_db->getWithDefault<double>("i0_anode",   1.0);
    alpha_a_c  = fuelcell_db->getWithDefault<double>("alpha_a_cathode", 0.5);
    alpha_c_c  = fuelcell_db->getWithDefault<double>("alpha_c_cathode", 0.5);
    alpha_a_a  = fuelcell_db->getWithDefault<double>("alpha_a_anode",   0.5);
    alpha_c_a  = fuelcell_db->getWithDefault<double>("alpha_c_anode",   0.5);
    E_eq       = fuelcell_db->getWithDefault<double>("E_eq", 1.23);
    V_cell     = fuelcell_db->getWithDefault<double>("V_cell", 0.6);
    sigma_s    = fuelcell_db->getWithDefault<double>("sigma_s", 300.0);
    sigma_e    = fuelcell_db->getWithDefault<double>("sigma_e", 10.0);
    electro_interval = fuelcell_db->getWithDefault<int>("electro_interval", 100);
    T_ref = fuelcell_db->getWithDefault<double>("T_ref", 353.15);
    k_thermal_gas       = fuelcell_db->getWithDefault<double>("k_thermal_gas", 0.03);
    k_thermal_liquid    = fuelcell_db->getWithDefault<double>("k_thermal_liquid", 0.6);
    k_thermal_GDL       = fuelcell_db->getWithDefault<double>("k_thermal_GDL", 1.7);
    k_thermal_membrane  = fuelcell_db->getWithDefault<double>("k_thermal_membrane", 0.2);
    h_fg  = fuelcell_db->getWithDefault<double>("h_fg", 2.26e6);
    cp_ref = fuelcell_db->getWithDefault<double>("cp_ref", 4180.0);
    lambda_membrane = fuelcell_db->getWithDefault<double>("lambda_membrane", 14.0);
    n_drag = fuelcell_db->getWithDefault<double>("n_drag", 1.0);

    if (rank == 0) {
        printf("========================================\n");
        printf("  LBPM Fuel Cell Simulator Parameters\n");
        printf("========================================\n");
        printf("Domain: %d x %d x %d, procs %d x %d x %d\n",
               Nx, Ny, Nz, nprocx, nprocy, nprocz);
        printf("Two-phase: tauA=%f, tauB=%f, rhoA=%f, rhoB=%f\n",
               tauA, tauB, rhoA, rhoB);
        printf("SC-EoS: G=%f, a=%f, b=%f, T/Tc=%f\n",
               sc_G, cs_a, cs_b, cs_T);
        printf("Species D: O2=%f, N2=%f, H2=%f, H2Ov=%f\n",
               D_O2, D_N2, D_H2, D_H2Ov);
        printf("Electrochem: i0c=%e, i0a=%e, Eeq=%f, Vcell=%f\n",
               i0_cathode, i0_anode, E_eq, V_cell);
        printf("Thermal: T_ref=%f K, h_fg=%e J/kg\n", T_ref, h_fg);
        printf("========================================\n");
    }
}

// =========================================================================
//  SetDomain
// =========================================================================
void ScaLBL_FuelCellModel::SetDomain()
{
    Dm = shared_ptr<Domain>(new Domain(domain_db, comm));
    Nx += 2; Ny += 2; Nz += 2;
    N = Nx * Ny * Nz;
    id = new char[N];
    for (int i = 0; i < N; i++) Dm->id[i] = 1;
    Distance.resize(Nx, Ny, Nz);
    MPI_Barrier(Dm->Comm);
    rank = Dm->rank();
}

// =========================================================================
//  ReadInput
// =========================================================================
void ScaLBL_FuelCellModel::ReadInput()
{
    Dm->ReadIDs();
    for (int i = 0; i < N; i++) id[i] = Dm->id[i];

    sprintf(LocalRankString, "%05d", rank);
    sprintf(LocalRankFilename, "%s%s", "ID.", LocalRankString);
    sprintf(LocalRestartFile, "%s%s", "Restart.", LocalRankString);

    Array<char> id_solid(Nx, Ny, Nz);
    for (int k = 0; k < Nz; k++)
        for (int j = 0; j < Ny; j++)
            for (int i = 0; i < Nx; i++) {
                int n = k * Nx * Ny + j * Nx + i;
                id_solid(i, j, k) = (Dm->id[n] > 0) ? 1 : 0;
            }
    if (rank == 0) printf("Computing signed distance function\n");
    CalcDist(Distance, id_solid, *Dm);
    if (rank == 0) printf("Domain set.\n");
}

// =========================================================================
//  AssignComponentLabels
// =========================================================================
void ScaLBL_FuelCellModel::AssignComponentLabels()
{
    double *PhaseLabel = new double[N];
    for (int n = 0; n < N; n++) {
        char val = id[n];
        if (val == REGION_SOLID)      PhaseLabel[n] = -1.0;
        else if (val == REGION_VOID)  PhaseLabel[n] =  0.0;
        else if (val == REGION_MPL)   PhaseLabel[n] = -0.5;
        else if (val == REGION_CL)    PhaseLabel[n] =  0.0;
        else if (val == REGION_MEMBRANE) PhaseLabel[n] = -1.0;
        else                          PhaseLabel[n] =  0.0;
        if (val == REGION_SOLID || val == REGION_MEMBRANE)
            Dm->id[n] = 0;
    }
    ScaLBL_CopyToDevice(Phi, PhaseLabel, N * sizeof(double));
    delete[] PhaseLabel;
}

// =========================================================================
//  AssignMaterialProperties
// =========================================================================
void ScaLBL_FuelCellModel::AssignMaterialProperties()
{
    double *poros_host = new double[Np];
    double *perm_host  = new double[Np];
    double *region_host = new double[Np];
    double *diff_host  = new double[Np];
    double *kth_host   = new double[Np];
    double *elec_host  = new double[Np];
    double *ion_host   = new double[Np];
    double *grey_grad_host = new double[3 * Np];

    double mpl_porosity = fuelcell_db->getWithDefault<double>("MPL_porosity", 0.5);
    double mpl_perm     = fuelcell_db->getWithDefault<double>("MPL_permeability", 1.0e-12);
    double cl_porosity  = fuelcell_db->getWithDefault<double>("CL_porosity", 0.3);
    double cl_perm      = fuelcell_db->getWithDefault<double>("CL_permeability", 1.0e-14);

    for (int k = 1; k < Nz - 1; k++) {
        for (int j = 1; j < Ny - 1; j++) {
            for (int i = 1; i < Nx - 1; i++) {
                int n_cart = k * Nx * Ny + j * Nx + i;
                int idx = Map(i, j, k);
                if (idx < 0) continue;

                char region = id[n_cart];
                region_host[idx] = (double)region;

                switch (region) {
                case REGION_VOID:
                    poros_host[idx] = 1.0;
                    perm_host[idx]  = 1.0;
                    diff_host[idx]  = 1.0;
                    kth_host[idx]   = k_thermal_gas;
                    elec_host[idx]  = 0.0;
                    ion_host[idx]   = 0.0;
                    break;
                case REGION_MPL:
                    poros_host[idx] = mpl_porosity;
                    perm_host[idx]  = mpl_perm;
                    diff_host[idx]  = mpl_porosity / 3.0;
                    kth_host[idx]   = k_thermal_GDL;
                    elec_host[idx]  = sigma_s;
                    ion_host[idx]   = 0.0;
                    break;
                case REGION_CL:
                    poros_host[idx] = cl_porosity;
                    perm_host[idx]  = cl_perm;
                    diff_host[idx]  = cl_porosity / 5.0;
                    kth_host[idx]   = k_thermal_GDL;
                    elec_host[idx]  = sigma_s * 0.3;
                    ion_host[idx]   = sigma_e;
                    break;
                case REGION_MEMBRANE:
                    poros_host[idx] = 0.3;
                    perm_host[idx]  = 1.0e-18;
                    diff_host[idx]  = 0.0;
                    kth_host[idx]   = k_thermal_membrane;
                    elec_host[idx]  = 0.0;
                    ion_host[idx]   = sigma_e;
                    break;
                default:
                    poros_host[idx] = 0.0;
                    perm_host[idx]  = 0.0;
                    diff_host[idx]  = 0.0;
                    kth_host[idx]   = k_thermal_GDL;
                    elec_host[idx]  = sigma_s;
                    ion_host[idx]   = 0.0;
                    break;
                }
                grey_grad_host[idx + 0 * Np] = 0.0;
                grey_grad_host[idx + 1 * Np] = 0.0;
                grey_grad_host[idx + 2 * Np] = 0.0;
            }
        }
    }

    ScaLBL_CopyToDevice(Poros, poros_host, Np * sizeof(double));
    ScaLBL_CopyToDevice(Perm, perm_host, Np * sizeof(double));
    ScaLBL_CopyToDevice(RegionID, region_host, Np * sizeof(double));
    ScaLBL_CopyToDevice(DiffCoeff, diff_host, Np * sizeof(double));
    ScaLBL_CopyToDevice(ThermalCond, kth_host, Np * sizeof(double));
    ScaLBL_CopyToDevice(ElecCond, elec_host, Np * sizeof(double));
    ScaLBL_CopyToDevice(IonCond, ion_host, Np * sizeof(double));
    ScaLBL_CopyToDevice(GreySolidGrad, grey_grad_host, 3 * Np * sizeof(double));

    delete[] poros_host;
    delete[] perm_host;
    delete[] region_host;
    delete[] diff_host;
    delete[] kth_host;
    delete[] elec_host;
    delete[] ion_host;
    delete[] grey_grad_host;

    if (rank == 0) printf("Material properties assigned to %d active voxels\n", Np);
}

// =========================================================================
//  Create
// =========================================================================
void ScaLBL_FuelCellModel::Create()
{
    Dm->CommInit();
    Np = Dm->PoreCount();
    if (rank == 0) printf("Creating FuelCell model: Np=%d, N=%d\n", Np, N);

    ScaLBL_Comm = shared_ptr<ScaLBL_Communicator>(new ScaLBL_Communicator(Dm));
    ScaLBL_Comm_Regular = shared_ptr<ScaLBL_Communicator>(new ScaLBL_Communicator(Dm));

    int Npad = (Np / 16 + 2) * 16;
    Map.resize(Nx, Ny, Nz);
    Map.fill(-2);
    auto neighborList = new int[18 * Npad];
    Np = ScaLBL_Comm->MemoryOptimizedLayoutAA(Map, neighborList, Dm->id, Np);
    MPI_Barrier(Dm->Comm);

    if (rank == 0) printf("Active voxels after layout: Np=%d\n", Np);
    dist_mem_size = Np * sizeof(double);
    neighborSize  = 18 * (Np * sizeof(int));

    // Neighbor list
    ScaLBL_AllocateDeviceMemory((void **)&NeighborList, neighborSize);
    ScaLBL_CopyToDevice(NeighborList, neighborList, neighborSize);
    delete[] neighborList;

    // Map
    ScaLBL_AllocateDeviceMemory((void **)&dvcMap, sizeof(int) * Np);
    int *TmpMap = new int[Np];
    for (int k = 1; k < Nz - 1; k++)
        for (int j = 1; j < Ny - 1; j++)
            for (int i = 1; i < Nx - 1; i++) {
                int idx = Map(i, j, k);
                if (!(idx < 0))
                    TmpMap[idx] = k * Nx * Ny + j * Nx + i;
            }
    for (int idx = 0; idx < ScaLBL_Comm->LastExterior(); idx++)
        if (TmpMap[idx] >= N) TmpMap[idx] = N - 1;
    for (int idx = ScaLBL_Comm->FirstInterior(); idx < ScaLBL_Comm->LastInterior(); idx++)
        if (TmpMap[idx] >= N) TmpMap[idx] = N - 1;
    ScaLBL_CopyToDevice(dvcMap, TmpMap, sizeof(int) * Np);
    ScaLBL_DeviceBarrier();
    delete[] TmpMap;

    // D3Q19 momentum
    ScaLBL_AllocateDeviceMemory((void **)&fq, 19 * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&Aq, 7 * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&Bq, 7 * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&Den, 2 * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&Phi, sizeof(double) * N);
    ScaLBL_AllocateDeviceMemory((void **)&Pressure, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&Velocity, 3 * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&ColorGrad, 3 * dist_mem_size);

    // Shan-Chen force arrays
    ScaLBL_AllocateDeviceMemory((void **)&ForceX, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&ForceY, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&ForceZ, dist_mem_size);

    // Greyscale
    ScaLBL_AllocateDeviceMemory((void **)&Poros, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&Perm, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&GreySolidGrad, 3 * dist_mem_size);

    // Species D3Q7
    for (int s = 0; s < NUM_SPECIES; s++) {
        ScaLBL_AllocateDeviceMemory((void **)&Cq[s], 7 * dist_mem_size);
        ScaLBL_AllocateDeviceMemory((void **)&ConcentrationDev[s], dist_mem_size);
    }
    ScaLBL_AllocateDeviceMemory((void **)&SourceO2, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&SourceN2, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&SourceH2, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&SourceH2O, dist_mem_size);

    // Thermal D3Q7
    ScaLBL_AllocateDeviceMemory((void **)&Tq, 7 * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&Temperature, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&SourceThermal, dist_mem_size);

    // Potential D3Q7
    ScaLBL_AllocateDeviceMemory((void **)&PhiSq, 7 * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&PhiEq, 7 * dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&PhiS, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&PhiE, dist_mem_size);

    // Phase change
    ScaLBL_AllocateDeviceMemory((void **)&SourcePhaseField, dist_mem_size);

    // Material properties
    ScaLBL_AllocateDeviceMemory((void **)&RegionID, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&DiffCoeff, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&ThermalCond, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&ElecCond, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&IonCond, dist_mem_size);
    ScaLBL_AllocateDeviceMemory((void **)&ReactionRate, dist_mem_size);

    // Cartesian output arrays
    PhaseField.resize(Nx, Ny, Nz);
    Pressure_Cart.resize(Nx, Ny, Nz);
    Velocity_x.resize(Nx, Ny, Nz);
    Velocity_y.resize(Nx, Ny, Nz);
    Velocity_z.resize(Nx, Ny, Nz);
    Temperature_Cart.resize(Nx, Ny, Nz);
    PhiS_Cart.resize(Nx, Ny, Nz);
    PhiE_Cart.resize(Nx, Ny, Nz);
    for (int s = 0; s < NUM_SPECIES; s++)
        Concentration_Cart[s].resize(Nx, Ny, Nz);

    if (rank == 0) printf("FuelCell model memory allocated\n");
}

// =========================================================================
//  Initialize
// =========================================================================
void ScaLBL_FuelCellModel::Initialize()
{
    if (rank == 0) printf("Initializing FuelCell model\n");

    // D3Q19 momentum equilibrium at rest
    ScaLBL_D3Q19_Init(fq, Np);

    // Material properties from label map
    AssignComponentLabels();
    AssignMaterialProperties();

    // Phase field D3Q7 from Phi (via Map)
    ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq,
                           0, ScaLBL_Comm->LastExterior(), Np);
    ScaLBL_PhaseField_Init(dvcMap, Phi, Den, Aq, Bq,
                           ScaLBL_Comm->FirstInterior(),
                           ScaLBL_Comm->LastInterior(), Np);

    // D3Q7 equilibrium initialisation for species
    // f0 = w0*C = 0.25*C, f1..6 = w1*C = 0.125*C
    double c_inlet[NUM_SPECIES] = {c_O2_in, c_N2_in, c_H2_in, c_H2Ov_in};
    for (int s = 0; s < NUM_SPECIES; s++) {
        double *fq_host = new double[7 * Np];
        double C = c_inlet[s];
        for (int n = 0; n < Np; n++) {
            fq_host[n]          = 0.25  * C; // f0
            fq_host[1 * Np + n] = 0.125 * C; // f1
            fq_host[2 * Np + n] = 0.125 * C; // f2
            fq_host[3 * Np + n] = 0.125 * C; // f3
            fq_host[4 * Np + n] = 0.125 * C; // f4
            fq_host[5 * Np + n] = 0.125 * C; // f5
            fq_host[6 * Np + n] = 0.125 * C; // f6
        }
        ScaLBL_CopyToDevice(Cq[s], fq_host, 7 * dist_mem_size);
        delete[] fq_host;

        double *c_host = new double[Np];
        for (int n = 0; n < Np; n++) c_host[n] = C;
        ScaLBL_CopyToDevice(ConcentrationDev[s], c_host, dist_mem_size);
        delete[] c_host;
    }

    // Thermal D3Q7 equilibrium at T=1.0 (normalised)
    {
        double *fq_host = new double[7 * Np];
        double T0 = 1.0;
        for (int n = 0; n < Np; n++) {
            fq_host[n]          = 0.25  * T0;
            fq_host[1 * Np + n] = 0.125 * T0;
            fq_host[2 * Np + n] = 0.125 * T0;
            fq_host[3 * Np + n] = 0.125 * T0;
            fq_host[4 * Np + n] = 0.125 * T0;
            fq_host[5 * Np + n] = 0.125 * T0;
            fq_host[6 * Np + n] = 0.125 * T0;
        }
        ScaLBL_CopyToDevice(Tq, fq_host, 7 * dist_mem_size);
        delete[] fq_host;

        double *T_host = new double[Np];
        for (int n = 0; n < Np; n++) T_host[n] = T0;
        ScaLBL_CopyToDevice(Temperature, T_host, dist_mem_size);
        delete[] T_host;
    }

    // Potential D3Q7: PhiS initialised to V_cell, PhiE to 0
    {
        double *fq_host = new double[7 * Np];

        // Electronic potential → V_cell
        for (int n = 0; n < Np; n++) {
            fq_host[n]          = 0.25  * V_cell;
            for (int q = 1; q <= 6; q++)
                fq_host[q * Np + n] = 0.125 * V_cell;
        }
        ScaLBL_CopyToDevice(PhiSq, fq_host, 7 * dist_mem_size);

        double *phi_host = new double[Np];
        for (int n = 0; n < Np; n++) phi_host[n] = V_cell;
        ScaLBL_CopyToDevice(PhiS, phi_host, dist_mem_size);

        // Protonic potential → 0
        for (int n = 0; n < Np; n++) {
            fq_host[n] = 0.0;
            for (int q = 1; q <= 6; q++)
                fq_host[q * Np + n] = 0.0;
        }
        ScaLBL_CopyToDevice(PhiEq, fq_host, 7 * dist_mem_size);
        for (int n = 0; n < Np; n++) phi_host[n] = 0.0;
        ScaLBL_CopyToDevice(PhiE, phi_host, dist_mem_size);

        delete[] fq_host;
        delete[] phi_host;
    }

    // Zero all source term arrays
    {
        double *zero = new double[Np];
        for (int n = 0; n < Np; n++) zero[n] = 0.0;
        ScaLBL_CopyToDevice(ForceX, zero, dist_mem_size);
        ScaLBL_CopyToDevice(ForceY, zero, dist_mem_size);
        ScaLBL_CopyToDevice(ForceZ, zero, dist_mem_size);
        ScaLBL_CopyToDevice(SourceO2, zero, dist_mem_size);
        ScaLBL_CopyToDevice(SourceN2, zero, dist_mem_size);
        ScaLBL_CopyToDevice(SourceH2, zero, dist_mem_size);
        ScaLBL_CopyToDevice(SourceH2O, zero, dist_mem_size);
        ScaLBL_CopyToDevice(SourceThermal, zero, dist_mem_size);
        ScaLBL_CopyToDevice(SourcePhaseField, zero, dist_mem_size);
        ScaLBL_CopyToDevice(ReactionRate, zero, dist_mem_size);
        delete[] zero;
    }

    if (rank == 0) printf("FuelCell model initialized\n");
}

// =========================================================================
//  SolveElectricPotentials — D3Q7 Laplace iteration to pseudo-steady state
// =========================================================================
void ScaLBL_FuelCellModel::SolveElectricPotentials()
{
    int pot_iterations = fuelcell_db->getWithDefault<int>("pot_iterations", 50);

    for (int iter = 0; iter < pot_iterations; iter++) {
        // --- Odd sub-step ---
        // Electronic potential (PhiSq with ElecCond)
        ScaLBL_Comm->BiSendD3Q7AA(PhiSq, PhiEq);
        ScaLBL_D3Q7_AAodd_FuelCell_Potential(
            NeighborList, dvcMap, PhiSq, PhiS, ElecCond,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_D3Q7_AAodd_FuelCell_Potential(
            NeighborList, dvcMap, PhiEq, PhiE, IonCond,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(PhiSq, PhiEq);
        ScaLBL_D3Q7_AAodd_FuelCell_Potential(
            NeighborList, dvcMap, PhiSq, PhiS, ElecCond,
            0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_D3Q7_AAodd_FuelCell_Potential(
            NeighborList, dvcMap, PhiEq, PhiE, IonCond,
            0, ScaLBL_Comm->LastExterior(), Np);

        ScaLBL_DeviceBarrier();

        // --- Even sub-step ---
        ScaLBL_Comm->BiSendD3Q7AA(PhiSq, PhiEq);
        ScaLBL_D3Q7_AAeven_FuelCell_Potential(
            dvcMap, PhiSq, PhiS, ElecCond,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_D3Q7_AAeven_FuelCell_Potential(
            dvcMap, PhiEq, PhiE, IonCond,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(PhiSq, PhiEq);
        ScaLBL_D3Q7_AAeven_FuelCell_Potential(
            dvcMap, PhiSq, PhiS, ElecCond,
            0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_D3Q7_AAeven_FuelCell_Potential(
            dvcMap, PhiEq, PhiE, IonCond,
            0, ScaLBL_Comm->LastExterior(), Np);

        ScaLBL_DeviceBarrier();
    }
}

// =========================================================================
//  ComputeButlerVolmerSources
// =========================================================================
void ScaLBL_FuelCellModel::ComputeButlerVolmerSources()
{
    ScaLBL_FuelCell_ButlerVolmer(
        PhiS, PhiE, Temperature, RegionID, ReactionRate,
        SourceO2, SourceN2, SourceH2, SourceH2O,
        i0_cathode, i0_anode,
        alpha_a_c, alpha_c_c, alpha_a_a, alpha_c_a,
        E_eq, F_const, R_gas, T_ref,
        ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
    ScaLBL_FuelCell_ButlerVolmer(
        PhiS, PhiE, Temperature, RegionID, ReactionRate,
        SourceO2, SourceN2, SourceH2, SourceH2O,
        i0_cathode, i0_anode,
        alpha_a_c, alpha_c_c, alpha_a_a, alpha_c_a,
        E_eq, F_const, R_gas, T_ref,
        0, ScaLBL_Comm->LastExterior(), Np);
}

// =========================================================================
//  ComputePhaseChangeSources
// =========================================================================
void ScaLBL_FuelCellModel::ComputePhaseChangeSources()
{
    double k_evap = fuelcell_db->getWithDefault<double>("k_evap", 0.001);
    double k_cond = fuelcell_db->getWithDefault<double>("k_cond", 0.001);
    double P_ref  = fuelcell_db->getWithDefault<double>("P_ref", 0.03167); // atm at 25C

    ScaLBL_FuelCell_PhaseChange(
        Phi, Temperature, ConcentrationDev[SP_H2OV],
        SourcePhaseField, SourceThermal,
        h_fg, R_gas, T_ref, P_ref, k_evap, k_cond,
        dvcMap,
        ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
    ScaLBL_FuelCell_PhaseChange(
        Phi, Temperature, ConcentrationDev[SP_H2OV],
        SourcePhaseField, SourceThermal,
        h_fg, R_gas, T_ref, P_ref, k_evap, k_cond,
        dvcMap,
        0, ScaLBL_Comm->LastExterior(), Np);
}

// =========================================================================
//  Run — fully coupled simulation loop
// =========================================================================
void ScaLBL_FuelCellModel::Run()
{
    int analysis_interval = analysis_db->getWithDefault<int>("analysis_interval", 1000);

    // D3Q7 tau for each species: tau = 0.5 + 4*D (cs^2=1/4)
    double tau_sp[NUM_SPECIES] = {
        0.5 + 4.0 * D_O2,
        0.5 + 4.0 * D_N2,
        0.5 + 4.0 * D_H2,
        0.5 + 4.0 * D_H2Ov
    };
    double tau_thermal = 0.5 + 4.0 * k_thermal_gas;

    // Source term device pointer array (ordered O2,N2,H2,H2O)
    double *SourceSpecies[NUM_SPECIES] = {SourceO2, SourceN2, SourceH2, SourceH2O};

    if (rank == 0) {
        printf("========================================\n");
        printf("  Starting FuelCell LBM simulation\n");
        printf("  timesteps: %d, analysis: %d, electro: %d\n",
               timestepMax, analysis_interval, electro_interval);
        printf("  Species tau: O2=%.3f N2=%.3f H2=%.3f H2Ov=%.3f\n",
               tau_sp[0], tau_sp[1], tau_sp[2], tau_sp[3]);
        printf("========================================\n");
    }

    auto tstart = MPI_Wtime();
    int START_TIMESTEP = timestep;

    while (timestep < timestepMax) {

        // ==========================================================
        //  ODD TIMESTEP
        // ==========================================================
        timestep++;

        // --- 1a. Phase field D3Q7 (odd) ---
        ScaLBL_Comm->BiSendD3Q7AA(Aq, Bq);
        ScaLBL_D3Q7_AAodd_PhaseField(NeighborList, dvcMap, Aq, Bq, Den, Phi,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(Aq, Bq);
        ScaLBL_D3Q7_AAodd_PhaseField(NeighborList, dvcMap, Aq, Bq, Den, Phi,
            0, ScaLBL_Comm->LastExterior(), Np);

        ScaLBL_Comm_Regular->SendHalo(Phi);
        ScaLBL_Comm_Regular->RecvHalo(Phi);

        // --- 1b. SC pseudopotential force (after Den is updated) ---
        ScaLBL_D3Q19_AAodd_ShanChen_Force(
            NeighborList, dvcMap, Den, Phi, ForceX, ForceY, ForceZ,
            sc_G, cs_a, cs_b, cs_T, rhoA, rhoB,
            Nx, Nx * Ny,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_D3Q19_AAodd_ShanChen_Force(
            NeighborList, dvcMap, Den, Phi, ForceX, ForceY, ForceZ,
            sc_G, cs_a, cs_b, cs_T, rhoA, rhoB,
            Nx, Nx * Ny,
            0, ScaLBL_Comm->LastExterior(), Np);

        // --- 1c. D3Q19 Color+Greyscale momentum (odd) ---
        ScaLBL_Comm->SendD3Q19AA(fq);
        ScaLBL_D3Q19_AAodd_GreyscaleColor(
            NeighborList, dvcMap, fq, Aq, Bq, Den, Phi,
            GreySolidGrad, Poros, Perm, Velocity,
            rhoA, rhoB, tauA, tauB, tauA_eff, tauB_eff,
            alpha, beta, Fx, Fy, Fz,
            Nx, Nx * Ny,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->RecvD3Q19AA(fq);
        ScaLBL_D3Q19_AAodd_GreyscaleColor(
            NeighborList, dvcMap, fq, Aq, Bq, Den, Phi,
            GreySolidGrad, Poros, Perm, Velocity,
            rhoA, rhoB, tauA, tauB, tauA_eff, tauB_eff,
            alpha, beta, Fx, Fy, Fz,
            Nx, Nx * Ny,
            0, ScaLBL_Comm->LastExterior(), Np);

        // Pressure BCs
        if (BoundaryCondition == 3) {
            ScaLBL_Comm->D3Q19_Pressure_BC_z(NeighborList, fq, din, timestep);
            ScaLBL_Comm->D3Q19_Pressure_BC_Z(NeighborList, fq, dout, timestep);
        }
        if (BoundaryCondition == 4) {
            ScaLBL_Comm->D3Q19_Flux_BC_z(NeighborList, fq, flux, timestep);
        }
        ScaLBL_DeviceBarrier();

        // --- 2. Species transport D3Q7 (odd, two BiSend rounds) ---
        // Round 1: O2 + N2
        ScaLBL_Comm->BiSendD3Q7AA(Cq[SP_O2], Cq[SP_N2]);
        ScaLBL_D3Q7_AAodd_FuelCell_Species(
            NeighborList, dvcMap, Cq[SP_O2], ConcentrationDev[SP_O2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_O2],
            tau_sp[SP_O2], ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_D3Q7_AAodd_FuelCell_Species(
            NeighborList, dvcMap, Cq[SP_N2], ConcentrationDev[SP_N2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_N2],
            tau_sp[SP_N2], ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(Cq[SP_O2], Cq[SP_N2]);
        ScaLBL_D3Q7_AAodd_FuelCell_Species(
            NeighborList, dvcMap, Cq[SP_O2], ConcentrationDev[SP_O2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_O2],
            tau_sp[SP_O2], 0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_D3Q7_AAodd_FuelCell_Species(
            NeighborList, dvcMap, Cq[SP_N2], ConcentrationDev[SP_N2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_N2],
            tau_sp[SP_N2], 0, ScaLBL_Comm->LastExterior(), Np);

        // Round 2: H2 + H2O
        ScaLBL_Comm->BiSendD3Q7AA(Cq[SP_H2], Cq[SP_H2OV]);
        ScaLBL_D3Q7_AAodd_FuelCell_Species(
            NeighborList, dvcMap, Cq[SP_H2], ConcentrationDev[SP_H2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_H2],
            tau_sp[SP_H2], ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_D3Q7_AAodd_FuelCell_Species(
            NeighborList, dvcMap, Cq[SP_H2OV], ConcentrationDev[SP_H2OV], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_H2OV],
            tau_sp[SP_H2OV], ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(Cq[SP_H2], Cq[SP_H2OV]);
        ScaLBL_D3Q7_AAodd_FuelCell_Species(
            NeighborList, dvcMap, Cq[SP_H2], ConcentrationDev[SP_H2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_H2],
            tau_sp[SP_H2], 0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_D3Q7_AAodd_FuelCell_Species(
            NeighborList, dvcMap, Cq[SP_H2OV], ConcentrationDev[SP_H2OV], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_H2OV],
            tau_sp[SP_H2OV], 0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_DeviceBarrier();

        // --- 3. Thermal D3Q7 (odd) ---
        // Use BiSendD3Q7AA with Tq twice (wastes bandwidth but correct)
        ScaLBL_Comm->BiSendD3Q7AA(Tq, Tq);
        ScaLBL_D3Q7_AAodd_FuelCell_Thermal(
            NeighborList, dvcMap, Tq, Temperature, Velocity,
            ThermalCond, SourceThermal,
            tau_thermal, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(Tq, Tq);
        ScaLBL_D3Q7_AAodd_FuelCell_Thermal(
            NeighborList, dvcMap, Tq, Temperature, Velocity,
            ThermalCond, SourceThermal,
            tau_thermal, 0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_DeviceBarrier();

        // ==========================================================
        //  EVEN TIMESTEP
        // ==========================================================
        timestep++;

        // --- 1a. Phase field D3Q7 (even) ---
        ScaLBL_Comm->BiSendD3Q7AA(Aq, Bq);
        ScaLBL_D3Q7_AAeven_PhaseField(dvcMap, Aq, Bq, Den, Phi,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(Aq, Bq);
        ScaLBL_D3Q7_AAeven_PhaseField(dvcMap, Aq, Bq, Den, Phi,
            0, ScaLBL_Comm->LastExterior(), Np);

        ScaLBL_Comm_Regular->SendHalo(Phi);
        ScaLBL_Comm_Regular->RecvHalo(Phi);

        // --- 1b. SC force (even) ---
        ScaLBL_D3Q19_AAeven_ShanChen_Force(
            dvcMap, Den, Phi, ForceX, ForceY, ForceZ,
            sc_G, cs_a, cs_b, cs_T, rhoA, rhoB,
            Nx, Nx * Ny,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_D3Q19_AAeven_ShanChen_Force(
            dvcMap, Den, Phi, ForceX, ForceY, ForceZ,
            sc_G, cs_a, cs_b, cs_T, rhoA, rhoB,
            Nx, Nx * Ny,
            0, ScaLBL_Comm->LastExterior(), Np);

        // --- 1c. D3Q19 Color+Greyscale (even) ---
        ScaLBL_Comm->SendD3Q19AA(fq);
        ScaLBL_D3Q19_AAeven_GreyscaleColor(
            dvcMap, fq, Aq, Bq, Den, Phi,
            GreySolidGrad, Poros, Perm, Velocity,
            rhoA, rhoB, tauA, tauB, tauA_eff, tauB_eff,
            alpha, beta, Fx, Fy, Fz,
            Nx, Nx * Ny,
            ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->RecvD3Q19AA(fq);
        ScaLBL_D3Q19_AAeven_GreyscaleColor(
            dvcMap, fq, Aq, Bq, Den, Phi,
            GreySolidGrad, Poros, Perm, Velocity,
            rhoA, rhoB, tauA, tauB, tauA_eff, tauB_eff,
            alpha, beta, Fx, Fy, Fz,
            Nx, Nx * Ny,
            0, ScaLBL_Comm->LastExterior(), Np);

        if (BoundaryCondition == 3) {
            ScaLBL_Comm->D3Q19_Pressure_BC_z(NeighborList, fq, din, timestep);
            ScaLBL_Comm->D3Q19_Pressure_BC_Z(NeighborList, fq, dout, timestep);
        }
        if (BoundaryCondition == 4) {
            ScaLBL_Comm->D3Q19_Flux_BC_z(NeighborList, fq, flux, timestep);
        }
        ScaLBL_DeviceBarrier();

        // --- 2. Species D3Q7 (even) ---
        ScaLBL_Comm->BiSendD3Q7AA(Cq[SP_O2], Cq[SP_N2]);
        ScaLBL_D3Q7_AAeven_FuelCell_Species(
            dvcMap, Cq[SP_O2], ConcentrationDev[SP_O2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_O2],
            tau_sp[SP_O2], ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_D3Q7_AAeven_FuelCell_Species(
            dvcMap, Cq[SP_N2], ConcentrationDev[SP_N2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_N2],
            tau_sp[SP_N2], ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(Cq[SP_O2], Cq[SP_N2]);
        ScaLBL_D3Q7_AAeven_FuelCell_Species(
            dvcMap, Cq[SP_O2], ConcentrationDev[SP_O2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_O2],
            tau_sp[SP_O2], 0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_D3Q7_AAeven_FuelCell_Species(
            dvcMap, Cq[SP_N2], ConcentrationDev[SP_N2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_N2],
            tau_sp[SP_N2], 0, ScaLBL_Comm->LastExterior(), Np);

        ScaLBL_Comm->BiSendD3Q7AA(Cq[SP_H2], Cq[SP_H2OV]);
        ScaLBL_D3Q7_AAeven_FuelCell_Species(
            dvcMap, Cq[SP_H2], ConcentrationDev[SP_H2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_H2],
            tau_sp[SP_H2], ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_D3Q7_AAeven_FuelCell_Species(
            dvcMap, Cq[SP_H2OV], ConcentrationDev[SP_H2OV], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_H2OV],
            tau_sp[SP_H2OV], ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(Cq[SP_H2], Cq[SP_H2OV]);
        ScaLBL_D3Q7_AAeven_FuelCell_Species(
            dvcMap, Cq[SP_H2], ConcentrationDev[SP_H2], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_H2],
            tau_sp[SP_H2], 0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_D3Q7_AAeven_FuelCell_Species(
            dvcMap, Cq[SP_H2OV], ConcentrationDev[SP_H2OV], Velocity,
            Phi, Poros, DiffCoeff, SourceSpecies[SP_H2OV],
            tau_sp[SP_H2OV], 0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_DeviceBarrier();

        // --- 3. Thermal D3Q7 (even) ---
        ScaLBL_Comm->BiSendD3Q7AA(Tq, Tq);
        ScaLBL_D3Q7_AAeven_FuelCell_Thermal(
            dvcMap, Tq, Temperature, Velocity,
            ThermalCond, SourceThermal,
            tau_thermal, ScaLBL_Comm->FirstInterior(), ScaLBL_Comm->LastInterior(), Np);
        ScaLBL_Comm->BiRecvD3Q7AA(Tq, Tq);
        ScaLBL_D3Q7_AAeven_FuelCell_Thermal(
            dvcMap, Tq, Temperature, Velocity,
            ThermalCond, SourceThermal,
            tau_thermal, 0, ScaLBL_Comm->LastExterior(), Np);
        ScaLBL_DeviceBarrier();

        // ==========================================================
        //  4. Electrochemistry (every electro_interval steps)
        // ==========================================================
        if (timestep % electro_interval == 0) {
            SolveElectricPotentials();
            ComputeButlerVolmerSources();
            ComputePhaseChangeSources();
        }

        // ==========================================================
        //  5. Analysis and output
        // ==========================================================
        if (timestep % analysis_interval == 0) {
            ScaLBL_D3Q19_Pressure(fq, Pressure, Np);
            ScaLBL_Comm->RegularLayout(Map, Pressure, Pressure_Cart);
            ScaLBL_Comm->RegularLayout(Map, &Velocity[0], Velocity_x);
            ScaLBL_Comm->RegularLayout(Map, &Velocity[Np], Velocity_y);
            ScaLBL_Comm->RegularLayout(Map, &Velocity[2 * Np], Velocity_z);

            double vax = 0, vay = 0, vaz = 0;
            int count = 0;
            for (int k = 1; k < Nz - 1; k++)
                for (int j = 1; j < Ny - 1; j++)
                    for (int i = 1; i < Nx - 1; i++) {
                        vax += Velocity_x(i, j, k);
                        vay += Velocity_y(i, j, k);
                        vaz += Velocity_z(i, j, k);
                        count++;
                    }

            double vax_g, vay_g, vaz_g;
            int count_g;
            MPI_Allreduce(&vax, &vax_g, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);
            MPI_Allreduce(&vay, &vay_g, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);
            MPI_Allreduce(&vaz, &vaz_g, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);
            MPI_Allreduce(&count, &count_g, 1, MPI_INT, MPI_SUM, Dm->Comm);

            // Extract species concentrations
            ScaLBL_Comm->RegularLayout(Map, ConcentrationDev[SP_O2], Concentration_Cart[SP_O2]);
            ScaLBL_Comm->RegularLayout(Map, ConcentrationDev[SP_H2], Concentration_Cart[SP_H2]);
            ScaLBL_Comm->RegularLayout(Map, Temperature, Temperature_Cart);
            ScaLBL_Comm->RegularLayout(Map, PhiS, PhiS_Cart);
            ScaLBL_Comm->RegularLayout(Map, PhiE, PhiE_Cart);

            // Compute average concentrations
            double cO2_avg = 0, cH2_avg = 0;
            for (int k = 1; k < Nz - 1; k++)
                for (int j = 1; j < Ny - 1; j++)
                    for (int i = 1; i < Nx - 1; i++) {
                        cO2_avg += Concentration_Cart[SP_O2](i, j, k);
                        cH2_avg += Concentration_Cart[SP_H2](i, j, k);
                    }
            double cO2_g, cH2_g;
            MPI_Allreduce(&cO2_avg, &cO2_g, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);
            MPI_Allreduce(&cH2_avg, &cH2_g, 1, MPI_DOUBLE, MPI_SUM, Dm->Comm);

            double elapsed = MPI_Wtime() - tstart;
            double MLUPS = double(Np) * double(timestep - START_TIMESTEP) / elapsed / 1.0e6;

            if (rank == 0) {
                printf("t=%d MLUPS=%.2f <vz>=%.4e <cO2>=%.4e <cH2>=%.4e\n",
                       timestep, MLUPS,
                       vaz_g / count_g,
                       cO2_g / count_g,
                       cH2_g / count_g);
            }
        }
    }

    auto tend = MPI_Wtime();
    if (rank == 0) {
        printf("========================================\n");
        printf("  FuelCell simulation complete\n");
        printf("  Total time: %.2f s\n", tend - tstart);
        printf("  MLUPS: %.2f\n",
               double(Np) * double(timestepMax - START_TIMESTEP) / (tend - tstart) / 1.0e6);
        printf("========================================\n");
    }
}

// =========================================================================
//  WriteDebug
// =========================================================================
void ScaLBL_FuelCellModel::WriteDebug()
{
    ScaLBL_Comm->RegularLayout(Map, Pressure, Pressure_Cart);
    ScaLBL_Comm->RegularLayout(Map, &Velocity[0], Velocity_x);
    ScaLBL_Comm->RegularLayout(Map, &Velocity[Np], Velocity_y);
    ScaLBL_Comm->RegularLayout(Map, &Velocity[2 * Np], Velocity_z);
    ScaLBL_Comm->RegularLayout(Map, Temperature, Temperature_Cart);
    ScaLBL_Comm->RegularLayout(Map, PhiS, PhiS_Cart);
    ScaLBL_Comm->RegularLayout(Map, PhiE, PhiE_Cart);
    for (int s = 0; s < NUM_SPECIES; s++)
        ScaLBL_Comm->RegularLayout(Map, ConcentrationDev[s], Concentration_Cart[s]);

    FILE *OUTFILE;
    char fname[100];

    sprintf(fname, "Pressure.%05d.raw", rank);
    OUTFILE = fopen(fname, "wb");
    fwrite(Pressure_Cart.data(), sizeof(double), N, OUTFILE);
    fclose(OUTFILE);

    sprintf(fname, "Velocity_z.%05d.raw", rank);
    OUTFILE = fopen(fname, "wb");
    fwrite(Velocity_z.data(), sizeof(double), N, OUTFILE);
    fclose(OUTFILE);

    sprintf(fname, "Temperature.%05d.raw", rank);
    OUTFILE = fopen(fname, "wb");
    fwrite(Temperature_Cart.data(), sizeof(double), N, OUTFILE);
    fclose(OUTFILE);

    sprintf(fname, "PhiS.%05d.raw", rank);
    OUTFILE = fopen(fname, "wb");
    fwrite(PhiS_Cart.data(), sizeof(double), N, OUTFILE);
    fclose(OUTFILE);

    sprintf(fname, "PhiE.%05d.raw", rank);
    OUTFILE = fopen(fname, "wb");
    fwrite(PhiE_Cart.data(), sizeof(double), N, OUTFILE);
    fclose(OUTFILE);

    sprintf(fname, "ConcO2.%05d.raw", rank);
    OUTFILE = fopen(fname, "wb");
    fwrite(Concentration_Cart[SP_O2].data(), sizeof(double), N, OUTFILE);
    fclose(OUTFILE);

    sprintf(fname, "ConcH2.%05d.raw", rank);
    OUTFILE = fopen(fname, "wb");
    fwrite(Concentration_Cart[SP_H2].data(), sizeof(double), N, OUTFILE);
    fclose(OUTFILE);

    if (rank == 0) printf("Debug output written\n");
}
