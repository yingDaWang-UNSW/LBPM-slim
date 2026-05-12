/*
  LBPMRelPermSimulator: relative permeability driver.

  Inherits from ScaLBL_ColorModel and rides the existing two-phase color
  LBM + automorph loop, but injects a per-phase single-phase BGK
  measurement at every automorph steady-state point via the
  OnSteadyStatePoint() hook.

  Workflow:
    1. lbpm_serial_decomp   -- decompose raw geometry into ID.xxxxx
    2. lbpm_morphopen_pp    -- morphological drainage to an *initial*
                               two-phase Sw (e.g. Sw=0.9, mostly WP)
    3. lbpm_relperm_simulator -- runs the color model with automorph
                               drainage/imbibition; at every steady
                               state hooks in single-phase BGK on each
                               phase's inlet->outlet connected blob to
                               record the effective absolute perm.

  Output:
    - relperm.csv          (standard color-model rel-perm log)
    - RelPermSummary.csv   (one row per (event, phase) with k_eff)
    - RelPermPhaseN_eventM.csv (per-phase BGK convergence trace)
*/
#ifndef LBPMRELPERMSIMULATOR_H
#define LBPMRELPERMSIMULATOR_H

#include "models/ColorModel.h"

class ScaLBL_LBPMRelPermSimulator : public ScaLBL_ColorModel {
public:
    ScaLBL_LBPMRelPermSimulator(int RANK, int NP, MPI_Comm COMM);
    ~ScaLBL_LBPMRelPermSimulator();

    // Reads color-model parameters via ScaLBL_ColorModel::ReadParams(),
    // then layers on the "RelPerm" section (single-phase BGK params).
    void ReadParamsRelPerm(std::string filename);

    // Hook called by ScaLBL_ColorModel::Run() on every automorph
    // steady-state event.  Runs single-phase BGK on each phase's
    // inlet->outlet connected component and records k_eff.
    virtual void OnSteadyStatePoint() override;

    // RelPerm-section parameters (single-phase BGK).
    double rp_tau;             // BGK relaxation time (>= 0.6)
    double rp_Fx, rp_Fy, rp_Fz; // body force (per-phase)
    int    rp_timestepMax;     // hard cap on single-phase iterations
    int    rp_analysis_interval;
    double rp_permTolerance;   // |dK/K| convergence test
    bool   rp_bgkFlag;         // BGK (true) or MRT (false) collision
    bool   rp_writeVisOnConv;  // dump rawVisVelP on convergence

    // Cumulative number of steady-state events seen so far.
    int rp_event_count;

private:
    int collateBlobs(int *&blobsGlob, std::vector<int> blobsLoc);

    // Build a single-phase Mask domain: id=1 only for voxels whose
    // current ColorModel id matches phaseID AND whose global blob ID
    // is in the inlet->outlet connected list.  Returns global pore
    // count.  The Mask is stored in `rp_Mask` (member, freed each call).
    long buildPhaseMask(int phaseID,
                        const IntArray &blob_label,
                        const std::vector<int> &connectedBlobs);

    // Allocate / free per-phase ScaLBL state.
    void createPhaseLayout();
    void freePhaseLayout();

    // BGK loop: returns k_eff in m^2.
    double runPhaseBGK(int phaseID);

    // Dump raw vel + pressure for the current phase.
    void writeVelPField(int phaseID);

    // Per-phase Domain + ScaLBL state (separate from ColorModel's).
    std::shared_ptr<Domain> rp_Mask;
    std::shared_ptr<ScaLBL_Communicator> rp_Comm;
    IntArray rp_Map;
    int *rp_NeighborList;
    double *rp_fq, *rp_Velocity, *rp_Pressure;
    int rp_Np;
    DoubleArray rp_Vx, rp_Vy, rp_Vz, rp_P;

    // Current ColorModel id snapshot (0 = solid, 1 = phase A, 2 = phase B)
    // built each hook fire from PhaseField + Distance + the original
    // immobile-component labels.
    std::vector<char> rp_id_full;
};

#endif
