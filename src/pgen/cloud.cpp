//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cloud.cpp
//! \brief Problem generator for cloud in wind simulation.
//!

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <basic_types.hpp>
#include <iomanip>
#include <ios>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>
#include <sstream>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "../eos/adiabatic_glmmhd.hpp"
#include "../eos/adiabatic_hydro.hpp"
#include "cloud.hpp"

namespace cloud {
using namespace parthenon::driver::prelude;

Real rho_wind, mom_wind, rhoe_wind, r_cloud, rho_cloud;
Real Bx = 0.0;
Real By = 0.0;
Real Bz = 0.0;

//========================================================================================
//! \fn void InitUserMeshData(Mesh *mesh, ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // no access to package in this function so we use a local units object
  Units units(pin);

  auto gamma = pin->GetReal("hydro", "gamma");
  auto gm1 = (gamma - 1.0);
  const auto &pkg = mesh->packages.Get("Hydro");
  const auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");

  r_cloud = pin->GetReal("problem/cloud", "r0_cgs") / units.code_length_cgs();
  rho_cloud = pin->GetReal("problem/cloud", "rho_cloud_cgs") / units.code_density_cgs();
  rho_wind = pin->GetReal("problem/cloud", "rho_wind_cgs") / units.code_density_cgs();
  auto T_wind = pin->GetReal("problem/cloud", "T_wind_cgs");
  auto v_wind = pin->GetReal("problem/cloud", "v_wind_cgs") /
                (units.code_length_cgs() / units.code_time_cgs());

  // mu_mh_gm1_by_k_B is already in code units
  rhoe_wind = T_wind * rho_wind / mbar_over_kb / gm1;
  const auto c_s_wind = std::sqrt(gamma * gm1 * rhoe_wind / rho_wind);
  const auto chi_0 = rho_cloud / rho_wind;               // cloud to wind density ratio
  const auto t_cc = r_cloud * std::sqrt(chi_0) / v_wind; // cloud crushting time (code)
  const auto pressure =
      gm1 * rhoe_wind; // one value for entire domain given initial pressure equil.

  const auto T_cloud = pressure / rho_cloud * mbar_over_kb;

  auto plasma_beta = pin->GetOrAddReal("problem/cloud", "plasma_beta", -1.0);

  auto mag_field_angle_str =
      pin->GetOrAddString("problem/cloud", "mag_field_angle", "undefined");
  // To support using the MHD integrator as Hydro (with B=0 indicated by plasma_beta = 0)
  // we avoid division by 0 here.
  if (plasma_beta > 0.0) {
    if (mag_field_angle_str == "aligned") {
      By = std::sqrt(2.0 * pressure / plasma_beta);
    } else if (mag_field_angle_str == "transverse") {
      Bx = std::sqrt(2.0 * pressure / plasma_beta);
    } else if (mag_field_angle_str == "oblique") {
      const auto B = std::sqrt(2.0 * pressure / plasma_beta);
      Bx = B / std::sqrt(5.0);
      Bz = 2 * Bx;
    } else {
      PARTHENON_FAIL("Unsupported problem/cloud/mag_field_angle. Please use either "
                     "'aligned', 'transverse', or 'oblique'.");
    }
  }

  const parthenon::Real He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
  const parthenon::Real H_mass_fraction = 1.0 - He_mass_fraction;
  const parthenon::Real mu =
      1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);

  pkg -> AddParam<Real>("singlecloud::mean_molecular_mass_by_kb", mu * units.atomic_mass_unit() / units.k_boltzmann());
  //Set frame speed as mutable
  pkg->AddParam<Real>("inertial_frame_v", 0., true);
  pkg -> AddParam<Real>("Tcloud", T_cloud);

  mom_wind = rho_wind * v_wind;

  std::stringstream msg;
  msg << std::setprecision(2);
  msg << "######################################" << std::endl;
  msg << "###### Cloud in wind problem generator" << std::endl;
  msg << "#### Input parameters" << std::endl;
  msg << "## Cloud radius: " << r_cloud / units.kpc() << " kpc" << std::endl;
  msg << "## Cloud density: " << rho_cloud / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Wind density: " << rho_wind / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Wind temperature: " << T_wind << " K" << std::endl;
  msg << "## Wind velocity: " << v_wind / units.km_s() << " km/s" << std::endl;
  msg << "#### Derived parameters" << std::endl;
  msg << "## Cloud temperature (from pressure equ.): " << T_cloud << " K" << std::endl;
  msg << "## Cloud to wind density ratio: " << chi_0 << std::endl;
  msg << "## Cloud to wind temperature ratio: " << T_cloud / T_wind << std::endl;
  msg << "## Uniform pressure (code units): " << pressure << std::endl;
  msg << "## Wind sonic Mach: " << v_wind / c_s_wind << std::endl;
  msg << "## Cloud crushing time: " << t_cc / units.myr() << " Myr" << std::endl;

  // (potentially) rescale global times only at the beginning of a simulation
  auto rescale_code_time_to_tcc =
      pin->GetOrAddBoolean("problem/cloud", "rescale_code_time_to_tcc", false);

  if (rescale_code_time_to_tcc) {
    msg << "#### INFO:" << std::endl;
    Real tlim_orig = pin->GetReal("parthenon/time", "tlim");
    Real tlim_rescaled = tlim_orig * t_cc;
    // rescale sim time limit
    pin->SetReal("parthenon/time", "tlim", tlim_rescaled);
    // rescale dt of each output block
    parthenon::InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
      if (pib->block_name.compare(0, 16, "parthenon/output") == 0) {
        auto dt = pin->GetReal(pib->block_name, "dt");
        pin->SetReal(pib->block_name, "dt", dt * t_cc);
      }
      pib = pib->pnext; // move to next input block name
    }

    msg << "## Interpreted time limits (partenon/time/tlim and dt for outputs) as in "
           "multiples of the cloud crushing time."
        << std::endl
        << "## Simulation will now run for " << tlim_rescaled
        << " [code_time] corresponding to " << tlim_orig << " [t_cc]." << std::endl;
    // Now disable rescaling of times so that this is done only once and not for restarts
    pin->SetBoolean("problem/cloud", "rescale_code_time_to_tcc", false);
  }
  if (parthenon::Globals::my_rank == 0) {
    msg << "######################################" << std::endl;

    std::cout << msg.str();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the cloud in wind setup

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  if (((Bx != 0.0) || (By != 0.0) || (Bz != 0.0)) && !mhd_enabled) {
    PARTHENON_FAIL("Requested to initialize magnetic fields by `cloud/plasma_beta > 0`, "
                   "but `hydro/fluid` is not supporting MHD.");
  }

  auto steepness = pin->GetOrAddReal("problem/cloud", "cloud_steepness", 10);

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  // Read problem parameters
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const Real rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));

        Real rho = rho_wind + 0.5 * (rho_cloud - rho_wind) *
                                  (1.0 - std::tanh(steepness * (rad / r_cloud - 1.0)));

        Real mom;
        // Factor 1.3 as used in Grønnow, Tepper-García, & Bland-Hawthorn 2018,
        // i.e., outside the cloud boundary region (for steepness 10)
        if (rad > 1.3 * r_cloud) {
          mom = mom_wind;
        } else {
          mom = 0.0;
        }

        u(IDN, k, j, i) = rho;
        u(IM2, k, j, i) = mom;
        // Can use rhoe_wind here as simulation is setup in pressure equil.
        u(IEN, k, j, i) = rhoe_wind + 0.5 * mom * mom / rho;

        if (mhd_enabled) {
          u(IB1, k, j, i) = Bx;
          u(IB2, k, j, i) = By;
          u(IB3, k, j, i) = Bz;
          u(IEN, k, j, i) += 0.5 * (Bx * Bx + By * By + Bz * Bz);
        }

        // Init passive scalars
        for (auto n = nhydro; n < nhydro + nscalars; n++) {
          if (rad <= r_cloud) {
            u(n, k, j, i) = 1.0 * rho;
          }
        }
      }
    }
  }

  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

void InflowWindX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  // TODO(pgrete) Add par_for_bndry to Parthenon without requiring nb
  const auto nb = IndexRange{0, 0};
  const auto rho_wind_ = rho_wind;
  const auto mom_wind_ = mom_wind;
  const auto rhoe_wind_ = rhoe_wind;
  const auto Bx_ = Bx;
  const auto By_ = By;
  const auto Bz_ = Bz;
  const bool fine = false;
  pmb->par_for_bndry(
      "InflowWindX2", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::CC,
      coarse, fine, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        cons(IDN, k, j, i) = rho_wind_;
        cons(IM2, k, j, i) = mom_wind_;
        cons(IEN, k, j, i) = rhoe_wind_ + 0.5 * mom_wind_ * mom_wind_ / rho_wind_;
        if (Bx_ != 0.0) {
          cons(IB1, k, j, i) = Bx_;
          cons(IEN, k, j, i) += 0.5 * Bx_ * Bx_;
        }
        if (By_ != 0.0) {
          cons(IB2, k, j, i) = By_;
          cons(IEN, k, j, i) += 0.5 * By_ * By_;
        }
        if (Bz_ != 0.0) {
          cons(IB3, k, j, i) = Bz_;
          cons(IEN, k, j, i) += 0.5 * Bz_ * Bz_;
        }
      });
}

parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd) {
  auto pmb = mbd->GetBlockPointer();
  auto w = mbd->Get("prim").data;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto nhydro = hydro_pkg->Param<int>("nhydro");

  Real maxscalar = 0.0;
  pmb->par_reduce(
      "cloud refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxscalar) {
        // scalar is first variable after hydro vars
        lmaxscalar = std::max(lmaxscalar, w(nhydro, k, j, i));
      },
      Kokkos::Max<Real>(maxscalar));

  if (maxscalar > 0.01) return parthenon::AmrTag::refine;
  if (maxscalar < 0.001) return parthenon::AmrTag::derefine;
  return parthenon::AmrTag::same;
};

// Compute frame_boosting velocity
parthenon::TaskStatus
compute_frame_v(parthenon::MeshData<parthenon::Real> *md) {

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
  IndexRange int_ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange int_jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange int_kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const auto units = hydro_pkg->Param<Units>("units");
  Real mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("singlecloud::mean_molecular_mass_by_kb");
  Real T_cloud = hydro_pkg->Param<Real>("Tcloud");
  std::stringstream msg;

  Kokkos::Array<Real, 2> sums{{0.0, 0.0}};

  Kokkos::parallel_reduce(
      "SingleCloud::frame_boosting_velocity", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e+1, jb.e+1, ib.e+1}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, 
      Real& local_IM_cold_gas, Real& local_cold_gas) { 
        auto &prim = prim_pack(b);
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        if ( coords.Xc<2>(j) < 0 ) {
          const Real temp =
              mean_molecular_mass_by_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i);

          if (temp <= 2*T_cloud) {

            if (k >= int_kb.s && k <= int_kb.e && j >= int_jb.s && j <= int_jb.e &&
                i >= int_ib.s && i <= int_ib.e) {
                  local_IM_cold_gas += cons(IM2, k, j, i);// * coords.CellVolume(k, j, i);
                  local_cold_gas += prim(IDN, k, j, i); //* coords.CellVolume(k, j, i);
            }
          }
        }
      },
      sums[0], sums[1]); //parthenon_output
#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 2, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL
  
  Real frame_v = sums[0]/sums[1];
  msg << "FRAME BOOST" << frame_v;
  if (frame_v != 0.) hydro_pkg->UpdateParam("inertial_frame_v", frame_v); 
  std::cout << msg.str();

  return TaskStatus::complete;

}


// Shift velocities to maintain intertial frame
parthenon::TaskStatus
apply_frame_boost(parthenon::MeshData<parthenon::Real> *md) {

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
  IndexRange int_ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange int_jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange int_kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto nhydro = hydro_pkg->Param<int>("nhydro");
  auto nscalars = hydro_pkg->Param<int>("nscalars");

  Real frame_v = hydro_pkg->Param<Real>("inertial_frame_v");
  if (fabs(frame_v) > 100.01 || frame_v < 0.0) frame_v = 0.;


  
  Kokkos::parallel_for(
    "SingleCloud::frame_boosting_velocity", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e+1, jb.e+1, ib.e+1}),  
    KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {

        auto &prim = prim_pack(b);
        auto &cons = cons_pack(b);

          if (k >= int_kb.s && k <= int_kb.e && j >= int_jb.s && j <= int_jb.e &&
                i >= int_ib.s && i <= int_ib.e) {
          
            cons(IEN, k, j, i) -= frame_v * cons(IM2, k, j, i);
            cons(IEN, k, j, i) += 0.5 * SQR(frame_v) * prim(IDN, k, j, i);
            cons(IM2, k, j, i) -= frame_v * prim(IDN, k, j, i);
            assert(("Negative densities after frame boost", prim(IDN, k, j, i) < 0.));

         }
      });

    return TaskStatus::complete; //compute only every 100 timesteps
  // output prior to every output
}

} // namespace cloud
