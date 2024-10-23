//! \file fractal_ism.cpp
//  \brief  Class for boosting frame in multicloud set-ups.

#include <cmath>
#include <fstream> // for ofstream
#include <limits>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/params.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../eos/adiabatic_glmmhd.hpp"
#include "../eos/adiabatic_hydro.hpp"
#include "../main.hpp"
#include "../units.hpp"
#include "fractal_ism.hpp"


namespace fractal_ism {
using namespace parthenon; 

// Compute frame_boosting velocity
parthenon::TaskStatus
compute_frame_v(parthenon::MeshData<parthenon::Real> *md) {

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  std::tuple<Real,Real> md_cold_gas{0.,0.};


  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "fractal_ism::frame_boosting_velocity",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &k, const int &j, const int &i, std::tuple<Real, Real>& local_cold_gas) {
        auto &prim = prim_pack(b);
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        if ( coords.Xc<1>(i) < 0 ) {

          const Real temp =
              mean_molecular_mass_by_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i);

          if (temp <= 2e4) {

            const Real cell_cold_mass = prim(IDN, k, j, i) * coords.CellVolume(k, j, i);
            const Real cold_v3 = pow(cons(IM3, k, j, i), 2);
            const Real cold_v2 = pow(cons(IM2, k, j, i), 2);
            const Real cold_v1 = pow(cons(IM1, k, j, i), 2);
            const Real cold_gas_speed = std::sqrt(v1 + v2 + v3);

            std::get<0>(local_cold_gas) += cell_cold_mass * cold_gas_speed;
            std::get<1>(local_cold_gas) += cell_cold_mass;

          }
        }
      },
      md_cold_gas); //parthenon_output

    Real frame_v =std::get<0>(md_cold_gas) / std::get<1>(md_cold_gas);
    if (frame_v != 0.) hydro_pkg->UpdateParam("fractal_ism::inertial_frame_v", frame_v); 

    return TaskStatus::complete;

}


// Shift velocities to maintain intertial frame
parthenon::TaskStatus
apply_frame_boost(parthenon::MeshData<parthenon::Real> *md) {

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const Real frame_v = hydro_pkg->Param<Real>("inertial_frame_v");
  if (fabs(frame_v) > 100.01 || frame_v < 0.0) frame_v = 0.;


  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "frame_boosting_velocity",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &k, const int &j, const int &i, const Real& frame_v) {
        auto &prim = prim_pack(b);
        auto &cons = cons_pack(b);

        cons(IM2, k, j, i) -= frame_v * prim(IDN, k, j, i);
        // Should add output to track x_shift?
      });

    return TaskStatus::complete; //compute only every 100 timesteps
    // output prior to every output

}


}