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


//Comute and store mean_molecularmass
void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // no access to package in this function so we use a local units object
  Units units(pin);
  auto hydro_pkg = mesh->packages.Get("Hydro");
  const parthenon::Real He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
  const parthenon::Real H_mass_fraction = 1.0 - He_mass_fraction;
  const parthenon::Real mu =
      1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);

  hydro_pkg -> AddParam<Real>("fractalism::mean_molecular_mass_by_kb", mu * units.atomic_mass_unit() / units.k_boltzmann());

}

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
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const auto units = hydro_pkg->Param<Units>("units");
  Real mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("fractalism::mean_molecular_mass_by_kb");
  Real IM_cold_gas;
  Real md_cold_gas;


  pmb->par_reduce(
      "Fractal_ism::frame_boosting_velocity", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, 
      Real &local_IM_cold_gas) {
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
            const Real cold_gas_speed = std::sqrt(cold_v1 + cold_v2 + cold_v3);

            local_IM_cold_gas += cell_cold_mass * cold_gas_speed;

          }
        }
      },
      Kokkos::Sum<Real>(IM_cold_gas)); //parthenon_output


  pmb->par_reduce(
      "Fractal_ism::frame_boosting_velocity", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
       Real &local_cold_gas) {
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
            const Real cold_gas_speed = std::sqrt(cold_v1 + cold_v2 + cold_v3);

            local_cold_gas+= cell_cold_mass;

          }
        }
      }, Kokkos::Sum<Real>(md_cold_gas));

    Real frame_v = IM_cold_gas / md_cold_gas;
    if (frame_v != 0.) hydro_pkg->UpdateParam("inertial_frame_v", frame_v); 

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
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  Real frame_v = hydro_pkg->Param<Real>("inertial_frame_v");
  if (fabs(frame_v) > 100.01 || frame_v < 0.0) frame_v = 0.;


  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FractalIMS::AdjustingFrameSpeed", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e + 1, 
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, const Real& frame_v) {
        auto &prim = prim_pack(b);
        auto &cons = cons_pack(b);

        cons(IM2, k, j, i) -= frame_v * prim(IDN, k, j, i);
        // Should add output to track x_shift?
      });

    return TaskStatus::complete; //compute only every 100 timesteps
  // output prior to every output
}


}