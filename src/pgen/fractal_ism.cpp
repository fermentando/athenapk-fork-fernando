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


//Comute and store mean_molecular_mass
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


// *******************************************************************************
//           Reading general initial conditions from binary file 
// *******************************************************************************

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Fractal ISM simulation set up

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  Units units(pin);
  auto d_cgs_factor = 1. / units.code_density_cgs();
  auto m_cgs_factor = 1. / ( units.code_density_cgs() * units.code_length_cgs() / units.code_time_cgs());
  auto e_cgs_factor = 1. / (m_cgs_factor*m_cgs_factor / units.code_density_cgs());

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const int dim1 = pmb->cellbounds.ncellsi(IndexDomain::interior);
  const int dim2 = pmb->cellbounds.ncellsj(IndexDomain::interior);
  const int dim3 = pmb->cellbounds.ncellsk(IndexDomain::interior);

  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  // Read ICs binary
    std::ifstream file("ICs.bin", std::ios::in | std::ios::binary);
    
    if (!file.is_open()) {
        PARTHENON_FAIL("Failed to open file");
    }
    
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    if (size != dim1 * dim2 * dim3 * 3) {
        PARTHENON_FAIL("File size mismatch");
    }
    
    double* data = new double[dim1 * dim2 * dim3 * 3];
    
    file.read(reinterpret_cast<char*>(data), size * sizeof(int64_t));
    
    // Assign values to primary variables
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                  u(IDN, kb.s+k, jb.s+j, ib.s+i) = data[i + j * dim1 + k * dim1 * dim2 + 0 * dim1 * dim2 * dim3] * d_cgs_factor;
                  u(IM2, kb.s+k, jb.s+j, ib.s+i) = data[i + j * dim1 + k * dim1 * dim2 + 0 * dim1 * dim2 * dim3] * m_cgs_factor;
                  u(IEN, kb.s+k, jb.s+j, ib.s+i) = data[i + j * dim1 + k * dim1 * dim2 + 0 * dim1 * dim2 * dim3] * e_cgs_factor;
            }
        }
    }
    
    delete[] data;
  


  // copy initialized vars to device
  u_dev.DeepCopy(u);
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


}