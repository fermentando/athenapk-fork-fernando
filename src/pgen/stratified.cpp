//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file stratified.cpp
//  \brief Idealized stratified box generator
//
//========================================================================================

// C headers

// C++ headers
#include <adios2.h>
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()
#include <fstream>   // bin file

// I/O for ICs reader
#include "globals.hpp"
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Parthenon headers
#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include <globals.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>
#include <sstream>
#include <string>

// AthenaPK headers
#include "../eos/adiabatic_glmmhd.hpp"
#include "../eos/adiabatic_hydro.hpp"
#include "../main.hpp"
#include "../tracers/tracers.hpp"
#include "../units.hpp"
#include "../utils/few_modes_ft.hpp"
#include "utils/error_checking.hpp"

namespace stratified_box {
using namespace parthenon;
using namespace parthenon::package::prelude;
using parthenon::DevMemSpace;
using parthenon::ParArray2D;
using utils::few_modes_ft::Complex;
using utils::few_modes_ft::FewModesFT;

bool drive_turbulence;
Real d_cgs_factor, m_cgs_factor, e_cgs_factor;
Real c_s;
int n_ghosts;

void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::Real beta_dt) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto a_over_H = hydro_pkg->Param<Real>("a_over_H");
  const auto surface_density = hydro_pkg->Param<Real>("surface_density");
  const auto H = hydro_pkg->Param<Real>("H_height");
  const auto units = hydro_pkg->Param<Units>("units");
  const auto code_units_length = units.code_length_cgs();
  const auto G = units.gravitational_constant();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravitationalFieldSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        auto y_norm = coords.Xc<2>(j) / (a_over_H * H);
        const Real g_z =
            2 * M_PI * G * surface_density * y_norm / std::sqrt(1 + y_norm * y_norm);

        // Apply g_r as a source term
        const Real den = prim(IDN, k, j, i);
        const Real src = (y_norm == 0) ? 0 : beta_dt * den * g_z;
        cons(IM2, k, j, i) -= src;
        cons(IEN, k, j, i) -= src * prim(IV2, k, j, i);
      });
}

//========================================================================================
//! \fn void ProblemInitPackageData(ParameterInput *pin, parthenon::State *hydro_pkg)
//  \brief Init package data from parameter input
//========================================================================================

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // no access to package in this function so we use a local units object
  Units units(pin);

  auto gamma = pin->GetReal("hydro", "gamma");
  auto gm1 = (gamma - 1.0);
  const auto &pkg = mesh->packages.Get("Hydro");
  const auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");

  auto a_over_H = pin->GetReal("problem/stratified_box", "a_over_H");
  auto surface_density = pin->GetReal("problem/stratified_box", "surface_density");
  auto T_base = pin->GetReal("problem/stratified_box", "T_base");
  auto T_cloud = pin->GetReal("problem/stratified_box", "T_cloud");
  drive_turbulence = pin->GetOrAddBoolean("problem/turbulence", "drive_turbulence", true);
  n_ghosts = pin->GetOrAddInteger("mesh", "nghost", 4);

  pkg->AddParam<Real>("a_over_H", a_over_H);
  pkg->AddParam<Real>("surface_density", surface_density);
  pkg->AddParam<Real>("T_cloud", T_cloud);
  pkg->AddParam<Real>("gamma", gamma);

  const auto g0 = 2 * M_PI * units.gravitational_constant() * surface_density;
  c_s = std::sqrt(T_base / mbar_over_kb);
  auto H_height = c_s * c_s / g0;
  auto rho0 = surface_density / 2 / a_over_H / H_height;

  pkg->AddParam<Real>("H_height", H_height);
  pkg->AddParam<Real>("rho0", rho0);

  // Relevant timescales
  auto t_ff = std::sqrt(2.0 * H_height / g0);

  std::stringstream msg;
  msg << std::setprecision(2);
  msg << "######################################" << std::endl;
  msg << "###### Stratified turbulent box" << std::endl;
  msg << "###### a/H = " << a_over_H << std::endl;
  msg << "###### surface_density = " << surface_density << " M_sun/pc^2" << std::endl;
  msg << "###### g0 = " << g0 / units.cm_s() * units.s() << " cm/s^2" << std::endl;
  msg << "###### T_base = " << T_base << " K" << std::endl;
  msg << "###### c_s = " << c_s / units.cm_s() / 1e5 << " km/s" << std::endl;
  msg << "###### H = " << H_height * 1e-3 * units.kpc() << " pc" << std::endl;
  std::cout << msg.str() << std::endl;

  // (potentially) rescale global times only at the beginning of a simulation
  auto rescale_code_time_to_tff =
      pin->GetOrAddBoolean("problem/stratified_box", "rescale_code_time_to_tff", false);

  if (rescale_code_time_to_tff) {
    msg << "#### INFO:" << std::endl;
    Real tlim_orig = pin->GetReal("parthenon/time", "tlim");
    Real tlim_rescaled = tlim_orig * t_ff;
    // rescale sim time limit
    pin->SetReal("parthenon/time", "tlim", tlim_rescaled);
    // rescale dt of each output block
    parthenon::InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
      if (pib->block_name.compare(0, 16, "parthenon/output") == 0) {
        auto dt = pin->GetReal(pib->block_name, "dt");
        pin->SetReal(pib->block_name, "dt", dt * t_ff);
      }
      pib = pib->pnext; // move to next input block name
    }
  }
}

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {

  Units units(pin);

  const std::string ics_filename = pin->GetString("job", "bin_input_file");
  std::string varname = ics_filename;
  size_t pos = varname.find(".bp");

  adios2::ADIOS adios(MPI_COMM_WORLD);

  adios2::IO get_var = adios.DeclareIO("GetVar");
  adios2::Engine bpReader = get_var.Open(ics_filename, adios2::Mode::Read);
  bpReader.BeginStep();
  adios2::Variable<double> myvar_in = get_var.InquireVariable<double>(varname.erase(pos));
  PARTHENON_REQUIRE_THROWS(myvar_in, "Could not find variable name in file.");

  d_cgs_factor = 1. / units.code_density_cgs();
  m_cgs_factor =
      1. / (units.code_density_cgs() * units.code_length_cgs() / units.code_time_cgs());
  e_cgs_factor = 1. / (units.code_density_cgs() * pow(units.code_length_cgs(), 2) /
                       pow(units.code_time_cgs(), 2));

  const auto nx = pmesh->GetDefaultBlockSize().nx(parthenon::X1DIR);
  const auto ny = pmesh->GetDefaultBlockSize().nx(parthenon::X2DIR);
  const auto nz = pmesh->GetDefaultBlockSize().nx(parthenon::X3DIR);

  const int fields = 5;

  std::vector<double> ICsdata(fields * nx * ny * nz);

  // Get meshblock for GPU
  for (int b = 0; b < md->NumBlocks(); b++) {

    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    auto hydro_pkg = pmb->packages.Get("Hydro");

    const auto mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
    const auto nhydro = hydro_pkg->Param<int>("nhydro");
    const auto nscalars = hydro_pkg->Param<int>("nscalars");
    const auto num_blocks = md->NumBlocks();
    const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

    const auto loc = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(pmb->loc);
    const auto gis = loc.lx1() * pmb->block_size.nx(X1DIR);
    const auto gjs = loc.lx2() * pmb->block_size.nx(X2DIR);
    const auto gks = loc.lx3() * pmb->block_size.nx(X3DIR);

    const int loc1 = loc.lx1();
    const int loc2 = loc.lx2();
    const int loc3 = loc.lx3();

    if ((loc.lx1() < 0) || (loc.lx2() < 0) || (loc.lx3() < 0)) {
      printf("Value of loc1 is not valid... \n");
      continue;
    }

    const adios2::Dims start{static_cast<unsigned long>(loc3),
                             static_cast<unsigned long>(loc2),
                             static_cast<unsigned long>(loc1),
                             0,
                             0,
                             0,
                             0};
    const adios2::Dims counts{1,
                              1,
                              1,
                              static_cast<unsigned long>(fields),
                              static_cast<unsigned long>(nz),
                              static_cast<unsigned long>(ny),
                              static_cast<unsigned long>(nx)};

    myvar_in.SetSelection({start, counts});
    bpReader.Get(myvar_in, ICsdata.data(), adios2::Mode::Sync);
    std::cerr << "Reading first item of meshblock data: " << ICsdata[0] << " from rank"
              << Globals::my_rank << std::endl;

    // Create initial conditions for meshblock from these values:
    // initialize conserved variables
    auto &mbd = pmb->meshblock_data.Get();
    auto &u_dev = mbd->Get("cons").data;
    // initializing on host
    auto u = u_dev.GetHostMirrorAndCopy();

    IndexRange ib = mbd->GetBoundsI(IndexDomain::interior);
    IndexRange jb = mbd->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = mbd->GetBoundsK(IndexDomain::interior);

    // Read problem parameters
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

          int index_base_0 = ((0 * nz + k) * ny + j) * nx + i;
          int index_base_1 = ((1 * nz + k) * ny + j) * nx + i;
          int index_base_2 = ((2 * nz + k) * ny + j) * nx + i;
          int index_base_3 = ((3 * nz + k) * ny + j) * nx + i;
          int index_base_4 = ((4 * nz + k) * ny + j) * nx + i;

          PARTHENON_REQUIRE_THROWS(ICsdata[index_base_0] > 0., "Densities below 0");

          u(IDN, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_0] * d_cgs_factor;
          u(IM1, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_1] * m_cgs_factor;
          u(IM2, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_2] * m_cgs_factor;
          u(IM3, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_3] * m_cgs_factor;
          u(IEN, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_4] * e_cgs_factor;
        }
      }
    }

    // copy initialized vars to device
    u_dev.DeepCopy(u);
  }
  bpReader.EndStep();
  bpReader.Close();
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;

  // For turbulence driving we need to initialize the velocity field
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const auto gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  const auto x3min = pmesh->mesh_size.xmin(X3DIR);
  const auto Lx = pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR);
  const auto Ly = pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR);
  const auto Lz = pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR);

  // already pack data here to get easy access to coords in kernels
  auto const &cons = md->PackVariables(std::vector<std::string>{"cons"});
  const auto num_blocks = md->NumBlocks();

  const auto init_vel =
      pin->GetOrAddVector<Real>("problem/turbulence", "v0", {0., 0., 0.});
  PARTHENON_REQUIRE_THROWS(init_vel.size() == 3,
                           "Initial velocity vector should have three components.");
  const auto v1 = init_vel.at(0);
  const auto v2 = init_vel.at(1);
  const auto v3 = init_vel.at(2);

  pmb->par_for(
      "Final norm. and init", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &u = cons(b);

        u(IM1, k, j, i) = u(IDN, k, j, i) * v1;
        u(IM2, k, j, i) = u(IDN, k, j, i) * v2;
        u(IM3, k, j, i) = u(IDN, k, j, i) * v3;

        u(IEN, k, j, i) += 0.5 * u(IDN, k, j, i) * (SQR(v1) + SQR(v2) + SQR(v3));
      });
}

void StratUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  GravitationalFieldSrcTerm(md, beta_dt);
}

///========================================================================================
/// Create boundary conditions from softness profile
///========================================================================================

KOKKOS_INLINE_FUNCTION
double rho_profile_Y(double Y, double rho0, double a, double H) {
  const double arg = Y / (a * H);
  return rho0 * exp(-a * (sqrt(1.0 + arg * arg) - 1.0));
}

void StratNoFlowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  // Local copies of parameters for device lambda
  auto surface_density = pmb->packages.Get("Hydro")->Param<Real>("surface_density");
  auto bc_a = pmb->packages.Get("Hydro")->Param<Real>("a_over_H");
  auto bc_H = pmb->packages.Get("Hydro")->Param<Real>("H_height");
  const auto mbar_over_kb = pmb->packages.Get("Hydro")->Param<Real>("mbar_over_kb");
  const double rho0 = surface_density / 2/bc_a/bc_H;  // midplane density
  const double a    = bc_a;
  const double H    = bc_H;
  const auto gamma = pmb->packages.Get("Hydro")->Param<Real>("gamma");
  const auto gm1 = gamma - 1.0;

  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto jg = pmb->cellbounds.GetBoundsJ(IndexDomain::inner_x2);


  pmb->par_for_bndry(
      "StratOutflowInnerX2", nb, IndexDomain::inner_x2,
      parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          const auto &coordsb = cons_pack.GetCoords();
          auto &cons = cons_pack;

          Real Y = coordsb.Xc<2>(j);
          double rhoY = rho_profile_Y(Y, rho0, a, H);
          double prsY = 1e6 * rhoY / mbar_over_kb;

          // Copy tangential velocities from last interior cell
          cons(IDN,k,j,i) = rhoY;


          // Mirror velocity profile 
          const auto j_mirror = jb.s + (jg.e - j);
          cons(IM1,k,j,i) = 0;//rhoY * cons(IM1,k,j_mirror,i) / cons(IDN,k,j_mirror,i);
          if (cons(IM2, k, jb.s, i) < 0.) cons(IM2,k,j,i) = 0;//rhoY * cons(IM2,k,j_mirror,i) / cons(IDN,k,j_mirror,i);
          else cons(IM2,k,j,i) = cons(IM2, k, jb.s, i);//rhoY * cons(IM2,k,j_mirror,i) / cons(IDN,k,j_mirror,i);
          cons(IM3,k,j,i) = 0;//rhoY * cons(IM3,k,j_mirror,i) / cons(IDN,k,j_mirror,i);

          //const auto e = (cons(IEN, k, j_mirror, i)  - 0.5 * ( SQR(cons(IM1,k,j_mirror,i)) + SQR(cons(IM2,k,j_mirror,i)) + SQR(cons(IM3,k,j_mirror,i)) ) / cons(IDN, k, j_mirror, i)) / cons(IDN, k, j_mirror, i);
          cons(IEN,k,j,i) = prsY / gm1 + 0.5 * ( SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) + SQR(cons(IM3,k,j,i)) ) / rhoY;
          
      });
}

void StratInflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  const auto nb = IndexRange{0,0};
  const bool fine = false;
  const auto gamma = pmb->packages.Get("Hydro")->Param<Real>("gamma");
  const auto gm1 = gamma - 1.0;

  // Local copies of parameters for device lambda
  auto surface_density = pmb->packages.Get("Hydro")->Param<Real>("surface_density");
  auto bc_a = pmb->packages.Get("Hydro")->Param<Real>("a_over_H");
  auto bc_H = pmb->packages.Get("Hydro")->Param<Real>("H_height");
  const double rho0 = surface_density / 2/bc_a/bc_H;  // midplane density
  const double a    = bc_a;
  const double H    = bc_H;

  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);

  pmb->par_for_bndry(
      "StratOutflowInnerX2", nb, IndexDomain::inner_x2, parthenon::TopologicalElement::CC,
      coarse, fine, KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        const auto &coordsb = cons_pack.GetCoords();
        auto &cons = cons_pack;
        Real Y = coordsb.Xc<2>(j);
        double rhoY = rho_profile_Y(Y, rho0, a, H);

          // Copy tangential velocities from last interior cell
          cons(IDN,k,j,i) = rhoY;
          Real T = cons(IPR,k,jb.s,i) / cons(IDN,k,jb.s,i);
          Real ci = sqrt( gamma * T );

          auto V_tot = sqrt( SQR(cons(IV1,k,jb.s,i)) + SQR(cons(IV2,k,jb.s,i)) + SQR(cons(IV3,k,jb.s,i)) );
          auto Hi = ci * ci / gm1 + 0.5 * V_tot * V_tot;
          auto J_riemann = -V_tot + 2.0 * ci / gm1;

          // Find biggest cb root 
          // Rewrite equation as: c_b^2/gm1 + 0.5*(J_riemann - 2*c_b/gm1)^2 - Hi = 0
          // Expanding: c_b^2/gm1 + 0.5*(J_riemann^2 - 4*J_riemann*c_b/gm1 + 4*c_b^2/gm1^2) - Hi = 0
          // Multiply by gm1: c_b^2 + 0.5*gm1*(J_riemann^2 - 4*J_riemann*c_b/gm1 + 4*c_b^2/gm1^2) - Hi*gm1 = 0
          // Simplify: c_b^2 + 0.5*gm1*J_riemann^2 - 2*J_riemann*c_b + 2*c_b^2/gm1 - Hi*gm1 = 0
          // Collect c_b terms: (1 + 2/gm1)*c_b^2 - 2*J_riemann*c_b + (0.5*gm1*J_riemann^2 - Hi*gm1) = 0

          auto a_coeff = 1.0 + 2.0 / gm1;
          auto b_coeff = -2.0 * J_riemann;
          auto c_coeff = 0.5 * gm1 * J_riemann * J_riemann - Hi * gm1;

          auto discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
          auto c_b = (-b_coeff + sqrt(discriminant)) / (2.0 * a_coeff);  // largest root

          auto Vn = -J_riemann + 2.0 * c_b / gm1;
          auto Mn = Vn / c_b;

          // Define inner pressure and temperature

          auto pb = cons(IPR,k,jb.s,i) * pow( (1.0 + 0.5 * gm1 * Mn * Mn), (gamma / gm1) );
          auto Tb = T * pow( (1.0 + 0.5 * gm1 * Mn * Mn), -1.0);
          auto rho_b = pb / Tb;

          // Set boundary conditions
          cons(IDN,k,j,i) = rho_b;
          cons(IV1,k,j,i) = 0;
          cons(IV3,k,j,i) = 0;
          cons(IV2,k,j,i) = Vn;
          cons(IPR,k,j,i) = pb;

      });
}

void StratNoFlowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  const auto nb = IndexRange{0, 0};
  const bool fine = false;

  auto surface_density = pmb->packages.Get("Hydro")->Param<Real>("surface_density");
  auto bc_a = pmb->packages.Get("Hydro")->Param<Real>("a_over_H");
  auto bc_H = pmb->packages.Get("Hydro")->Param<Real>("H_height");
  const auto mbar_over_kb = pmb->packages.Get("Hydro")->Param<Real>("mbar_over_kb");
  const double rho0 = surface_density / 2/ bc_a/bc_H;  // midplane density
  const double a    = bc_a;
  const double H    = bc_H;
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto jg = pmb->cellbounds.GetBoundsJ(IndexDomain::outer_x2);
  const auto gamma = pmb->packages.Get("Hydro")->Param<Real>("gamma");
  const auto gm1 = gamma - 1.0;

  pmb->par_for_bndry(
      "StratOutflowInnerX2", nb, IndexDomain::outer_x2,
      parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          const auto &coordsb = cons_pack.GetCoords();
          auto &cons = cons_pack;
          Real Y = coordsb.Xc<2>(j);
          double rhoY = rho_profile_Y(Y, rho0, a, H);
          auto prsY = 1e6 * rhoY / mbar_over_kb;

          // Copy tangential velocities from last interior cell
          cons(IDN,k,j,i) = rhoY;

          // Mirror velocity profile 
          const auto j_mirror = jb.e - (j - jg.s);
          cons(IM1,k,j,i) = 0;
          if (cons(IM2, k, jb.e, i) > 0.) cons(IM2,k,j,i) = 0;//rhoY * cons(IM2,k,j_mirror,i) / cons(IDN,k,j_mirror,i);
          else cons(IM2,k,j,i) = cons(IM2, k, jb.e, i);//rhoY * cons(IM2,k,j_mirror,i) / cons(IDN,k,j_mirror,i);
          cons(IM3,k,j,i) = 0;

          //const auto e = (cons(IEN, k, j_mirror, i)  - 0.5 * ( SQR(cons(IM1,k,j_mirror,i)) + SQR(cons(IM2,k,j_mirror,i)) + SQR(cons(IM3,k,j_mirror,i)) ) / cons(IDN, k, j_mirror, i)) / cons(IDN, k, j_mirror, i);
          cons(IEN,k,j,i) = prsY / gm1 + 0.5 * ( SQR(cons(IM1,k,j,i)) + SQR(cons(IM2,k,j,i)) + SQR(cons(IM3,k,j,i)) ) / rhoY;

      });
}

void StratInflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  const auto nb = IndexRange{0,0};
  const bool fine = false;
  const auto gamma = pmb->packages.Get("Hydro")->Param<Real>("gamma");
  const auto gm1 = gamma - 1.0;

  auto surface_density = pmb->packages.Get("Hydro")->Param<Real>("surface_density");
  auto bc_a = pmb->packages.Get("Hydro")->Param<Real>("a_over_H");
  auto bc_H = pmb->packages.Get("Hydro")->Param<Real>("H_height");
  const double rho0 = surface_density / 2/ bc_a/bc_H;  // midplane density
  const double a    = bc_a;
  const double H    = bc_H;
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);


  pmb->par_for_bndry(
      "StratOutflowOuterX2", nb, IndexDomain::inner_x2,
      parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          const auto &coordsb = cons_pack.GetCoords();
          auto &cons = cons_pack;
          Real Y = coordsb.Xc<2>(j);
          double rhoY = rho_profile_Y(Y, rho0, a, H);

          // Copy tangential velocities from last interior cell
          cons(IDN,k,j,i) = rhoY;
          Real T = cons(IPR,k,jb.e,i) / cons(IDN,k,jb.e,i);
          Real ci = sqrt( gamma * T );

          auto V_tot = sqrt( SQR(cons(IV1,k,jb.e,i)) + SQR(cons(IV2,k,jb.e,i)) + SQR(cons(IV3,k,jb.e,i)) );
          auto Hi = ci * ci / gm1 + 0.5 * V_tot * V_tot;
          auto J_riemann = -V_tot + 2.0 * ci / gm1;

          // Find biggest cb root 
          // Rewrite equation as: c_b^2/gm1 + 0.5*(J_riemann - 2*c_b/gm1)^2 - Hi = 0
          // Expanding: c_b^2/gm1 + 0.5*(J_riemann^2 - 4*J_riemann*c_b/gm1 + 4*c_b^2/gm1^2) - Hi = 0
          // Multiply by gm1: c_b^2 + 0.5*gm1*(J_riemann^2 - 4*J_riemann*c_b/gm1 + 4*c_b^2/gm1^2) - Hi*gm1 = 0
          // Simplify: c_b^2 + 0.5*gm1*J_riemann^2 - 2*J_riemann*c_b + 2*c_b^2/gm1 - Hi*gm1 = 0
          // Collect c_b terms: (1 + 2/gm1)*c_b^2 - 2*J_riemann*c_b + (0.5*gm1*J_riemann^2 - Hi*gm1) = 0

          auto a_coeff = 1.0 + 2.0 / gm1;
          auto b_coeff = -2.0 * J_riemann;
          auto c_coeff = 0.5 * gm1 * J_riemann * J_riemann - Hi * gm1;

          auto discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
          auto c_b = (-b_coeff + sqrt(discriminant)) / (2.0 * a_coeff);  // largest root

          auto Vn = -J_riemann + 2.0 * c_b / gm1;
          auto Mn = Vn / c_b;

          // Define inner pressure and temperature

          auto pb = cons(IPR,k,jb.e,i) * pow( (1.0 + 0.5 * gm1 * Mn * Mn), (gamma / gm1) );
          auto Tb = T * pow( (1.0 + 0.5 * gm1 * Mn * Mn), -1.0);
          auto rho_b = pb / Tb;

          // Set boundary conditions
          cons(IDN,k,j,i) = rho_b;
          cons(IV1,k,j,i) = 0;
          cons(IV3,k,j,i) = 0;
          cons(IV2,k,j,i) = Vn;
          cons(IPR,k,j,i) = pb;

      });
}

void InjectBlob(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");

  const auto inject_once_at_time = pkg->Param<Real>("turbulence/inject_once_at_time");
  const auto inject_once_at_cycle = pkg->Param<int>("turbulence/inject_once_at_cycle");
  const auto inject_once_on_restart =
      pkg->Param<bool>("turbulence/inject_once_on_restart");

  // Check if any condition is met for injecting
  if (!((inject_once_at_time >= tm.time && inject_once_at_time < tm.time + dt) ||
        (inject_once_at_cycle == tm.ncycle) || inject_once_on_restart)) {
    return;
  }

  // Always disable injecting as the original value doesn't matter
  pkg->UpdateParam("turbulence/inject_once_at_time", -1.0);
  pkg->UpdateParam("turbulence/inject_once_at_cycle", -1);
  pkg->UpdateParam("turbulence/inject_once_on_restart", false);

  const auto radius = pkg->Param<Real>("stratified_box/r_cloud_inserted");
  const auto chi = pkg->Param<Real>("stratified_box/chi_cloud_inserted");
  const auto loc = pkg->Param<std::vector<Real>>("stratified_box/loc_cloud_inserted");

  // redef vars for easier capture (std::vector does not work)
  const auto loc_x = loc[0];
  const auto loc_y = loc[1];
  const auto loc_z = loc[2];
  if (parthenon::Globals::my_rank == 0) {
    std::stringstream msg;
    msg << std::setprecision(2);
    msg << "\n# Turbulence driver: injecting cloud";
    msg << " at location " << loc_x << " " << loc_y << " " << loc_z
        << " with overdensity " << chi << ".\n\n ";
    std::cout << msg.str();
  }

  const auto *const error_msg =
      "Blob bounds crossing domain bounds currently not supported.";
  PARTHENON_REQUIRE_THROWS(loc_x + radius < pmb->pmy_mesh->mesh_size.xmax(X1DIR),
                           error_msg)
  PARTHENON_REQUIRE_THROWS(loc_x - radius > pmb->pmy_mesh->mesh_size.xmin(X1DIR),
                           error_msg)
  PARTHENON_REQUIRE_THROWS(loc_y + radius < pmb->pmy_mesh->mesh_size.xmax(X2DIR),
                           error_msg)
  PARTHENON_REQUIRE_THROWS(loc_y - radius > pmb->pmy_mesh->mesh_size.xmin(X2DIR),
                           error_msg)
  PARTHENON_REQUIRE_THROWS(loc_z + radius < pmb->pmy_mesh->mesh_size.xmax(X3DIR),
                           error_msg)
  PARTHENON_REQUIRE_THROWS(loc_z - radius > pmb->pmy_mesh->mesh_size.xmin(X3DIR),
                           error_msg)

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  const auto fluid = pkg->Param<Fluid>("fluid");
  // To fix this, we'd just have to account for the magnetic energy in the reduction
  PARTHENON_REQUIRE(fluid == Fluid::euler,
                    "Injecting only supported for hydro sims at the moment.");

  const auto gamma = pkg->Param<Real>("AdiabaticIndex");

  pmb->par_for(
      "turbulence: inject blob", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);

        const auto x = coords.Xc<1>(i) - loc_x;
        const auto y = coords.Xc<2>(j) - loc_y;
        const auto z = coords.Xc<3>(k) - loc_z;
        const auto r = std::sqrt(SQR(x) + SQR(y) + SQR(z));

        if (r < radius) {
          const auto kin_en_density = 0.5 *
                                      (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                       SQR(cons(IM3, k, j, i))) /
                                      cons(IDN, k, j, i);
          auto rho_e = cons(IEN, k, j, i) - kin_en_density;

          // increase density according to overdensity
          cons(IDN, k, j, i) *= chi;
          // adjust momentum (so that the velocity remains constant)
          cons(IM1, k, j, i) *= chi;
          cons(IM2, k, j, i) *= chi;
          cons(IM3, k, j, i) *= chi;
          // adjust total energy density (using original rho_e translates to an increase
          // of 1/chi in temperature)
          cons(IEN, k, j, i) = kin_en_density * chi + rho_e;
        }
      });

  // Update cooling routine
  // pkg->UpdateParam("enable_cooling", Cooling::tabular);
}

void Rescale(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");

  const auto rescale_once_at_time = pkg->Param<Real>("turbulence/rescale_once_at_time");
  const auto rescale_once_at_cycle = pkg->Param<int>("turbulence/rescale_once_at_cycle");
  const auto rescale_once_on_restart =
      pkg->Param<bool>("turbulence/rescale_once_on_restart");

  // Check if any condition is met for rescaling
  if (!((rescale_once_at_time >= tm.time && rescale_once_at_time < tm.time + dt) ||
        (rescale_once_at_cycle == tm.ncycle) || rescale_once_on_restart)) {
    return;
  }

  // Always disable rescaling as the original value doesn't matter
  pkg->UpdateParam("turbulence/rescale_once_at_time", -1.0);
  pkg->UpdateParam("turbulence/rescale_once_at_cycle", -1);
  pkg->UpdateParam("turbulence/rescale_once_on_restart", false);

  const auto rescale_to_rms_Ms = pkg->Param<Real>("turbulence/rescale_to_rms_Ms");
  PARTHENON_REQUIRE_THROWS(rescale_to_rms_Ms > 0.0, "What's a negative Mach number?");

  if (parthenon::Globals::my_rank == 0) {
    std::stringstream msg;
    msg << std::setprecision(2);
    msg << "\n# Turbulence driver: rescaling to an RMS Ms of " << rescale_to_rms_Ms;
    msg << " by resetting the temperature.\n\n";
    std::cout << msg.str();
  }

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  const auto fluid = pkg->Param<Fluid>("fluid");
  // To fix this, we'd just have to account for the magnetic energy in the reduction
  PARTHENON_REQUIRE(fluid == Fluid::euler,
                    "Rescaling only supported for hydro sims at the moment.");

  const auto gamma = pkg->Param<Real>("AdiabaticIndex");

  Real Ms2_sum;
  Kokkos::parallel_reduce(
      "turbulence: calc RMS Ms",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lMs2_sum) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);

        const auto kin_en_density = 0.5 *
                                    (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                     SQR(cons(IM3, k, j, i))) /
                                    cons(IDN, k, j, i);
        auto pres = (gamma - 1.0) * (cons(IEN, k, j, i) - kin_en_density);
        lMs2_sum += 2.0 * kin_en_density / (gamma * pres) * coords.CellVolume(k, j, i);
      },
      Ms2_sum);

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &Ms2_sum, 1, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  const auto Lx =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  const auto Ly =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);
  const auto Lz =
      pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);
  auto norm = SQR(rescale_to_rms_Ms) / (Ms2_sum / (Lx * Ly * Lz));

  pmb->par_for(
      "Rescale temperature to target rms Ms", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);

        const auto kin_en_density = 0.5 *
                                    (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                     SQR(cons(IM3, k, j, i))) /
                                    cons(IDN, k, j, i);

        auto e = (cons(IEN, k, j, i) - kin_en_density) / cons(IDN, k, j, i);

        cons(IEN, k, j, i) = kin_en_density + e / norm * cons(IDN, k, j, i);
      });
}

//----------------------------------------------------------------------------------------
//! \fn void StratHst(MeshData<Real> *md)
//  \brief Hst file initialiser for new variables

// TODO(?) until we are able to process multiple variables in a single hst function call
// we'll use this enum to identify the various vars.
enum class HstQuan { mc, mbw, Mcx1, Mcx2, Mcx3, mcout, mwout, Ms, Ma, pb };

// Compute the local sum of cloud mass
template <HstQuan hst_quan>
Real StratHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  Real mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  const auto T_cloud_ = hydro_pkg->Param<Real>("T_cloud");
  const auto gamma = hydro_pkg->Param<Real>("AdiabaticIndex");

  const auto &prims_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // after this function is called the result is MPI_SUMed across all procs/meshblocks
  // thus, we're only concerned with local sums
  Real sum;

  if (hst_quan == HstQuan::mcout || hst_quan == HstQuan::mwout) {
    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::outer_x2);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::outer_x2);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::outer_x2);

    auto pmesh = pmb->pmy_mesh;
    const auto x2max = pmesh->mesh_size.xmax(X2DIR);
    const auto H_height = hydro_pkg->Param<Real>("H_height");

    pmb->par_reduce(
        "WTopenrun::outflowing_gas", 0, prims_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          const auto &prims = prims_pack(b);
          const auto &cons = cons_pack(b);
          const auto &coords = prims_pack.GetCoords(b);
          const Real rho = prims(IDN, k, j, i);
          const Real My = cons(IM2, k, j, i);
          const Real temp = mean_molecular_mass_by_kb * prims(IPR, k, j, i) / rho;

          if (coords.Xc<2>(j) > x2max && My > 0.0) {
            if (hst_quan == HstQuan::mcout && temp <= 5 * T_cloud_) {
              const Real mass = rho * coords.CellVolume(k, j, i);
              lsum += mass;
            }
            if (hst_quan == HstQuan::mwout && temp > 5 * T_cloud_ &&
                temp <= 10 * T_cloud_) {
              const Real mass = rho * coords.CellVolume(k, j, i);
              lsum += mass;
            }
          }
        },
        sum);
  }

  else {

    pmb->par_reduce(
        "WTOpenRun::hst_calc", 0, prims_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          const auto &cons = cons_pack(b);
          const auto &prim = prims_pack(b);
          const auto &coords = prims_pack.GetCoords(b);
          const Real temp =
              mean_molecular_mass_by_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i);

          const auto vel2 = (prim(IV1, k, j, i) * prim(IV1, k, j, i) +
                             prim(IV2, k, j, i) * prim(IV2, k, j, i) +
                             prim(IV3, k, j, i) * prim(IV3, k, j, i));

          const auto c_s = std::sqrt(gamma * prim(IPR, k, j, i) /
                                     prim(IDN, k, j, i)); // speed of sound

          const auto e_kin = 0.5 * prim(IDN, k, j, i) * vel2;

          if (hst_quan == HstQuan::Ms) { // Ms
            lsum += std::sqrt(vel2) / c_s * coords.CellVolume(k, j, i);
          }

          if (fluid == Fluid::glmmhd) {
            const auto B2 = (prim(IB1, k, j, i) * prim(IB1, k, j, i) +
                             prim(IB2, k, j, i) * prim(IB2, k, j, i) +
                             prim(IB3, k, j, i) * prim(IB3, k, j, i));

            const auto e_mag = 0.5 * B2;

            if (hst_quan == HstQuan::Ma) { // Ma
              lsum += std::sqrt(e_kin / e_mag) * coords.CellVolume(k, j, i);
            } else if (hst_quan == HstQuan::pb) { // plasma beta
              lsum += prim(IPR, k, j, i) / e_mag * coords.CellVolume(k, j, i);
            }
          }

          if (temp <= 2 * T_cloud_) {

            if (hst_quan == HstQuan::mc) {
              lsum += prim(IDN, k, j, i) * coords.CellVolume(k, j, i);
            }
            if (hst_quan == HstQuan::Mcx1) {
              lsum += cons(IM1, k, j, i) * coords.CellVolume(k, j, i);
            }
            if (hst_quan == HstQuan::Mcx2) {
              lsum += cons(IM2, k, j, i) * coords.CellVolume(k, j, i);
            }
            if (hst_quan == HstQuan::Mcx3) {
              lsum += cons(IM3, k, j, i) * coords.CellVolume(k, j, i);
            }
          }
          if (temp <= 10 * T_cloud_) {
            if (hst_quan == HstQuan::mbw) {
              lsum += prim(IDN, k, j, i) * coords.CellVolume(k, j, i);
            }
          }
        },
        sum);
  }

  return sum;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {

  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  const auto fluid = pkg->Param<Fluid>("fluid");

  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    StratHst<HstQuan::mc>, "mc"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    StratHst<HstQuan::mc>, "mbw"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    StratHst<HstQuan::Mcx1>, "Mcx1"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    StratHst<HstQuan::Mcx2>, "Mcx2"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    StratHst<HstQuan::Mcx3>, "Mcx3"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    StratHst<HstQuan::mcout>, "mcout"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    StratHst<HstQuan::mwout>, "mwout"));

  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    StratHst<HstQuan::Ms>, "Ms"));
  if (fluid == Fluid::glmmhd) {
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, StratHst<HstQuan::Ma>, "Ma"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, StratHst<HstQuan::pb>, "plasma_beta"));
  }

  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

  // Driving turbulence parameters
  Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
             std::vector<int>({3}));
  pkg->AddField("acc", m);

  auto num_modes =
      pin->GetInteger("problem/turbulence", "num_modes"); // number of wavemodes

  uint32_t rseed =
      pin->GetOrAddInteger("problem/turbulence", "rseed", -1); // seed for random number.
  pkg->AddParam<>("turbulence/rseed", rseed);

  auto k_peak =
      pin->GetOrAddReal("problem/turbulence", "kpeak", 0.0); // peak of the forcing spec
  pkg->AddParam<>("turbulence/kpeak", k_peak);

  auto accel_rms =
      pin->GetReal("problem/turbulence", "accel_rms"); // turbulence amplitude
  pkg->AddParam<>("turbulence/accel_rms", accel_rms);

  auto t_corr =
      pin->GetReal("problem/turbulence", "corr_time"); // forcing autocorrelation time
  pkg->AddParam<>("turbulence/t_corr", t_corr);

  Real sol_weight = pin->GetReal("problem/turbulence", "sol_weight"); // solenoidal weight
  pkg->AddParam<>("turbulence/sol_weight", sol_weight);

  // list of wavenumber vectors
  auto k_vec = ParArray2D<Real>("k_vec", 3, num_modes);
  auto k_vec_host = Kokkos::create_mirror_view(k_vec);
  for (int j = 0; j < 3; j++) {
    for (int i = 1; i <= num_modes; i++) {
      k_vec_host(j, i - 1) =
          pin->GetInteger("modes", "k_" + std::to_string(i) + "_" + std::to_string(j));
    }
  }
  Kokkos::deep_copy(k_vec, k_vec_host);

  auto few_modes_ft = FewModesFT(pin, pkg, "turbulence", num_modes, k_vec, k_peak,
                                 sol_weight, t_corr, rseed);
  // object must be mutable to update the internal state of the RNG
  pkg->AddParam<>("turbulence/few_modes_ft", few_modes_ft, true);

  // Check if this is is a restart and restore previous state
  if (pin->DoesParameterExist("problem/turbulence", "accel_hat_0_0_r")) {
    // Need to extract mutable object from Params here as the original few_modes_ft above
    // and the one in Params are different instances
    auto *pfew_modes_ft = pkg->MutableParam<FewModesFT>("turbulence/few_modes_ft");
    // Restore (common) acceleration field in spectral space
    auto accel_hat = pfew_modes_ft->GetVarHat();
    auto accel_hat_host = Kokkos::create_mirror_view(accel_hat);
    for (int i = 0; i < 3; i++) {
      for (int m = 0; m < num_modes; m++) {
        auto real =
            pin->GetReal("problem/turbulence", "accel_hat_" + std::to_string(i) + "_" +
                                                   std::to_string(m) + "_r");
        auto imag =
            pin->GetReal("problem/turbulence", "accel_hat_" + std::to_string(i) + "_" +
                                                   std::to_string(m) + "_i");
        accel_hat_host(i, m) = Complex(real, imag);
      }
    }
    Kokkos::deep_copy(accel_hat, accel_hat_host);

    // Restore state of random number gen
    {
      std::istringstream iss(pin->GetString("problem/turbulence", "state_rng"));
      pfew_modes_ft->RestoreRNG(iss);
    }
    // Restore state of dist
    {
      std::istringstream iss(pin->GetString("problem/turbulence", "state_dist"));
      pfew_modes_ft->RestoreDist(iss);
    }
  }


  // Parameters to rescale the simulation to a target Mach number at a given cycle,
  // time, or restart
  auto rescale_once_at_time =
      pin->GetOrAddReal("problem/stratified_box", "rescale_once_at_time", -1.0);
  auto rescale_once_at_cycle =
      pin->GetOrAddInteger("problem/stratified_box", "rescale_once_at_cycle", -1);
  auto rescale_once_on_restart =
      pin->GetOrAddBoolean("problem/stratified_box", "rescale_once_on_restart", false);

  PARTHENON_REQUIRE_THROWS(
      (rescale_once_at_time < 0.0 && rescale_once_at_cycle < 0 &&
       !rescale_once_on_restart) ||
          (rescale_once_at_cycle * rescale_once_at_time < 0.0 &&
           !rescale_once_on_restart) ||
          (rescale_once_at_cycle * rescale_once_at_time > 0.0 && rescale_once_on_restart),
      "Rescaling should only be set for one option (or none at all).");
  // Make Params mutable as they're reset after rescale
  pkg->AddParam<>("turbulence/rescale_once_at_time", rescale_once_at_time, true);
  pkg->AddParam<>("turbulence/rescale_once_at_cycle", rescale_once_at_cycle, true);
  pkg->AddParam<>("turbulence/rescale_once_on_restart", rescale_once_on_restart, true);

  auto rescale_to_rms_Ms =
      pin->GetOrAddReal("problem/stratified_box", "rescale_to_rms_Ms", -1.0);
  pkg->AddParam<>("turbulence/rescale_to_rms_Ms", rescale_to_rms_Ms);

  // Parameters to inject overdense blobs into the simulation with a target overdensity
  // and radius at a given cycle, time, or restart
  auto inject_once_at_time =
      pin->GetOrAddReal("problem/stratified_box", "inject_once_at_time", -1.0);
  auto inject_once_at_cycle =
      pin->GetOrAddInteger("problem/stratified_box", "inject_once_at_cycle", -1);
  auto inject_once_on_restart =
      pin->GetOrAddBoolean("problem/stratified_box", "inject_once_on_restart", false);

  PARTHENON_REQUIRE_THROWS(
      (inject_once_at_time < 0.0 && inject_once_at_cycle < 0 &&
       !inject_once_on_restart) ||
          (inject_once_at_cycle * inject_once_at_time < 0.0 && !inject_once_on_restart) ||
          (inject_once_at_cycle * inject_once_at_time > 0.0 && inject_once_on_restart),
      "injectng should only be set for one option (or none at all).");
  // Make Params mutable as they're reset after inject
  pkg->AddParam<>("turbulence/inject_once_at_time", inject_once_at_time, true);
  pkg->AddParam<>("turbulence/inject_once_at_cycle", inject_once_at_cycle, true);
  pkg->AddParam<>("turbulence/inject_once_on_restart", inject_once_on_restart, true);


  auto inject_blob_radius = pin->GetReal("problem/stratified_box", "r_cloud_inserted");
  pkg->AddParam<>("stratified_box/r_cloud_inserted", inject_blob_radius);

  auto inject_blob_loc =
      pin->GetVector<Real>("problem/stratified_box", "loc_cloud_inserted");
  pkg->AddParam<>("stratified_box/loc_cloud_inserted", inject_blob_loc);

  auto inject_blob_chi = pin->GetReal("problem/stratified_box", "chi_cloud_inserted");
  pkg->AddParam<>("stratified_box/chi_cloud_inserted", inject_blob_chi);

  // Frame tracking parameters for cold gas drift
  auto enable_cold_gas_frame_track = pin->GetOrAddBoolean(
      "problem/stratified_box", "enable_cold_gas_frame_track", false);
  pkg->AddParam<>("stratified_box/enable_cold_gas_frame_track",
                  enable_cold_gas_frame_track);

  // Cumulative displacement of the frame (in code units)
  // This is mutable to allow updates during simulation
  pkg->AddParam<Real>("stratified_box/frame_displacement_y", 0.0, true);
}

void ProblemInitTracerData(ParameterInput * /*pin*/,
                           parthenon::StateDescriptor *tracer_pkg) {
  // Number of lookback times to be stored (in powers of 2,
  // i.e., 12 allows to go from 0, 2^0 = 1, 2^1 = 2, 2^2 = 4, ..., 2^10 = 1024 cycles)
  const int n_lookback = 12; // could even be made an input parameter if required/desired
                             // (though it should probably not be changeable for restarts)
  tracer_pkg->AddParam("turbulence/n_lookback", n_lookback);

  const auto swarm_name = tracer_pkg->Param<std::string>("swarm_name");
  // Using a vector to reduce code duplication.
  Metadata vreal_swarmvalue_metadata(
      {Metadata::Real, Metadata::Vector, Metadata::Restart},
      std::vector<int>{n_lookback});
  tracer_pkg->AddSwarmValue("s", swarm_name, vreal_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot", swarm_name, vreal_swarmvalue_metadata);
  // Timestamps for the lookback entries
  tracer_pkg->AddParam<>("turbulence/t_lookback", std::vector<Real>(n_lookback),
                         Params::Mutability::Restart);
}

// SetPhases is used as InitMeshBlockUserData because phases need to be reset on remeshing
void SetPhases(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto few_modes_ft = hydro_pkg->Param<FewModesFT>("turbulence/few_modes_ft");
  few_modes_ft.SetPhases(pmb, pin);
}

//----------------------------------------------------------------------------------------
//! \fn void Generate()
//  \brief Generate velocity pertubation.

void Generate(MeshData<Real> *md, Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  // Must be mutable so the internal RNG state is updated
  auto *few_modes_ft = hydro_pkg->MutableParam<FewModesFT>("turbulence/few_modes_ft");
  few_modes_ft->Generate(md, dt, "acc");
}

//----------------------------------------------------------------------------------------
//! \fn void Perturb(Real dt)
//  \brief Add velocity perturbation to the hydro variables

void Perturb(MeshData<Real> *md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto acc_pack = md->PackVariables(std::vector<std::string>{"acc"});

  Kokkos::Array<Real, 4> sums{{0.0, 0.0, 0.0, 0.0}};
  Kokkos::parallel_reduce(
      "forcing: calc mean momenum",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmass_sum,
                    Real &lim1_sum, Real &lim2_sum, Real &lim3_sum) {
        const auto &coords = cons_pack.GetCoords(b);
        auto den = cons_pack(b, IDN, k, j, i);
        lmass_sum += den * coords.CellVolume(k, j, i);
        lim1_sum += den * acc_pack(b, 0, k, j, i) * coords.CellVolume(k, j, i);
        lim2_sum += den * acc_pack(b, 1, k, j, i) * coords.CellVolume(k, j, i);
        lim3_sum += den * acc_pack(b, 2, k, j, i) * coords.CellVolume(k, j, i);
      },
      sums[0], sums[1], sums[2], sums[3]);

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 4, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  pmb->par_reduce(
      "forcing: remove mean momentum and calc norm", 0, acc_pack.GetDim(5) - 1, 0, 2,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
                    Real &lampl_sum) {
        const auto &coords = acc_pack.GetCoords(b);
        acc_pack(b, n, k, j, i) -= sums[n + 1] / sums[0];
        lampl_sum += SQR(acc_pack(b, n, k, j, i)) * coords.CellVolume(k, j, i);
      },
      sums[0]);

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 1, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  const auto Lx =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  const auto Ly =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);
  const auto Lz =
      pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);
  const auto accel_rms = hydro_pkg->Param<Real>("turbulence/accel_rms");
  auto norm = accel_rms / std::sqrt(sums[0] / (Lx * Ly * Lz));

  pmb->par_for(
      "apply momemtum perturb", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &acc = acc_pack(b);

        auto &acc_0 = acc(0, k, j, i);
        auto &acc_1 = acc(1, k, j, i);
        auto &acc_2 = acc(2, k, j, i);

        // normalizing accel field here so that the actual values are used in the output
        acc_0 *= norm;
        acc_1 *= norm;
        acc_2 *= norm;

        Real qa = dt * cons(IDN, k, j, i);
        cons(IEN, k, j, i) +=
            (cons(IM1, k, j, i) * dt * acc_0 + cons(IM2, k, j, i) * dt * acc_1 +
             cons(IM3, k, j, i) * dt * acc_2 +
             (SQR(acc_0) + SQR(acc_1) + SQR(acc_2)) * qa * qa / (2 * cons(IDN, k, j, i)));

        cons(IM1, k, j, i) += qa * acc_0;
        cons(IM2, k, j, i) += qa * acc_1;
        cons(IM3, k, j, i) += qa * acc_2;
      });
}

//----------------------------------------------------------------------------------------
//! \fn void FewModesTurbulenceDriver::Driving(void)
//  \brief Generate and Perturb the velocity field

// Forward declarations
void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
void ColdGasFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);

//----------------------------------------------------------------------------------------
//! \fn void DrivingAndFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm, const
//! Real dt)
//  \brief Wrapper function that calls both turbulence driving and cold gas frame tracking
//  This is used as the ProblemSourceFirstOrder callback

void DrivingAndFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm,
                          const Real dt) {
  // Call turbulence driving
  Driving(md, tm, dt);

  // NOTE: Temporarily disabled due to issues with boundary conditions
  // (see conversation and issue tracking). To re-enable, uncomment the
  // following line. Leaving the implementation in place so this can be
  // restored without further edits.
  // Call frame tracking for cold gas drift
  ColdGasFrameTrack(md, tm, dt);
}

void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  // evolve forcing

  if (drive_turbulence) {
    Generate(md, dt);

    // actually drive turbulence
    Perturb(md, dt);

    // Magic rescaling of simulation to target regime
    Rescale(md, tm, dt);


    // Magic injection of blobs into the simulation
    InjectBlob(md, tm, dt);
  }
}

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime & /*tm*/) {
  auto hydro_pkg = pmb->packages.Get("Hydro");

  // Store (common) acceleration field in spectral space
  auto few_modes_ft = hydro_pkg->Param<FewModesFT>("turbulence/few_modes_ft");
  auto var_hat = few_modes_ft.GetVarHat();
  auto accel_hat_host =
      Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), var_hat);

  const auto num_modes = few_modes_ft.GetNumModes();
  for (int i = 0; i < 3; i++) {
    for (int m = 0; m < num_modes; m++) {
      pin->SetReal("problem/turbulence",
                   "accel_hat_" + std::to_string(i) + "_" + std::to_string(m) + "_r",
                   accel_hat_host(i, m).real());
      pin->SetReal("problem/turbulence",
                   "accel_hat_" + std::to_string(i) + "_" + std::to_string(m) + "_i",
                   accel_hat_host(i, m).imag());
    }
  }
  // store state of random number gen
  auto state_rng = few_modes_ft.GetRNGState();
  pin->SetString("problem/turbulence", "state_rng", state_rng);
  // store state of distribution
  auto state_dist = few_modes_ft.GetDistState();
  pin->SetString("problem/turbulence", "state_dist", state_dist);
}

TaskStatus ProblemFillTracers(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real dt) {
  const auto current_cycle = tm.ncycle;

  auto tracers_pkg = md->GetParentPointer()->packages.Get("tracers");
  const auto n_lookback = tracers_pkg->Param<int>("turbulence/n_lookback");
  // Params (which is storing t_lookback) is shared across all blocks so we update it
  // outside the block loop. Note, that this is a standard vector, so it cannot be used
  // in the kernel (but also don't need to be used as can directly update it)
  auto t_lookback = tracers_pkg->Param<std::vector<Real>>("turbulence/t_lookback");
  auto dncycle = static_cast<int>(Kokkos::pow(2, n_lookback - 2));
  auto idx = n_lookback - 1;
  while (dncycle > 0) {
    if (current_cycle % dncycle == 0) {
      t_lookback[idx] = t_lookback[idx - 1];
    }
    dncycle /= 2;
    idx -= 1;
  }
  t_lookback[0] = tm.time;
  // Write data back to Params dict
  tracers_pkg->UpdateParam("turbulence/t_lookback", t_lookback);

  // TODO(pgrete) Benchmark atomic and potentially update to proper reduction instead of
  // atomics.
  //  Used for the parallel reduction. Could be reused but this way it's initalized to
  //  0.
  // n_lookback + 1 as it also carries <s> and <sdot>
  parthenon::ParArray2D<Real> corr("tracer correlations", 2, n_lookback + 1);
  int64_t num_particles_total = 0;

  for (int b = 0; b < md->NumBlocks(); b++) {
    auto *pmb = md->GetBlockData(b)->GetBlockPointer();
    auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
    auto &swarm = sd->Get("tracers");

    // TODO(pgrete) cleanup once get swarm packs (currently in development upstream)
    // pull swarm vars
    auto &rho = swarm->Get<Real>("rho").Get();
    auto &s = swarm->Get<Real>("s").Get();
    auto &sdot = swarm->Get<Real>("sdot").Get();

    auto swarm_d = swarm->GetDeviceContext();

    // update loop.
    const int max_active_index = swarm->GetMaxActiveIndex();
    pmb->par_for(
        "Turbulence::Fill Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            auto dncycle = static_cast<int>(Kokkos::pow(2, n_lookback - 2));
            auto s_idx = n_lookback - 1;
            while (dncycle > 0) {
              if (current_cycle % dncycle == 0) {
                s(s_idx, n) = s(s_idx - 1, n);
                sdot(s_idx, n) = sdot(s_idx - 1, n);
              }
              dncycle /= 2;
              s_idx -= 1;
            }
            s(0, n) = Kokkos::log(rho(n));
            sdot(0, n) = (s(0, n) - s(1, n)) / dt;

            // Now that all s and sdot entries are updated, we calculate the (mean)
            // correlations
            for (s_idx = 0; s_idx < n_lookback; s_idx++) {
              Kokkos::atomic_add(&corr(0, s_idx), s(0, n) * s(s_idx, n));
              Kokkos::atomic_add(&corr(1, s_idx), sdot(0, n) * sdot(s_idx, n));
            }
            Kokkos::atomic_add(&corr(0, n_lookback), s(0, n));
            Kokkos::atomic_add(&corr(1, n_lookback), sdot(0, n));
          }
        });
    num_particles_total += swarm->GetNumActive();
  } // loop over all blocks on this rank (this MeshData container)

  // Results still live in device memory. Copy to host for global reduction and output.
  auto corr_h = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), corr);
#ifdef MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, corr_h.data(), corr_h.GetSize(),
                                   MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &num_particles_total, 1, MPI_INT64_T,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(corr_h.data(), corr_h.data(), corr_h.GetSize(),
                                   MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(&num_particles_total, &num_particles_total, 1,
                                   MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif
  if (parthenon::Globals::my_rank == 0) {
    // Turn sum into mean
    for (int i = 0; i < n_lookback + 1; i++) {
      corr_h(0, i) /= static_cast<Real>(num_particles_total);
      corr_h(1, i) /= static_cast<Real>(num_particles_total);
    }

    // and write data
    std::ofstream outfile;
    const std::string fname("correlations.csv");
    // On startup, write header
    if (current_cycle == 0) {
      outfile.open(fname, std::ofstream::out);
      outfile << "# cycle, time, s, sdot";
      for (const auto &var : {"corr_s", "corr_sdot", "t_lookback"}) {
        for (int i = 0; i < n_lookback; i++) {
          outfile << ", " << var << "[" << i << "]";
        }
        outfile << std::endl;
      }
    } else {
      outfile.open(fname, std::ofstream::out | std::ofstream::app);
    }

    outfile << tm.ncycle << "," << tm.time;

    // <s> and <sdot>
    outfile << "," << corr_h(0, n_lookback);
    outfile << "," << corr_h(1, n_lookback);
    // <corr(s)> and <corr(sdot)>
    for (int j = 0; j < 2; j++) {
      for (int i = 0; i < n_lookback; i++) {
        outfile << "," << corr_h(j, i);
      }
    }
    for (int i = 0; i < n_lookback; i++) {
      outfile << "," << t_lookback[i];
    }
    outfile << std::endl;

    outfile.close();
  }

  return TaskStatus::complete;
}

//========================================================================================
//! \fn void ColdGasFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm, const
//! Real dt)
//  \brief Track and shift frame when cold gas drifts (using cumulative displacement)
//
//  When cold gas moves inward by one cell width in Y-direction, this function:
//  1. Removes the trailing row (outermost in Y)
//  2. Adds a new row at the inner Y boundary with density from rho_profile_Y
//  3. Maintains velocity continuity and recomputes energy
//========================================================================================

void ColdGasFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Check if frame tracking is enabled
  const auto enable_frame_track =
      hydro_pkg->Param<bool>("stratified_box/enable_cold_gas_frame_track");
  if (!enable_frame_track) return;

  // Get pointer to cumulative displacement
  Real* const p_frame_disp =
      hydro_pkg->MutableParam<Real>("stratified_box/frame_displacement_y");
  Real& frame_disp = *p_frame_disp;

  // Get density profile parameters
  const auto surface_density = hydro_pkg->Param<Real>("surface_density");
  const auto bc_a = hydro_pkg->Param<Real>("a_over_H");
  const auto bc_H = hydro_pkg->Param<Real>("H_height");
  const auto gamma = hydro_pkg->Param<Real>("gamma");
  const auto mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  const double rho0 = surface_density / 2.0 / bc_a / bc_H;
  const double a = bc_a;
  const double H = bc_H;
  const double gm1 = gamma - 1.0;

  // Pack all blocks' conserved variables for parallel reduction
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // Compute mass-weighted average v2 at inner boundary (jb.s)
  // Only include gas cells with temperature < 2e5 K
  Kokkos::Array<Real, 2> sums{{0.0, 0.0}};
  const Real T_cut = 2e5; // Kelvin

  Kokkos::parallel_reduce(
      "InnerBoundary::cold_gas_v2_mass_weighted",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
          {0, kb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, ib.e + 1}
      ),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &i,
                    Real &local_momentum_sum, Real &local_mass_sum) {
          auto &cons = cons_pack(b);
          const int j = jb.s; // Inner boundary in Y-direction
          
          const Real rho_cell = cons(IDN, k, j, i);
          if (rho_cell <= 0.0) return; // Skip invalid cells
          
          const Real T_cell = mean_molecular_mass_by_kb * cons(IPR, k, j, i) / rho_cell;
          if (T_cell < T_cut) {
              const Real v2_cell = cons(IM2, k, j, i) / rho_cell;
              local_momentum_sum += rho_cell * v2_cell; // Mass-weighted velocity
              local_mass_sum += rho_cell;
          }
      },
      Kokkos::Sum<Real>(sums[0]), // Sum of rho * v2
      Kokkos::Sum<Real>(sums[1])  // Sum of rho (total cold gas mass)
  );

#ifdef MPI_PARALLEL
  // Sum over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 2, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  Real v2_avg = 0.0;
  if (sums[1] > 0.0) {
      v2_avg = sums[0] / sums[1]; // Mass-weighted average velocity
  } else {
      v2_avg = 0.0; // No cold gas at boundary
  }

  // Update cumulative displacement (inward is negative)
  frame_disp += v2_avg * dt;

  // Get cell width in Y-direction from first block
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  Real dy = (pmb->pmy_mesh->mesh_size.xmax(X2DIR) - 
             pmb->pmy_mesh->mesh_size.xmin(X2DIR)) /
            pmb->pmy_mesh->GetDefaultBlockSize().nx(parthenon::X2DIR);

  // Check if cumulative displacement exceeds one cell width
  if (std::abs(frame_disp) >= dy) {
    // Determine shift direction
    int num_shifts = static_cast<int>(std::floor(std::abs(frame_disp) / dy));
    int shift_dir = (frame_disp < 0.0) ? -1 : 1; // -1 for inward, +1 for outward

    // Perform shifts on device
    for (int shift = 0; shift < num_shifts; shift++) {
      if (shift_dir == -1) {
        // Inward shift: row j ← row j+1 for j = [jb.s, jb.e-1]
        // Shift all data one row inward
        const int num_vars = cons_pack.GetDim(4);
        Kokkos::parallel_for(
            "InnerBoundary::shift_inward",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
                {0, kb.s, jb.s, ib.s},
                {cons_pack.GetDim(5), kb.e + 1, jb.e, ib.e + 1}
            ),
            KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
                auto &cons = cons_pack(b);
                for (int n = 0; n < num_vars; n++) {
                  cons(n, k, j, i) = cons(n, k, j + 1, i);
                }
            }
        );

        // Populate new row at jb.s with profile values
        Kokkos::parallel_for(
            "InnerBoundary::populate_inner",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0, kb.s, ib.s},
                {cons_pack.GetDim(5), kb.e + 1, ib.e + 1}
            ),
            KOKKOS_LAMBDA(const int &b, const int &k, const int &i) {
                auto &cons = cons_pack(b);
                const auto &coords = cons_pack.GetCoords(b);
                
                const Real Y = coords.Xc<2>(jb.s);
                const Real rhoY = rho_profile_Y(Y, rho0, a, H);

                // Set density
                cons(IDN, k, jb.s, i) = rhoY;

                // Copy tangential velocities from next interior cell
                cons(IM1, k, jb.s, i) = cons(IM1, k, jb.s + 1, i);
                cons(IM3, k, jb.s, i) = cons(IM3, k, jb.s + 1, i);

                // Set normal velocity from nearest interior cell
                cons(IM2, k, jb.s, i) = cons(IM2, k, jb.s + 1, i);

                // Compute pressure/energy: use temperature from nearest interior cell
                const Real T = cons(IPR, k, jb.s + 1, i) / cons(IDN, k, jb.s + 1, i);
                const Real ke = 0.5 *
                          (cons(IM1, k, jb.s, i) * cons(IM1, k, jb.s, i) +
                           cons(IM2, k, jb.s, i) * cons(IM2, k, jb.s, i) +
                           cons(IM3, k, jb.s, i) * cons(IM3, k, jb.s, i)) /
                          rhoY;
                const Real ie = T / gm1; // specific internal energy
                cons(IEN, k, jb.s, i) = rhoY * (ie + ke);

                // Pressure for storage
                cons(IPR, k, jb.s, i) = rhoY * T;
            }
        );
      } else {
        // Outward shift: row j ← row j-1 for j = [jb.e, jb.s+1]
        // Shift all data one row outward
        const int num_vars = cons_pack.GetDim(4);
        Kokkos::parallel_for(
            "InnerBoundary::shift_outward",
            Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
                {0, kb.s, jb.s + 1, ib.s},
                {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1}
            ),
            KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
                auto &cons = cons_pack(b);
                const int j_target = jb.e - (j - jb.s - 1); // Reverse iteration
                const int j_source = j_target - 1;
                for (int n = 0; n < num_vars; n++) {
                  cons(n, k, j_target, i) = cons(n, k, j_source, i);
                }
            }
        );

        // Populate new row at jb.e with profile values
        Kokkos::parallel_for(
            "InnerBoundary::populate_outer",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0, kb.s, ib.s},
                {cons_pack.GetDim(5), kb.e + 1, ib.e + 1}
            ),
            KOKKOS_LAMBDA(const int &b, const int &k, const int &i) {
                auto &cons = cons_pack(b);
                const auto &coords = cons_pack.GetCoords(b);
                
                const Real Y = coords.Xc<2>(jb.e);
                const Real rhoY = rho_profile_Y(Y, rho0, a, H);

                // Set density
                cons(IDN, k, jb.e, i) = rhoY;

                // Copy tangential velocities from nearest interior cell
                cons(IM1, k, jb.e, i) = cons(IM1, k, jb.e - 1, i);
                cons(IM3, k, jb.e, i) = cons(IM3, k, jb.e - 1, i);

                // Set normal velocity from nearest interior cell
                cons(IM2, k, jb.e, i) = cons(IM2, k, jb.e - 1, i);

                // Compute energy
                const Real T = cons(IPR, k, jb.e - 1, i) / cons(IDN, k, jb.e - 1, i);
                const Real ke = 0.5 *
                          (cons(IM1, k, jb.e, i) * cons(IM1, k, jb.e, i) +
                           cons(IM2, k, jb.e, i) * cons(IM2, k, jb.e, i) +
                           cons(IM3, k, jb.e, i) * cons(IM3, k, jb.e, i)) /
                          rhoY;
                const Real ie = T / gm1;
                cons(IEN, k, jb.e, i) = rhoY * (ie + ke);

                // Pressure for storage
                cons(IPR, k, jb.e, i) = rhoY * T;
            }
        );
      }
      
      // Ensure shifts complete before next iteration
      Kokkos::fence();
    }

    // Reset displacement counter
    frame_disp -= shift_dir * num_shifts * dy;
  }
}

} // namespace stratified_box
