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
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()
#include <fstream>   // bin file
#include <adios2.h>

//I/O for ICs reader
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h> 
#include "globals.hpp"

// Parthenon headers
#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include <iomanip>
#include <ios>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>
#include <sstream>
#include <iostream>
#include <string>
#include <globals.hpp>


// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "../eos/adiabatic_glmmhd.hpp"
#include "../eos/adiabatic_hydro.hpp"
#include "../tracers/tracers.hpp"
#include "../utils/few_modes_ft.hpp"
#include "utils/error_checking.hpp"



namespace stratified_box_simple {
using namespace parthenon;
using namespace parthenon::package::prelude;
using parthenon::DevMemSpace;
using parthenon::ParArray2D;
using utils::few_modes_ft::Complex;
using utils::few_modes_ft::FewModesFT;

Real c_s, gm1;

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

        auto y_norm = coords.Xc<2>(j) / (a_over_H * H) * code_units_length;
        const Real g_z =  2 * M_PI * G * surface_density * y_norm / std::sqrt(1 + y_norm * y_norm);

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
  gm1 = (gamma - 1.0);
  const auto &pkg = mesh->packages.Get("Hydro");
  const auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");

  auto a_over_H= pin->GetReal("problem/stratified_box", "a_over_H");
  auto surface_density = pin->GetReal("problem/stratified_box", "surface_density");
  auto T_base = pin->GetReal("problem/stratified_box", "T_base");
  auto T_cloud = pin->GetReal("problem/stratified_box", "T_cloud");

  pkg->AddParam<Real>("a_over_H", a_over_H );
  pkg->AddParam<Real>("surface_density", surface_density);
  pkg->AddParam<Real>("T_cloud", T_cloud);

  const auto G = units.gravitational_constant();
  const auto g0 = 2.0 * M_PI * G * surface_density; 

  c_s = std::sqrt(T_base / mbar_over_kb); 
  const auto H_height = c_s * c_s / g0;              

  pkg->AddParam<Real>("H_height", H_height);

  const auto t_ff = std::sqrt(2.0 * H_height / g0); 
  

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


void ProblemGenerator(Mesh *pmesh, ParameterInput *pin,  MeshData<Real> *md) {

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

  auto d_cgs_factor = 1. / units.code_density_cgs();
  auto m_cgs_factor = 1. / ( units.code_density_cgs() * units.code_length_cgs() / units.code_time_cgs());
  auto e_cgs_factor = 1. / ( units.code_density_cgs() * pow(units.code_length_cgs(),2) / pow(units.code_time_cgs(),2));


  const auto nx = pmesh->GetDefaultBlockSize().nx(parthenon::X1DIR);
  const auto ny = pmesh->GetDefaultBlockSize().nx(parthenon::X2DIR);
  const auto nz = pmesh->GetDefaultBlockSize().nx(parthenon::X3DIR);

  
  const int fields = 5;


  std::vector<double> ICsdata(fields * nx * ny * nz);


  //Get meshblock for GPU
  for (int b = 0; b < md->NumBlocks() ; b++) {

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


    if (( loc.lx1() < 0) || ( loc.lx2() < 0) || ( loc.lx3() < 0)) {
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
    std::cerr << "Reading first item of meshblock data: " << ICsdata[0] << " from rank" << Globals::my_rank << std::endl;
    


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
          u(IM2, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_1] * m_cgs_factor;
          u(IM3, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_1] * m_cgs_factor;
          u(IEN, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_2] * e_cgs_factor;


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
    return rho0 * exp(-a * (sqrt(1.0 + arg*arg) - 1.0));
}

void StratOutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  const auto nb = IndexRange{0,0};
  const bool fine = false;

  // Local copies of parameters for device lambda
  auto surface_density = pmb->packages.Get("Hydro")->Param<Real>("surface_density");
  auto bc_a = pmb->packages.Get("Hydro")->Param<Real>("a_over_H");
  auto bc_H = pmb->packages.Get("Hydro")->Param<Real>("H_height");
  const double rho0 = surface_density / 2/bc_H;  // midplane density
  const double a    = bc_a;
  const double H    = bc_H;

  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);


  pmb->par_for_bndry(
      "StratOutflowInnerX2", nb, IndexDomain::inner_x2,
      parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          const auto &coordsb = cons_pack.GetCoords();
          auto &cons = cons_pack;
          Real Y = coordsb.Xc<2>(j);
          double rhoY = rho_profile_Y(Y, rho0, a, H);

          // Copy tangential velocities from last interior cell
          cons(IDN,k,j,i) = rhoY;
          cons(IV1,k,j,i) = cons(IV1,k,jb.s,i);
          cons(IV3,k,j,i) = cons(IV3,k,jb.s,i);

          // Normal velocity: zero if inflow
          //if (cons(IV2,k,jb.s,i) >= 0.0) cons(IV2,k,j,i) = 0.0;
          cons(IV2,k,j,i) = cons(IV2,k,jb.s,i);

          Real T = cons(IPR,k,jb.s,i) / cons(IDN,k,jb.s,i);
          cons(IPR,k,j,i) = rhoY * T;
      });
}

void StratOutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  const auto nb = IndexRange{0,0};
  const bool fine = false;

  auto surface_density = pmb->packages.Get("Hydro")->Param<Real>("surface_density");
  auto bc_a = pmb->packages.Get("Hydro")->Param<Real>("a_over_H");
  auto bc_H = pmb->packages.Get("Hydro")->Param<Real>("H_height");
  const double rho0 = surface_density / 2/ bc_H;  // midplane density
  const double a    = bc_a;
  const double H    = bc_H;
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);


  pmb->par_for_bndry(
      "StratOutflowInnerX2", nb, IndexDomain::outer_x2,
      parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          const auto &coordsb = cons_pack.GetCoords();
          auto &cons = cons_pack;
          Real Y = coordsb.Xc<2>(j);
          double rhoY = rho_profile_Y(Y, rho0, a, H);

          // Copy tangential velocities from last interior cell
          cons(IDN,k,j,i) = rhoY;
          cons(IV1,k,j,i) = cons(IV1,k,jb.s,i);
          cons(IV3,k,j,i) = cons(IV3,k,jb.s,i);

          // Normal velocity: zero if inflow
          //if (cons(IV2,k,jb.s,i) <= 0.0) cons(IV2,k,j,i) = 0.0;
          cons(IV2,k,j,i) = cons(IV2,k,jb.s,i);

          Real T = cons(IPR,k,jb.s,i) / cons(IDN,k,jb.s,i);
          cons(IPR,k,j,i) = rhoY * T;
      });
}



//----------------------------------------------------------------------------------------
//! \fn void StratHst(MeshData<Real> *md)
//  \brief Hst file initialiser for new variables

// TODO(?) until we are able to process multiple variables in a single hst function call
// we'll use this enum to identify the various vars.
enum class HstQuan {mc, mbw, Mcx1, Mcx2, Mcx3, mcout, mwout,  Ms, Ma, pb };

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

  if (hst_quan == HstQuan::mcout || hst_quan == HstQuan::mwout)
  {
    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::outer_x2);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::outer_x2);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::outer_x2);

    auto pmesh = pmb->pmy_mesh;
    const auto x2max = pmesh->mesh_size.xmax(X2DIR);
    const auto H_height = hydro_pkg->Param<Real>("H_height");



    pmb->par_reduce(
      "WTopenrun::outflowing_gas", 0, prims_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &prims = prims_pack(b);
        const auto &cons = cons_pack(b);
        const auto &coords = prims_pack.GetCoords(b);
        const Real rho = prims(IDN, k, j, i);    
        const Real My = cons(IM2, k, j, i);      
        const Real temp = mean_molecular_mass_by_kb * prims(IPR, k, j, i) / rho; 

        if (coords.Xc<2>(j) > x2max && My > 0.0) {
          if (hst_quan == HstQuan::mcout && temp <= 5 * T_cloud_){
            const Real mass = rho * coords.CellVolume(k, j, i); 
            lsum += mass; 
          }
          if (hst_quan == HstQuan::mwout && temp > 5 * T_cloud_ && temp <= 10 * T_cloud_){
            const Real mass = rho * coords.CellVolume(k, j, i); 
            lsum += mass; 
          }
        }
      },
    sum);
  }

  else{

  pmb->par_reduce(
    "WTOpenRun::hst_calc", 0, prims_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
    KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &prim = prims_pack(b);
        const auto &coords = prims_pack.GetCoords(b);
        const Real temp = mean_molecular_mass_by_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i);


        const auto vel2 = (prim(IV1, k, j, i) * prim(IV1, k, j, i) +
                           prim(IV2, k, j, i) * prim(IV2, k, j, i) +
                           prim(IV3, k, j, i) * prim(IV3, k, j, i));

        const auto c_s =
            std::sqrt(gamma * prim(IPR, k, j, i) / prim(IDN, k, j, i)); // speed of sound

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


        if (temp <= 2*T_cloud_) { 

          if (hst_quan == HstQuan::mc) {
            lsum += prim(IDN, k, j, i) * coords.CellVolume(k, j, i);
          }
          if (hst_quan == HstQuan::Mcx1) {
            lsum += cons(IM1, k, j, i) *  coords.CellVolume(k, j, i);
          }
          if (hst_quan == HstQuan::Mcx2) {
            lsum +=  cons(IM2, k, j, i) * coords.CellVolume(k, j, i);
          }
          if (hst_quan == HstQuan::Mcx3) {
            lsum +=  cons(IM3, k, j, i) * coords.CellVolume(k, j, i);
          }
          
        }
        if (temp <= 10 * T_cloud_){
          if (hst_quan == HstQuan::mbw){
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


}

}

