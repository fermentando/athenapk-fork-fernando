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

namespace stratified_box {
using namespace parthenon;

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
  const auto G = units.gravitational_constant();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravitationalFieldSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        auto y_norm = coords.Xc<2>(j) / (a_over_H * H);
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
  auto gm1 = (gamma - 1.0);
  const auto &pkg = mesh->packages.Get("Hydro");
  const auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");

  auto a_over_H= pin->GetReal("problem/stratified_box", "a_over_H");
  auto surface_density = pin->GetReal("problem/stratified_box", "surface_density");
  auto T_base = pin->GetReal("problem/stratified_box", "T_base");

  pkg->AddParam<Real>("a_over_H", a_over_H );
  pkg->AddParam<Real>("surface_density", surface_density);

  const auto g0 = 2 * M_PI * units.gravitational_constant() * surface_density;
  auto c_s = std::sqrt(T_base / mbar_over_kb);
  auto H_height = c_s*c_s/ g0;

  pkg->AddParam<Real>("H_height", H_height);


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

  
  const int fields = 3;


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
    auto &coords = pmb->coords;
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

          PARTHENON_REQUIRE_THROWS(ICsdata[index_base_0] > 0., "Densities below 0");

          u(IDN, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_0] * d_cgs_factor;
          u(IM2, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_1] * m_cgs_factor;
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


}


void StratUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                           const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  GravitationalFieldSrcTerm(md, beta_dt);
}
}

