
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sn.cpp
//  \brief Problem generator for supernova problem.  Reads initial conditions from a 
//         .bp file using ADIOS2.  Adapted from blast.cpp with input reading from wtopenrun.cpp
//

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <adios2.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// Parthenon headers
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// AthenaPK headers
#include "../main.hpp"
#include "parthenon/prelude.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
#include "../units.hpp"
#include "globals.hpp"

using namespace parthenon::package::prelude;

namespace sn {

//========================================================================================
//! \fn void InitUserMeshData(Mesh *mesh, ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.
//========================================================================================

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // Placeholder for any mesh-level initialization needed
  // Can be extended in the future
}

//========================================================================================
//! \fn void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md)
//  \brief Supernova problem generator that reads initial conditions from a .bp file
//========================================================================================

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {

  Units units(pin);

  // Get the input filename from the job section
  const std::string ics_filename = pin->GetString("job", "bin_input_file");
  std::string varname = ics_filename;
  size_t pos = varname.find(".bp");

  // Initialize ADIOS2 reader
  adios2::ADIOS adios(MPI_COMM_WORLD);

  adios2::IO get_var = adios.DeclareIO("GetVar");
  adios2::Engine bpReader = get_var.Open(ics_filename, adios2::Mode::Read);
  bpReader.BeginStep();
  adios2::Variable<double> myvar_in = get_var.InquireVariable<double>(varname.erase(pos));
  PARTHENON_REQUIRE_THROWS(myvar_in, "Could not find variable name in file.");

  // Get unit conversion factors
  auto d_cgs_factor = 1. / units.code_density_cgs();
  auto m_cgs_factor = 1. / ( units.code_density_cgs() * units.code_length_cgs() / units.code_time_cgs());
  auto e_cgs_factor = 1. / ( units.code_density_cgs() * pow(units.code_length_cgs(),2) / pow(units.code_time_cgs(),2));

  const auto nx = pmesh->GetDefaultBlockSize().nx(parthenon::X1DIR);
  const auto ny = pmesh->GetDefaultBlockSize().nx(parthenon::X2DIR);
  const auto nz = pmesh->GetDefaultBlockSize().nx(parthenon::X3DIR);

  const int fields = 3;  // density, momentum, energy

  std::vector<double> ICsdata(fields * nx * ny * nz);

  // Get meshblock for GPU and initialize with data from .bp file
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
      printf("Value of loc is not valid... \n");
      continue;
    }

    // Set up ADIOS2 read parameters for this block
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

    // Initialize conserved variables from the .bp file data
    auto &mbd = pmb->meshblock_data.Get();
    auto &u_dev = mbd->Get("cons").data;
    auto &coords = pmb->coords;
    // Initialize on host
    auto u = u_dev.GetHostMirrorAndCopy();

    IndexRange ib = mbd->GetBoundsI(IndexDomain::interior);
    IndexRange jb = mbd->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = mbd->GetBoundsK(IndexDomain::interior);

    // Read problem parameters and fill conserved variables
    for (int k = 0; k < nz; k++) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {

          int index_base_0 = ((0 * nz + k) * ny + j) * nx + i;  // density
          int index_base_1 = ((1 * nz + k) * ny + j) * nx + i;  // momentum
          int index_base_2 = ((2 * nz + k) * ny + j) * nx + i;  // energy

          PARTHENON_REQUIRE_THROWS(ICsdata[index_base_0] > 0., "Densities below 0");

          u(IDN, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_0] * d_cgs_factor;
          u(IM2, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_1] * m_cgs_factor;
          u(IEN, kb.s + k, jb.s + j, ib.s + i) = ICsdata[index_base_2] * e_cgs_factor;

          // Initialize velocities in other directions to zero
          u(IM1, kb.s + k, jb.s + j, ib.s + i) = 0.0;
          u(IM3, kb.s + k, jb.s + j, ib.s + i) = 0.0;

          if (mhd_enabled) {
            // Initialize magnetic field to zero (can be modified if needed)
            u(IB1, kb.s + k, jb.s + j, ib.s + i) = 0.0;
            u(IB2, kb.s + k, jb.s + j, ib.s + i) = 0.0;
            u(IB3, kb.s + k, jb.s + j, ib.s + i) = 0.0;
          }
        }
      }
    }

    // Copy initialized variables to device
    u_dev.DeepCopy(u);
  }

  bpReader.EndStep();
  bpReader.Close();
}

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  // Placeholder for any cleanup needed after simulation
}

} // namespace sn
