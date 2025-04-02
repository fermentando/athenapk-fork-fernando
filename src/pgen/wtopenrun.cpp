//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wtopenrun.cpp
//! \brief Open problem generator for wind tunnel simulations.
//!

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()
#include <fstream>   // bin file

//I/O for ICs reader
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h> 

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
#include <string>
#include <globals.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "../eos/adiabatic_glmmhd.hpp"
#include "../eos/adiabatic_hydro.hpp"



namespace wtopenrun {
using namespace parthenon;
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;


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

  r_cloud = pin->GetReal("problem/wtopenrun", "r0_cgs") / units.code_length_cgs();
  rho_cloud = pin->GetReal("problem/wtopenrun", "rho_cloud_cgs") / units.code_density_cgs();
  rho_wind = pin->GetReal("problem/wtopenrun", "rho_wind_cgs") / units.code_density_cgs();
  auto T_wind = pin->GetReal("problem/wtopenrun", "T_wind_cgs");
  auto Mach_wind = pin->GetReal("problem/wtopenrun", "Mach_wind");
  auto bool_boost = pin->GetOrAddBoolean("parthenon/mesh", "tracking", false);
  auto wfrac = pin->GetOrAddReal("parthenon/mesh", "wfrac", 1.);
  auto depth = pin->GetOrAddReal("parthenon/wtopenrun", "depth", 1.);


  // mu_mh_gm1_by_k_B is already in code units
  rhoe_wind = T_wind * rho_wind / mbar_over_kb / gm1;
  const auto c_s_wind = std::sqrt(gamma * gm1 * rhoe_wind / rho_wind);
  const auto chi_0 = rho_cloud / rho_wind;               // cloud to wind density ratio
  const auto v_wind = c_s_wind * Mach_wind;
  const auto t_cc = r_cloud * std::sqrt(chi_0) / v_wind * depth; // cloud crushting time (code)
  const auto pressure =
      gm1 * rhoe_wind; // one value for entire domain given initial pressure equil.

  const auto T_cloud = pressure / rho_cloud * mbar_over_kb;

  auto plasma_beta = pin->GetOrAddReal("problem/wtopenrun", "plasma_beta", -1.0);

  auto mag_field_angle_str =
      pin->GetOrAddString("problem/wtopenrun", "mag_field_angle", "undefined");
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
      PARTHENON_FAIL("Unsupported problem/wtopenrun/mag_field_angle. Please use either "
                     "'aligned', 'transverse', or 'oblique'.");
    }
  }


  //Set frame speed as mutable
  pkg->AddParam<Real>("dv_v", 0., true);
  pkg->AddParam<Real>("v_boost", 0., true);
  pkg->AddParam<Real>("Tcloud", T_cloud);
  pkg->AddParam<Real>("wfrac", wfrac);
  pkg->AddParam<bool>("tracking", bool_boost);

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
  msg << "## Tracking on: " << std::boolalpha << bool_boost << std::endl;

  // (potentially) rescale global times only at the beginning of a simulation
  auto rescale_code_time_to_tcc =
      pin->GetOrAddBoolean("problem/wtopenrun", "rescale_code_time_to_tcc", false);

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
    pin->SetBoolean("problem/wtopenrun", "rescale_code_time_to_tcc", false);
  }
  if (parthenon::Globals::my_rank == 0) {
    msg << "######################################" << std::endl;

    std::cout << msg.str();
  }

  // Check if frame boosting is on
  //BoostBool = pin->GetOrAddBoolean("problem/wtopenrun", "frame_boost", false);
}




std::pair<Real, Real> cold_gas_extent_y(MeshData<Real> *md) {

  using parthenon::Real;

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto pmesh = pmb->pmy_mesh;

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  Real T_cloud = hydro_pkg->Param<Real>("Tcloud");
  Real mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("mbar_over_kb");

  Real cgymin = std::numeric_limits<Real>::max();  // Local min for reduction
  Real cgymax = std::numeric_limits<Real>::lowest();  // Local max for reduction

  Kokkos::Min<Real> reducer_min(cgymin);
  Kokkos::Max<Real> reducer_max(cgymax);

  Kokkos::parallel_reduce(
      "WTOpenRun::cold_gas_extent_y",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e+1, jb.e+1, ib.e+1}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, 
      Real &thread_min, Real &thread_max) {

          const auto &cons = cons_pack(b);
          const auto &coords = cons_pack.GetCoords(b);

          const Real temp = mean_molecular_mass_by_kb * cons(IPR, k, j, i) / cons(IDN, k, j, i);


          if (temp <= 2 * T_cloud) {
              Real y = coords.Xc<2>(j);  
              
              thread_min = fmin(thread_min, y);
              thread_max = fmax(thread_max, y);
          }
      },
      reducer_min, reducer_max);

  // After reduction, if no cold gas was found (min is still max value), return ymax and 0.0
  if (cgymin == std::numeric_limits<Real>::max()) {
      // No cold gas found
      return {pmesh->mesh_size.xmax(X2DIR), 0.0};
  }

  return {cgymin, cgymax - cgymin};
}

// Function to check available memory (only works for CUDA/HIP devices)
bool checkGpuMemory(size_t required_bytes) {
#if defined(KOKKOS_ENABLE_CUDA)
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed!" << std::endl;
        return false;
    }
    std::cout << "GPU Memory: " << free_mem / (1024.0 * 1024) << " MiB free, "
              << total_mem / (1024.0 * 1024) << " MiB total." << std::endl;
    return free_mem >= required_bytes;
#elif defined(KOKKOS_ENABLE_HIP)
    size_t free_mem, total_mem;
    if (hipMemGetInfo(&free_mem, &total_mem) != hipSuccess) {
        std::cerr << "hipMemGetInfo failed!" << std::endl;
        return false;
    }
    std::cout << "HIP GPU Memory: " << free_mem / (1024.0 * 1024) << " MiB free, "
              << total_mem / (1024.0 * 1024) << " MiB total." << std::endl;
    return free_mem >= required_bytes;
#else
    return true; // Assume CPU has enough memory
#endif
}



//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the cloud in wind setup

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin,  MeshData<Real> *md) {

  Units units(pin);

  const std::string ics_filename = pin->GetString("job", "bin_input_file");

  auto d_cgs_factor = 1. / units.code_density_cgs();
  auto m_cgs_factor = 1. / ( units.code_density_cgs() * units.code_length_cgs() / units.code_time_cgs());
  auto e_cgs_factor = 1. / ( units.code_density_cgs() * pow(units.code_length_cgs(),2) / pow(units.code_time_cgs(),2));

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");
  const auto num_blocks = md->NumBlocks();
 
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto Ncellx1 = pmesh->mesh_size.nx(X1DIR);
  const auto Ncellx2 = pmesh->mesh_size.nx(X2DIR);
  const auto Ncellx3 = pmesh->mesh_size.nx(X3DIR);

  printf("Dimensions of interior domain: %d, %d, %d. \n", Ncellx1, Ncellx2, Ncellx3);


  const auto lsizex1 = (pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR))/Ncellx1;
  const auto lsizex2 = (pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR))/Ncellx2;
  const auto lsizex3 = (pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR))/Ncellx3;

  const auto x1min = pmesh->mesh_size.xmin(X1DIR);
  const auto x2min = pmesh->mesh_size.xmin(X2DIR);
  const auto x3min = pmesh->mesh_size.xmin(X3DIR);


  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto const &cons = md->PackVariables(std::vector<std::string>{"cons"});


  
  // Quantities to initialize
  int Nq = 4;
  size_t size = Ncellx1 * Ncellx2 * Ncellx3 * Nq;
  size_t total_bytes = size * sizeof(double);

  // Allocate host memory in chunks
  using HostMemSpace = Kokkos::HostSpace;
  using HostPinnedArr = Kokkos::View<double*, HostMemSpace>;

  size_t chunk_size = 64 * 1024;  // Max chunk size (adjust as needed)
  size_t num_chunks = (size + chunk_size - 1) / chunk_size;

  // Create a vector of chunked Views
  std::vector<HostPinnedArr> hICs_chunks(num_chunks);
  for (size_t i = 0; i < num_chunks; i++) {
      size_t this_chunk_size = std::min(chunk_size, size - i * chunk_size);
      hICs_chunks[i] = HostPinnedArr("hICs_chunk", this_chunk_size);
  }

  // Open and memory-map the file
  int fd = open(ics_filename.c_str(), O_RDONLY);
  if (fd == -1) {
      std::cerr << "Failed to open ICs file." << std::endl;
      return;
  }

  // Get file size
  struct stat file_stat;
  if (fstat(fd, &file_stat) == -1) {
      close(fd);
      std::cerr << "Failed to get file size." << std::endl;
      return;
  }
  if (file_stat.st_size < total_bytes) {
      close(fd);
      std::cerr << "ICs file is smaller than expected data size." << std::endl;
      return;
  }

  // Memory-map the file in chunks
  size_t bytes_read = 0;
  size_t doubles_read = 0;

  while (bytes_read < total_bytes) {
      size_t bytes_to_read = std::min(chunk_size * sizeof(double), total_bytes - bytes_read);

      void* file_data = mmap(nullptr, bytes_to_read, PROT_READ, MAP_PRIVATE, fd, bytes_read);
      if (file_data == MAP_FAILED) {
          close(fd);
          std::cerr << "Memory mapping failed." << std::endl;
          return;
      }

      madvise(file_data, bytes_to_read, MADV_SEQUENTIAL);

      double* src = reinterpret_cast<double*>(file_data);

      size_t doubles_to_read = bytes_to_read / sizeof(double);
      std::memcpy(hICs_chunks[doubles_read / chunk_size].data(), src, bytes_to_read);

      doubles_read += doubles_to_read;
      bytes_read += bytes_to_read;

      munmap(file_data, bytes_to_read);  // Free memory after reading each chunk
  }

  close(fd);

  // Allocate execution-space memory (on the device)
  using DeviceArr = Kokkos::View<double*, Kokkos::DefaultExecutionSpace>;
  DeviceArr ICsdata("ICsdata", size);
  Kokkos::fence();

  // Copy chunked data to device memory
  size_t offset = 0;
  for (size_t i = 0; i < num_chunks; i++) {
      size_t chunk_extent = hICs_chunks[i].extent(0);
      Kokkos::deep_copy(Kokkos::subview(ICsdata, std::make_pair(offset, offset + chunk_extent)), hICs_chunks[i]);
      offset += chunk_extent;
  }

  std::cout << "Initialized ICs data of size: " << ICsdata.extent(0) << " elements." << std::endl;

  // Parallel assignment of initial conditions
  Kokkos::parallel_for(
      "WtOpenRun::ProblemGenerator",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {num_blocks, kb.e + 1, jb.e + 1, ib.e + 1}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {

          const auto &u = cons(b); 
          const auto &coords = cons.GetCoords(b);

          const int global_x = (coords.Xc<1>(i) - lsizex1 / 2 - x1min) / lsizex1;
          const int global_y = (coords.Xc<2>(j) - lsizex2 / 2 - x2min) / lsizex2;
          const int global_z = (coords.Xc<3>(k) - lsizex3 / 2 - x3min) / lsizex3;

          if (global_x < 0 || global_x >= Ncellx1 ||
            global_y < 0 || global_y >= Ncellx2 ||
            global_z < 0 || global_z >= Ncellx3) {
            printf("Invalid global indices: x=%d, y=%d, z=%d\n", global_x, global_y, global_z);
            return;
          }

          int index_base = ((global_z * Ncellx2 + global_y) * Ncellx1 + global_x) * Nq;
          if (index_base < 0 || index_base >= size) {
                printf("Out of bounds: index_base=%d, size=%zu\n", index_base, size);
                return;
            }
          u(IDN, k, j, i) = ICsdata(index_base) * d_cgs_factor;
          u(IM2, k, j, i) = ICsdata(index_base + 1) * m_cgs_factor;
          u(IEN, k, j, i) = ICsdata(index_base + 2) * e_cgs_factor + ICsdata(index_base + 3) / mbar_over_kb * d_cgs_factor;

      });
  Kokkos::fence();
  std::cout << "Initial conditions finalized." << std::endl;




    //auto cg_width = cold_gas_extent_y(md);
    //parthenon::Real cgmin = cg_width.first;
    //parthenon::Real cg_extent = cg_width.second;
    //parthenon::Real wfrac = hydro_pkg->Param<Real>("wfrac");


    //hydro_pkg->AddParam("y0boost", wfrac * cg_extent + cgmin);

    //std::cout << "Benchmark for frame boost calculation at "<< (wfrac * cg_extent + abs(cgmin)) / (lsizex2 * Ncellx2) << "L_y,box. \n" << std::endl;
  
}

void InflowWindX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto nb = IndexRange{0, 0};
  const auto rho_wind_ = rho_wind;
  const auto mom_wind_init = mom_wind;
  const auto rhoe_wind_ = rhoe_wind;
  const auto Bx_ = Bx;
  const auto By_ = By;
  const auto Bz_ = Bz;
  const bool fine = false;
  auto v_boost = hydro_pkg->Param<Real>("v_boost");

  auto mom_wind_ = mom_wind_init - rho_wind_ * v_boost;

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



//========================================================================================
//! \fn void ApplyFrameBoost(parthenon::MeshData<parthenon::Real> *md)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

// Compute frame_boosting velocity
Real ComputeCloudMassWeightedVel(parthenon::MeshData<parthenon::Real> *md) {

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto pmesh = pmb->pmy_mesh;

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  //Skip calculation if tracking is off
  auto bool_boost = hydro_pkg->Param<bool>("tracking");
  if (!bool_boost) return 0.;

  const auto units = hydro_pkg->Param<Units>("units");
  const auto x2min = pmesh->mesh_size.xmin(X2DIR);
  const auto lsizex2 = (pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR))/ pmesh->mesh_size.nx(X2DIR);


  Real mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  Real T_cloud = hydro_pkg->Param<Real>("Tcloud");
  auto v_boost = hydro_pkg->Param<Real>("v_boost");
  int alert_stop_boost = 0;
  Real frame_dv;

  //const auto x2centre = hydro_pkg->Param<Real>("y0boost");
  //const auto x2centre = (pmesh->mesh_size.xmax(X2DIR) + pmesh->mesh_size.xmin(X2DIR))/2;


  Kokkos::Array<Real, 2> sums{{0.0, 0.0}};

  Kokkos::parallel_reduce(
      "WTOpenRun::frame_boosting_velocity", 
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, 
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1} 
      ),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, 
      Real& local_IM_cold_gas, Real& local_cold_gas, int& cold_gas_found) { 
          auto &cons = cons_pack(b);
          const auto &coords = cons_pack.GetCoords(b);

          const Real temp =
              mean_molecular_mass_by_kb * cons(IPR, k, j, i) / cons(IDN, k, j, i);


          if (temp <= 5 * T_cloud) {
              const int global_y = (coords.Xc<2>(j) - lsizex2 / 2 - x2min) / lsizex2;
              local_IM_cold_gas += cons(IM2, k, j, i);
              local_cold_gas += cons(IDN, k, j, i);
              
              // Check if it's in the first 3 x cells 
              if (global_y <= 3) {
                  cold_gas_found = 1;  // Mark that cold gas was found in the first 3 x cells
              }
          }
      },
      Kokkos::Sum<Real>(sums[0]), Kokkos::Sum<Real>(sums[1]),
      Kokkos::Sum<int>(alert_stop_boost)  // Reduce the cold gas found flag
  );
#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 2, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &alert_stop_boost, 1, MPI_INT,
                                  MPI_MAX, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  if (sums[1] > 0. && sums[0] > 0. && alert_stop_boost == 0) {
    frame_dv = sums[0]/sums[1];
  } else {
    frame_dv = 0.;
  }
  v_boost += frame_dv;
  hydro_pkg->UpdateParam("dv_v", frame_dv); 
  hydro_pkg->UpdateParam("v_boost", v_boost);

  return v_boost;

}



// Shift velocities to maintain intertial frame
void ApplyFrameBoost(parthenon::MeshData<parthenon::Real> *md) {

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);


  Real frame_v = hydro_pkg->Param<Real>("dv_v");
  
  if (fabs(frame_v) > 100.01 || frame_v < 0.0) frame_v = 0.;

 
  Kokkos::parallel_for(
    "WTOpenRun::frame_boosting_velocity", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e+1, jb.e+1, ib.e+1}),  
    KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {

        auto &cons = cons_pack(b);

          
            cons(IEN, k, j, i) -= frame_v * cons(IM2, k, j, i);
            cons(IEN, k, j, i) += 0.5 * SQR(frame_v) * cons(IDN, k, j, i);
            cons(IM2, k, j, i) -= frame_v * cons(IDN, k, j, i);
    
         
      });

  
}

void FrameBoosting(parthenon::MeshData<parthenon::Real> *md, const parthenon::SimTime &tm,
                         const Real dt){
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  bool bool_boost = hydro_pkg->Param<bool>("tracking");

  if (bool_boost){
    Real boost = ComputeCloudMassWeightedVel(md);
    ApplyFrameBoost(md);
  }

}

//----------------------------------------------------------------------------------------
//! \fn void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg)
//  \brief Hst file initialiser for new variables

// TODO(?) until we are able to process multiple variables in a single hst function call
// we'll use this enum to identify the various vars.
enum class HstQuan {mc, Mcx1, Mcx2, Mcx3, vboost, mcout, mwout};

// Compute the local sum of cloud mass
template <HstQuan hst_quan>
Real WindTunnelHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  Real T_cloud = hydro_pkg->Param<Real>("Tcloud");
  Real mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("mbar_over_kb");

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
          if (hst_quan == HstQuan::mcout && temp <= 5 * T_cloud){
            const Real mass = rho * coords.CellVolume(k, j, i); 
            lsum += mass; 
          }
          if (hst_quan == HstQuan::mwout && temp > 5 * T_cloud && temp <= 10 * T_cloud){
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
        const auto &prims = prims_pack(b);
        const auto &coords = prims_pack.GetCoords(b);
        const Real temp = mean_molecular_mass_by_kb * prims(IPR, k, j, i) / prims(IDN, k, j, i);

        if (temp <= 2*T_cloud) { 

          if (hst_quan == HstQuan::mc) {
            lsum += prims(IDN, k, j, i) * coords.CellVolume(k, j, i);
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
      },
      sum);
  }

  return sum;
}


void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {

  
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);

  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    WindTunnelHst<HstQuan::mc>, "mc"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    WindTunnelHst<HstQuan::Mcx1>, "Mcx1"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    WindTunnelHst<HstQuan::Mcx2>, "Mcx2"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    WindTunnelHst<HstQuan::Mcx3>, "Mcx3"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::max,
                                                    ComputeCloudMassWeightedVel, "vboost"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    WindTunnelHst<HstQuan::mcout>, "mcout"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    WindTunnelHst<HstQuan::mwout>, "mwout"));
  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

}



} // namespace wtopenrun
