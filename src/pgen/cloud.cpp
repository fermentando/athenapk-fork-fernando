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
#include <fstream>   // bin file

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

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin,  MeshData<Real> *md) {

  Units units(pin);
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


  const auto lsizex1 = (pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR))/Ncellx1;
  const auto lsizex2 = (pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR))/Ncellx2;
  const auto lsizex3 = (pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR))/Ncellx3;

  const auto x1min = pmesh->mesh_size.xmin(X1DIR);
  const auto x2min = pmesh->mesh_size.xmin(X2DIR);
  const auto x3min = pmesh->mesh_size.xmin(X3DIR);


  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto const &cons = md->PackVariables(std::vector<std::string>{"cons"});


  //Quantities to initialise
  int Nq = 4;

// Read ICs binary
/*
  const size_t size_buf = 1024;
  char buffer[size_buf];  
  if (getcwd(buffer, size_buf) == nullptr) {
    std::cout << "Error getting current working directory" << std::endl;
  }else{
    std::cout << "This is the cwd: " << buffer << std::endl;
  }
*/
  std::ifstream infile("ICs.bin",  std::ios::in | std::ios::binary);
  if (!infile.is_open()) {
      PARTHENON_FAIL("Failed to open ICs bin file.");
  }
  


  //View for ICs Kokkos initialization
  size_t size = Ncellx1 * Ncellx2 * Ncellx3 * Nq;
  typedef Kokkos::View<double*> BinArr;
  BinArr ICsdata("data", size); 
  BinArr::HostMirror hICs = Kokkos::create_mirror_view(ICsdata);


  //Get data from Binary
  std::vector<double> temp_data(size);
  
  infile.read(reinterpret_cast<char*>(temp_data.data()), size * sizeof(double));
  infile.close();

  for (size_t i = 0; i < size; ++i) {
    hICs(i) = temp_data[i];
  }

  //Pass data onto dev memory space 
  Kokkos::deep_copy(ICsdata, hICs);

  
  // Assign values to primary variables

  Kokkos::parallel_for( "Cloud::ProblemGenerator", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, kb.s , jb.s, ib.s },{num_blocks, kb.e + 1, jb.e + 1, ib.e + 1}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {

      const auto &u = cons(b);
      const auto &coords = cons.GetCoords(b);
      const int global_x = (coords.Xc<1>(i) - lsizex1/2 - x1min)/lsizex1;
      const int global_y = (coords.Xc<2>(j) - lsizex2/2 - x2min)/lsizex2;
      const int global_z = (coords.Xc<3>(k) - lsizex3/2 - x3min)/lsizex3;

      int indexDN = ((global_z * Ncellx2 + global_y) * Ncellx1 + global_x) * Nq + 0;
      int indexM2 = ((global_z * Ncellx2 + global_y) * Ncellx1 + global_x) * Nq + 1;
      int indexIEN1 = ((global_z * Ncellx2 + global_y) * Ncellx1 + global_x) * Nq + 2;
      int indexIEN2 = ((global_z * Ncellx2 + global_y) * Ncellx1 + global_x) * Nq + 3;

      u(IDN, k, j, i) = ICsdata(indexDN)* d_cgs_factor;
      u(IM2, k, j, i) =  ICsdata(indexM2)* m_cgs_factor;
      u(IEN, k, j, i) =  (ICsdata(indexIEN1) + ICsdata(indexIEN2)/mbar_over_kb ) * e_cgs_factor;

    });
  
  
}




void InflowWindX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
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

//========================================================================================
//! \fn void ApplyFrameBoost(parthenon::MeshData<parthenon::Real> *md)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

// Compute frame_boosting velocity
void ComputeCloudMassWeightedVel(parthenon::MeshData<parthenon::Real> *md) {

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  //const auto x2centre = (pmesh->mesh_size.xmax(X2DIR) + pmesh->mesh_size.xmin(X2DIR))/2;


  const auto units = hydro_pkg->Param<Units>("units");
  Real mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("singlecloud::mean_molecular_mass_by_kb");
  Real T_cloud = hydro_pkg->Param<Real>("Tcloud");
  Real frame_v;

  Kokkos::Array<Real, 2> sums{{0.0, 0.0}};

  Kokkos::parallel_reduce(
      "SingleCloud::frame_boosting_velocity", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e+1, jb.e+1, ib.e+1}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, 
      Real& local_IM_cold_gas, Real& local_cold_gas) { 
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        //if ( coords.Xc<2>(j) < x2centre ) {
          const Real temp =
              mean_molecular_mass_by_kb * cons(IPR, k, j, i) / cons(IDN, k, j, i);

          if (temp <= 2*T_cloud) {

                  local_IM_cold_gas += cons(IM2, k, j, i);
                  local_cold_gas += cons(IDN, k, j, i); 
          }
        //}
      },
      Kokkos::Sum<Real>(sums[0]), Kokkos::Sum<Real>(sums[1])); 
#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 2, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  if (sums[1] > 0. && sums[0] > 0.) {
  frame_v = sums[0]/sums[1];
  } else {
  frame_v = 0.;
  }
  hydro_pkg->UpdateParam("inertial_frame_v", frame_v); 

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


  Real frame_v = hydro_pkg->Param<Real>("inertial_frame_v");
  
  if (fabs(frame_v) > 100.01 || frame_v < 0.0) frame_v = 0.;

 
  Kokkos::parallel_for(
    "SingleCloud::frame_boosting_velocity", Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e+1, jb.e+1, ib.e+1}),  
    KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {

        auto &cons = cons_pack(b);

          
            cons(IEN, k, j, i) -= frame_v * cons(IM2, k, j, i);
            cons(IEN, k, j, i) += 0.5 * SQR(frame_v) * cons(IDN, k, j, i);
            cons(IM2, k, j, i) -= frame_v * cons(IDN, k, j, i);
    
         
      });

  
}
void FrameBoosting(parthenon::MeshData<parthenon::Real> *md, const parthenon::SimTime &tm,
                         const Real dt){
  ComputeCloudMassWeightedVel(md);
  ApplyFrameBoost(md);

                         }



//----------------------------------------------------------------------------------------
//! \fn void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg)
//  \brief Hst file initialiser for new variables

// TODO(?) until we are able to process multiple variables in a single hst function call
// we'll use this enum to identify the various vars.
enum class HstQuan {mc};

// Compute the local sum of cloud mass
template <HstQuan hst_quan>
Real CloudHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  Real T_cloud = hydro_pkg->Param<Real>("Tcloud");
  Real mean_molecular_mass_by_kb = hydro_pkg->Param<Real>("singlecloud::mean_molecular_mass_by_kb");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);


  // after this function is called the result is MPI_SUMed across all procs/meshblocks
  // thus, we're only concerned with local sums
  Real sum;

  pmb->par_reduce(
      "hst_cloud", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);


        if (hst_quan == HstQuan::mc) { 
          const Real temp = mean_molecular_mass_by_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i);

          if (temp <= 2*T_cloud) {
            lsum += prim(IDN, k, j, i) * coords.CellVolume(k, j, i);
          }
        }
      },
      sum);

  return sum;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {

  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    CloudHst<HstQuan::mc>, "Mcloud (code units)"));
  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

  }
} // namespace cloud
