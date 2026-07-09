#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace linear_wave {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace linear_wave

namespace linear_wave_mhd {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
} // namespace linear_wave_mhd

namespace cpaw {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace cpaw

namespace cloud {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void InflowWindX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd);
} // namespace cloud

namespace wtopenrun {
using namespace parthenon;

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md);
void InflowWindX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd);
void FrameBoosting(parthenon::MeshData<parthenon::Real> *md, const parthenon::SimTime &tm,
                   const Real dt);
} // namespace wtopenrun

namespace stratified_box {
using namespace parthenon;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md);
void StratUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real beta_dt);
void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::Real beta_dt);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void ProblemInitTracerData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void StratInflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratInflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratNoFlowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratNoFlowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
void DrivingAndFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm,
                          const Real dt);
void SetPhases(MeshBlock *pmb, ParameterInput *pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
TaskStatus ProblemFillTracers(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real dt);
void ColdGasFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
} // namespace stratified_box

namespace stratified_box_noio {
using namespace parthenon;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md);
void StratUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real beta_dt);
void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::Real beta_dt);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void ProblemInitTracerData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void StratInflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratInflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratNoFlowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratNoFlowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
void DrivingAndFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm,
                          const Real dt);
void SetPhases(MeshBlock *pmb, ParameterInput *pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
TaskStatus ProblemFillTracers(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real dt);
void ColdGasFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
} // namespace stratified_box

namespace stratified_box_exponential {
using namespace parthenon;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md);
void StratUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real beta_dt);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void ProblemInitTracerData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void StratInflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratInflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratNoFlowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratNoFlowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void DrivingAndFrameTrack(MeshData<Real> *md, const parthenon::SimTime &tm,
                          const Real dt);
void SetPhases(MeshBlock *pmb, ParameterInput *pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
TaskStatus ProblemFillTracers(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real dt);
} // namespace stratified_box_exponential

namespace stratified_box_simple {
using namespace parthenon;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md);
void StratUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real beta_dt);
void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::Real beta_dt);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void StratOutflowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratOutflowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratNoFlowInnerX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void StratNoFlowOuterX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);

} // namespace stratified_box_simple
namespace blast {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace blast

namespace advection {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace advection

namespace orszag_tang {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace orszag_tang

namespace diffusion {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace diffusion

namespace field_loop {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
} // namespace field_loop

namespace kh {
using namespace parthenon::driver::prelude;

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
} // namespace kh

namespace lw_implode {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace lw_implode

namespace rand_blast {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void RandomBlasts(MeshData<Real> *md, const parthenon::SimTime &tm, const Real);
} // namespace rand_blast

namespace cluster {
using namespace parthenon::driver::prelude;

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void InitUserMeshData(ParameterInput *pin);
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
void ClusterUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                           const Real beta_dt);
void ClusterSplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real beta_dt);
parthenon::Real ClusterEstimateTimestep(MeshData<Real> *md);
} // namespace cluster

namespace shattering {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace shattering

namespace sod {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace sod

namespace turbulence {
using namespace parthenon::driver::prelude;

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void ProblemInitTracerData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
void SetPhases(MeshBlock *pmb, ParameterInput *pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
TaskStatus ProblemFillTracers(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real dt);
void Cleanup();
} // namespace turbulence

namespace turbulence_grav {
using namespace parthenon::driver::prelude;

void StratUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real beta_dt);
void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::Real beta_dt);
void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void ProblemInitTracerData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
void SetPhases(MeshBlock *pmb, ParameterInput *pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
TaskStatus ProblemFillTracers(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real dt);
void Cleanup();
} // namespace turbulence

namespace sn {
using namespace parthenon;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin,  MeshData<Real> *md);
void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm);
} // namespace sn
#endif // PGEN_PGEN_HPP_
