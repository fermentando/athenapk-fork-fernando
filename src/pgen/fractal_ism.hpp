//! \file fractal_ism.hpp
//  \brief  Class for boosting frame in multicloud set-ups.

// parthenon headers
#include <basic_types.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../units.hpp"
#include "utils/error_checking.hpp"

namespace fractal_ism {

parthenon::TaskStatus compute_frame_v(parthenon::MeshData<parthenon::Real> *md);

parthenon::TaskStatus apply_frame_boost(parthenon::MeshData<parthenon::Real> *md);

}