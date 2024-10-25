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
#include "../units.hpp"
#include "utils/error_checking.hpp"

namespace fractal_ism {

int64_t*** create_4d_array(int x1, int x2, int x3, int layers);

void delete_4d_array(int64_t*** array, int x1, int x2, int x3);

int64_t*** ICsReader(const std::string& filename, int dim1, int dim2, int dim3);
        
parthenon::TaskStatus compute_frame_v(parthenon::MeshData<parthenon::Real> *md);

parthenon::TaskStatus apply_frame_boost(parthenon::MeshData<parthenon::Real> *md);

}