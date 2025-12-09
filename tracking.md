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

  // Get mutable reference to cumulative displacement
  auto *p_frame_disp =
      hydro_pkg->MutableParam<Real>("stratified_box/frame_displacement_y");

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

  // Loop over blocks
  for (int b = 0; b < md->NumBlocks(); b++) {
    auto &mbd = md->GetBlockData(b);
    auto pmb = mbd->GetBlockPointer();
    auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"});

    IndexRange ib = mbd->GetBoundsI(IndexDomain::interior);
    IndexRange jb = mbd->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = mbd->GetBoundsK(IndexDomain::interior);

    const auto &coords = cons_pack.GetCoords();

    // Calculate average inward velocity at inner Y boundary to update displacement
    // We do this on host side by reading interior data
    auto cons = mbd->Get("cons").data;
    auto cons_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons);

    // Compute average v2 (velocity in Y-direction) at inner boundary
    // Only include gas cells with temperature < 2e5 K
    Real v2_avg = 0.0;
    int count = 0;
    const Real T_cut = 2e5; // Kelvin
    for (int k = kb.s; k <= kb.e; k++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Real rho_cell = cons_h(IDN, k, jb.s, i);
        if (rho_cell <= 0.0) continue; // skip invalid
        Real T_cell = mean_molecular_mass_by_kb * cons_h(IPR, k, jb.s, i) / rho_cell;
        if (T_cell < T_cut) {
          Real v2_cell = cons_h(IM2, k, jb.s, i) / rho_cell;
          v2_avg += v2_cell;
          count++;
        }
      }
    }
    if (count > 0) {
      v2_avg /= static_cast<Real>(count);
    } else {
      v2_avg = 0.0; // no cold gas at boundary, no displacement from this block
    }

    // Update cumulative displacement (inward is negative)
    *p_frame_disp += v2_avg * dt;

    // Get cell width in Y-direction
    Real dy = (pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR))/pmb->pmy_mesh->GetDefaultBlockSize().nx(parthenon::X2DIR); // cell width

    // Check if cumulative displacement exceeds one cell width
    if (std::abs(*p_frame_disp) >= dy) {
      // Determine shift direction
      int num_shifts = static_cast<int>(std::floor(std::abs(*p_frame_disp) / dy));
      int shift_dir = (*p_frame_disp < 0.0) ? -1 : 1; // -1 for inward, +1 for outward

      // Perform shifts
      for (int shift = 0; shift < num_shifts; shift++) {
        if (shift_dir == -1) {
          // Inward shift: remove last row, add at first row
          // Shift data: row j ← row j+1 for j = [jb.s, jb.e-1]
          for (int k = kb.s; k <= kb.e; k++) {
            for (int j = jb.s; j < jb.e; j++) {
              for (int i = ib.s; i <= ib.e; i++) {
                for (int n = 0; n < cons_h.GetDim(0); n++) {
                  cons_h(n, k, j, i) = cons_h(n, k, j + 1, i);
                }
              }
            }
          }

          // Populate new row at jb.s with profile values
          for (int k = kb.s; k <= kb.e; k++) {
            for (int i = ib.s; i <= ib.e; i++) {
              Real Y = coords.Xc<2>(jb.s);
              double rhoY = rho_profile_Y(Y, rho0, a, H);

              // Set density
              cons_h(IDN, k, jb.s, i) = rhoY;

              // Copy tangential velocities from next interior cell
              cons_h(IM1, k, jb.s, i) = cons_h(IM1, k, jb.s + 1, i);
              cons_h(IM3, k, jb.s, i) = cons_h(IM3, k, jb.s + 1, i);

              // Set normal velocity from nearest interior cell (preserves flow pattern)
              cons_h(IM2, k, jb.s, i) = cons_h(IM2, k, jb.s + 1, i);

              // Compute pressure/energy: use temperature from nearest interior cell
              Real T = cons_h(IPR, k, jb.s + 1, i) / cons_h(IDN, k, jb.s + 1, i);
              Real ke = 0.5 *
                        (cons_h(IM1, k, jb.s, i) * cons_h(IM1, k, jb.s, i) +
                         cons_h(IM2, k, jb.s, i) * cons_h(IM2, k, jb.s, i) +
                         cons_h(IM3, k, jb.s, i) * cons_h(IM3, k, jb.s, i)) /
                        rhoY;
              Real ie = T / gm1; // specific internal energy
              cons_h(IEN, k, jb.s, i) = rhoY * (ie + 0.5 * ke);

              // Pressure for storage
              cons_h(IPR, k, jb.s, i) = rhoY * T;
            }
          }
        } else {
          // Outward shift: remove first row, add at last row
          // Shift data: row j ← row j-1 for j = [jb.e, jb.s+1]
          for (int k = kb.s; k <= kb.e; k++) {
            for (int j = jb.e; j > jb.s; j--) {
              for (int i = ib.s; i <= ib.e; i++) {
                for (int n = 0; n < cons_h.GetDim(0); n++) {
                  cons_h(n, k, j, i) = cons_h(n, k, j - 1, i);
                }
              }
            }
          }

          // Populate new row at jb.e with profile values
          for (int k = kb.s; k <= kb.e; k++) {
            for (int i = ib.s; i <= ib.e; i++) {
              Real Y = coords.Xc<2>(jb.e);
              double rhoY = rho_profile_Y(Y, rho0, a, H);

              // Set density
              cons_h(IDN, k, jb.e, i) = rhoY;

              // Copy tangential velocities from nearest interior cell
              cons_h(IM1, k, jb.e, i) = cons_h(IM1, k, jb.e - 1, i);
              cons_h(IM3, k, jb.e, i) = cons_h(IM3, k, jb.e - 1, i);

              // Set normal velocity from nearest interior cell
              cons_h(IM2, k, jb.e, i) = cons_h(IM2, k, jb.e - 1, i);

              // Compute energy
              Real T = cons_h(IPR, k, jb.e - 1, i) / cons_h(IDN, k, jb.e - 1, i);
              Real ke = 0.5 *
                        (cons_h(IM1, k, jb.e, i) * cons_h(IM1, k, jb.e, i) +
                         cons_h(IM2, k, jb.e, i) * cons_h(IM2, k, jb.e, i) +
                         cons_h(IM3, k, jb.e, i) * cons_h(IM3, k, jb.e, i)) /
                        rhoY;
              Real ie = T / gm1;
              cons_h(IEN, k, jb.e, i) = rhoY * (ie + 0.5 * ke);

              // Pressure for storage
              cons_h(IPR, k, jb.e, i) = rhoY * T;
            }
          }
        }
      }

      // Copy back to device
      cons.DeepCopy(cons_h);

      // Reset displacement counter
      *p_frame_disp -= shift_dir * num_shifts * dy;
    }
  }
}