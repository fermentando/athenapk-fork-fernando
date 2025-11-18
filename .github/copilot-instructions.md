# AthenaPK - AI Coding Instructions

## Quick Reference

**AthenaPK** is a performance-portable block-structured AMR astrophysical MHD code built on **Parthenon** (mesh/AMR framework) and **Kokkos** (GPU/CPU portability). It implements compressible (magneto)hydrodynamics with various reconstruction methods, Riemann solvers, and optional diffusive processes (conduction, viscosity, resistivity).

**Key Files:**
- `src/main.cpp`: Simulation driver, callback registration, parameter initialization
- `src/pgen/stratified.cpp`: Problem generator for stratified atmosphere simulations
- `src/hydro/`: Hydrodynamic core (fluxes, diffusion, EOS)
- `CMakeLists.txt`: Build configuration (Kokkos/Parthenon integration)

---

## Architecture Overview

### Three-Layer Design
1. **Physics Layer** (`src/hydro/`, `src/eos/`): Flux calculations, equation of state, diffusion
2. **Problem-Specific Layer** (`src/pgen/*.cpp`): Initialization, boundary conditions, source terms via callbacks
3. **Framework Layer** (Parthenon/Kokkos): Mesh management, AMR, parallelization, I/O

### Data Flow
```
Input file (.in)
  ↓
main.cpp (callback setup)
  ↓
Problem Generator (pgen/*.cpp) - initializes variables & parameters
  ↓
Hydro loop: each stage calls
  - Reconstruction → Riemann solver → Flux divergence
  - Unsplit source terms (gravity, user callbacks)
  - Boundary conditions (user callbacks)
  ↓
Output (HDF5 via Parthenon)
```

---

## Problem Generator System

### Adding New Problem Physics
**All problem generators must follow this pattern:**

1. **File**: `src/pgen/newname.cpp` in namespace `newname_box { ... }`
2. **Callbacks** (declared in `src/pgen/pgen.hpp`):
   - `ProblemGenerator()` - initialize conserved variables for all blocks
   - Optional unsplit source: `void MySource(MeshData<Real> *md, const Real beta_dt)`
   - Optional boundary conditions: `void MyBC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse)`
   - Optional first/second-order split sources

3. **Registration** (in `src/main.cpp`):
   ```cpp
   if (problem_id == "myname") {
     packages.Get("Hydro")->StartPackage(...)->ProblemSourceUnsplit = newname_box::MySource;
     /* ... other callbacks */
   }
   ```

4. **Compilation**: Add to `src/pgen/CMakeLists.txt`

5. **Input file** (`inputs/myname.in`):
   ```ini
   <job>
   problem_id = myname
   
   <problem/myname>
   param1 = value1
   ```

### Key Patterns from `stratified.cpp`

**Stratified atmosphere example** (see `src/pgen/stratified.cpp:372-456`):
- Density profile using `rho_profile_Y(Y, rho0, a, H)` - applies hydrostatic equilibrium
- Boundary conditions `StratOutflowInnerX2/OuterX2` use `par_for_bndry()` for ghost zones
- Source term `StratUnsplitSrcTerm()` wraps gravitational acceleration
- Key params stored via `pkg->AddParam<Real>(...)` - accessible from `pkg->Param<Type>("name")`

**Accessing Parameters in Lambdas:**
```cpp
auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
auto my_param = hydro_pkg->Param<Real>("param_name");  // in host code
// Copy to lambda variables for device access
const auto param_copy = my_param;
par_for(..., KOKKOS_LAMBDA(...) { use(param_copy); });
```

---

## Kokkos/GPU Patterns

### Parallel Loops
**Host-side (CPU, serial):**
```cpp
for (int j = jb.s; j <= jb.e; j++) { ... }  // standard C++ loop
```

**Device-portable (GPU or CPU):**
```cpp
par_for(DEFAULT_LOOP_PATTERN, "name", DevExecSpace(),
        0, cons_pack.GetDim(5) - 1,  // blocks
        kb.s, kb.e,                   // k range
        jb.s, jb.e,                   // j range
        ib.s, ib.e,                   // i range
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          // Device code here - read/write cons_pack, prim_pack
        });
```

### Memory Layout
- Conserved variables: `cons(field_idx, k, j, i)` - 4D array
- Field indices: `IDN` (density), `IM1/IM2/IM3` (momenta), `IEN` (energy), `IPR` (pressure, primitive)
- Host-device transfer: `.GetHostMirrorAndCopy()` then `.DeepCopy()`

---

## Building & Testing

### Build
```bash
cmake -S. -Bbuild -DKokkos_ARCH_BDW=ON -DPARTHENON_DISABLE_MPI=ON
cd build && make -j8
```

### Run
```bash
./bin/athenaPK -i ../inputs/linear_wave3d.in
# Override parameters:
./bin/athenaPK -i ../inputs/file.in hydro/reconstruction=ppm parthenon/mesh/nx1=256
```

### Code Style
- Format with `make format-athenapk` (clang-format for C++, black for Python)
- Checked automatically in CI

---

## Units System

**Purpose**: Maintain code-unit / CGS conversions consistently.

**Usage** (see `src/units.hpp`):
```cpp
Units units(pin);  // initialized from <units> block in input file
Real density_cgs = code_value * units.code_density_cgs();
Real length_pc = code_value * units.kpc();  // convert to parsecs
```

**Common conversions stored in `InitUserMeshData()` (e.g., `stratified.cpp:147-151`)**:
```cpp
d_cgs_factor = 1. / units.code_density_cgs();  // multiply code density to get CGS
```

---

## Boundary Conditions & Frame Tracking

### Standard BC Pattern (e.g., `StratOutflowInnerX2`)
```cpp
void MyBC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  auto pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables({"cons"}, coarse);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  
  pmb->par_for_bndry("BCName", IndexRange{0, 0}, IndexDomain::inner_x2,
      parthenon::TopologicalElement::CC, coarse, false,
      KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack;
        // Set ghost zone values based on interior at jb.s
        cons(IDN, k, j, i) = /* new value */;
      });
}
```

### Frame Tracking Pattern
When domain moves (e.g., cold gas drifts inward in Y):
1. **Detect motion**: Monitor boundary flow or displacement
2. **Shift data**: Remove trailing row, add new row at leading boundary
3. **Populate new row**: Use profile function (e.g., `rho_profile_Y()`) with appropriate boundary position
4. **Maintain physics**: Copy/compute velocities and energy to match interior state

**Example for Y-axis frame tracking** (inward moving boundary):
```cpp
// Remove last row in j-direction
// Shift all j-indices down: j → j-1
// Add new row at j_min with:
// - ρ from rho_profile_Y(Y_min, rho0, a, H)  
// - v from copy of nearest interior cells
// - E computed from ρ and temperature profile
```

---

## Common Patterns to Avoid

1. **Don't access `pkg->Param()` inside device lambdas** → copy to host variable first
2. **Don't assume cell indices match global indices** → use `loc.lx1()`, `loc.lx2()`, `loc.lx3()` for block offsets
3. **Don't modify ghost zones in source terms** → they're updated elsewhere; modify only interior cells unless it's an explicit BC function
4. **Don't call MPI inside `par_for`** → write reduction logic outside loop or use Kokkos reductions
5. **Don't assume 3D** → code supports 1D/2D/3D; use `Globals::ndim` and `pmesh->GetDefaultBlockSize().nx(X*DIR)`

---

## Input File Structure

### Minimal Example (`inputs/mytest.in`)
```ini
<job>
problem_id = myname
bin_input_file = path/to/file.bp

<parthenon/mesh>
nx1 = 256
nx2 = 128  
nx3 = 64
nghost = 4

<parthenon/meshblock>
nx1 = 32
nx2 = 32
nx3 = 32

<hydro>
gamma = 1.4
reconstruction = ppm
riemann = hllc
fluid = euler

<units>  
code_length_cgs = 3.086e18  # 1 pc
code_mass_cgs = 1.989e33    # 1 M_sun

<problem/myname>
param1 = 1.0

<parthenon/time>
tlim = 1.0
```

---

## Variable Naming Conventions

| Prefix/Suffix | Meaning |
|---|---|
| `nx/ny/nz` | Grid cell counts |
| `nb` / `pmb` / `mbd` | `MeshBlock` / block data pointers |
| `*_pack` | Parthenon VariablePack (multi-block view) |
| `ib/jb/kb` | IndexRange for i/j/k interior cells |
| `*_cgs` | CGS units |
| `rho/den` | Density |
| `mom/vel` | Momentum / velocity |

---

## References & Further Reading

- **Parthenon Docs**: Mesh management, VariablePacks, callbacks
- `docs/pgen.md`: Detailed problem generator API
- `docs/input.md`: Full input parameter reference
- `docs/development.md`: Code formatting, dev container setup
- Test examples: `tst/regression/test_suites/*/`

---

## Quick Commands

```bash
# Format code
make -C build format-athenapk

# Build from scratch (CPU, no MPI)
cmake -S. -Bbuild-new -DKokkos_ARCH_BDW=ON -DPARTHENON_DISABLE_MPI=ON
cd build-new && make -j

# Run with custom parameters
./bin/athenaPK -i inputs/stratified.in \
  problem/stratified_box/a_over_H=2.0 \
  parthenon/mesh/nx2=512

# Check for errors/warnings
cd build && make 2>&1 | grep -E "error|warning"
```
