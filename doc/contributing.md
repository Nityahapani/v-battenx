# Contributing to V-BATTEN-X

## Build Instructions

### Prerequisites

- g++ ≥ 13 or clang++ ≥ 16
- Eigen3 ≥ 3.4 (`apt install libeigen3-dev`)
- Python ≥ 3.10
- Optional: CUDA toolkit ≥ 12.0, MPI, R ≥ 4.3

### CPU build

```bash
git clone https://github.com/Nityahapani/v-battenx.git
cd v-battenx

# Build shared library
g++ -std=c++17 -O2 -shared -fPIC \
  -I. -Iinclude -I/usr/include/eigen3 \
  src/vbatten_x_impl.cc src/c_api.cc \
  -o python-package/vbatten_x/_vbatten_x.so

# Install Python package
pip install -e python-package/[dev]
```

### CMake build (recommended for production)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### CUDA build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
cmake --build build -j$(nproc)
```

## Code Style

- **C++**: `clang-format` with the project `.clang-format` (LLVM style, 4-space indent)
- **Python**: `black --line-length 100` + `isort --profile black`
- **R**: `styler::style_pkg()`

Run all formatters before committing:
```bash
clang-format -i src/**/*.cc include/**/*.h
black python-package/ --line-length 100
isort python-package/
```

## Adding a New PDE Evaluator

1. Create `src/physics/pde/my_pde.cc` implementing `PhysicsEvaluator`
2. Add `#include "src/physics/pde/my_pde.cc"` to `src/vbatten_x_impl.cc`
3. Register in `src/physics/evaluator_registry.cc` under the appropriate `PDETypeTag`
4. Add `PDETypeTag::MyPde` to the enum in `include/vbatten_x/data.h`
5. Add Python enum value `PDEType.MY_PDE` in `python-package/vbatten_x/physics.py`
6. Write a test in `tests/physics/test_my_pde.py`

## Adding a New Mutation Type

1. Add the enum value to `MutationType` in `include/vbatten_x/mutation_result.h`
2. Add the name to `MutationTypeName()` in the same file
3. Implement the atomic mutation in `src/dtdo/mutations/my_mutation.cc`
4. Declare it in `src/dtdo/mutations/mutations.h`
5. Add `#include` in `src/vbatten_x_impl.cc`
6. Handle it in `DtdoNet::ApplyAction` in `src/dtdo/learned/dtdo_net_apply.cc`
7. Handle it in all rule-based operators that should fire it
8. Write a test in `tests/cpp/test_dtdo_mutations.cc`

## PR Checklist

- [ ] All existing tests pass (`pytest tests/ -q`)
- [ ] New tests written for new functionality
- [ ] No `TODO` or `FIXME` in merged code (open an issue instead)
- [ ] No ABI break in `include/vbatten_x/` (no struct layout changes, no virtual method reorder)
- [ ] New public symbols added to `src/c_api.cc` if needed
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] Documentation updated if behaviour changed

## Commit Convention

```
feat(scope): add thing
fix(scope): fix thing
test(scope): add tests for thing
docs: update thing
perf(scope): speed up thing
chore: update dependency
```

Scopes: `core`, `dtdo`, `physics`, `booster`, `encoder`, `python`, `R`, `build`, `docs`

## Running Tests

```bash
# Python tests
pytest tests/python/ tests/physics/ -v

# C++ dim transition tests
g++ -std=c++17 -O2 -I. -Iinclude -I/usr/include/eigen3 \
  tests/cpp/test_dim_transitions.cc -o /tmp/test_dt && /tmp/test_dt
```
