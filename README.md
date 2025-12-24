<p align="center">
  <img src="docs/logo.png" height="240" alt="玄松 EquivFusion Logo">
</p>
<h2 align="center">EquivFusion from 玄松</h2>
<p align="center"><em>深理如松 · 验证无声</em></p>
<p align="center"><em>Slient as Pine, Precise as Logic</em></p>

# EquivFusion
EquivFusion: Unifying Formal Verification from Algorithms to Netlists for High-Efficiency Signoff.

# Contributors
- Min Li
- Baoqiz
- Mengxia Tao   <taomengxia@nctieda.com>

# Build
To get started quickly, run the following commands:

Linux/MacOS
```bash
# Clone the the repository.
git clone https://github.com/FORMiND-Lab/EquivFusion.git
cd EquivFusion

# Configure the build.
mkdir build
cd build
cmake .. -G Ninja
ninja

# Install solvers.
# This command automatically retrieves and builds dependencies such as aiger, bitwuzla, kissat, and boolector. 
# The resulting binaries are installed in the EquivFusion/build/bin directory.
ninja install_solvers

# Add 'EquivFusion/build/bin' to your PATH environment variable
export PATH="$PWD/bin/:$PATH"
```
# Dependencies

If you have `git`, `ninja`, `python3`, `cmake`, and a C++ toolchain installed, you should be able to build EquivFusion. 
Additionally, there are also some dependencies require configuration:

- **readline:** If `readline` is not in the system search path, you can specify its location during the CMake configuration step. Use `-DREADLINE_INCLUDE_ABSOLUTE_DIRECTORY` for the header files path and `-DREADLINE_LIBRARY_ABSOLUTE_PATH` for the library file path.

- **z3:** Ensure the Z3 solver is in your system `PATH`.



