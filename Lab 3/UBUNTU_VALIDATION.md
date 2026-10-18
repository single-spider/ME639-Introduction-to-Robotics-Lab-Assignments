# Ubuntu validation

The package was tested under **Ubuntu 24.04.4 LTS, Linux x86-64 (WSL2),
Python 3.12.3**, using a fresh Linux virtual environment and the exact versions
in `requirements.txt`. No Windows Python environment was used for these checks.

The submission ZIP was extracted to a temporary directory on Ubuntu's native
filesystem. All source, configuration and mesh files were loaded from that
extracted copy, independently of the original workspace. This also checks
case-sensitive asset paths.

## Completed checks

- All five submission tests passed: both model/Jacobian checks, DLS/QP agreement,
  QP bounds, both table layouts and complete historical evaluation groups.
- Workspace and method-comparison plots were regenerated from the supplied data.
- Both robots completed Mink, DLS and QP trials on the workspace-sized tables
  at seed 3100: **6/6 constraint-safe placements**.
- Both robots completed the three methods on the additional placement layouts
  at seed 2000: **6/6 constraint-safe placements**.
- The documented Heal QP placement-layout failure at seed 2004 reproduced a
  pre-grasp timeout. Its nonzero exit status is expected, not a setup failure.
- Both models rendered successfully with **`MUJOCO_GL=egl`**, without a display
  window.
- Complete physical recordings were generated for Franka/DLS and Heal/QP on
  the workspace-sized tables at seed 3100.

Key installed versions: MuJoCo 3.14.0, Mink 1.3.0, NumPy 2.4.6, SciPy 1.17.1,
qpsolvers 4.13.0 and quadprog 0.1.13.

These are Linux compatibility and smoke checks, not a rerun of the entire
300-episode historical evaluation or a hardware experiment. Different Ubuntu
graphics drivers may require GLFW or OSMesa instead of EGL; the README provides
the corresponding options. The runtime code and final submission files have no
Windows font or drive-path dependencies.
