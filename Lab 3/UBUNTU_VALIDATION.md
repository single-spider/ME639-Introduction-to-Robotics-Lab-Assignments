# Ubuntu validation

## Native dual-boot Ubuntu — verified 19 October 2026

This repository's `Lab 3` copy was tested on the **native Ubuntu 22.04.5 LTS**
installation (Linux 6.8.0-138-generic, x86-64) with a fresh Python **3.12.15**
virtual environment on the Linux ext4 filesystem. Python 3.12 was installed
with `uv`; the packages were installed from the unmodified pinned
`requirements.txt`. `uv pip check` reported all 27 installed packages
compatible. No Windows environment or parent development modules were used.

Commands and observed outcomes:

| Check | Result |
| --- | --- |
| `python tools/check_submission.py` | 5/5 tests passed |
| `MUJOCO_GL=egl python tools/check_submission.py --render` | 5/5 tests passed; Franka and Heal PNGs rendered and visually inspected |
| `python run_lab.py --robot both --method all --layout sizing --seed 3100` | 6/6 constraint-safe successes |
| `python run_lab.py --robot both --method all --layout placement --seed 2000` | 6/6 constraint-safe successes |
| `python plot_results.py` | Plots regenerated under ignored `runs/plots/` |
| `MUJOCO_GL=egl python run_lab.py --robot both --method dls --layout sizing --seed 3100 --record` | 2/2 constraint-safe successes; both MP4s decoded without errors |
| `python run_lab.py --robot heal --method qp --layout placement --seed 2004` | Expected `RuntimeError:timeout:pregrasp`, exit status 1 |

The rendering backend used here was **EGL**. The fresh recordings and test
output remain under ignored `runs/` and are not part of the submitted
historical evaluation data. The report and supplied demonstration MP4 were
also inspected; the latter is 2 min 13 s, 1280×1080, H.264 with AAC audio.

These checks establish native Ubuntu execution for both models, methods,
layouts, rendering and recording. They do not rerun the full 300-episode
historical evaluation or test physical hardware. The two documented Heal QP
placement failures (seeds 2004 and 2018) remain part of the supplied results.

## Earlier WSL2 validation

The original package was separately tested on Ubuntu 24.04.4 LTS under WSL2,
Python 3.12.3, with the same pinned dependencies. Five tests, both six-case
layout runs, EGL rendering, full Franka/Heal recordings, and plot regeneration
passed. The known Heal QP seed-2004 timeout reproduced there too. WSL2
results were kept distinct from the native dual-boot checks above.
