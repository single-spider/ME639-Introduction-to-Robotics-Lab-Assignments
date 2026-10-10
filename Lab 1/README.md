# Lab 1: Crazyflie and TurtleBot3 control

This lab runs entirely in MuJoCo. The app offers Crazyflie 2, TurtleBot3
Waffle Pi, and TurtleBot3 Burger in one window. The model assets needed by
these three robots are included locally under `mujoco_menagerie/` and
`robotis_mujoco_menagerie/`; no Gazebo installation is needed.

## Demonstrations

The [UI recording](media/lab1_controls_and_matrices.mp4) shows the controls and
changing matrices for all three robots in the actual MuJoCo app. Run
`python lab1_ui.py --record-demo media/lab1_controls_and_matrices.mp4` to
regenerate it. [Animated UI preview](media/lab1_controls_and_matrices.gif).

Separate demos: [Crazyflie](media/crazyflie.mp4),
[Waffle Pi](media/turtlebot_waffle_pi.mp4), and
[Burger](media/turtlebot_burger.mp4). They show scripted simulation motion
with live state labels; the README displays their animated previews inline.

## Setup and run

Use Python 3.10 or newer and a desktop with OpenGL support. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python lab1_ui.py
```

On Windows, activate with `.venv\Scripts\activate` and use `python` in place
of `python3` to create the environment.

To check the robot models and controls without opening a window:

```bash
python lab1_ui.py --check
```

## Controls and live matrices

The Robot menu offers Crazyflie, Waffle Pi, and Burger. Hold the movement keys
to command motion. For Crazyflie, W/S move forward/back, A/D move left/right,
R/F move up/down, and Q/E turn. For TurtleBot, W/S move forward/back and A/D
turn; arrow keys also work. Release the keys to stop commanding motion.
The sliders provide continuous velocity commands. Space stops either robot.

The **Live pose and matrices** panel shows the world-to-body homogeneous
transform, rotation matrix, quaternion, roll/pitch/yaw, and MuJoCo's
world-frame floating-base Jacobian. TurtleBot also shows the ideal
differential-drive wheel Jacobian and wheel state. These two Jacobians have
different inputs: floating-base velocities versus wheel rates.

`show_turtlebot3_frame.py` is a separate viewer for the TurtleBot models;
Waffle Pi is its default, and `--variant burger` selects Burger.

## Model attribution

Crazyflie assets come from MuJoCo Menagerie and TurtleBot3 assets from the
Robotis MuJoCo Menagerie. Their license files are preserved in the copied
model directories.
