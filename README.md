# ME 639: Introduction to Robotics — Lab Assignments

**Student:** Gaurav Srivastava

**Course:** ME 639, Introduction to Robotics

**Term:** Autumn 2026

MuJoCo implementations and demonstrations for Lab 1 and Lab 2. The animations
play directly in this README; each full resolution MP4 is linked below its preview.

## Lab 1 — Drone and TurtleBot3

[Code, installation, controls, and model details](Lab%201/README.md)

### Controls and changing matrices

The recording captures the actual app window while velocity controls change.
The right panel updates the world-to-body transform, rotation, quaternion,
Jacobian, and wheel state as each robot moves.

![Lab 1 controls and live matrices](Lab%201/media/lab1_controls_and_matrices.gif)

[Watch or download the full UI recording](Lab%201/media/lab1_controls_and_matrices.mp4)

### Crazyflie 2 drone

![Crazyflie demo](Lab%201/media/crazyflie.gif)

[Full MP4](Lab%201/media/crazyflie.mp4)

### TurtleBot3 Waffle Pi

![Waffle Pi demo](Lab%201/media/turtlebot_waffle_pi.gif)

[Full MP4](Lab%201/media/turtlebot_waffle_pi.mp4)

### TurtleBot3 Burger

![Burger demo](Lab%201/media/turtlebot_burger.gif)

[Full MP4](Lab%201/media/turtlebot_burger.mp4)

**Run:** `cd "Lab 1"`, install `requirements.txt` in a Python environment,
then run `python lab1_ui.py`. See the [Lab 1 README](Lab%201/README.md) for
controls and verification commands.

## Lab 2 — Franka and Heal robot arm control

[Code, installation, DH forward kinematics, and model details](Lab%202/README.md)

### Franka Panda

![Franka demo](Lab%202/media/franka.gif)

[Full MP4](Lab%202/media/franka.mp4)

### Heal arm

![Heal demo](Lab%202/media/heal.gif)

[Full MP4](Lab%202/media/heal.mp4)

### Part 3: handwritten DH derivation

[Open the two-page handwritten PDF](Lab%202/report/handwritten-dh-derivation.pdf)

[![First page of handwritten DH derivation](Lab%202/media/dh_derivation_preview.jpg)](Lab%202/report/handwritten-dh-derivation.pdf)

**Run:** `cd "Lab 2"`, install `franka_joint_ui/requirements.txt`, then run
`python franka_joint_ui/app.py`. The app controls both arms.
`python_script_for_dh_tran.py` calculates end-effector poses from DH parameters.
See the [Lab 2 README](Lab%202/README.md) for exact commands.
