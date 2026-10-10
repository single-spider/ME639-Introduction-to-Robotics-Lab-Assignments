# ME 639: Introduction to Robotics lab assignments

MuJoCo implementations of Lab 1 and Lab 2, with separate videos for each of
the five robot models. Click a preview image to open its short MP4 demo.

## Lab 1: Drone and TurtleBot3

[Code, setup, controls, and model details](Lab%201/README.md)

| Crazyflie 2 | TurtleBot3 Waffle Pi | TurtleBot3 Burger |
| --- | --- | --- |
| [![Crazyflie demo](Lab%201/media/crazyflie.jpg)](Lab%201/media/crazyflie.mp4) | [![Waffle Pi demo](Lab%201/media/turtlebot_waffle_pi.jpg)](Lab%201/media/turtlebot_waffle_pi.mp4) | [![Burger demo](Lab%201/media/turtlebot_burger.jpg)](Lab%201/media/turtlebot_burger.mp4) |
| [Video](Lab%201/media/crazyflie.mp4) | [Video](Lab%201/media/turtlebot_waffle_pi.mp4) | [Video](Lab%201/media/turtlebot_burger.mp4) |

**Run:** `cd "Lab 1"`, create a Python environment, install
`requirements.txt`, then run `python lab1_ui.py`. The app offers live keyboard
and slider control for all three models and a live pose/Jacobian panel. See
the [Lab 1 README](Lab%201/README.md) for commands and headless checks.

## Lab 2: Franka and Heal arms

[Code, setup, DH forward kinematics, and model details](Lab%202/README.md)

| Franka Panda | Heal arm |
| --- | --- |
| [![Franka demo](Lab%202/media/franka.jpg)](Lab%202/media/franka.mp4) | [![Heal demo](Lab%202/media/heal.jpg)](Lab%202/media/heal.mp4) |
| [Video](Lab%202/media/franka.mp4) | [Video](Lab%202/media/heal.mp4) |

**Run:** `cd "Lab 2"`, create a Python environment, install
`franka_joint_ui/requirements.txt`, then run `python franka_joint_ui/app.py`.
The UI controls both arms. `python_script_for_dh_tran.py` calculates their
end-effector poses from DH parameters. See the [Lab 2 README](Lab%202/README.md)
for exact setup and verification commands.

The videos are scripted MuJoCo demonstrations with live state labels; they
are separate from recordings of mouse interaction with the apps.
