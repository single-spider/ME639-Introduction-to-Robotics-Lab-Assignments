"""Drive a TurtleBot3 Waffle Pi or Burger in MuJoCo.

Run with ``python show_turtlebot3_frame.py``. Use ``--check`` for a headless
wheel-motion smoke test.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np


SCENES = {
    "waffle": Path(__file__).parent / "robotis_mujoco_menagerie/robotis_tb3/scene_turtlebot3_waffle_pi.xml",
    "burger": Path(__file__).parent / "robotis_mujoco_menagerie/robotis_tb3/scene_turtlebot3_burger.xml",
}
WHEEL_RADIUS = 0.033
WHEEL_SEPARATIONS = {"waffle": 0.288, "burger": 0.16}
LINEAR_SPEED = 0.12
ANGULAR_SPEED = 0.8


def wheel_speeds(linear: float, angular: float, variant: str = "waffle") -> tuple[float, float]:
    """Convert planar body velocity to left and right wheel speeds (rad/s)."""
    half_track = WHEEL_SEPARATIONS[variant] / 2
    return ((linear - angular * half_track) / WHEEL_RADIUS,
            (linear + angular * half_track) / WHEEL_RADIUS)


def make_simulation(variant: str = "waffle"):
    model = mujoco.MjModel.from_xml_path(str(SCENES[variant]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = model.body("base").id
    left = model.actuator("wheel_left").id
    right = model.actuator("wheel_right").id
    return model, data, base, left, right


def draw_arrow(scene, start, end, color):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                       np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                       np.asarray(color, dtype=np.float32))
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.008,
                         np.asarray(start), np.asarray(end))
    scene.ngeom += 1


def draw_frame(scene, origin, rotation, length=0.13):
    for axis, color in enumerate(((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1))):
        draw_arrow(scene, origin, origin + length * rotation[:, axis], color)


def check_motion(variant: str = "waffle"):
    model, data, base, left, right = make_simulation(variant)
    start = data.xpos[base].copy()
    data.ctrl[left], data.ctrl[right] = wheel_speeds(LINEAR_SPEED, 0.0, variant)
    for _ in range(round(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    displacement = data.xpos[base] - start
    if not displacement[0] > 0.05:
        raise RuntimeError(f"TurtleBot did not advance: displacement={displacement}")
    print(f"TurtleBot3 {variant} forward displacement in 2 s: {displacement[0]:.3f} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run without a viewer")
    parser.add_argument("--variant", choices=tuple(SCENES), default="waffle")
    args = parser.parse_args()
    if args.check:
        check_motion(args.variant)
        return

    import mujoco.viewer

    model, data, base, left, right = make_simulation(args.variant)
    keys = set()

    def on_key(key):
        # Passive viewer callbacks report presses. Each key sets a motion mode;
        # space stops the robot, avoiding reliance on key-release events.
        if key in (glfw.KEY_UP, glfw.KEY_W):
            keys.clear(); keys.add("forward")
        elif key in (glfw.KEY_DOWN, glfw.KEY_S):
            keys.clear(); keys.add("backward")
        elif key in (glfw.KEY_LEFT, glfw.KEY_A):
            keys.clear(); keys.add("left")
        elif key in (glfw.KEY_RIGHT, glfw.KEY_D):
            keys.clear(); keys.add("right")
        elif key == glfw.KEY_SPACE:
            keys.clear()

    print("TurtleBot3: arrows/WASD select forward, reverse, or turn; Space stops.")
    print("Frame colors: X red, Y green, Z blue. Close the viewer to exit.")
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base
        viewer.cam.distance = 1.4
        viewer.cam.elevation = -35
        while viewer.is_running():
            started = time.perf_counter()
            linear = LINEAR_SPEED * (int("forward" in keys) - int("backward" in keys))
            angular = ANGULAR_SPEED * (int("left" in keys) - int("right" in keys))
            data.ctrl[left], data.ctrl[right] = wheel_speeds(linear, angular, args.variant)
            mujoco.mj_step(model, data)
            position = data.xpos[base].copy()
            rotation = data.xmat[base].reshape(3, 3).copy()
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_frame(viewer.user_scn, position + np.array([0, 0, 0.14]), rotation)
                draw_frame(viewer.user_scn, position + np.array([0.25, -0.25, 0.14]), np.eye(3))
            viewer.sync()
            remaining = model.opt.timestep - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
