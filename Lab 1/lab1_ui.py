"""Single-window MuJoCo controls for Lab 1's Crazyflie and TurtleBot3.

Run ``.venv/bin/python lab1_ui.py`` or ``lab1_ui.py --check``.
"""

from __future__ import annotations

import argparse
import math
import time

import glfw
import imgui
import mujoco
import numpy as np
from imgui.integrations.glfw import GlfwRenderer

from lab1_drone import Drone, check as check_drone
from show_turtlebot3_frame import (
    WHEEL_RADIUS, WHEEL_SEPARATIONS, check_motion, draw_frame,
    make_simulation, wheel_speeds,
)


def kinematics(model, data, body):
    """World pose and the 6x6 world-frame free-body Jacobian."""
    position = data.xpos[body].copy()
    rotation = data.xmat[body].reshape(3, 3).copy()
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position
    jac_position = np.zeros((3, model.nv))
    jac_rotation = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jac_position, jac_rotation, body)
    joint = model.body_jntadr[body]
    dof = model.jnt_dofadr[joint]
    jacobian = np.vstack((jac_position, jac_rotation))[:, dof:dof + 6]
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1., 1.)))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return pose, jacobian, np.degrees([roll, pitch, yaw])


def matrix_rows(matrix):
    for row in matrix:
        imgui.text("  ".join(f"{value:+.3f}" for value in row))


class LabWindow:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize")
        self.window = glfw.create_window(1400, 780, "Lab 1 - MuJoCo robot control", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Could not create OpenGL window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        imgui.create_context()
        self.imgui = GlfwRenderer(self.window, attach_callbacks=False)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_char_callback(self.window, self.imgui.char_callback)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor)
        glfw.set_scroll_callback(self.window, self.on_scroll)
        self.old_cursor = None
        self.robot_index = 0
        self.variant = None
        self.pending_robot = None
        self.drone_velocity = [0., 0., 0., 0.]
        self.drone_was_driving = False
        self.linear = 0.
        self.angular = 0.
        self.status = "Hold keys or adjust live speed sliders."
        self.load_robot(0)

    def load_robot(self, index):
        if hasattr(self, "context"):
            del self.context
            del self.scene
        self.robot_index = index
        if index == 0:
            self.drone = Drone()
            self.drone_velocity = [0., 0., 0., 0.]
            self.drone_was_driving = False
            self.model, self.data, self.body = self.drone.model, self.drone.data, self.drone.body
        else:
            self.variant = "waffle" if index == 1 else "burger"
            self.model, self.data, self.body, self.left, self.right = make_simulation(self.variant)
            self.linear = self.angular = 0.
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = self.body
        self.camera.distance = 1.2 if index == 0 else 1.5
        self.camera.azimuth = 130
        self.camera.elevation = -25
        self.options = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)

    def on_key(self, window, key, scancode, action, mods):
        self.imgui.keyboard_callback(window, key, scancode, action, mods)
        if action != glfw.PRESS or imgui.get_io().want_capture_keyboard:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self.stop()

    def stop(self):
        self.drone_velocity = [0., 0., 0., 0.]
        self.linear = self.angular = 0.
        if self.robot_index == 0:
            self.drone.hold()
            self.drone_was_driving = False

    def held(self, key):
        return glfw.get_key(self.window, key) == glfw.PRESS

    def live_command(self):
        """Return held-key commands; releasing a key releases its command."""
        if imgui.get_io().want_capture_keyboard:
            return (0., 0., 0., 0.)
        if self.robot_index == 0:
            return (
                .15 * (self.held(glfw.KEY_W) or self.held(glfw.KEY_UP))
                - .15 * (self.held(glfw.KEY_S) or self.held(glfw.KEY_DOWN)),
                .15 * (self.held(glfw.KEY_A) or self.held(glfw.KEY_LEFT))
                - .15 * (self.held(glfw.KEY_D) or self.held(glfw.KEY_RIGHT)),
                .12 * self.held(glfw.KEY_R) - .12 * self.held(glfw.KEY_F),
                math.radians(70) * int(self.held(glfw.KEY_Q))
                - math.radians(70) * int(self.held(glfw.KEY_E)),
            )
        return (
            .12 * (self.held(glfw.KEY_W) or self.held(glfw.KEY_UP))
            - .12 * (self.held(glfw.KEY_S) or self.held(glfw.KEY_DOWN)),
            .8 * (self.held(glfw.KEY_A) or self.held(glfw.KEY_LEFT))
            - .8 * (self.held(glfw.KEY_D) or self.held(glfw.KEY_RIGHT)),
        )

    def on_cursor(self, window, x, y):
        if self.old_cursor is None:
            self.old_cursor = (x, y)
            return
        old_x, old_y = self.old_cursor
        self.old_cursor = (x, y)
        if imgui.get_io().want_capture_mouse:
            return
        _, height = glfw.get_window_size(window)
        if not height:
            return
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.model, action, (x-old_x)/height, (y-old_y)/height, self.camera)

    def on_scroll(self, window, x_offset, y_offset):
        self.imgui.scroll_callback(window, x_offset, y_offset)
        if not imgui.get_io().want_capture_mouse:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                                 0, -0.05*y_offset, self.camera)

    def panel(self):
        imgui.set_next_window_position(15, 15, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(340, 430, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Lab 1 controls")
        if expanded:
            changed, selected = imgui.combo("Robot", self.robot_index,
                                             ["Crazyflie 2 drone", "TurtleBot3 Waffle Pi", "TurtleBot3 Burger"])
            if changed:
                self.pending_robot = selected
            if self.robot_index == 0:
                for i, (label, lo, hi) in enumerate((
                    ("Forward (m/s)", -.15, .15), ("Left (m/s)", -.15, .15),
                    ("Up (m/s)", -.12, .12), ("Yaw rate (deg/s)", -70., 70.),
                )):
                    changed, value = imgui.slider_float(label, self.drone_velocity[i], lo, hi)
                    if changed:
                        self.drone_velocity[i] = value
                if imgui.button("Stop and hover"):
                    self.stop()
                imgui.text("Hold W/S forward; A/D left; R/F up/down; Q/E yaw")
                imgui.text("Arrow keys also move in the horizontal plane")
                pos, rotation, velocity = self.drone.state()
                yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
                imgui.text(f"Position: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f} m")
                imgui.text(f"Yaw: {yaw:.1f} deg | Speed: {np.linalg.norm(velocity):.2f} m/s")
            else:
                _, self.linear = imgui.slider_float("Linear (m/s)", self.linear, -.18, .18)
                _, self.angular = imgui.slider_float("Angular (rad/s)", self.angular, -1., 1.)
                if imgui.button("Stop"):
                    self.stop()
                imgui.text("Hold arrows/WASD to move; release to stop")
                pos = self.data.xpos[self.body]
                rotation = self.data.xmat[self.body].reshape(3, 3)
                yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
                imgui.text(f"Position: {pos[0]:.2f}, {pos[1]:.2f} m | Yaw: {yaw:.1f} deg")
            imgui.separator()
            imgui.text("World frame is offset from robot; body frame follows it.")
            imgui.text("X red | Y green | Z blue")
            imgui.text_wrapped(self.status)
        imgui.end()

    def matrices_panel(self):
        width, _ = glfw.get_window_size(self.window)
        imgui.set_next_window_position(max(350, width - 465), 15, imgui.ALWAYS)
        imgui.set_next_window_size(450, 745, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Live pose and matrices")
        if expanded:
            pose, jacobian, rpy = kinematics(self.model, self.data, self.body)
            imgui.text("World -> body homogeneous transform (m)")
            matrix_rows(pose)
            imgui.separator()
            imgui.text("Body rotation matrix R (world frame)")
            matrix_rows(pose[:3, :3])
            quat = self.data.xquat[self.body]
            imgui.text("Quaternion [w, x, y, z]: " + ", ".join(f"{x:+.3f}" for x in quat))
            imgui.text("Roll / pitch / yaw (deg): " + ", ".join(f"{x:+.1f}" for x in rpy))
            imgui.separator()
            imgui.text("Body spatial Jacobian, world frame (6 x 6)")
            imgui.text("Rows: vx vy vz wx wy wz; columns: free-joint qvel")
            matrix_rows(jacobian)
            if self.robot_index == 0:
                imgui.separator()
                imgui.text("Drone target [x, y, z] (m)")
                imgui.text(", ".join(f"{x:+.3f}" for x in self.drone.target))
                imgui.text(f"Target yaw: {math.degrees(self.drone.yaw):+.1f} deg")
                imgui.text("Actuator controls [thrust, Mx, My, Mz]")
                imgui.text(", ".join(f"{self.data.ctrl[i]:+.3f}" for i in self.drone.ids))
            else:
                imgui.separator()
                track = WHEEL_SEPARATIONS[self.variant]
                wheel_jacobian = np.array([[WHEEL_RADIUS/2, WHEEL_RADIUS/2],
                                           [-WHEEL_RADIUS/track, WHEEL_RADIUS/track]])
                imgui.text("Ideal wheel Jacobian [v, yaw rate] / [wL, wR]")
                matrix_rows(wheel_jacobian)
                left_joint = self.model.joint("wheel_left").id
                right_joint = self.model.joint("wheel_right").id
                angles = [self.data.qpos[self.model.jnt_qposadr[j]] for j in (left_joint, right_joint)]
                rates = [self.data.qvel[self.model.jnt_dofadr[j]] for j in (left_joint, right_joint)]
                imgui.text("Wheel angles L/R (rad): " + ", ".join(f"{x:+.2f}" for x in angles))
                imgui.text("Wheel rates L/R (rad/s): " + ", ".join(f"{x:+.2f}" for x in rates))
        imgui.end()

    def run(self):
        previous = time.monotonic()
        remainder = 0.
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                now = time.monotonic()
                remainder += min(now - previous, .05)
                previous = now
                while remainder >= self.model.opt.timestep:
                    if self.robot_index == 0:
                        keys = self.live_command()
                        command = [
                            self.drone_velocity[i] + keys[i] * (180 / math.pi if i == 3 else 1)
                            for i in range(4)
                        ]
                        active = any(abs(value) > 1e-5 for value in command)
                        if active:
                            self.drone.drive(command[0], command[1], command[2],
                                             math.radians(command[3]), self.model.opt.timestep)
                        elif self.drone_was_driving:
                            self.drone.hold()
                        self.drone_was_driving = active
                        self.drone.step()
                    else:
                        keys = self.live_command()
                        linear = float(np.clip(self.linear + keys[0], -.18, .18))
                        angular = float(np.clip(self.angular + keys[1], -1., 1.))
                        self.data.ctrl[self.left], self.data.ctrl[self.right] = wheel_speeds(linear, angular, self.variant)
                        mujoco.mj_step(self.model, self.data)
                    remainder -= self.model.opt.timestep
                self.imgui.process_inputs()
                imgui.new_frame()
                self.panel()
                self.matrices_panel()
                width, height = glfw.get_framebuffer_size(self.window)
                if width and height:
                    mujoco.mjv_updateScene(self.model, self.data, self.options, None,
                                          self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                    pos = self.data.xpos[self.body].copy()
                    rotation = self.data.xmat[self.body].reshape(3, 3).copy()
                    rise = .12 if self.robot_index == 1 else 0.
                    draw_frame(self.scene, pos + [0, 0, rise], rotation)
                    draw_frame(self.scene, pos + [.25, -.25, rise], np.eye(3))
                    mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), self.scene, self.context)
                    imgui.render()
                    self.imgui.render(imgui.get_draw_data())
                    glfw.swap_buffers(self.window)
                else:
                    time.sleep(.05)
                if self.pending_robot is not None:
                    self.load_robot(self.pending_robot)
                    self.pending_robot = None
                    remainder = 0.
                    previous = time.monotonic()
        finally:
            self.imgui.shutdown()
            del self.context
            del self.scene
            glfw.destroy_window(self.window)
            glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run both model checks without a GUI")
    args = parser.parse_args()
    if args.check:
        check_drone()
        check_motion("waffle")
        check_motion("burger")
        for index in (0, 1, 2):
            if index == 0:
                robot = Drone()
                model, data, body = robot.model, robot.data, robot.body
            else:
                model, data, body, _, _ = make_simulation("waffle" if index == 1 else "burger")
            pose, jacobian, rpy = kinematics(model, data, body)
            assert pose.shape == (4, 4) and jacobian.shape == (6, 6)
            assert np.all(np.isfinite(pose)) and np.all(np.isfinite(jacobian))
    else:
        LabWindow().run()


if __name__ == "__main__":
    main()
