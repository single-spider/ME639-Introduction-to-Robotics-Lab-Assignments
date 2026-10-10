"""Single-window MuJoCo renderer with in-window joint controls.

This uses MuJoCo's classic OpenGL rendering API and Dear ImGui in one GLFW
window, avoiding the separate Tk/browser UI and the WSLg Studio renderer.
"""

from __future__ import annotations

import argparse
import math
import time

import glfw
import imgui
import mujoco
from imgui.integrations.glfw import GlfwRenderer

from robot import ROBOTS, RobotArm


class JointWindow:
    def __init__(self, robot: str = "franka") -> None:
        self.robot_keys = list(ROBOTS)
        self.robot_index = self.robot_keys.index(robot)
        self.pending_robot: str | None = None
        self.arm = RobotArm(robot)
        self.values = self.arm.target_degrees()
        self.home = self.values.copy()
        self.limits = self.arm.limit_degrees()
        self.status = "Enter angles or drag sliders, then click Move."
        self.last_cursor: tuple[float, float] | None = None

        if not glfw.init():
            raise RuntimeError("GLFW could not initialize. Check WSLg / DISPLAY.")
        self.window = glfw.create_window(
            1150, 730, "Robot joint control - MuJoCo", None, None
        )
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Could not create the OpenGL window.")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        imgui.create_context()
        imgui.get_io().font_global_scale = 1.1
        self.imgui = GlfwRenderer(self.window, attach_callbacks=False)
        glfw.set_key_callback(self.window, self.on_key)
        glfw.set_char_callback(self.window, self.imgui.char_callback)
        glfw.set_cursor_pos_callback(self.window, self.on_cursor)
        glfw.set_scroll_callback(self.window, self.on_scroll)

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = self.arm.model.stat.center
        self.camera.distance = 2.5 * self.arm.model.stat.extent
        self.camera.azimuth = 90
        self.camera.elevation = -20
        self.options = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.arm.model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            self.arm.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)

    def switch_robot(self, robot: str) -> None:
        """Replace the model and MuJoCo rendering resources in the same window."""
        if robot == self.arm.spec.key:
            return
        new_arm = RobotArm(robot)
        del self.context
        del self.scene
        self.arm = new_arm
        self.robot_index = self.robot_keys.index(robot)
        self.values = self.arm.target_degrees()
        self.home = self.values.copy()
        self.limits = self.arm.limit_degrees()
        self.status = "Robot loaded. Enter target angles, then click Move."
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = self.arm.model.stat.center
        self.camera.distance = 2.5 * self.arm.model.stat.extent
        self.camera.azimuth = 90
        self.camera.elevation = -20
        self.options = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.arm.model, maxgeom=10000)
        self.context = mujoco.MjrContext(
            self.arm.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        glfw.set_window_title(self.window, f"{self.arm.spec.label} - MuJoCo joint control")

    def on_key(self, window: object, key: int, scancode: int, action: int, mods: int) -> None:
        self.imgui.keyboard_callback(window, key, scancode, action, mods)
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            if not imgui.get_io().want_capture_keyboard:
                glfw.set_window_should_close(self.window, True)

    def on_cursor(self, _window: object, x: float, y: float) -> None:
        if self.last_cursor is None:
            self.last_cursor = (x, y)
            return
        old_x, old_y = self.last_cursor
        self.last_cursor = (x, y)
        if imgui.get_io().want_capture_mouse:
            return
        _, height = glfw.get_window_size(self.window)
        if not height:
            return
        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        else:
            return
        mujoco.mjv_moveCamera(
            self.arm.model, action, (x - old_x) / height, (y - old_y) / height,
            self.camera
        )

    def on_scroll(self, window: object, x_offset: float, y_offset: float) -> None:
        self.imgui.scroll_callback(window, x_offset, y_offset)
        if not imgui.get_io().want_capture_mouse:
            mujoco.mjv_moveCamera(
                self.arm.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset,
                self.camera
            )

    def validate(self) -> str | None:
        for name, value, (lo, hi) in zip(self.arm.joint_names, self.values, self.limits):
            if not math.isfinite(value) or not lo <= value <= hi:
                return f"{name}: use {lo:.1f} to {hi:.1f} degrees."
        return None

    def draw_panel(self) -> None:
        imgui.set_next_window_position(15, 15, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(380, 680, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Joint configuration")
        if expanded:
            changed, selected = imgui.combo(
                "Robot", self.robot_index, [ROBOTS[key].label for key in self.robot_keys]
            )
            if changed:
                self.pending_robot = self.robot_keys[selected]
            imgui.text(self.arm.spec.label)
            if self.arm.spec.key == "franka":
                imgui.text("Position-actuated; gripper stays open.")
            else:
                imgui.text("Kinematic FK control; 6 joint angles.")
            imgui.text("Type a value or drag its slider.")
            imgui.separator()
            actuals = self.arm.current_degrees()
            for i, name in enumerate(self.arm.joint_names):
                lo, hi = self.limits[i]
                imgui.text(f"{name} [{lo:.0f}, {hi:.0f}]  actual {actuals[i]:.2f}")
                imgui.push_item_width(145)
                changed, value = imgui.input_float(
                    f"##target_{name}", self.values[i], 1.0, 10.0, "%.2f"
                )
                if changed:
                    self.values[i] = value
                    self.status = "Target edited. Click Move to apply."
                imgui.pop_item_width()
                imgui.same_line()
                imgui.push_item_width(160)
                changed, value = imgui.slider_float(
                    f"##range_{name}", self.values[i], lo, hi, "%.1f"
                )
                if changed:
                    self.values[i] = value
                    self.status = "Target edited. Click Move to apply."
                imgui.pop_item_width()
                imgui.separator()
            if imgui.button("Move to configuration"):
                error = self.validate()
                if error:
                    self.status = error
                else:
                    self.arm.set_target_degrees(self.values)
                    self.status = "Moving..."
            imgui.same_line()
            if imgui.button("Return home"):
                self.arm.go_home()
                self.values = self.home.copy()
                self.status = "Returning home..."
            if all(abs(a - b) < 0.01 for a, b in zip(self.values, self.arm.target_degrees())) and all(
                abs(a - b) < 1.0
                for a, b in zip(actuals, self.arm.target_degrees())
            ):
                self.status = "Configuration reached (within 1 deg)."
            imgui.text_wrapped(self.status)
        imgui.end()

    def run(self) -> None:
        print("Single MuJoCo window with in-window angle inputs is open.", flush=True)
        last_time = time.monotonic()
        remainder = 0.0
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                now = time.monotonic()
                remainder += min(now - last_time, 0.05)
                last_time = now
                timestep = self.arm.model.opt.timestep
                while remainder >= timestep:
                    self.arm.step()
                    remainder -= timestep

                self.imgui.process_inputs()
                imgui.new_frame()
                self.draw_panel()

                width, height = glfw.get_framebuffer_size(self.window)
                if width and height:
                    viewport = mujoco.MjrRect(0, 0, width, height)
                    mujoco.mjv_updateScene(
                        self.arm.model, self.arm.data, self.options, None,
                        self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
                    )
                    mujoco.mjr_render(viewport, self.scene, self.context)
                    imgui.render()
                    self.imgui.render(imgui.get_draw_data())
                    glfw.swap_buffers(self.window)
                else:
                    time.sleep(0.05)
                if self.pending_robot is not None:
                    robot = self.pending_robot
                    self.pending_robot = None
                    self.switch_robot(robot)
                    remainder = 0.0
                    last_time = time.monotonic()
        except KeyboardInterrupt:
            pass
        finally:
            self.imgui.shutdown()
            del self.context
            del self.scene
            glfw.destroy_window(self.window)
            glfw.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-robot in-window joint control")
    parser.add_argument("--check", action="store_true", help="test model without a window")
    parser.add_argument("--robot", choices=tuple(ROBOTS), default="franka")
    args = parser.parse_args()
    if args.check:
        arm = RobotArm(args.robot)
        targets = {
            "franka": [15, -20, 10, -100, 20, 100, -30],
            "heal": [20, -30, 15, 40, -25, 35],
        }
        arm.set_target_degrees(targets[args.robot])
        for _ in range(int(4 / arm.model.opt.timestep)):
            arm.step()
        error = max(
            abs(a - b)
            for a, b in zip(arm.current_degrees(), arm.target_degrees())
        )
        print(f"Maximum joint error: {error:.2f} degrees")
        if error >= 2:
            raise SystemExit("Position-control check failed.")
        print("PASS")
    else:
        JointWindow(args.robot).run()


if __name__ == "__main__":
    main()
