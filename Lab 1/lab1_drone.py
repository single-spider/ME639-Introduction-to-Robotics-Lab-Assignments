"""Testable Crazyflie position controller for the Lab 1 MuJoCo scene."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np


SCENE = Path(__file__).parent / "mujoco_menagerie/bitcraze_crazyflie_2/scene.xml"


class Drone:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.body = self.model.body("cf2").id
        self.ids = [self.model.actuator(name).id for name in
                    ("body_thrust", "x_moment", "y_moment", "z_moment")]
        self.gyro = self.model.sensor("body_gyro").adr[0]
        joint = self.model.body_jntadr[self.body]
        self.qpos = self.model.jnt_qposadr[joint]
        self.dof = self.model.jnt_dofadr[joint]
        self.mass = float(self.model.body_mass[self.body])
        self.gravity = abs(float(self.model.opt.gravity[2]))
        # The original model has little rotational authority for keyboard flight.
        self.model.actuator_gear[self.ids[1:]] *= 4.0
        self.data.qpos[self.qpos:self.qpos+3] = [0., 0., 0.5]
        self.data.qpos[self.qpos+3:self.qpos+7] = [1., 0., 0., 0.]
        mujoco.mj_forward(self.model, self.data)
        self.target = np.array([0., 0., 0.5])
        self.yaw = 0.
        self.velocity_command = None

    def state(self):
        position = self.data.xpos[self.body].copy()
        rotation = self.data.xmat[self.body].reshape(3, 3).copy()
        velocity = self.data.qvel[self.dof:self.dof+3].copy()
        return position, rotation, velocity

    def set_target(self, x: float, y: float, z: float, yaw_deg: float):
        self.velocity_command = None
        self.target[:] = np.clip([x, y, z], [-0.5, -0.5, 0.25], [0.5, 0.5, 0.9])
        self.yaw = math.radians(yaw_deg)

    def hold(self):
        """Brake toward the current pose after a live command is released."""
        position, rotation, _ = self.state()
        self.velocity_command = None
        self.target[:] = position
        self.yaw = math.atan2(rotation[1, 0], rotation[0, 0])

    def drive(self, forward: float, left: float, up: float, yaw_rate: float, dt: float):
        """Integrate live body-relative velocity commands into the pose target."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        world_velocity = np.array([c * forward - s * left,
                                   s * forward + c * left, up])
        position, _, _ = self.state()
        # Stop commanding velocity at the defined flying-volume boundary.
        for axis, (low, high) in enumerate(((-.5, .5), (-.5, .5), (.25, .9))):
            if position[axis] <= low and world_velocity[axis] < 0:
                world_velocity[axis] = 0.
            if position[axis] >= high and world_velocity[axis] > 0:
                world_velocity[axis] = 0.
        self.velocity_command = world_velocity
        self.target[:] = position
        self.yaw = (self.yaw + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi

    def step(self):
        pos, rotation, vel = self.state()
        if self.velocity_command is None:
            desired_vel = np.array([0.7, 0.7, 1.8]) * (self.target - pos)
        else:
            desired_vel = self.velocity_command.copy()
        xy_speed = np.linalg.norm(desired_vel[:2])
        if xy_speed > 0.18:
            desired_vel[:2] *= 0.18 / xy_speed
        desired_vel[2] = np.clip(desired_vel[2], -0.2, 0.2)
        desired_acc = np.array([1.2, 1.2, 3.0]) * (desired_vel - vel)
        xy_acc = np.linalg.norm(desired_acc[:2])
        if xy_acc > 0.35:
            desired_acc[:2] *= 0.35 / xy_acc
        desired_acc[2] = np.clip(desired_acc[2], -1., 1.)
        force = self.mass * (desired_acc + [0., 0., self.gravity])
        body_z = force / np.linalg.norm(force)
        heading = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.])
        body_y = np.cross(body_z, heading)
        body_y /= np.linalg.norm(body_y)
        body_x = np.cross(body_y, body_z)
        desired_rotation = np.column_stack((body_x, body_y, body_z))
        skew = 0.5 * (desired_rotation.T @ rotation - rotation.T @ desired_rotation)
        error = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        gyro = self.data.sensordata[self.gyro:self.gyro+3]
        torque = -np.array([2e-4, 2e-4, 1.5e-4]) * error - np.array([9e-5, 9e-5, 7e-5]) * gyro
        thrust = max(0., float(force @ rotation[:, 2]))
        values = [thrust / self.model.actuator_gear[self.ids[0], 2]]
        values += [torque[i] / self.model.actuator_gear[self.ids[i+1], i+3] for i in range(3)]
        for actuator, value in zip(self.ids, values):
            lo, hi = self.model.actuator_ctrlrange[actuator]
            self.data.ctrl[actuator] = np.clip(value, lo, hi)
        mujoco.mj_step(self.model, self.data)


def check():
    drone = Drone()
    dt = drone.model.opt.timestep
    for _ in range(round(3 / dt)):
        drone.step()
    hover, _, _ = drone.state()
    print("hover", np.round(hover, 3))
    if np.linalg.norm(hover - [0, 0, 0.5]) > 0.08:
        raise AssertionError("hover did not settle")
    drone.set_target(0.2, 0, 0.5, 0)
    for _ in range(round(5 / dt)):
        drone.step()
    moved, _, _ = drone.state()
    print("move", np.round(moved, 3))
    if moved[0] < 0.12 or abs(moved[1]) > 0.1:
        raise AssertionError("position command did not work")
    drone.set_target(0.2, 0, 0.5, 45)
    for _ in range(round(5 / dt)):
        drone.step()
    _, rotation, _ = drone.state()
    yaw = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    print("yaw", round(yaw, 1))
    if abs(yaw - 45) > 15:
        raise AssertionError("yaw command did not work")
    live = Drone()
    for _ in range(round(2 / live.model.opt.timestep)):
        live.drive(0.15, 0, 0, 0, live.model.opt.timestep)
        live.step()
    before_release = live.state()[0][0]
    live.hold()
    for _ in range(round(4 / live.model.opt.timestep)):
        live.step()
    after_release, _, release_velocity = live.state()
    print("live move/release", round(before_release, 3),
          round(after_release[0], 3), round(release_velocity[0], 3))
    if before_release < 0.1 or abs(release_velocity[0]) > 0.04:
        raise AssertionError("live drive or release-to-hover failed")


if __name__ == "__main__":
    check()
