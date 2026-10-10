"""Joint-configuration control for the two instructor-supplied MuJoCo arms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import mujoco


MODELS = Path(__file__).resolve().parent / "models"
MAX_SETPOINT_SPEED = math.radians(60.0)  # rad/s


@dataclass(frozen=True)
class RobotSpec:
    key: str
    label: str
    xml: str
    joints: tuple[str, ...]
    actuators: tuple[str, ...]
    home: tuple[float, ...]  # radians
    control: str


ROBOTS = {
    "franka": RobotSpec(
        key="franka",
        label="Franka Panda (7 DoF)",
        xml="franka/scene.xml",
        joints=tuple(f"joint{i}" for i in range(1, 8)),
        actuators=tuple(f"actuator{i}" for i in range(1, 8)),
        home=(0, 0, 0, -1.57079, 0, 1.57079, -0.7853),
        control="position",
    ),
    "heal": RobotSpec(
        key="heal",
        label="Heal (6 DoF)",
        xml="single_arm_heal_effort_actuation_rs_mj.xml",
        joints=tuple(f"joint_{i}" for i in range(1, 7)),
        actuators=("turret", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"),
        home=(0, 0, 0, 0, 0, 0),
        control="kinematic",
    ),
}


class RobotArm:
    def __init__(self, robot: str = "franka") -> None:
        if robot not in ROBOTS:
            raise ValueError(f"Unknown robot {robot!r}; choose one of {tuple(ROBOTS)}")
        self.spec = ROBOTS[robot]
        self.model = mujoco.MjModel.from_xml_path(str(MODELS / self.spec.xml))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [self.model.joint(name).id for name in self.spec.joints]
        self.actuator_ids = [self.model.actuator(name).id for name in self.spec.actuators]
        self.qpos_ids = [int(self.model.jnt_qposadr[j]) for j in self.joint_ids]
        self.limits = [tuple(map(float, self.model.jnt_range[j])) for j in self.joint_ids]

        self.home = list(self.spec.home)
        self.target = self.home.copy()
        self.command = self.home.copy()
        for qpos_id, angle in zip(self.qpos_ids, self.home):
            self.data.qpos[qpos_id] = angle
        if self.spec.key == "franka":
            # The supplied Panda scene has two coupled finger joints and one
            # gripper actuator. Keep the fingers open for this FK assignment.
            self.data.qpos[7:9] = 0.04
            self.data.ctrl[self.model.actuator("actuator8").id] = 255.0
            for actuator_id, angle in zip(self.actuator_ids, self.home):
                self.data.ctrl[actuator_id] = angle
        mujoco.mj_forward(self.model, self.data)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.spec.joints

    def limit_degrees(self) -> list[tuple[float, float]]:
        return [(math.degrees(lo), math.degrees(hi)) for lo, hi in self.limits]

    def current_degrees(self) -> list[float]:
        return [math.degrees(float(self.data.qpos[i])) for i in self.qpos_ids]

    def target_degrees(self) -> list[float]:
        return [math.degrees(q) for q in self.target]

    def set_target_degrees(self, values: Sequence[float]) -> None:
        if len(values) != len(self.joint_names):
            raise ValueError(f"Enter {len(self.joint_names)} joint angles.")
        angles = [math.radians(float(value)) for value in values]
        for name, angle, (lower, upper) in zip(self.joint_names, angles, self.limits):
            if not math.isfinite(angle):
                raise ValueError(f"{name}: angle must be a finite number.")
            if not lower - 1e-5 <= angle <= upper + 1e-5:
                raise ValueError(
                    f"{name}: use {math.degrees(lower):.1f}° to "
                    f"{math.degrees(upper):.1f}°."
                )
        self.target = [
            max(lower, min(upper, angle))
            for angle, (lower, upper) in zip(angles, self.limits)
        ]

    def go_home(self) -> None:
        self.target = self.home.copy()

    def step(self) -> None:
        """Advance one model timestep toward the requested joint configuration.

        Franka uses its supplied position actuators and physics. Heal's supplied
        actuators are torque motors, so for the FK lab we interpolate qpos and
        call mj_forward; this reaches arbitrary valid joint configurations.
        """
        max_delta = MAX_SETPOINT_SPEED * self.model.opt.timestep
        for index in range(len(self.joint_names)):
            delta = self.target[index] - self.command[index]
            self.command[index] += max(-max_delta, min(max_delta, delta))

        if self.spec.control == "position":
            for actuator_id, angle in zip(self.actuator_ids, self.command):
                self.data.ctrl[actuator_id] = angle
            mujoco.mj_step(self.model, self.data)
        else:
            for qpos_id, angle in zip(self.qpos_ids, self.command):
                self.data.qpos[qpos_id] = angle
            self.data.qvel[:] = 0.0
            self.data.time += self.model.opt.timestep
            mujoco.mj_forward(self.model, self.data)
