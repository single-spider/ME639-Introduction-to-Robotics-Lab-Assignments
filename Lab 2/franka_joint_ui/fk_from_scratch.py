"""Forward kinematics from the supplied XML, without MuJoCo's FK routines.

Usage: python fk_from_scratch.py franka 0 0 0 -90 0 90 -45
       python fk_from_scratch.py heal 0 0 0 0 0 0
       python fk_from_scratch.py --verify
Angles are in degrees. Positions are in metres in the model world frame.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robot import MODELS, ROBOTS


SOURCES = {
    "franka": (MODELS / "franka/panda.xml", "hand"),
    "heal": (MODELS / "single_arm_heal_effort_actuation_rs_mj.xml", "end_effector"),
}


def xyz(value: str | None, default=(0., 0., 0.)) -> np.ndarray:
    return np.fromstring(value, sep=" ") if value is not None else np.array(default)


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    v = 1 - c
    return np.array([
        [c+x*x*v, x*y*v-z*s, x*z*v+y*s],
        [y*x*v+z*s, c+y*y*v, y*z*v-x*s],
        [z*x*v-y*s, z*y*v+x*s, c+z*z*v],
    ])


def static_rotation(body: ET.Element) -> np.ndarray:
    if "quat" in body.attrib:
        w, x, y, z = np.fromstring(body.attrib["quat"], sep=" ")
        w, x, y, z = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])
    if "euler" in body.attrib:
        rx, ry, rz = xyz(body.attrib["euler"])
        return (axis_rotation(np.array([1., 0., 0.]), rx)
                @ axis_rotation(np.array([0., 1., 0.]), ry)
                @ axis_rotation(np.array([0., 0., 1.]), rz))
    return np.eye(3)


def transform(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


def body_chain(root: ET.Element, target: str) -> list[ET.Element]:
    def visit(body: ET.Element, parents: list[ET.Element]):
        chain = parents + [body]
        if body.get("name") == target:
            return chain
        for child in body.findall("body"):
            found = visit(child, chain)
            if found is not None:
                return found
        return None

    for body in root.findall("./worldbody/body"):
        found = visit(body, [])
        if found is not None:
            return found
    raise ValueError(f"No body named {target!r}")


def forward_kinematics(robot: str, angles_deg: list[float]) -> dict[str, np.ndarray]:
    """Return world-frame joint/body poses and end-effector pose."""
    spec = ROBOTS[robot]
    if len(angles_deg) != len(spec.joints):
        raise ValueError(f"{robot} requires {len(spec.joints)} joint angles")
    angles = dict(zip(spec.joints, map(math.radians, angles_deg)))
    source, end_body = SOURCES[robot]
    chain = body_chain(ET.parse(source).getroot(), end_body)
    result: dict[str, np.ndarray] = {}
    pose = np.eye(4)
    for body in chain:
        pose = pose @ transform(static_rotation(body), xyz(body.get("pos")))
        for joint in body.findall("joint"):
            name = joint.get("name")
            if name not in angles:
                continue
            pivot = xyz(joint.get("pos"))
            axis = xyz(joint.get("axis"), (0., 0., 1.))
            pose = (pose @ transform(np.eye(3), pivot)
                    @ transform(axis_rotation(axis, angles[name]), np.zeros(3))
                    @ transform(np.eye(3), -pivot))
            result[name] = pose.copy()
        result[body.get("name", "unnamed")] = pose.copy()
    result["end_effector"] = pose.copy()
    return result


def verify() -> None:
    """Use MuJoCo only as an independent reference for the manual calculation."""
    import mujoco
    samples = {
        "franka": [[0, 0, 0, -90, 0, 90, -45], [15, -20, 10, -100, 20, 100, -30]],
        "heal": [[0]*6, [30, -35, 40, 15, -20, 25]],
    }
    for robot, configurations in samples.items():
        spec = ROBOTS[robot]
        _, end_body = SOURCES[robot]
        model = mujoco.MjModel.from_xml_path(str(MODELS / spec.xml))
        data = mujoco.MjData(model)
        for angles in configurations:
            for name, angle in zip(spec.joints, angles):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = math.radians(angle)
            mujoco.mj_forward(model, data)
            expected = transform(data.body(end_body).xmat.reshape(3, 3), data.body(end_body).xpos)
            actual = forward_kinematics(robot, angles)["end_effector"]
            error = float(np.max(np.abs(actual - expected)))
            print(f"{robot} {angles}: maximum pose matrix error {error:.3g}")
            if error > 1e-4:
                raise AssertionError(f"FK mismatch for {robot}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", nargs="?", choices=tuple(ROBOTS))
    parser.add_argument("angles", nargs="*", type=float)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
        return
    if args.robot is None:
        parser.error("choose franka or heal, or use --verify")
    poses = forward_kinematics(args.robot, args.angles)
    for name in ROBOTS[args.robot].joints:
        print(f"{name}: position {np.round(poses[name][:3, 3], 5)}")
        print(np.round(poses[name][:3, :3], 5))
    print("End effector pose:\n", np.round(poses["end_effector"], 5))


if __name__ == "__main__":
    main()
