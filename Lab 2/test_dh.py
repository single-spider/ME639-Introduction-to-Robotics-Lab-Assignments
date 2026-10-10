"""Compare the DH calculations with the supplied MuJoCo arm models."""

import math
import unittest

import mujoco
import numpy as np

from franka_joint_ui.robot import MODELS
from python_script_for_dh_tran import franka_7dof, heal_6dof


class DHValidation(unittest.TestCase):
    def compare(self, robot, configurations):
        filename = ("franka/scene.xml" if robot == "franka"
                    else "single_arm_heal_effort_actuation_rs_mj.xml")
        model = mujoco.MjModel.from_xml_path(str(MODELS / filename))
        for angles in configurations:
            with self.subTest(robot=robot, angles=angles):
                data = mujoco.MjData(model)
                for i, angle in enumerate(angles, 1):
                    name = f"joint{i}" if robot == "franka" else f"joint_{i}"
                    data.qpos[model.jnt_qposadr[model.joint(name).id]] = math.radians(angle)
                mujoco.mj_forward(model, data)
                body = data.body("hand" if robot == "franka" else "end_effector")
                reference = np.eye(4)
                reference[:3, :3] = body.xmat.reshape(3, 3)
                reference[:3, 3] = body.xpos
                if robot == "franka":
                    reference[:3, 3] -= data.body("link0").xpos
                    calculated = franka_7dof(angles, verbose=False)
                else:
                    calculated = heal_6dof(angles, verbose=False)
                np.testing.assert_allclose(calculated, reference, atol=1e-6)

    def test_franka(self):
        self.compare("franka", [[0, 0, 0, -90, 0, 90, -45],
                                [15, -20, 10, -100, 20, 100, -30]])

    def test_heal(self):
        self.compare("heal", [[0] * 6, [30, -35, 40, 15, -20, 25],
                              [90, -90, 90, 90, -90, 90]])


if __name__ == "__main__":
    unittest.main()
