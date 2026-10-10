"""Headless checks for the instructor-supplied model integration."""

import unittest

from robot import RobotArm


class FrankaTests(unittest.TestCase):
    def test_supplied_scene_and_position_control(self) -> None:
        arm = RobotArm("franka")
        self.assertEqual(len(arm.joint_names), 7)
        self.assertEqual(arm.spec.xml, "franka/scene.xml")
        self.assertEqual(arm.spec.control, "position")
        arm.set_target_degrees([15, -20, 10, -100, 20, 100, -30])
        for _ in range(int(4 / arm.model.opt.timestep)):
            arm.step()
        errors = [
            abs(actual - target)
            for actual, target in zip(arm.current_degrees(), arm.target_degrees())
        ]
        self.assertLess(max(errors), 1.0)

    def test_rejects_out_of_range_joint(self) -> None:
        arm = RobotArm("franka")
        with self.assertRaisesRegex(ValueError, "joint4"):
            arm.set_target_degrees([0, 0, 0, 0, 0, 90, 0])


class HealTests(unittest.TestCase):
    def test_six_joint_fk_motion(self) -> None:
        arm = RobotArm("heal")
        self.assertEqual(len(arm.joint_names), 6)
        self.assertEqual(arm.spec.control, "kinematic")
        arm.set_target_degrees([90, -90, 90, 90, -90, 90])
        for _ in range(int(2 / arm.model.opt.timestep)):
            arm.step()
        errors = [
            abs(actual - target)
            for actual, target in zip(arm.current_degrees(), arm.target_degrees())
        ]
        self.assertLess(max(errors), 0.01)

    def test_invalid_input_and_home(self) -> None:
        arm = RobotArm("heal")
        with self.assertRaisesRegex(ValueError, "joint_1"):
            arm.set_target_degrees([200, 0, 0, 0, 0, 0])
        arm.set_target_degrees([20, 0, 0, 0, 0, 0])
        arm.go_home()
        self.assertEqual(arm.target_degrees(), [0] * 6)


if __name__ == "__main__":
    unittest.main()
