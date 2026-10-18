"""Physical, table-local layout trials with per-physics-step instrumentation."""
import argparse
from dataclasses import asdict
import json
import platform
import time
import mujoco
import numpy as np
from ik import Solver, target_rotation
from layout import BASELINE, Layout, TRAY_XY, grasp_yaw, scene_for_seed, source_hash
from run import Episode
from scene import ROOT


def settled_success(layout, cube_xyz, linear_velocity):
    p = layout.local(cube_xyz)
    return bool(abs(p[0] - TRAY_XY[0]) < .052 and abs(p[1] - TRAY_XY[1]) < .047
                and .022 < p[2] < .040 and np.linalg.norm(linear_velocity) < .03)


class LayoutEpisode(Episode):
    def __init__(self, robot, method, seed, layout=BASELINE, record=None):
        # Baseline class supplies the unchanged stage-reaching method. Build a
        # fresh transformed model instead of mutating a running simulation.
        self.scene, self.data, self.initial_local, self.initial = scene_for_seed(robot, layout, seed)
        self.layout, self.method, self.seed = layout, method, seed
        self.solver = Solver(self.scene, method)
        self.lifted = False
        self.record = record
        self.writer = self.renderer = None
        self.steps = self.joint_violations = self.velocity_violations = 0
        self.obstacle_contact_samples = 0
        self.min_clearance = 1.
        self.contact_pairs = {}
        self.trace = []
        self.stage = 'initialization'
        self.max_joint_speed = 0.
        self.joint_travel = 0.
        self.last_q = self.data.qpos[self.scene.qids].copy()
        self.state_samples = []
        m = self.scene.model
        self.obstacle_ids = {m.geom(n).id for n in ('floor', 'table', 'tray_bottom',
                             'tray_wall0', 'tray_wall1', 'tray_wall2', 'tray_wall3')}
        if record:
            import imageio.v2 as imageio
            self.renderer = mujoco.Renderer(m, height=480, width=640)
            self.writer = imageio.get_writer(str(record), fps=25, codec='libx264', quality=7)
            self.camera = mujoco.MjvCamera()
            self.camera.lookat[:] = [layout.center_x / 2, layout.center_y / 2, layout.height + .15]
            self.camera.distance = 1.8
            self.camera.azimuth = 135
            self.camera.elevation = -25

    def observe(self):
        s, d = self.scene, self.data
        q = d.qpos[s.qids]
        speed = float(np.max(np.abs(d.qvel[s.dids])))
        self.max_joint_speed = max(self.max_joint_speed, speed)
        self.joint_travel += float(np.linalg.norm(q - self.last_q, ord=1))
        self.last_q = q.copy()
        self.joint_violations += int(np.any((q < s.limits[:, 0] - .001) | (q > s.limits[:, 1] + .001)))
        self.velocity_violations += int(speed > 1.05)
        forbidden = False
        for c in d.contact:
            if c.dist > 0:
                continue
            a, b = int(c.geom1), int(c.geom2)
            obstacle = a if a in self.obstacle_ids else b if b in self.obstacle_ids else None
            if obstacle is None:
                continue
            other = b if obstacle == a else a
            bodyid = int(s.model.geom_bodyid[other])
            # Fixed robot-base/floor mounting and static environment contacts
            # are expected supports, not moving-arm obstacle collisions.
            if s.model.body_weldid[bodyid] == 0 or s.model.body(bodyid).name == 'cube':
                continue
            forbidden = True
            label = f'{s.model.body(bodyid).name}/{s.model.geom(obstacle).name}'
            self.contact_pairs[label] = self.contact_pairs.get(label, 0) + 1
        self.obstacle_contact_samples += int(forbidden)
        for k in range(4):
            dist = mujoco.mj_geomDistance(s.model, d, s.model.geom('cube_geom').id,
                                         s.model.geom(f'tray_wall{k}').id, 1., None)
            self.min_clearance = min(self.min_clearance, float(dist))

    def control(self, velocity, grip):
        # Same tracking laws as preserved run.py. Only observations are made
        # more frequent; this is not a controller tuning experiment.
        s, d = self.scene, self.data
        if s.robot == 'heal':
            M = np.zeros((s.model.nv, s.model.nv))
            mujoco.mj_fullM(s.model, d, M)
            accel = (velocity - d.qvel[s.dids]) / .04
            tau = M[np.ix_(s.dids, s.dids)] @ accel + d.qfrc_bias[s.dids]
            tau += s.model.dof_damping[s.dids] * velocity
            tau += s.model.dof_frictionloss[s.dids] * np.tanh(velocity / .005)
            d.ctrl[s.aids] = np.clip(tau, -150, 150)
        else:
            kp = s.model.actuator_gainprm[s.aids, 0]
            kd = -s.model.actuator_biasprm[s.aids, 2]
            d.ctrl[s.aids] = np.clip(d.qpos[s.qids] + (kd / kp + .01) * velocity,
                                    s.limits[:, 0], s.limits[:, 1])
            d.qfrc_applied[s.dids] = d.qfrc_bias[s.dids]
        d.ctrl[s.grip] = grip
        for _ in range(5):
            mujoco.mj_step(s.model, d)
            self.steps += 1
            if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
                raise RuntimeError('nonfinite simulation')
            self.observe()
            if self.writer and self.steps % 20 == 0:
                self.renderer.update_scene(d, camera=self.camera)
                self.writer.append_data(self.renderer.render())
        if self.steps % 50 == 0:
            self.state_samples.append({'time': float(d.time), 'stage': self.stage,
                                       'tcp': d.site_xpos[s.tcp].tolist(),
                                       'cube': d.qpos[s.cube_q:s.cube_q + 3].tolist()})

    def run(self):
        s, d = self.scene, self.data
        layout = self.layout
        started = time.perf_counter()
        success = False
        reason = ''
        stage = self.stage = 'settle'
        try:
            self.observe()
            if self.obstacle_contact_samples:
                raise RuntimeError('initial_robot_obstacle_contact')
            self.hold(.3, False)
            u, v, _ = self.initial_local
            R = target_rotation(grasp_yaw(s.robot, self.initial[2]))
            stage = 'pregrasp'; self.move(stage, layout.world(u, v, .14), R)
            stage = 'descend'; self.move(stage, layout.world(u, v, .025), R)
            stage = self.stage = 'close'; self.hold(1., True)
            stage = 'lift'; self.move(stage, layout.world(u, v, .18), R, True)
            self.hold(.2, True)
            self.lifted = bool(d.qpos[s.cube_q + 2] > layout.height + .09)
            if not self.lifted:
                raise RuntimeError('grasp_not_lifted')
            tray_R = target_rotation(grasp_yaw(s.robot, layout.angle))
            stage = 'transit'; self.move(stage, layout.world(*TRAY_XY, .18), tray_R, True)
            stage = 'place'; self.move(stage, layout.world(*TRAY_XY, .07), tray_R, True)
            stage = self.stage = 'release'; self.hold(.8, False)
            stage = 'retreat'; self.move(stage, layout.world(*TRAY_XY, .18), tray_R)
            self.hold(.5, False)
            vi = int(s.model.jnt_dofadr[s.model.joint('cube_free').id])
            success = settled_success(layout, d.qpos[s.cube_q:s.cube_q + 3], d.qvel[vi:vi + 3])
            if not success:
                reason = 'not_settled_in_tray'
        except Exception as exc:
            reason = f'{type(exc).__name__}:{exc}'
        finally:
            if self.writer:
                self.writer.close()
            if self.renderer:
                self.renderer.close()
        safe = success and not (self.joint_violations or self.velocity_violations or self.obstacle_contact_samples)
        return {'robot': s.robot, 'method': self.method, 'seed': self.seed,
                'layout': asdict(layout), 'layout_key': layout.key,
                'initial_pose_local': self.initial_local, 'initial_pose': self.initial,
                'success': bool(success), 'constraint_safe_success': bool(safe),
                'collision_free_success': bool(success and not self.obstacle_contact_samples),
                'failure_reason': reason, 'last_stage': stage, 'lift_verified': self.lifted,
                'trace': self.trace, 'state_samples': self.state_samples,
                'final_cube': d.qpos[s.cube_q:s.cube_q + 3].tolist(),
                'min_cube_tray_wall_clearance_m': self.min_clearance,
                'max_joint_speed_rad_s': self.max_joint_speed,
                'joint_travel_rad': self.joint_travel,
                'solve_time_s': self.solver.solve_seconds, 'ik_iterations': self.solver.iterations,
                'sim_time_s': float(d.time), 'wall_time_s': time.perf_counter() - started,
                'joint_limit_violation_samples': self.joint_violations,
                'velocity_limit_violation_samples': self.velocity_violations,
                'robot_obstacle_contact_samples': self.obstacle_contact_samples,
                'robot_obstacle_contact_pairs': self.contact_pairs,
                'metric_sample_period_s': .002, 'command_velocity_bound_rad_s': .6,
                'measured_velocity_threshold_rad_s': 1.05,
                'config_sha256': source_hash(), 'python_version': platform.python_version(),
                'mujoco_version': mujoco.__version__}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--robot', choices=['franka', 'heal'], required=True)
    p.add_argument('--method', choices=['mink', 'dls', 'qp'], default='mink')
    p.add_argument('--layout', help='JSON layout file; omitted means original layout')
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--record', action='store_true')
    p.add_argument('--tag', default='layout_baseline_v1')
    args = p.parse_args()
    layout = Layout.load(args.layout) if args.layout else BASELINE
    out = ROOT / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed, args.seed + args.episodes):
        video = out / f'{args.robot}_{args.method}_{seed}.mp4' if args.record else None
        result = LayoutEpisode(args.robot, args.method, seed, layout, video).run()
        with (out / f'{args.robot}_{args.method}.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(result) + '\n')
        print(json.dumps({k: result[k] for k in ('robot', 'method', 'seed', 'success',
                         'constraint_safe_success', 'failure_reason', 'max_joint_speed_rad_s')}), flush=True)
