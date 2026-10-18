"""Level-table configurations and rigid task-frame transforms.

This module does not modify the preserved baseline scene/runner source files.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import mujoco
import numpy as np
from scene import ROOT
from layout_scene import build


def wrap_yaw(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def grasp_yaw(robot, angle):
    """Choose an equivalent square-face grasp in the baseline wrist branch."""
    return angle % (np.pi / 2) if robot == 'heal' else (angle + np.pi / 4) % (np.pi / 2) - np.pi / 4


@dataclass(frozen=True)
class Layout:
    height: float = 0.20
    center_x: float = 0.48
    center_y: float = 0.0
    yaw_deg: float = 0.0

    def __post_init__(self):
        values = list(asdict(self).values())
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Layout values must be finite')
        if not 0.08 <= self.height <= 0.60:
            raise ValueError('Top height outside allowed study bounds [0.08,0.60] m')
        if not -1 <= self.center_x <= 1 or not -1 <= self.center_y <= 1:
            raise ValueError('Table center outside study bounds')

    @property
    def angle(self):
        return np.deg2rad(self.yaw_deg)

    @property
    def rotation(self):
        c, s = np.cos(self.angle), np.sin(self.angle)
        return np.array([[c, -s], [s, c]])

    @property
    def origin(self):
        return np.array([self.center_x, self.center_y])

    @property
    def key(self):
        return f'h{self.height:.3f}_x{self.center_x:.3f}_y{self.center_y:.3f}_a{self.yaw_deg:.2f}'

    def world(self, u, v, z_offset):
        xy = self.origin + self.rotation @ np.array([u, v])
        return np.r_[xy, self.height + z_offset]

    def local(self, xyz):
        xyz = np.asarray(xyz)
        return np.r_[self.rotation.T @ (xyz[:2] - self.origin), xyz[2] - self.height]

    @classmethod
    def load(cls, path):
        obj = json.loads(open(path, encoding='utf8').read())
        return cls(**obj.get('layout', obj))


BASELINE = Layout()
PICK_X = (-0.08, 0.02)
PICK_Y = (-0.18, -0.09)
TRAY_XY = (0.0, 0.16)


def build_layout(robot, layout=BASELINE):
    def transform(spec):
        spec.geom('table').pos = layout.world(0, 0, -.02)
        spec.geom('table').quat = [np.cos(layout.angle / 2), 0, 0, np.sin(layout.angle / 2)]
        spec.body('tray').pos = layout.world(*TRAY_XY, 0)
        spec.body('tray').quat = [np.cos(layout.angle / 2), 0, 0, np.sin(layout.angle / 2)]
    scene = build(robot, spec_transform=transform)
    scene.table_height = layout.height
    scene.layout = layout
    return scene


def sample_pose(seed, layout):
    rng = np.random.default_rng(seed)
    u = float(rng.uniform(*PICK_X))
    v = float(rng.uniform(*PICK_Y))
    yaw = float(rng.uniform(-np.pi, np.pi))
    xyz = layout.world(u, v, .0205)
    return [u, v, yaw], [float(xyz[0]), float(xyz[1]), wrap_yaw(yaw + layout.angle)]


def scene_for_seed(robot, layout, seed):
    scene = build_layout(robot, layout)
    local, world = sample_pose(seed, layout)
    return scene, scene.reset(*world), local, world


def source_hash():
    files = ('scene.py', 'ik.py', 'run.py', 'layout_scene.py', 'layout.py', 'layout_run.py', 'optimize_layout.py')
    h = hashlib.sha256()
    for name in files:
        path = ROOT / 'code' / name
        if path.exists():
            h.update(name.encode()); h.update(b'\0'); h.update(path.read_bytes()); h.update(b'\0')
    return h.hexdigest()
