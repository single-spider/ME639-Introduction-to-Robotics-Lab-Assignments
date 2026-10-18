"""World-frame pose error and interchangeable differential IK backends."""
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from qpsolvers import solve_qp
import mink


def target_rotation(yaw):
    return Rotation.from_euler('z', yaw).as_matrix() @ Rotation.from_euler('x', np.pi).as_matrix()


def error_jacobian(scene, data, position, rotation):
    current = data.site_xmat[scene.tcp].reshape(3,3)
    error = np.r_[position-data.site_xpos[scene.tcp],
                  Rotation.from_matrix(rotation @ current.T).as_rotvec()]
    jp,jr = np.zeros((3,scene.model.nv)),np.zeros((3,scene.model.nv))
    mujoco.mj_jacSite(scene.model,data,jp,jr,scene.tcp)
    return error,np.vstack((jp[:,scene.dids],jr[:,scene.dids]))


def dls_velocity(J, desired, damping=1e-3):
    # Solve the system, rather than explicitly computing a matrix inverse.
    return np.linalg.solve(J.T@J+damping*np.eye(J.shape[1]),J.T@desired)


def qp_velocity(J, desired, lower, upper, damping=1e-3):
    # Variable is joint velocity. Both quadratic and linear terms are halved
    # consistently relative to ||Jv-desired||² + damping ||v||².
    answer = solve_qp(J.T@J+damping*np.eye(J.shape[1]),-J.T@desired,
                      lb=lower,ub=upper,solver='quadprog')
    if answer is None:
        raise RuntimeError('QP infeasible')
    return answer


class Solver:
    def __init__(self, scene, method, dt=0.01):
        self.scene,self.method,self.dt=scene,method,dt
        self.iterations=0
        self.solve_seconds=0.
        if method == 'mink':
            self.configuration=mink.Configuration(scene.model)
            self.task=mink.FrameTask('tcp','site',position_cost=1.,orientation_cost=1.,
                                     gain=0.04,lm_damping=1e-3)
            self.bounds=[mink.ConfigurationLimit(scene.model),
                         mink.VelocityLimit(scene.model,{n:0.6 for n in scene.joint_names})]

    def step(self,data,position,rotation):
        start=time.perf_counter()
        error,J=error_jacobian(self.scene,data,position,rotation)
        q=data.qpos[self.scene.qids]
        lower=np.maximum(-0.6,(self.scene.limits[:,0]+1e-4-q)/self.dt)
        upper=np.minimum(0.6,(self.scene.limits[:,1]-1e-4-q)/self.dt)
        if self.method == 'mink':
            self.configuration.update(data.qpos)
            self.task.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(rotation),np.asarray(position)))
            velocity=mink.solve_ik(self.configuration,[self.task],self.dt,
                solver='quadprog',damping=1e-3,limits=self.bounds)[self.scene.dids]
        elif self.method == 'dls':
            velocity=np.clip(dls_velocity(J,4.*error),lower,upper)
        elif self.method == 'qp':
            velocity=qp_velocity(J,4.*error,lower,upper)
        else:
            raise ValueError(self.method)
        self.iterations+=1
        self.solve_seconds+=time.perf_counter()-start
        return velocity,error


def solve_pose(scene,position,rotation,method='dls',initial=None,iterations=600):
    """Kinematic probe only; never used to teleport the live robot during episodes."""
    data=scene.reset()
    if initial is not None:
        data.qpos[scene.qids]=initial
        mujoco.mj_forward(scene.model,data)
    solver=Solver(scene,method)
    for _ in range(iterations):
        v,e=solver.step(data,np.asarray(position),rotation)
        if np.linalg.norm(e[:3])<0.002 and np.linalg.norm(e[3:])<0.03:
            return data.qpos[scene.qids].copy(),e,solver.iterations
        data.qpos[scene.qids]+=solver.dt*v
        mujoco.mj_forward(scene.model,data)
    return None,e,solver.iterations
