"""Headless physical trials, incrementally logged. No pose teleportation in motion."""
import argparse
import json
import time
import hashlib
import platform
from pathlib import Path
import mujoco
import numpy as np
from scene import build,ROOT
from ik import Solver,target_rotation


class Episode:
    def __init__(self,robot,method,seed,record=None):
        self.scene=build(robot)
        self.method,self.seed=method,seed
        rng=np.random.default_rng(seed)
        self.initial=[float(rng.uniform(.40,.50)),float(rng.uniform(-.18,-.09)),
                      float(rng.uniform(-np.pi,np.pi))]
        self.data=self.scene.reset(*self.initial)
        self.solver=Solver(self.scene,method)
        self.lifted=False
        self.record=record
        self.writer=self.renderer=None
        self.steps=0
        self.min_clearance=1.
        self.joint_violations=0
        self.velocity_violations=0
        self.trace=[]
        self.obstacle_contact_samples=0
        self.contact_pairs={}
        self.stage='initialization'
        if record:
            import imageio.v2 as imageio
            self.renderer=mujoco.Renderer(self.scene.model,height=480,width=640)
            self.writer=imageio.get_writer(str(record),fps=25,codec='libx264',quality=7)
            self.camera=mujoco.MjvCamera()
            self.camera.lookat[:]=[.25,0,.35]
            self.camera.distance=1.8
            self.camera.azimuth=135
            self.camera.elevation=-25

    def control(self,velocity,grip):
        s,d=self.scene,self.data
        if s.robot=='heal':
            M=np.zeros((s.model.nv,s.model.nv))
            mujoco.mj_fullM(s.model,d,M)
            accel=(velocity-d.qvel[s.dids])/0.04
            # The supplied viscous damping is large. Feed it the desired
            # velocity (rather than cancelling old measured damping once per
            # five physics substeps, which stalls the very light wrist).
            tau=M[np.ix_(s.dids,s.dids)]@accel+d.qfrc_bias[s.dids]
            tau+=s.model.dof_damping[s.dids]*velocity
            # Dry joint friction is solved as a constraint and is not included
            # in qfrc_passive. Feedforward helps the light wrist leave stiction.
            tau+=s.model.dof_frictionloss[s.dids]*np.tanh(velocity/0.005)
            d.ctrl[s.aids]=np.clip(tau,-150,150)
        else:
            kp=s.model.actuator_gainprm[s.aids,0]
            kd=-s.model.actuator_biasprm[s.aids,2]
            d.ctrl[s.aids]=np.clip(d.qpos[s.qids]+(kd/kp+0.01)*velocity,
                                  s.limits[:,0],s.limits[:,1])
            # Physical external feedforward force, not a state update.
            d.qfrc_applied[s.dids]=d.qfrc_bias[s.dids]
        d.ctrl[s.grip]=grip
        for _ in range(5):
            mujoco.mj_step(s.model,d)
            self.steps+=1
            if not np.all(np.isfinite(d.qpos)):
                raise RuntimeError('nonfinite simulation')
            if self.writer and self.steps%20==0:
                self.renderer.update_scene(d,camera=self.camera)
                self.writer.append_data(self.renderer.render())
        q=d.qpos[s.qids]
        self.joint_violations+=int(np.any((q<s.limits[:,0]-1e-3)|(q>s.limits[:,1]+1e-3)))
        self.velocity_violations+=int(np.any(np.abs(d.qvel[s.dids])>1.05))
        obstacle_ids={s.model.geom('table').id,s.model.geom('tray_bottom').id,
                      *(s.model.geom(f'tray_wall{k}').id for k in range(4))}
        forbidden=False
        for contact in d.contact:
            if contact.dist>0:continue
            a,b=int(contact.geom1),int(contact.geom2)
            obstacle=a if a in obstacle_ids else b if b in obstacle_ids else None
            if obstacle is None:continue
            other=b if obstacle==a else a
            body=s.model.body(int(s.model.geom_bodyid[other])).name
            if body in ('world','cube','tray'):continue
            forbidden=True
            label=f"{body}/{s.model.geom(obstacle).name}"
            self.contact_pairs[label]=self.contact_pairs.get(label,0)+1
        self.obstacle_contact_samples+=int(forbidden)
        # Clearance between transported cube and tray walls only. Expected
        # cube/support and finger/cube contacts are not obstacle violations.
        for k in range(4):
            dist=mujoco.mj_geomDistance(s.model,d,s.model.geom('cube_geom').id,
                                         s.model.geom(f'tray_wall{k}').id,1.,None)
            self.min_clearance=min(self.min_clearance,float(dist))

    def move(self,stage,position,rotation,closed=False,timeout=12.):
        s,d=self.scene,self.data
        self.stage=stage
        start=d.time
        e=None
        stable=0
        while d.time-start<timeout:
            v,e=self.solver.step(d,np.asarray(position),rotation)
            self.control(v,s.close_command if closed else s.open_command)
            stable=stable+1 if np.linalg.norm(e[:3])<.003 and np.linalg.norm(e[3:])<.04 else 0
            if stable>=10:
                self.trace.append({'stage':stage,'sim_time':float(d.time),'error':e.tolist(),
                                   'cube':d.qpos[s.cube_q:s.cube_q+3].tolist()})
                return
        self.trace.append({'stage':stage,'sim_time':float(d.time),'error':e.tolist(),
                           'cube':d.qpos[s.cube_q:s.cube_q+3].tolist()})
        raise RuntimeError(f'timeout:{stage}')

    def hold(self,seconds,closed):
        until=self.data.time+seconds
        while self.data.time<until:
            self.control(np.zeros(len(self.scene.qids)),self.scene.close_command if closed else self.scene.open_command)

    def run(self):
        s,d=self.scene,self.data
        reason='';stage='settle';success=False
        start=time.perf_counter()
        try:
            self.hold(.3,False)
            x,y,yaw=self.initial
            # Square cubes have fourfold symmetry. Choose an equivalent face
            # grasp within a wrist-friendly branch, without changing cube pose.
            yaw=yaw%(np.pi/2) if s.robot=='heal' else (yaw+np.pi/4)%(np.pi/2)-np.pi/4
            R=target_rotation(yaw)
            stage='pregrasp';self.move(stage,[x,y,.34],R)
            stage='descend';self.move(stage,[x,y,.225],R)
            stage='close';self.hold(1.,True)
            stage='lift';self.move(stage,[x,y,.38],R,True)
            self.hold(.2,True)
            self.lifted=bool(d.qpos[s.cube_q+2]>.29)
            if not self.lifted:
                raise RuntimeError('grasp_not_lifted')
            tray_R=target_rotation(0.)
            # Align parallel jaws with tray edges while above the rim. Release
            # above the floor so fingers do not press into its bottom/walls.
            stage='transit';self.move(stage,[.48,.16,.38],tray_R,True)
            stage='place';self.move(stage,[.48,.16,.27],tray_R,True)
            stage='release';self.hold(.8,False)
            stage='retreat';self.move(stage,[.48,.16,.38],tray_R)
            self.hold(.5,False)
            cube=d.qpos[s.cube_q:s.cube_q+3]
            cv=d.qvel[s.model.jnt_dofadr[s.model.joint('cube_free').id]:][:3]
            success=bool(abs(cube[0]-.48)<.052 and abs(cube[1]-.16)<.047 and
                         .222<cube[2]<.24 and np.linalg.norm(cv)<.03)
            if not success:reason='not_settled_in_tray'
        except Exception as exc:
            reason=f'{type(exc).__name__}:{exc}'
        finally:
            if self.writer:self.writer.close()
            if self.renderer:self.renderer.close()
        return {'robot':s.robot,'method':self.method,'seed':self.seed,
                'initial_pose':self.initial,'success':success,'failure_reason':reason,
                'last_stage':stage,'lift_verified':self.lifted,'trace':self.trace,
                'final_cube':d.qpos[s.cube_q:s.cube_q+3].tolist(),
                'min_cube_tray_wall_clearance_m':self.min_clearance,
                'solve_time_s':self.solver.solve_seconds,'ik_iterations':self.solver.iterations,
                'sim_time_s':float(d.time),'wall_time_s':time.perf_counter()-start,
                'joint_limit_violation_samples':self.joint_violations,
                'velocity_limit_violation_samples':self.velocity_violations,
                'robot_obstacle_contact_samples':self.obstacle_contact_samples,
                'robot_obstacle_contact_pairs':self.contact_pairs,
                'collision_free_success':success and self.obstacle_contact_samples==0,
                'config_sha256':hashlib.sha256(b''.join((ROOT/f).read_bytes() for f in
                                                      ('scene.py','ik.py','run.py'))).hexdigest(),
                'python_version':platform.python_version(),'mujoco_version':mujoco.__version__}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--robot',choices=['franka','heal'],required=True)
    p.add_argument('--method',choices=['mink','dls','qp'],default='mink')
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--record',action='store_true')
    p.add_argument('--tag',default='pilot')
    args=p.parse_args()
    output=ROOT/'results'/args.tag;output.mkdir(parents=True,exist_ok=True)
    log=output/f'{args.robot}_{args.method}.jsonl'
    for seed in range(args.seed,args.seed+args.episodes):
        recording=output/f'{args.robot}_{args.method}_{seed}.mp4' if args.record else None
        result=Episode(args.robot,args.method,seed,recording).run()
        with log.open('a',encoding='utf8') as f:f.write(json.dumps(result)+'\n')
        print(json.dumps(result),flush=True)
