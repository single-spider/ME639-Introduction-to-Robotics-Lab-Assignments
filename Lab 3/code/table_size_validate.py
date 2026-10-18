"""Resized-scene endpoint and physical validation for the alternative size study.

No frozen experiment sources or global constants are modified. The final tracker
is reused; task geometry is fitted to each proposed table and logged explicitly.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import time

import mujoco
import numpy as np

from control_study import SlewEpisode, fingerprint as control_fingerprint
from ik import Solver, target_rotation
from layout import Layout, grasp_yaw, wrap_yaw
from layout_scene import build
from optimize_layout import blocked
from scene import ROOT
from table_size_study import OUT, orientations, probe


def fingerprint():
    return hashlib.sha256(control_fingerprint().encode()+open(__file__,'rb').read()).hexdigest()


def geometry(candidate):
    length,width=candidate['length_m'],candidate['width_m']
    if length<.28-1e-9 or width<.42-1e-9:
        raise ValueError('Cannot pack the unchanged 22cm tray footprint and a separated cube region')
    tray=np.array([0.,width/2-.13])
    pick_u=(-min(.08,length/2-.05),min(.08,length/2-.05))
    pick_v=(-width/2+.05,min(-width/2+.14,tray[1]-.16))
    if pick_v[1]-pick_v[0]<.04-1e-9:raise ValueError('Pick band too small')
    return tray,pick_u,pick_v


def build_sized(robot,candidate,empty_surface=False):
    layout=Layout(**{k:candidate[k] for k in ('height','center_x','center_y','yaw_deg')})
    tray=np.array([0.,candidate['width_m']/2-.13])
    def transform(spec):
        geom=spec.geom('table')
        geom.size=[candidate['length_m']/2,candidate['width_m']/2,.02]
        geom.pos=layout.world(0,0,-.02)
        geom.quat=[np.cos(layout.angle/2),0,0,np.sin(layout.angle/2)]
        spec.body('tray').pos=layout.world(*tray,0)
        spec.body('tray').quat=[np.cos(layout.angle/2),0,0,np.sin(layout.angle/2)]
        if empty_surface:
            for name in ('cube_geom','tray_bottom',*(f'tray_wall{k}' for k in range(4))):
                spec.geom(name).contype=0;spec.geom(name).conaffinity=0
    scene=build(robot,spec_transform=transform)
    scene.table_height=layout.height;scene.layout=layout
    return scene,layout


def endpoint_check(candidate):
    start=time.perf_counter();robot=candidate['robot']
    scene,layout=build_sized(robot,candidate,empty_surface=True);data=scene.reset()
    all_yaws,full,task=orientations(robot)
    yaws=all_yaws[full if candidate['policy']=='full_yaw_8' else task]
    rows=[]
    # Separate surface-only workspace support from vertical approach/lift support.
    for offset in (.025,.14,.18):
        for u in np.linspace(-candidate['length_m']/2+.025,candidate['length_m']/2-.025,3):
            for v in np.linspace(-candidate['width_m']/2+.025,candidate['width_m']/2-.025,3):
                for yaw in yaws:
                    q,e,n=probe(scene,data,layout.world(u,v,offset),yaw,[scene.home,(scene.limits[:,0]+scene.limits[:,1])/2])
                    rows.append({'u':float(u),'v':float(v),'offset_m':offset,'yaw_rad':float(yaw),
                                 'clear':q is not None,'position_error_m':float(np.linalg.norm(e[:3])),
                                 'rotation_error_rad':float(np.linalg.norm(e[3:])),'iterations':n})
    try:
        tray,us,vs=geometry(candidate)
        actual,layout=build_sized(robot,candidate);data=actual.reset()
        initial_clear=not blocked(actual,data)
        task_rows=[]
        for u in us:
            for v in vs:
                for yaw in all_yaws[task]:
                    guess=actual.home
                    for offset in (.14,.025,.18):
                        q,e,n=probe(actual,data,layout.world(u,v,offset),yaw,[guess,actual.home])
                        task_rows.append({'u':u,'v':v,'offset_m':offset,'yaw_rad':float(yaw),'clear':q is not None})
                        if q is not None:guess=q
        for offset in (.07,.18):
            q,e,n=probe(actual,data,layout.world(*tray,offset),grasp_yaw(robot,layout.angle),[actual.home])
            task_rows.append({'stage':'tray','offset_m':offset,'clear':q is not None})
        packing={'tray_local_xy_m':tray.tolist(),'pick_u_m':list(us),'pick_v_m':list(vs)}
    except ValueError as exc:
        initial_clear=False;task_rows=[];packing={'reject_reason':str(exc)}
    result={'candidate':candidate,'source_sha256':fingerprint(),'initial_clear':initial_clear,
            'footprint_probes':rows,'footprint_all_three_heights_clear':all(r['clear'] for r in rows),
            'footprint_grasp_clear':all(r['clear'] for r in rows if r['offset_m']==.025),
            'task_probes':task_rows,'task_all_clear':bool(task_rows) and all(r['clear'] for r in task_rows),
            'packing':packing,'wall_time_s':time.perf_counter()-start}
    result['physical_eligible']=result['initial_clear'] and result['task_all_clear'] and result['footprint_all_three_heights_clear']
    return result


class SizedEpisode(SlewEpisode):
    def __init__(self,candidate,method,seed):
        robot=candidate['robot']
        layout=Layout(**{k:candidate[k] for k in ('height','center_x','center_y','yaw_deg')})
        acceleration=json.loads((ROOT/'results/control_study_v1/selected_policy.json').read_text())['acceleration'] if robot=='franka' else 0.
        super().__init__(robot,method,seed,layout,acceleration)
        self.scene,self.layout=build_sized(robot,candidate)
        self.candidate=candidate
        self.tray_xy,us,vs=geometry(candidate)
        rng=np.random.default_rng(seed)
        u,v,yaw=float(rng.uniform(*us)),float(rng.uniform(*vs)),float(rng.uniform(-np.pi,np.pi))
        self.initial_local=[u,v,yaw]
        xyz=self.layout.world(u,v,.0205)
        self.initial=[float(xyz[0]),float(xyz[1]),wrap_yaw(yaw+self.layout.angle)]
        self.data=self.scene.reset(*self.initial)
        self.solver=Solver(self.scene,method)
        self.last_q=self.data.qpos[self.scene.qids].copy()
        self.obstacle_ids={self.scene.model.geom(n).id for n in ('floor','table','tray_bottom',*(f'tray_wall{k}' for k in range(4)))}

    def run(self):
        # Same physical motion sequence and success thresholds as LayoutEpisode;
        # only tray and seeded pick coordinates come from the variable-size table.
        s,d=self.scene,self.data;layout=self.layout;started=time.perf_counter()
        success=False;reason='';self.stage='settle'
        try:
            self.observe()
            if self.obstacle_contact_samples:raise RuntimeError('initial_robot_obstacle_contact')
            self.hold(.3,False)
            u,v,_=self.initial_local;R=target_rotation(grasp_yaw(s.robot,self.initial[2]))
            self.move('pregrasp',layout.world(u,v,.14),R)
            self.move('descend',layout.world(u,v,.025),R)
            self.stage='close';self.hold(1.,True)
            self.move('lift',layout.world(u,v,.18),R,True);self.hold(.2,True)
            self.lifted=bool(d.qpos[s.cube_q+2]>layout.height+.09)
            if not self.lifted:raise RuntimeError('grasp_not_lifted')
            tray_R=target_rotation(grasp_yaw(s.robot,layout.angle))
            self.move('transit',layout.world(*self.tray_xy,.18),tray_R,True)
            self.move('place',layout.world(*self.tray_xy,.07),tray_R,True)
            self.stage='release';self.hold(.8,False)
            self.move('retreat',layout.world(*self.tray_xy,.18),tray_R);self.hold(.5,False)
            vi=int(s.model.jnt_dofadr[s.model.joint('cube_free').id])
            p=layout.local(d.qpos[s.cube_q:s.cube_q+3])
            success=bool(abs(p[0]-self.tray_xy[0])<.052 and abs(p[1]-self.tray_xy[1])<.047
                         and .022<p[2]<.040 and np.linalg.norm(d.qvel[vi:vi+3])<.03)
            if not success:reason='not_settled_in_tray'
        except Exception as exc:reason=f'{type(exc).__name__}:{exc}'
        safe=success and not(self.joint_violations or self.velocity_violations or self.obstacle_contact_samples)
        return {'robot':s.robot,'method':self.method,'seed':self.seed,'candidate':self.candidate,
                'initial_pose_local':self.initial_local,'initial_pose':self.initial,'success':success,
                'constraint_safe_success':bool(safe),'failure_reason':reason,'last_stage':self.stage,
                'sim_time_s':float(d.time),'wall_time_s':time.perf_counter()-started,'trace':self.trace,
                'max_joint_speed_rad_s':self.max_joint_speed,'joint_limit_violation_samples':self.joint_violations,
                'velocity_limit_violation_samples':self.velocity_violations,'robot_obstacle_contact_samples':self.obstacle_contact_samples,
                'min_cube_tray_wall_clearance_m':self.min_clearance,'joint_travel_rad':self.joint_travel,
                'final_cube':d.qpos[s.cube_q:s.cube_q+3].tolist(),'source_sha256':fingerprint(),
                'frozen_control_source_sha256':control_fingerprint(),'franka_command_acceleration_rad_s2':self.acceleration,
                'tray_local_xy_m':self.tray_xy.tolist(),'scope':'variable-size table/task geometry; not paired with old fixed-size episodes'}


def physical_one(candidate,method,seed):return SizedEpisode(candidate,method,seed).run()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--top',type=int,default=3,help='Distinct area-ranked candidates per robot and policy')
    p.add_argument('--physical',action='store_true')
    args=p.parse_args()
    candidates=[]
    for robot in ('franka','heal'):
        study=json.loads((OUT/f'{robot}_rectangles.json').read_text())
        for policy in ('full_yaw_8','task_policy_5'):
            seen=set();count=0
            for c in study['candidates']:
                if c['policy']!=policy or c['size_constraint']!='task_packable_size':continue
                key=tuple(round(c[k],5) for k in ('height','center_x','center_y','yaw_deg','length_m','width_m'))
                if key in seen:continue
                seen.add(key);candidates.append(c);count+=1
                if count>=args.top:break
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(endpoint_check,c) for c in candidates]
        for f in as_completed(futures):
            r=f.result();results.append(r)
            print(r['candidate']['robot'],r['candidate']['policy'],r['candidate']['area_m2'],'eligible',r['physical_eligible'],flush=True)
            (OUT/'endpoint_validation.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    chosen=[]
    for robot in ('franka','heal'):
        for policy in ('full_yaw_8','task_policy_5'):
            eligible=[r for r in results if r['candidate']['robot']==robot and r['candidate']['policy']==policy and r['physical_eligible']]
            if eligible:chosen.append(max(eligible,key=lambda r:r['candidate']['area_m2'])['candidate'])
    (OUT/'endpoint_selected.json').write_text(json.dumps({'candidates':chosen,'source_sha256':fingerprint(),
        'selection':'maximum screened area among top-N candidates passing all resized-scene checks; no physical outcome selection',
        'development_seeds':list(range(3000,3005)),'test_seeds':list(range(3100,3125))},indent=2),encoding='utf8')
    if args.physical and chosen:
        path=OUT/'physical.jsonl'
        done={(r['robot'],r['candidate']['policy'],r['method'],r['seed']) for r in
              [json.loads(s) for s in path.read_text().splitlines()]} if path.exists() else set()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(physical_one,c,m,s) for c in chosen for m in ('mink','dls','qp')
                     for s in [*range(3000,3005),*range(3100,3125)] if (c['robot'],c['policy'],m,s) not in done]
            for f in as_completed(futures):
                r=f.result()
                with path.open('a',encoding='utf8') as stream:stream.write(json.dumps(r)+'\n')
                print(r['robot'],r['candidate']['policy'],r['method'],r['seed'],r['success'],r['constraint_safe_success'],flush=True)


if __name__=='__main__':main()
