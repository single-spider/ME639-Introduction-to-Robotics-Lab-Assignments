"""Alternative largest-table study, isolated from all frozen experiments.

Independent kinematic LM probes are NOT the physical controller. Finite yaw,
height and XY sampling gives a bounded search, not continuous certification.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from ik import target_rotation
from layout import Layout, grasp_yaw, source_hash
from layout_scene import build
from optimize_layout import blocked
from scene import ROOT

OUT = ROOT / 'results/table_size_study_v1'
HEIGHTS = [.10,.15,.20,.25,.30,.35,.40]
XS = np.round(np.arange(.20,.8001,.075),6)
YS = np.round(np.arange(-.375,.3751,.075),6)
PITCH = .075
RASTER = .025
ANGLES = list(range(0,180,15))


def fingerprint():
    return hashlib.sha256(source_hash().encode()+Path(__file__).read_bytes()).hexdigest()


def orientations(robot):
    full = np.linspace(-np.pi,np.pi,8,endpoint=False)
    task = np.array([grasp_yaw(robot,y) for y in np.linspace(-np.pi,np.pi,5,endpoint=False)])
    all_yaws = np.array(sorted(set(round(float(y),9) for y in [*full,*task])))
    full_ids = [int(np.argmin(abs(all_yaws-y))) for y in full]
    task_ids = [int(np.argmin(abs(all_yaws-y))) for y in task]
    return all_yaws, full_ids, task_ids


def workspace_scene(robot):
    def remove_task_obstacles(spec):
        for name in ('table','cube_geom','tray_bottom',*(f'tray_wall{k}' for k in range(4))):
            geom=spec.geom(name);geom.contype=0;geom.conaffinity=0
    return build(robot,spec_transform=remove_task_obstacles)


def pose_residual(scene,data,position,rotation):
    current=data.site_xmat[scene.tcp].reshape(3,3)
    return np.r_[position-data.site_xpos[scene.tcp],Rotation.from_matrix(rotation@current.T).as_rotvec()]


def probe(scene,data,position,yaw,guesses,iterations=120):
    """Multi-start bounded kinematic LM, using an isolated data object only."""
    rotation=target_rotation(yaw)
    lo,hi=scene.limits[:,0]+.002,scene.limits[:,1]-.002
    jp,jr=np.zeros((3,scene.model.nv)),np.zeros((3,scene.model.nv))
    weights=np.array([1,1,1,.25,.25,.25])
    best=(float('inf'),None,None)
    count=0
    for guess in guesses:
        data.qpos[scene.qids]=np.clip(guess,lo,hi)
        mujoco.mj_forward(scene.model,data)
        damping=.003
        stagnant=0
        for _ in range(iterations):
            count+=1
            error=pose_residual(scene,data,position,rotation)
            cost=float(np.linalg.norm(error*weights))
            if cost<best[0]:best=(cost,data.qpos[scene.qids].copy(),error.copy())
            if np.linalg.norm(error[:3])<.002 and np.linalg.norm(error[3:])<.03 and not blocked(scene,data):
                return data.qpos[scene.qids].copy(),error,count
            mujoco.mj_jacSite(scene.model,data,jp,jr,scene.tcp)
            J=np.vstack((jp[:,scene.dids],jr[:,scene.dids]))*weights[:,None]
            delta=np.linalg.solve(J.T@J+damping*np.eye(len(scene.qids)),J.T@(error*weights))
            delta*=min(1,.18/max(float(np.max(abs(delta))),1e-12))
            old=data.qpos[scene.qids].copy()
            data.qpos[scene.qids]=np.clip(old+delta,lo,hi)
            mujoco.mj_forward(scene.model,data)
            new=float(np.linalg.norm(pose_residual(scene,data,position,rotation)*weights))
            if new>=cost-1e-9:
                data.qpos[scene.qids]=old;mujoco.mj_forward(scene.model,data)
                damping=min(.5,damping*3);stagnant+=1
            else:
                damping=max(.00001,damping*.65);stagnant=0
            if stagnant>=12:break
    return None,best[2],count


def map_slice(robot,height):
    start=time.perf_counter()
    s=workspace_scene(robot);d=s.reset()
    yaws,full_ids,task_ids=orientations(robot)
    shape=(len(YS),len(XS),len(yaws))
    ok=np.zeros(shape,dtype=bool)
    qs=np.full((*shape,len(s.qids)),np.nan)
    errors=np.full((*shape,2),np.nan)
    iterations=np.zeros(shape,dtype=np.int32)
    rng=np.random.default_rng(7391)
    restarts=[s.home,(s.limits[:,0]+s.limits[:,1])/2,
              rng.uniform(s.limits[:,0]+.05,s.limits[:,1]-.05)]
    for iy,y in enumerate(YS):
        for ix,x in enumerate(XS):
            for k,yaw in enumerate(yaws):
                guesses=[]
                if ix and ok[iy,ix-1,k]:guesses.append(qs[iy,ix-1,k])
                if iy and ok[iy-1,ix,k]:guesses.append(qs[iy-1,ix,k])
                if k and ok[iy,ix,k-1]:guesses.append(qs[iy,ix,k-1])
                # Limit deterministic starts to four; never use trial outcomes.
                guesses=(guesses[:2]+restarts)[:4]
                q,e,n=probe(s,d,np.array([x,y,height+.025]),yaw,guesses)
                ok[iy,ix,k]=q is not None;iterations[iy,ix,k]=n
                if q is not None:qs[iy,ix,k]=q
                if e is not None:errors[iy,ix,k]=[np.linalg.norm(e[:3]),np.linalg.norm(e[3:])]
    path=OUT/f'{robot}_h{height:.2f}.npz'
    np.savez_compressed(path,ok=ok,q=qs,errors=errors,iterations=iterations,x=XS,y=YS,yaws=yaws,
                        full_ids=full_ids,task_ids=task_ids,height=height)
    row={'robot':robot,'height':height,'pose_count':int(ok.size),'reachable_clear_poses':int(ok.sum()),
         'full_yaw_xy':int(ok[:,:,full_ids].all(axis=2).sum()),
         'task_policy_xy':int(ok[:,:,task_ids].all(axis=2).sum()),
         'wall_time_s':time.perf_counter()-start,'source_sha256':fingerprint()}
    path.with_suffix('.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    return row


def largest_rectangle(mask,min_width_cells=1,min_height_cells=1):
    """Largest all-true axis-aligned rectangle, O(rows*cols), half-open bounds."""
    heights=np.zeros(mask.shape[1],dtype=int)
    best=None
    for row in range(mask.shape[0]):
        heights=np.where(mask[row],heights+1,0)
        stack=[]
        for col in range(mask.shape[1]+1):
            h=int(heights[col]) if col<mask.shape[1] else 0
            left=col
            while stack and stack[-1][1]>h:
                start,height=stack.pop();left=start
                width=col-start
                if width>=min_width_cells and height>=min_height_cells:
                    obj={'area_cells':width*height,'x0':start,'x1':col,'y0':row-height+1,'y1':row+1}
                    if best is None or obj['area_cells']>best['area_cells']:best=obj
            if not stack or stack[-1][1]<h:stack.append((left,h))
    return best


def rotated_candidates(nodes,height,robot,policy):
    # A coarse cell is feasible only if all four vertex IK probes pass.
    cells=nodes[:-1,:-1]&nodes[1:,:-1]&nodes[:-1,1:]&nodes[1:,1:]
    edges=np.arange(-1.0,1.0001,RASTER)
    centers=(edges[:-1]+edges[1:])/2
    U,V=np.meshgrid(centers,centers)
    candidates=[]
    for angle in ANGLES:
        a=np.deg2rad(angle);c,s=np.cos(a),np.sin(a)
        fine=np.ones(U.shape,dtype=bool)
        # Nine checks per fine cell, with a tiny outward numerical margin.
        for du in (-RASTER/2-1e-8,0,RASTER/2+1e-8):
            for dv in (-RASTER/2-1e-8,0,RASTER/2+1e-8):
                X=c*(U+du)-s*(V+dv);Y=s*(U+du)+c*(V+dv)
                ix=np.floor((X-XS[0])/PITCH).astype(int);iy=np.floor((Y-YS[0])/PITCH).astype(int)
                inside=(ix>=0)&(ix<cells.shape[1])&(iy>=0)&(iy<cells.shape[0])
                fine &= inside & cells[np.clip(iy,0,cells.shape[0]-1),np.clip(ix,0,cells.shape[1]-1)]
        for name,mw,mh in [('unconstrained_size',1,1),('task_packable_size',math.ceil(.28/RASTER),math.ceil(.42/RASTER))]:
            best=largest_rectangle(fine,mw,mh)
            if best is None:continue
            u=(edges[best['x0']]+edges[best['x1']])/2
            v=(edges[best['y0']]+edges[best['y1']])/2
            length=(best['x1']-best['x0'])*RASTER;width=(best['y1']-best['y0'])*RASTER
            candidates.append({'robot':robot,'policy':policy,'size_constraint':name,'height':height,
                               'center_x':float(c*u-s*v),'center_y':float(s*u+c*v),'yaw_deg':angle,
                               'length_m':length,'width_m':width,'area_m2':length*width,
                               'coarse_feasible_xy':int(nodes.sum()),'coarse_feasible_cells':int(cells.sum())})
    return candidates


def analyze(robot):
    candidates=[];slices=[]
    fig,axes=plt.subplots(2,len(HEIGHTS),figsize=(21,7),sharex=True,sharey=True)
    for col,height in enumerate(HEIGHTS):
        path=OUT/f'{robot}_h{height:.2f}.npz'
        data=np.load(path)
        for row,(policy,key) in enumerate([('full_yaw_8','full_ids'),('task_policy_5','task_ids')]):
            mask=data['ok'][:,:,data[key]].all(axis=2)
            found=rotated_candidates(mask,height,robot,policy);candidates.extend(found)
            ax=axes[row,col]
            ax.scatter(*np.meshgrid(XS,YS),c=mask.ravel(),vmin=0,vmax=1,cmap='RdYlGn',s=22)
            best=max((r for r in found if r['size_constraint']=='unconstrained_size'),key=lambda r:r['area_m2'],default=None)
            if best:
                layout=Layout(best['height'],best['center_x'],best['center_y'],best['yaw_deg'])
                corners=[layout.world(u,v,0)[:2] for u,v in [(-best['length_m']/2,-best['width_m']/2),
                         (best['length_m']/2,-best['width_m']/2),(best['length_m']/2,best['width_m']/2),(-best['length_m']/2,best['width_m']/2)]]
                ax.add_patch(Polygon(corners,fill=False,ec='#173c7d',lw=2))
            ax.set_title(f'H={height:.2f}m\nA={best["area_m2"]:.3f}m²' if best else f'H={height:.2f}m\nNo rectangle')
            ax.set_aspect('equal');ax.set_xlabel('X [m]')
            if col==0:ax.set_ylabel(policy+'\nY [m]')
            slices.append({'height':height,'policy':policy,'feasible_xy':int(mask.sum()),'best':best})
    fig.suptitle(f'{robot.title()}: alternative largest level-table footprint | coarse kinematic screen')
    fig.tight_layout();fig.savefig(OUT/f'{robot}_height_comparison.png',dpi=140);plt.close(fig)
    ranked=sorted(candidates,key=lambda r:r['area_m2'],reverse=True)
    result={'robot':robot,'source_sha256':fingerprint(),'slices':slices,'candidates':ranked,
            'best_by_policy':{policy:{kind:next((r for r in ranked if r['policy']==policy and r['size_constraint']==kind),None)
                                    for kind in ('unconstrained_size','task_packable_size')}
                              for policy in ('full_yaw_8','task_policy_5')},
            'stage':'kinematic_screen_only; resized-scene and physical validation pending'}
    (OUT/f'{robot}_rectangles.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--robot',choices=['franka','heal','both'],default='both')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    manifest={'source_sha256':fingerprint(),'frozen_layout_source_sha256':source_hash(),
              'heights_m':HEIGHTS,'x_m':XS.tolist(),'y_m':YS.tolist(),'coarse_pitch_m':PITCH,
              'rectangle_raster_m':RASTER,'rectangle_yaws_deg':ANGLES,'grasp_tcp_offset_m':.025,
              'position_tolerance_m':.002,'orientation_tolerance_rad':.03,'joint_margin_rad':.002,
              'floor_collision_check':True,'self_collision_certification':False,
              'objective':'maximize sampled feasible rectangular area independently per robot',
              'original_experiments_unchanged':True,'mujoco_version':mujoco.__version__}
    target=OUT/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:
        raise RuntimeError('Study configuration/source changed; use a new version or deliberately clear only new screen results.')
    target.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    robots=('franka','heal') if args.robot=='both' else (args.robot,)
    if not args.analyze_only:
        tasks=[(r,h) for r in robots for h in HEIGHTS if not (OUT/f'{r}_h{h:.2f}.json').exists()]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(map_slice,*task) for task in tasks]
            for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    for robot in robots:
        result=analyze(robot)
        print(robot,json.dumps(result['best_by_policy']),flush=True)


if __name__=='__main__':main()
