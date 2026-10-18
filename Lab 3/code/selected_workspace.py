"""Workspace evidence for frozen per-robot layouts, separate from dynamic trials."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import mujoco
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from ik import solve_pose, target_rotation
from layout import Layout, build_layout, grasp_yaw, source_hash
from optimize_layout import blocked
from scene import ROOT

OUT=ROOT/'results'/'selected_workspace_v1'
LAYOUTS=ROOT/'results'/'layout_study_v2'


def probe(s,position,yaw,guess):
    q,e,n=solve_pose(s,position,target_rotation(yaw),initial=guess,iterations=600)
    if q is None:
        q,e,n2=solve_pose(s,position,target_rotation(yaw),initial=s.home,iterations=800);n+=n2
    clear=False
    if q is not None:
        d=s.reset();d.qpos[s.qids]=q;mujoco.mj_forward(s.model,d)
        clear=not blocked(s,d)
    return q,{'world_xyz_m':position.tolist(),'world_tcp_yaw_rad':float(yaw),'reachable':q is not None,
              'endpoint_clear':clear,'position_error_m':float(np.linalg.norm(e[:3])),
              'rotation_error_rad':float(np.linalg.norm(e[3:])),'iterations':n}


def run_robot(robot):
    layout=Layout.load(LAYOUTS/f'{robot}_selected.json')
    s=build_layout(robot,layout)
    rng=np.random.default_rng(501)
    points=[];d=s.reset()
    for q in rng.uniform(s.limits[:,0],s.limits[:,1],(8000,len(s.qids))):
        d.qpos[s.qids]=q;mujoco.mj_forward(s.model,d);points.append(d.site_xpos[s.tcp].copy())
    points=np.array(points)
    task=[];arbitrary=[];tray=[]
    for u in (-.08,-.03,.02):
        for v in (-.18,-.135,-.09):
            for yaw in np.linspace(-np.pi,np.pi,5,endpoint=False):
                guess=s.home.copy()
                for z in (.14,.025,.18):
                    q,row=probe(s,layout.world(u,v,z),grasp_yaw(robot,yaw+layout.angle),guess)
                    row.update({'local_u':u,'local_v':v,'local_cube_yaw_rad':float(yaw),'height_offset_m':z})
                    task.append(row)
                    if q is not None:guess=q
            for yaw in np.linspace(-np.pi,np.pi,8,endpoint=False):
                _,row=probe(s,layout.world(u,v,.025),yaw+layout.angle,s.home)
                row.update({'local_u':u,'local_v':v,'table_relative_tcp_yaw_rad':float(yaw)})
                arbitrary.append(row)
    for z in (.07,.18):
        _,row=probe(s,layout.world(0,.16,z),grasp_yaw(robot,layout.angle),s.home)
        tray.append(row)
    result={'robot':robot,'layout':json.loads((LAYOUTS/f'{robot}_selected.json').read_text())['layout'],
            'layout_source_sha256':source_hash(),'fk_samples':8000,
            'task_endpoint_count':len(task),'task_endpoint_clear':sum(r['endpoint_clear'] for r in task),
            'arbitrary_yaw_count':len(arbitrary),'arbitrary_yaw_endpoint_clear':sum(r['endpoint_clear'] for r in arbitrary),
            'arbitrary_yaw_all8_xy':sum(all(r['endpoint_clear'] for r in arbitrary if r['local_u']==u and r['local_v']==v)
                                      for u in (-.08,-.03,.02) for v in (-.18,-.135,-.09)),
            'task_probes':task,'arbitrary_yaw_probes':arbitrary,'tray_probes':tray,
            'limitations':['Seeded FK cloud is collision-unscreened, not guaranteed reachable free space.',
                           'Task135 endpoints use square-cube symmetry; arbitrary72 yaw endpoints do not.',
                           'Finite local IK starts and endpoint environment screening are not full dexterity/swept-path proofs.',
                           'This probes a fixed task region, not the entire table surface.']}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/f'{robot}.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    np.savez_compressed(OUT/f'{robot}_fk.npz',points=points)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].scatter(points[:,0],points[:,1],s=1,alpha=.08,color='#587da0')
    corners=np.array([layout.world(u,v,0)[:2] for u,v in [(-.25,-.30),(.25,-.30),(.25,.30),(-.25,.30)]])
    axes[0].add_patch(Polygon(corners,fill=False,color='#a16b2b',lw=2,label='table'))
    taskxy=np.array([layout.world(u,v,0)[:2] for u,v in [(-.08,-.18),(.02,-.18),(.02,-.09),(-.08,-.09)]])
    axes[0].add_patch(Polygon(taskxy,fill=False,color='#c43333',lw=2,label='pick region'))
    axes[0].scatter(*layout.world(0,.16,0)[:2],marker='s',s=80,color='#d6ac24',label='tray')
    axes[0].set(xlabel='World X [m]',ylabel='World Y [m]',title='Seeded FK cloud + selected layout')
    axes[0].axis('equal');axes[0].legend(fontsize=7)
    for ax,rows,ncol,title in [(axes[1],task,15,'Task: 3 heights x5 cube yaws'),
                               (axes[2],arbitrary,8,'Arbitrary TCP: 8 yaw samples')]:
        arr=np.array([r['endpoint_clear'] for r in rows]).reshape(9,ncol)
        ax.imshow(arr,aspect='auto',vmin=0,vmax=1,cmap='RdYlGn',interpolation='nearest')
        ax.set(xlabel='Pose sample index',ylabel='Task-local XY grid point',title=title)
        ax.set_yticks(range(9),[f'{u:.2f},{v:.3f}' for u in (-.08,-.03,.02) for v in (-.18,-.135,-.09)],fontsize=7)
    fig.suptitle(f'{robot.title()} | H={layout.height:.2f}m, center=({layout.center_x:.2f},{layout.center_y:.2f})m, yaw={layout.yaw_deg:g}deg')
    fig.tight_layout();fig.savefig(OUT/f'{robot}_workspace.png',dpi=160);plt.close(fig)
    print(robot,result['task_endpoint_clear'],'/135 task;',result['arbitrary_yaw_endpoint_clear'],'/72 arbitrary',flush=True)
    return result


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_robot,('franka','heal')))
