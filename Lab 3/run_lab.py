"""Run the submitted physical pick-and-place controller on either table design."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from control_study import SlewEpisode
from layout import Layout
from table_size_validate import SizedEpisode


def make_episode(robot,method,seed,layout='sizing'):
    if layout=='sizing':
        choices=json.loads((ROOT/'results/table_size_study_v1/refined_selected.json').read_text())['candidates']
        c=next(c for c in choices if c['robot']==robot and c['policy']=='full_yaw_8')
        return SizedEpisode(c,method,seed)
    selected=Layout.load(ROOT/f'results/layout_study_v2/{robot}_selected.json')
    acceleration=json.loads((ROOT/'results/control_study_v1/selected_policy.json').read_text())['acceleration'] if robot=='franka' else 0.
    return SlewEpisode(robot,method,seed,selected,acceleration)


def enable_recording(ep,path):
    import imageio.v2 as imageio
    import mujoco
    ep.renderer=mujoco.Renderer(ep.scene.model,height=480,width=640)
    ep.camera=mujoco.MjvCamera()
    ep.camera.lookat[:]=[ep.layout.center_x/2,ep.layout.center_y/2,ep.layout.height+.15]
    ep.camera.distance=1.8;ep.camera.azimuth=135;ep.camera.elevation=-25
    ep.writer=imageio.get_writer(str(path),fps=25,codec='libx264',quality=7)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',choices=('franka','heal','both'),default='both')
    p.add_argument('--method',choices=('mink','dls','qp','all'),default='all')
    p.add_argument('--layout',choices=('sizing','placement'),default='sizing')
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--record',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'runs')
    a=p.parse_args()
    if a.episodes<1:p.error('--episodes must be positive')
    a.output.mkdir(parents=True,exist_ok=True)
    robots=('franka','heal') if a.robot=='both' else (a.robot,)
    methods=('mink','dls','qp') if a.method=='all' else (a.method,)
    seed=a.seed if a.seed is not None else (3100 if a.layout=='sizing' else 2000)
    failures=[]
    for robot in robots:
        for method in methods:
            for s in range(seed,seed+a.episodes):
                ep=make_episode(robot,method,s,a.layout)
                try:
                    if a.record:enable_recording(ep,a.output/f'{a.layout}_{robot}_{method}_{s}.mp4')
                    row=ep.run()
                    row.update(solve_time_s=ep.solver.solve_seconds,ik_iterations=ep.solver.iterations,table_study=a.layout)
                finally:
                    if ep.writer:ep.writer.close()
                    if ep.renderer:ep.renderer.close()
                with (a.output/f'{a.layout}_{robot}_{method}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({k:row[k] for k in ('robot','method','seed','success','constraint_safe_success','failure_reason','solve_time_s','ik_iterations')}),flush=True)
                if not row['constraint_safe_success']:failures.append((robot,method,s,row['failure_reason']))
    if failures:
        print('Unsuccessful episodes:',json.dumps(failures),file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
