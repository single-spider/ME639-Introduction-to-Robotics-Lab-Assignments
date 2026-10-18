"""Recreate workspace and IK comparison plots from the supplied evaluation data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'runs/plots');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    candidates=json.loads((ROOT/'results/table_size_study_v1/refined_selected.json').read_text())['candidates']
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for col,robot in enumerate(('franka','heal')):
        c=next(c for c in candidates if c['robot']==robot and c['policy']=='full_yaw_8')
        points=np.load(ROOT/f'results/selected_workspace_v1/{robot}_fk.npz')['points']
        axs[0,col].scatter(points[:,0],points[:,1],s=1,alpha=.15)
        axs[0,col].set_title(robot.title()+': FK position samples')
        grid=np.load(ROOT/f'results/table_size_study_v1/{robot}_h{c["height"]:.2f}.npz')
        X,Y=np.meshgrid(grid['x'],grid['y']);ok=grid['ok'][:,:,grid['full_ids']].all(axis=2)
        axs[1,col].scatter(X.ravel(),Y.ravel(),c=np.where(ok.ravel(),'#27875e','#c8665a'),s=16)
        axs[1,col].set_title(f'{robot.title()}: eight-yaw feasibility, H={c["height"]:.2f} m')
        L,W=c['length_m'],c['width_m'];angle=np.deg2rad(c['yaw_deg']);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        corners=np.array([[-L/2,-W/2],[L/2,-W/2],[L/2,W/2],[-L/2,W/2]])@R.T+[c['center_x'],c['center_y']]
        for ax in axs[:,col]:
            ax.add_patch(Polygon(corners,fill=False,color='#183d77',lw=2,label='Selected table'))
            ax.set(xlabel='World X [m]',ylabel='World Y [m]');ax.set_aspect('equal');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(a.output/'workspace.png',dpi=160);plt.close(fig)
    metrics=json.loads((ROOT/'results/metrics.json').read_text())
    for branch,label in (('workspace_sized_alternative','sizing'),('fixed_size_selected','placement')):
        fig,axs=plt.subplots(1,2,figsize=(10,3))
        for ax,field,factor,ylabel in zip(axs,('mean_solver_time_s','mean_ik_iterations'),(1000,1),('IK solve time per episode [ms]','IK updates per episode')):
            for j,robot in enumerate(('franka','heal')):
                values=[next(r for r in metrics if r['branch']==branch and r['robot']==robot and r['method']==m)[field]*factor for m in ('mink','dls','qp')]
                ax.bar(np.arange(3)+(j-.5)*.35,values,.33,label=robot.title())
            ax.set_xticks(np.arange(3),['Mink','DLS','QP']);ax.set_ylabel(ylabel);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.tight_layout();fig.savefig(a.output/f'{label}_comparison.png',dpi=160);plt.close(fig)
    print(a.output)

if __name__=='__main__':main()
