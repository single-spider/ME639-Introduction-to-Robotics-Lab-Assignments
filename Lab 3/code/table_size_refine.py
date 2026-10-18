"""Stronger endpoint restarts using saved workspace branches, not physical outcomes.

Preserves the initial two-start validation and its trials. Reuses identical
geometry/seed/method episodes instead of counting duplicates as new evidence.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from layout import grasp_yaw
from optimize_layout import blocked
from table_size_study import OUT, HEIGHTS, orientations, probe
from table_size_validate import build_sized, geometry, physical_one, fingerprint as initial_fingerprint

CACHE={}


def fingerprint():
    return hashlib.sha256(initial_fingerprint().encode()+Path(__file__).read_bytes()).hexdigest()


def warm_probe(scene,data,position,yaw,guesses):
    key=scene.robot
    if key not in CACHE:CACHE[key]=[dict(np.load(OUT/f'{key}_h{h:.2f}.npz')) for h in HEIGHTS]
    choices=[]
    for grid in CACHE[key]:
        k=int(np.argmin(abs(np.angle(np.exp(1j*(grid['yaws']-yaw))))))
        good=grid['ok'][:,:,k]
        iy,ix=np.where(good)
        if not len(ix):continue
        dist=(grid['x'][ix]-position[0])**2+(grid['y'][iy]-position[1])**2+(float(grid['height'])+.025-position[2])**2
        order=np.argsort(dist)[:2]
        choices.extend((float(dist[i]),grid['q'][iy[i],ix[i],k]) for i in order)
    nearest=[q for _,q in sorted(choices,key=lambda item:item[0])[:3]]
    return probe(scene,data,position,yaw,[*nearest,*guesses],iterations=160)


def check(candidate):
    start=time.perf_counter();robot=candidate['robot']
    scene,layout=build_sized(robot,candidate,empty_surface=True);data=scene.reset()
    all_yaws,full,task=orientations(robot)
    yaws=all_yaws[full if candidate['policy']=='full_yaw_8' else task]
    rows=[]
    for offset in (.025,.14,.18):
        for u in np.linspace(-candidate['length_m']/2+.025,candidate['length_m']/2-.025,3):
            for v in np.linspace(-candidate['width_m']/2+.025,candidate['width_m']/2-.025,3):
                for yaw in yaws:
                    q,e,n=warm_probe(scene,data,layout.world(u,v,offset),yaw,[scene.home,(scene.limits[:,0]+scene.limits[:,1])/2])
                    rows.append({'u':float(u),'v':float(v),'offset_m':offset,'yaw_rad':float(yaw),'clear':q is not None,
                                 'position_error_m':float(np.linalg.norm(e[:3])),'rotation_error_rad':float(np.linalg.norm(e[3:])),
                                 'iterations':n})
    tray,us,vs=geometry(candidate)
    actual,layout=build_sized(robot,candidate);data=actual.reset();initial_clear=not blocked(actual,data)
    tasks=[]
    for u in us:
        for v in vs:
            for yaw in all_yaws[task]:
                guess=actual.home
                for offset in (.14,.025,.18):
                    q,e,n=warm_probe(actual,data,layout.world(u,v,offset),yaw,[guess,actual.home])
                    tasks.append({'u':u,'v':v,'offset_m':offset,'yaw_rad':float(yaw),'clear':q is not None})
                    if q is not None:guess=q
    for offset in (.07,.18):
        q,e,n=warm_probe(actual,data,layout.world(*tray,offset),grasp_yaw(robot,layout.angle),[actual.home])
        tasks.append({'stage':'tray','offset_m':offset,'clear':q is not None})
    r={'candidate':candidate,'source_sha256':fingerprint(),'initial_clear':initial_clear,
       'footprint_probes':rows,'footprint_all_three_heights_clear':all(x['clear'] for x in rows),
       'footprint_grasp_clear':all(x['clear'] for x in rows if x['offset_m']==.025),
       'task_probes':tasks,'task_all_clear':all(x['clear'] for x in tasks),
       'packing':{'tray_local_xy_m':tray.tolist(),'pick_u_m':list(us),'pick_v_m':list(vs)},
       'wall_time_s':time.perf_counter()-start,'restart_policy':'three nearest saved workspace branches, then home and midpoint'}
    r['physical_eligible']=r['initial_clear'] and r['task_all_clear'] and r['footprint_all_three_heights_clear']
    return r


def episode_key(c,m,s):
    return (c['robot'],*(round(c[k],6) for k in ('height','center_x','center_y','yaw_deg','length_m','width_m')),m,s)


def main():
    initial=json.loads((OUT/'endpoint_validation.json').read_text())
    candidates=[r['candidate'] for r in initial]
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(check,c) for c in candidates]
        for f in as_completed(futures):
            r=f.result();results.append(r)
            print(r['candidate']['robot'],r['candidate']['policy'],r['candidate']['height'],'eligible',r['physical_eligible'],flush=True)
    (OUT/'refined_endpoint_validation.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    chosen=[]
    for robot in ('franka','heal'):
        for policy in ('full_yaw_8','task_policy_5'):
            good=[r['candidate'] for r in results if r['physical_eligible'] and r['candidate']['robot']==robot and r['candidate']['policy']==policy]
            if good:chosen.append(max(good,key=lambda c:(c['area_m2'],-c['height'],-c['yaw_deg'])))
    selection={'candidates':chosen,'source_sha256':fingerprint(),
               'selection':'largest area among the preselected top-three footprints passing stronger endpoint restarts; ties prefer lower height/yaw',
               'no_physical_outcomes_used':True,'development_seeds':list(range(3000,3005)),'test_seeds':list(range(3100,3125))}
    (OUT/'refined_selected.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
    # Cache identical physics, independent of the workspace policy label.
    cached={}
    for line in (OUT/'physical.jsonl').read_text().splitlines():
        r=json.loads(line);cached[episode_key(r['candidate'],r['method'],r['seed'])]=r
    jobs={}
    for c in chosen:
        for m in ('mink','dls','qp'):
            for s in [*range(3000,3005),*range(3100,3125)]:
                key=episode_key(c,m,s)
                if key not in cached:jobs.setdefault(key,(c,m,s))
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(physical_one,*args) for args in jobs.values()]
        for f in as_completed(futures):
            r=f.result();cached[episode_key(r['candidate'],r['method'],r['seed'])]=r
            print('new physical',r['robot'],r['method'],r['seed'],r['success'],flush=True)
    output=[]
    for c in chosen:
        for m in ('mink','dls','qp'):
            for s in [*range(3000,3005),*range(3100,3125)]:
                old=cached[episode_key(c,m,s)]
                output.append({**old,'candidate':c,'refinement_source_sha256':fingerprint(),
                               'shared_geometry_key':list(episode_key(c,m,s)),
                               'evidence_note':'Same geometry/method/seed is one physical episode even when shared across workspace policies.'})
    (OUT/'refined_physical.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output),encoding='utf8')
    print('Policy-labelled rows',len(output),'unique geometry episodes',len({episode_key(r['candidate'],r['method'],r['seed']) for r in output}),flush=True)


if __name__=='__main__':main()
