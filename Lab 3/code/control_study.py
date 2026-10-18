"""Separate, bounded command-slew study; frozen layout study is never rewritten.

Development: seeds300-304. Validation:400-409. New untouched test:2000-2024.
Only Franka command slew changes; task, solver and physical success checks stay fixed.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from layout import BASELINE, Layout, source_hash
from layout_run import LayoutEpisode
from optimize_layout import append, records, physical_rank
from scene import ROOT

OUT = ROOT / 'results' / 'control_study_v1'
LAYOUTS = ROOT / 'results' / 'layout_study_v2'
SEEDS = {'development': list(range(300,305)), 'validation': list(range(400,410)),
         'test': list(range(2000,2025))}
POLICIES = [0., 1.5, 3., 6.]


def fingerprint():
    return hashlib.sha256(source_hash().encode() + Path(__file__).read_bytes()).hexdigest()


class SlewEpisode(LayoutEpisode):
    def __init__(self, robot, method, seed, layout=BASELINE, acceleration=0., record=None):
        super().__init__(robot, method, seed, layout, record)
        self.acceleration = acceleration
        self.filtered_command = np.zeros(len(self.scene.qids))
        self.peak_event = None

    def control(self, velocity, grip):
        if self.scene.robot == 'franka' and self.acceleration > 0:
            delta = self.acceleration * self.solver.dt
            self.filtered_command += np.clip(velocity - self.filtered_command, -delta, delta)
            velocity = self.filtered_command.copy()
        super().control(velocity, grip)

    def observe(self):
        old_max = self.max_joint_speed
        super().observe()
        if self.max_joint_speed > old_max:
            self.peak_event = {'time_s': float(self.data.time), 'stage': self.stage,
                               'q': self.data.qpos[self.scene.qids].tolist(),
                               'qvel': self.data.qvel[self.scene.dids].tolist(),
                               'tcp': self.data.site_xpos[self.scene.tcp].tolist()}

    def run(self):
        row = super().run()
        row['layout_source_sha256'] = row['config_sha256']
        row['config_sha256'] = fingerprint()
        row['franka_command_acceleration_rad_s2'] = self.acceleration
        row['peak_speed_event'] = self.peak_event
        row['policy'] = f'slew_{self.acceleration:g}'
        return row


def trial(robot, method, seed, obj, acceleration):
    return SlewEpisode(robot, method, seed, Layout(**obj), acceleration).run()


def rank_policies(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['layout_key'] != BASELINE.key:
            groups[r['franka_command_acceleration_rad_s2']].append(r)
    ranked = [{'acceleration': a, 'rank': list(physical_rank(g)), 'n': len(g),
               'successes': sum(r['success'] for r in g),
               'safe_successes': sum(r['constraint_safe_success'] for r in g)} for a,g in groups.items()]
    return sorted(ranked, key=lambda x: x['rank'], reverse=True)


def run(stage, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'source_sha256': fingerprint(), 'layout_source_sha256': source_hash(),
                'seeds': SEEDS, 'policies_rad_s2': POLICIES,
                'invariant': 'Selected per-robot layouts frozen. Only Franka command slew changes.',
                'ranking': 'same safe-success-first rule on selected Franka layout, no tuning on test outcomes'}
    path = OUT / 'manifest.json'
    if path.exists() and json.loads(path.read_text())['source_sha256'] != fingerprint():
        raise RuntimeError('Source changed; use new experiment version')
    path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    if stage == 'development':
        policies, robots = POLICIES, ['franka']
    elif stage == 'validation':
        ranked = json.loads((OUT/'development_ranking.json').read_text())
        policies = sorted(set([0.] + [r['acceleration'] for r in ranked[:2]]))
        robots = ['franka']
    else:
        policies = [json.loads((OUT/'selected_policy.json').read_text())['acceleration']]
        robots = ['franka', 'heal']
    resultpath = OUT / f'{stage}.jsonl'
    done = {(r['robot'],r['layout_key'],r['method'],r['seed'],r['franka_command_acceleration_rad_s2'])
            for r in records(resultpath)}
    tasks = []
    for robot in robots:
        selected = json.loads((LAYOUTS/f'{robot}_selected.json').read_text())['layout']
        for obj in [asdict(BASELINE), selected]:
            for method in ('mink','dls','qp'):
                for seed in SEEDS[stage]:
                    for a in policies:
                        effective = a if robot == 'franka' else 0.
                        key = (robot,Layout(**obj).key,method,seed,effective)
                        if key not in done:
                            tasks.append((robot,method,seed,obj,effective))
                            done.add(key)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(trial,*t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();append(resultpath,r)
            print(f'{stage} {i}/{len(tasks)} {r["robot"]} {r["policy"]} {r["layout_key"]} '
                  f'{r["method"]} seed{r["seed"]} success={r["success"]} '
                  f'safe={r["constraint_safe_success"]} maxv={r["max_joint_speed_rad_s"]:.3f}',flush=True)
    ranking = rank_policies(records(resultpath))
    (OUT/f'{stage}_ranking.json').write_text(json.dumps(ranking,indent=2),encoding='utf8')
    if stage == 'validation':
        (OUT/'selected_policy.json').write_text(json.dumps({**ranking[0],
            'selected_using': 'validation seeds400-409 only', 'source_sha256':fingerprint()},indent=2),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','validation','test'],default='development')
    p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.stage,a.workers)
