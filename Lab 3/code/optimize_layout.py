"""Resumeable independent layout study. Search results are not final-test claims."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import itertools
import json
import platform
from pathlib import Path
import time
import mujoco
import numpy as np
from ik import error_jacobian, solve_pose, target_rotation
from layout import BASELINE, Layout, TRAY_XY, build_layout, grasp_yaw, sample_pose, source_hash
from layout_run import LayoutEpisode
from scene import ROOT

OUT = ROOT / 'results' / 'layout_study_v2'
METHODS = ('mink', 'dls', 'qp')
SEEDS = {'search': list(range(100, 105)), 'validation': list(range(200, 210)),
         'test': list(range(1000, 1025))}
GRID = {'height': [.15, .20, .25, .30], 'center_x': [.38, .48, .58],
        'center_y': [-.10, 0., .10], 'yaw_deg': [0., 30., 60., 90.]}


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(row) + '\n')


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def blocked(scene, data):
    """Penetrating moving-arm/environment contacts; expected base support excluded."""
    m = scene.model
    names = ('floor', 'table', 'tray_bottom', *(f'tray_wall{k}' for k in range(4)))
    obstacles = {m.geom(n).id for n in names}
    for c in data.contact:
        if c.dist >= -1e-4:
            continue
        a, b = int(c.geom1), int(c.geom2)
        other = b if a in obstacles else a if b in obstacles else None
        if other is None:
            continue
        body = int(m.geom_bodyid[other])
        if m.body_weldid[body] != 0 and m.body(body).name != 'cube':
            return True
    return False


def screen_one(robot, obj):
    start = time.perf_counter()
    layout = Layout(**obj)
    s = build_layout(robot, layout)
    initial = s.reset(*sample_pose(100, layout)[1])
    row = {'robot': robot, 'layout': obj, 'layout_key': layout.key,
           'source_sha256': source_hash(), 'initial_clear': not blocked(s, initial),
           'reachable': 0, 'endpoint_clear': 0, 'probes': [], 'count': 0}
    if not row['initial_clear']:
        row.update({'reject_reason': 'initial_robot_obstacle_penetration',
                    'reach_fraction': 0., 'clear_fraction': 0.,
                    'minimum_joint_margin_fraction': 0., 'minimum_translation_sigma_m': 0.})
        return row
    tasks = []
    for seed in SEEDS['search'][:3]:
        local, world = sample_pose(seed, layout)
        yaw = grasp_yaw(robot, world[2])
        for offset, stage in ((.14, 'pregrasp'), (.025, 'grasp'), (.18, 'lift')):
            tasks.append((seed, stage, layout.world(local[0], local[1], offset), yaw))
    tray_yaw = grasp_yaw(robot, layout.angle)
    for offset in (.18, .07):
        tasks.append((-1, 'tray', layout.world(*TRAY_XY, offset), tray_yaw))
    margins, sigmas = [], []
    last_seed = None
    guess = s.home.copy()
    for seed, stage, position, yaw in tasks:
        if seed != last_seed:
            guess = s.home.copy()
        last_seed = seed
        q, e, it = solve_pose(s, position, target_rotation(yaw), initial=guess, iterations=400)
        if q is None:
            q, e, it2 = solve_pose(s, position, target_rotation(yaw), initial=s.home, iterations=600)
            it += it2
        clear = False
        if q is not None:
            row['reachable'] += 1
            d = s.reset()
            d.qpos[s.qids] = q
            mujoco.mj_forward(s.model, d)
            clear = not blocked(s, d)
            row['endpoint_clear'] += int(clear)
            margin = np.min(np.minimum(q - s.limits[:, 0], s.limits[:, 1] - q)
                            / (s.limits[:, 1] - s.limits[:, 0]))
            _, J = error_jacobian(s, d, position, target_rotation(yaw))
            sigma = np.linalg.svd(J[:3], compute_uv=False)[-1]
            margins.append(float(margin)); sigmas.append(float(sigma))
            guess = q
        row['probes'].append({'seed': seed, 'stage': stage, 'position': position.tolist(),
                              'yaw_rad': float(yaw), 'reachable': q is not None,
                              'endpoint_clear': clear, 'iterations': it,
                              'position_error_m': float(np.linalg.norm(e[:3])),
                              'orientation_error_rad': float(np.linalg.norm(e[3:]))})
    row['count'] = len(tasks)
    row['reach_fraction'] = row['reachable'] / len(tasks)
    row['clear_fraction'] = row['endpoint_clear'] / len(tasks)
    row['minimum_joint_margin_fraction'] = min(margins, default=0.)
    row['minimum_translation_sigma_m'] = min(sigmas, default=0.)
    row['wall_time_s'] = time.perf_counter() - start
    return row


def screen_rank(row):
    return (row['clear_fraction'], row['reach_fraction'], row['minimum_joint_margin_fraction'],
            row['minimum_translation_sigma_m'])


def physical_rank(rows):
    """Lexicographic; failures cannot win simply by finishing early."""
    if not rows:
        return (-1., -1., -1., -1., -1e30, -1e30)
    groups = [[r for r in rows if r['method'] == m] for m in METHODS]
    if any(not g for g in groups):
        return (-1., -1., -1., -1., -1e30, -1e30)
    safe = [sum(bool(r['constraint_safe_success']) for r in g) / len(g) for g in groups]
    success = [sum(bool(r['success']) for r in g) / len(g) for g in groups]
    completed = [r for r in rows if r['success']]
    # Worst method comes before average; only successful runs inform efficiency.
    return (min(safe), float(np.mean(safe)), min(success), float(np.mean(success)),
            -float(np.mean([r['sim_time_s'] for r in completed])) if completed else -1e30,
            -float(np.mean([r['joint_travel_rad'] for r in completed])) if completed else -1e30)


def physical_one(robot, obj, method, seed):
    return LayoutEpisode(robot, method, seed, Layout(**obj)).run()


def ensure_manifest():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'source_sha256': source_hash(), 'grid': GRID,
                'seeds': SEEDS, 'methods': METHODS, 'robots_optimized_independently': True,
                'fixed_table_size_m': [.50, .60], 'fixed_robot_home_policy': 'baseline joint configuration',
                'controllers': 'unchanged from preserved run.py',
                'pose_screen': '3 search seeds x3 pick heights +2 tray heights; endpoint-only',
                'ranking': ['worst_method_safe_success', 'mean_safe_success',
                            'worst_method_settled_success', 'mean_settled_success',
                            'successful_sim_time', 'successful_joint_travel'],
                'limitations': ['No tilt, no shared compromise, finite bounded search, not a global optimum.',
                                'Safety screen covers robot/environment, not exhaustive self/swept collision proofs.',
                                'Measured 2ms observations differ from historical 10ms metric samples.'],
                'python': platform.python_version(), 'mujoco': mujoco.__version__}
    path = OUT / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['source_sha256'] != manifest['source_sha256']:
            raise RuntimeError('Source changed since this experiment began; choose a new --output directory')
    else:
        path.write_text(json.dumps(manifest, indent=2), encoding='utf8')


def run_screen(robot, workers):
    path = OUT / f'{robot}_screen.jsonl'
    done = {r['layout_key'] for r in records(path)}
    grid = [Layout(*values) for values in itertools.product(*GRID.values())]
    pending = [x for x in grid if x.key not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(screen_one, robot, asdict(x)): x.key for x in pending}
        for f in as_completed(futures):
            result = f.result()
            append(path, result)
            print(f'{robot} screen {len(done)+1}/144 {result["layout_key"]} '
                  f'clear={result["clear_fraction"]:.2f}', flush=True)
            done.add(result['layout_key'])
    rows = sorted(records(path), key=screen_rank, reverse=True)
    # The baseline is always included, even if it is not in the top shortlist.
    selected = [r['layout'] for r in rows[:2]]
    # A margin-only top eight can miss an entire yaw/height family. Reserve
    # representation for every in-plane angle, then unseen heights.
    for angle in GRID['yaw_deg']:
        candidate = next(r['layout'] for r in rows if r['layout']['yaw_deg'] == angle)
        if candidate not in selected:
            selected.append(candidate)
    for height in GRID['height']:
        candidate = next(r['layout'] for r in rows if r['layout']['height'] == height)
        if candidate not in selected:
            selected.append(candidate)
    for r in rows:
        if len(selected) >= 8:
            break
        if r['layout'] not in selected:
            selected.append(r['layout'])
    if asdict(BASELINE) not in selected:
        selected.append(asdict(BASELINE))
    (OUT / f'{robot}_shortlist.json').write_text(json.dumps(selected, indent=2), encoding='utf8')


def run_physical(robot, stage, workers):
    if stage == 'search':
        layouts = json.loads((OUT / f'{robot}_shortlist.json').read_text())
    elif stage == 'refine':
        best = json.loads((OUT / f'{robot}_search_ranking.json').read_text())[0]['layout']
        layouts = [best]
        for parameter, delta in [('height', .025), ('center_x', .03), ('center_y', .03), ('yaw_deg', 15.)]:
            for sign in (-1, 1):
                candidate = dict(best)
                candidate[parameter] += sign * delta
                layouts.append(candidate)
    elif stage == 'validation':
        refined = OUT / f'{robot}_refine_ranking.json'
        layouts = json.loads(refined.read_text() if refined.exists() else (OUT / f'{robot}_search_ranking.json').read_text())[:3]
        layouts = [x['layout'] for x in layouts]
        original_best = json.loads((OUT / f'{robot}_search_ranking.json').read_text())[0]['layout']
        if original_best not in layouts:
            layouts.append(original_best)
        if asdict(BASELINE) not in layouts:
            layouts.append(asdict(BASELINE))
    else:
        chosen = json.loads((OUT / f'{robot}_selected.json').read_text())['layout']
        layouts = [asdict(BASELINE)] if chosen == asdict(BASELINE) else [chosen, asdict(BASELINE)]
    path = OUT / f'{robot}_{stage}.jsonl'
    existing = records(path)
    done = {(r['layout_key'], r['method'], r['seed']) for r in existing}
    seeds = SEEDS['search' if stage == 'refine' else stage]
    tasks = [(obj, method, seed) for obj in layouts for method in METHODS for seed in seeds
             if (Layout(**obj).key, method, seed) not in done]
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(physical_one, robot, obj, method, seed) for obj, method, seed in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            append(path, r)
            print(f'{robot} {stage} {i}/{total} {r["layout_key"]} {r["method"]} '
                  f'seed{r["seed"]} success={r["success"]} safe={r["constraint_safe_success"]}', flush=True)
    grouped = defaultdict(list)
    for r in records(path):
        grouped[r['layout_key']].append(r)
    rank = [{'layout': group[0]['layout'], 'layout_key': key, 'episodes': len(group),
             'rank': list(physical_rank(group)),
             'successes': sum(r['success'] for r in group),
             'safe_successes': sum(r['constraint_safe_success'] for r in group)}
            for key, group in grouped.items()]
    rank.sort(key=lambda r: r['rank'], reverse=True)
    (OUT / f'{robot}_{stage}_ranking.json').write_text(json.dumps(rank, indent=2), encoding='utf8')
    if stage == 'validation':
        (OUT / f'{robot}_selected.json').write_text(json.dumps({'robot': robot, 'layout': rank[0]['layout'],
            'selection_source': 'validation seeds only; final test untouched',
            'source_sha256': source_hash()}, indent=2), encoding='utf8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--robot', choices=['franka', 'heal', 'both'], default='both')
    p.add_argument('--stage', choices=['screen', 'search', 'refine', 'validation', 'test'], default='screen')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--output', type=Path, default=OUT)
    args = p.parse_args()
    OUT = args.output
    if args.workers < 1:
        p.error('workers must be positive')
    ensure_manifest()
    for robot in ('franka', 'heal') if args.robot == 'both' else (args.robot,):
        run_screen(robot, args.workers) if args.stage == 'screen' else run_physical(robot, args.stage, args.workers)
