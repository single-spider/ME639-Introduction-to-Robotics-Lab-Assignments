"""Small submission tests: mathematical behavior, packaged assets and geometry."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from run_lab import make_episode


class PackageTests(unittest.TestCase):
    def test_both_table_designs_load_for_both_arms(self):
        for robot in ('franka','heal'):
            for layout in ('sizing','placement'):
                with self.subTest(robot=robot,layout=layout):
                    ep=make_episode(robot,'dls',3100 if layout=='sizing' else 2000,layout)
                    self.assertTrue(np.isfinite(ep.data.qpos).all())
                    self.assertEqual(len(ep.scene.qids),7 if robot=='franka' else 6)

    def test_historical_comparison_is_complete(self):
        import json
        rows=json.loads((ROOT/'results/metrics.json').read_text())
        self.assertEqual(len(rows),12)
        self.assertTrue(all(r['n']==25 for r in rows))
        for filename in ('sizing_episodes.jsonl','placement_episodes.jsonl'):
            episodes=[json.loads(s) for s in (ROOT/'results'/filename).read_text().splitlines()]
            self.assertEqual(len(episodes),150)
            self.assertEqual(len({(r['robot'],r['method'],r['seed']) for r in episodes}),150)

if __name__=='__main__':unittest.main()
