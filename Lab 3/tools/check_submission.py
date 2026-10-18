"""Run concise package tests and optionally check offscreen rendering."""
import argparse
import json
from pathlib import Path
import platform
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--render',action='store_true');a=p.parse_args()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():return 1
    if a.render:
        import mujoco
        from run_lab import make_episode
        from PIL import Image
        output=ROOT/'runs';output.mkdir(exist_ok=True)
        for robot in ('franka','heal'):
            ep=make_episode(robot,'dls',3100)
            with mujoco.Renderer(ep.scene.model,width=640,height=480) as renderer:
                cam=mujoco.MjvCamera();cam.lookat[:]=[.25,0,ep.layout.height+.15]
                cam.distance=1.8;cam.azimuth=135;cam.elevation=-25
                renderer.update_scene(ep.data,camera=cam)
                Image.fromarray(renderer.render()).save(output/f'{robot}_render.png')
    print(json.dumps({'platform':platform.platform(),'python':platform.python_version(),'tests_passed':result.testsRun,'render_checked':a.render}))
    return 0

if __name__=='__main__':raise SystemExit(main())
