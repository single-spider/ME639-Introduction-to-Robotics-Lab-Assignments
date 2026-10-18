import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
import mujoco
from scene import build
from ik import error_jacobian,dls_velocity,qp_velocity
from scipy.spatial.transform import Rotation


class FoundationTests(unittest.TestCase):
    def test_qp_matches_dls_without_active_bounds(self):
        rng=np.random.default_rng(41)
        for n in (6,7):
            J=rng.normal(size=(6,n)); desired=rng.normal(size=6)
            np.testing.assert_allclose(qp_velocity(J,desired,-100*np.ones(n),100*np.ones(n)),
                                       dls_velocity(J,desired),atol=1e-8)

    def test_qp_respects_bounds(self):
        v=qp_velocity(np.eye(6),10*np.ones(6),-0.2*np.ones(6),0.3*np.ones(6))
        self.assertTrue(np.all(v<=0.3+1e-10));self.assertTrue(np.all(v>=-0.2-1e-10))

    def test_both_models_and_world_jacobians(self):
        for robot in ('franka','heal'):
            with self.subTest(robot=robot):
                s=build(robot); d=s.reset(); R=d.site_xmat[s.tcp].reshape(3,3).copy()
                p=d.site_xpos[s.tcp].copy()
                e,J=error_jacobian(s,d,p,R)
                self.assertLess(np.linalg.norm(e),1e-10)
                for k,qi in enumerate(s.qids):
                    old=d.qpos[qi];d.qpos[qi]+=1e-6;mujoco.mj_forward(s.model,d)
                    dp=(d.site_xpos[s.tcp]-p)/1e-6
                    dr=Rotation.from_matrix(d.site_xmat[s.tcp].reshape(3,3)@R.T).as_rotvec()/1e-6
                    np.testing.assert_allclose(np.r_[dp,dr],J[:,k],atol=2e-5)
                    d.qpos[qi]=old;mujoco.mj_forward(s.model,d)
                # Cube must be dynamic, not welded or mocap controlled.
                self.assertEqual(s.model.jnt_type[s.model.joint('cube_free').id],mujoco.mjtJoint.mjJNT_FREE)
                self.assertEqual(s.model.body_mocapid[s.model.body('cube').id],-1)


if __name__=='__main__':
    unittest.main(verbosity=2)
