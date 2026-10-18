# Robot assets

- `franka/`: supplied Franka scene, arm model and mesh assets; its included license
  is retained.
- `single_arm_heal_effort_actuation_rs_mj.xml` and `heal_meshes/`: supplied Heal
  effort-actuated arm and the seven meshes referenced by that model.
- `robotiq_2f85/`: MuJoCo Menagerie Robotiq 2F-85, with its license and source
  revision in `PROVENANCE.json`. This replaces the custom Heal gripper whose
  meshes were absent from the supplied files.

The Python scene builders add the table, free cube, tray and TCP site. They also
place the Franka base at the analysis origin and make the supplied Heal visual
material opaque. These scene choices are part of the reported simulation setup.
All mesh files are local; model loading does not fetch assets from the internet.
