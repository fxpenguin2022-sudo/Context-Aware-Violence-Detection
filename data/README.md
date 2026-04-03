# Data Layout

Datasets are not redistributed in this repository.

Default configs assume repository-local directories such as:

```text
data/
  rwf2000/
  rwf2000_pose_hq/
  hockey_fight/
  hockey_fight_pose_hq/
  violent_flow/
  violent_flow_pose_hq/
  rlvs/
  rlvs_pose_hq/
```

Video directories are expected to contain split/class subfolders, for example `train/Fight/*.avi` and `val/Normal/*.avi`. Pose directories are expected to mirror that structure with `.npz` files.
